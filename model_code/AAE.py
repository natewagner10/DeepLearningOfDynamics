# New College of Florida
# Amanda Bucklin, Rosa Gradilla, Justin Tienken-Harder, Dominic Ventura, Nate Wagner

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import autograd
import torch.nn as nn
import os
from PIL import Image
from torchvision.utils import save_image
import argparse

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--EPS', default=1e-15, type=float, help='EPS')
    parser.add_argument('--dim_z', default=40, type=int, help='latent dimension of autoencoder')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')    
    parser.add_argument('--epochs', default=150, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--path_to_save_dir', default='/home/nwagner/critter_aae_models/', type=str, help='where to save trained models & test images')
    parser.add_argument('--path_to_images', default='/home/rgradilla/critter_images/critter/', type=str, help='path to critter images') 
    parser.add_argument('--width', default=160, type=int, help='image resize width')
    parser.add_argument('--height', default=90, type=int, help='image resize height')                             
    parser.set_defaults(feature=True)
    return parser.parse_args()    
    
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else: 
    device = torch.device("cpu")
    print("Running on the CPU")

def buildTrainVal():
    critter_images = []
    for img in os.listdir(args.path_to_images):
        im = Image.open(args.path_to_images + str(img))
        im = im.resize((args.width, args.height))
        im_arr = np.array(im).flatten() / 255
        critter_images.append(torch.from_numpy(im_arr))  
            
    train_set, val_set = torch.utils.data.random_split(critter_images, [round(len(critter_images) * 0.80), round(len(critter_images) * 0.20)])
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    
    return train_loader, val_loader

class Encoder(nn.Module):
    def __init__(self, dim_input , dim_z):
        super(Encoder, self).__init__()
        self.dim_input = dim_input
        self.dim_z = dim_z
        self.network = []
        self.network.extend([
            nn.Linear(self.dim_input, 10000),
            nn.ReLU(),
            nn.Linear(10000, 4000),
            nn.ReLU(),
            nn.Linear(4000, 1000),
            nn.ReLU(),            
            nn.Linear(1000, 512),
            nn.ReLU(), 
            nn.Linear(512, 128),
            nn.ReLU(),             
            nn.Linear(128, self.dim_z),
            nn.ReLU(),
        ])
        self.network = nn.Sequential(*self.network)
    def forward(self, x):
        z = self.network(x)
        return z

class Decoder(nn.Module):
    def __init__(self, dim_input , dim_z):
        super(Decoder, self).__init__()
        self.dim_input = dim_input
        self.dim_z = dim_z
        self.network = []
        self.network.extend([
            nn.Linear(self.dim_z, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),            
            nn.Linear(512, 1000),
            nn.ReLU(),
            nn.Linear(1000, 4000),
            nn.ReLU(), 
            nn.Linear(4000, 10000),
            nn.ReLU(),             
            nn.Linear(10000, self.dim_input),
            nn.ReLU(),
        ])
        self.network = nn.Sequential(*self.network)
    def forward(self, z):
        x_recon = self.network(z)
        return x_recon

class Discriminator(nn.Module):
    def __init__(self, dim_z , dim_h):
        super(Discriminator,self).__init__()
        self.dim_z = dim_z
        self.dim_h = dim_h
        self.network = []
        self.network.extend([
            nn.Linear(self.dim_z, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU(),            
            nn.Linear(5,1),
            nn.Sigmoid(),
        ])
        self.network = nn.Sequential(*self.network)

    def forward(self, z):
        disc = self.network(z)
        return disc

def aae_loss(encoder, decoder, Disc, dataloader, optim_encoder, optim_decoder, optim_D, train, optim_encoder_reg):
    ae_criterion = nn.MSELoss()
    total_rec_loss = 0
    total_disc_loss = 0
    total_gen_loss = 0
    if train:
        encoder.train()
        decoder.train()
        Disc.train()
    else:
        encoder.eval()
        decoder.eval()
        Disc.eval()

    for i, data in enumerate(dataloader):
        """ Reconstruction loss """
        for p in Disc.parameters():
            p.requires_grad = False

        real_data_v = autograd.Variable(data).to(device)
        real_data_v = real_data_v.view(-1, args.width*args.height)
        encoding = encoder(real_data_v.float())
        fake = decoder(encoding)
        ae_loss = ae_criterion(fake.float(), real_data_v.float())
        total_rec_loss += ae_loss.item()
        if train:
            optim_encoder.zero_grad()
            optim_decoder.zero_grad()
            ae_loss.backward()
            optim_encoder.step()
            optim_decoder.step()

        """ Discriminator loss """
        encoder.eval()
        z_real_gauss = autograd.Variable(torch.randn(data.size()[0], args.dim_z) * 5.).to(device)
        D_real_gauss = Disc(z_real_gauss)

        z_fake_gauss = encoder(real_data_v.float())
        D_fake_gauss = Disc(z_fake_gauss)
        
        D_loss = -torch.mean(torch.log(D_real_gauss.float() + args.EPS) + torch.log(1 - D_fake_gauss.float() + args.EPS))
        total_disc_loss += D_loss.item()

        if train:
            optim_D.zero_grad()
            D_loss.backward()
            optim_D.step()

        """ Generator loss """
        if train:
            encoder.train()
        else:
            encoder.eval()
        z_fake_gauss = encoder(real_data_v.float())
        D_fake_gauss = Disc(z_fake_gauss)

        G_loss = -torch.mean(torch.log(D_fake_gauss.float() + args.EPS))
        total_gen_loss += G_loss.item()

        if train:
            optim_encoder_reg.zero_grad()
            G_loss.backward()
            optim_encoder_reg.step()

        if i % 100 == 0:
            print ('\n Step [%d], recon_loss: %.4f, discriminator_loss :%.4f , generator_loss:%.4f'
                    %(i, ae_loss.item(), D_loss.item(), G_loss.item()))
            
    return total_rec_loss, total_disc_loss, total_gen_loss

dtype = torch.cuda.FloatTensor

def reconstruct(encoder, decoder, device, dtype, loader_val):
    encoder.eval()
    decoder.eval()
    X_val = next(iter(loader_val))
    X_val = X_val.view(-1,args.width*args.height)
    X_val = X_val.type(dtype)
    X_val = X_val.to(device)
    z_val = encoder(X_val)
    X_hat_val = decoder(z_val)

    X_val = X_val[:10].cpu().view(10 * args.height, args.width)
    X_hat_val = X_hat_val[:10].cpu().view(10 * args.height, args.width)
    comparison = torch.cat((X_val, X_hat_val), 1).view(10 * args.height, 2 * args.width)
    return comparison

def train(train_loader, val_loader):
    encoder = Encoder(args.width*args.height,args.dim_z).to(device)
    decoder = Decoder(args.width*args.height,args.dim_z).to(device)
    Disc = Discriminator(args.dim_z,500).to(device)

    # encode/decode optimizers
    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    optim_decoder = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    optim_D = torch.optim.Adam(Disc.parameters(), lr=args.lr)
    optim_encoder_reg = torch.optim.Adam(encoder.parameters(), lr=0.000001)
    
    schedulerDisc = torch.optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.99)
    schedulerD = torch.optim.lr_scheduler.ExponentialLR(optim_decoder, gamma=0.99)
    schedulerE = torch.optim.lr_scheduler.ExponentialLR(optim_encoder, gamma=0.99)
    
    train_rec_loss = []
    train_disc_loss = []
    train_gen_loss = []
    val_rec_loss = []
    val_disc_loss = []
    val_gen_loss = []
    for epoch in range(args.epochs):
        l1,l2,l3 = aae_loss(encoder, decoder, Disc, train_loader, optim_encoder, optim_decoder, optim_D, True, optim_encoder_reg)
        print('\n epoch:{} ---- training loss:{}'.format(epoch, l1))
        train_rec_loss.append(l1)
        train_disc_loss.append(l2)
        train_gen_loss.append(l3)
    
        l1, l2, l3 = aae_loss(encoder, decoder, Disc, val_loader, optim_encoder, optim_decoder, optim_D, False, optim_encoder_reg)
        print('\n epoch:{} ---- validation loss loss:{}'.format(epoch, l1))
        val_rec_loss.append(l1)
        val_disc_loss.append(l2)
        val_gen_loss.append(l3)    
    
        if epoch % 10 == 0:
            comparison = reconstruct(encoder,decoder, device, dtype, val_loader)
            save_image(comparison, args.path_to_save_dir + 'AAE_test_comparison_z_{}_epch_{}.png'.format(args.dim_z,epoch))
    
    plt.rcParams['figure.figsize'] = 5, 5
    plt.plot(np.arange(len(train_rec_loss)), train_rec_loss/np.max(train_rec_loss), label='train_rec')
    plt.plot(np.arange(len(train_rec_loss)), train_disc_loss/np.max(train_disc_loss), label='train_disc')
    plt.plot(np.arange(len(train_rec_loss)), train_gen_loss/np.max(train_gen_loss), label='train_gen')
    plt.plot(np.arange(len(val_rec_loss)), val_rec_loss/np.max(val_rec_loss), label='val_rec')
    plt.plot(np.arange(len(val_rec_loss)), val_disc_loss/np.max(val_disc_loss), label='val_disc')
    plt.plot(np.arange(len(val_rec_loss)), val_gen_loss/np.max(val_gen_loss), label='val_rec')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(args.path_to_save_dir + 'training_' + str(args.epochs) + 'epochs.png')
    
    return encoder, decoder, Disc

if __name__ == "__main__":
    args = get_args()
    train_loader, val_loader = buildTrainVal()
    encoder, decoder, Disc = train(train_loader, val_loader)

    # save the models
    torch.save(encoder.state_dict(),args.path_to_save_dir + 'encoder_z'+ str(args.dim_z)+'_epch'+str(args.epochs)+'.pt')
    torch.save(decoder.state_dict(),args.path_to_save_dir +'decoder_z'+ str(args.dim_z)+'_epch'+str(args.epochs)+'.pt')
    torch.save(Disc.state_dict(),args.path_to_save_dir +'disc_z'+ str(args.dim_z)+'_epch'+str(args.epochs)+'.pt')
