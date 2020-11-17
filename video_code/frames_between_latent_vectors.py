import pandas as pd
import numpy as np
import torch
from torch import autograd
import torch.nn as nn
eps = np.finfo(float).eps
import os
from PIL import Image
from torchvision.utils import save_image
from PIL import Image
from matplotlib import cm
import os
import moviepy.video.io.ImageSequenceClip
import cv2
import numpy as np
import glob
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os.path

encodings = pd.read_csv('C:/Users/Manda/Documents/NCF/Practical_Data_Science/critter_encodings.csv')
encodings = encodings.drop(['file', 'frame_num'], axis = 1)
vecs = encodings.values.tolist()
vecs = np.array(vecs)
first_frame = vecs[0]
next_frame = vecs[30]
ratios = np.linspace(0,1,30) 
vectors = list()
for ratio in ratios:
    v = (1.0-ratio) * first_frame + ratio * next_frame
    vectors.append(v)
print(torch.tensor(vectors))

val_loader = torch.tensor(vectors)
######## has weights that he already trained on

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else: 
    device = torch.device("cpu")
    print("Running on the CPU")
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
dim_z = 20
decoder = Decoder(14400,dim_z).to(device)
decoder.load_state_dict(torch.load('C:/Users/Manda/Documents/NCF/Practical_Data_Science/decoder_z20_epch140.pt'))

#val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True, num_workers=1)
dtype = torch.cuda.FloatTensor

list_of_arrays = []
def reconstruct(decoder, device, dtype, val_loader):
    #encoder.eval()
    decoder.eval()
    for val in val_loader:
        #X_val = next(iter(val))
        X_val = val.type(dtype)
        X_val = X_val.to(device)
        X_hat_val = decoder(X_val)
        X_hat_val = X_hat_val.cpu().view(10*9, 160)
        X_hat_val = X_hat_val.detach().numpy()
        list_of_arrays.append(X_hat_val)
    return list_of_arrays    


reconstruct(decoder, device, dtype, val_loader)
cnt = 0
for item in list_of_arrays:
    #print(item)  
    im = Image.fromarray(np.uint8(cm.gray(item)*255))
    new_img = im.resize((480,270))
    new_img = new_img.convert('RGB')
    if cnt in range(10):
        new_img.save("C:/Users/Manda/Documents/NCF/Practical_Data_Science/critters/latent_images/0"+str(cnt)+".jpg") 
    else:
        new_img.save("C:/Users/Manda/Documents/NCF/Practical_Data_Science/critters/latent_images/"+str(cnt)+".jpg")
    cnt +=1


image_folder='C:/Users/Manda/Documents/NCF/Practical_Data_Science/critters/latent_images/'
fps=5
image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".jpg")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('C:/Users/Manda/Documents/NCF/Practical_Data_Science/critters/latent_images/my_video_racoon.mp4')
