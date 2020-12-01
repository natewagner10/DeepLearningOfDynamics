# New College of Florida
# Amanda Bucklin, Rosa Gradilla, Justin Tienken-Harder, Dominic Ventura, Nate Wagner

import numpy as np
import torch
from torch.distributions import Normal, Independent
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from PIL import Image
import moviepy.video.io.ImageSequenceClip
from moviepy.editor import VideoFileClip, concatenate_videoclips

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
    
class DPN(nn.Module):
    """AGENTS"""
    def __init__(self, alpha, input_size):
        super(DPN, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, input_size)
        self.fc41 = nn.Linear(input_size, input_size)
        self.fc42 = nn.Linear(input_size, input_size)
        self.fc51 = nn.Linear(input_size, input_size)
        self.fc52 = nn.Linear(input_size, input_size)
        self.fc61 = nn.Linear(input_size, input_size)
        self.fc62 = nn.Linear(input_size, input_size)
        self.optimizer = optim.SGD(self.parameters(), lr=alpha, weight_decay = 5)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self, x):
        residual = x
        h = F.tanh(self.fc1(x)) + x
        h = F.tanh(self.fc2(h)) + h 
        h = F.tanh(self.fc3(h)) + h
        #Mu side
        mu = F.tanh(self.fc41(h)) + h
        mu = F.tanh(self.fc51(mu)) + h
        mu = F.leaky_relu(self.fc61(mu)) + residual
        #logsigma side
        logvar = F.leaky_relu(self.fc42(h))
        logvar = F.leaky_relu(self.fc52(logvar))
        logvar = F.elu(self.fc62(logvar))
        return mu, logvar

def generate_latent_future(dpn, decoder, first_ls_vector, seconds, default_frame_skip = 10, interpolate = True, clip = True, is_log_sigma = False, save_video_path='/Users/natewagner/Documents/ArmyResearchLab/new_movie4.mp4'):
    '''
    Use this function to generate the predicted the future latent vectors from a trained policy network.

    Assumes output of the network is the actual next frame, and not the epsilon perturbation.

    Parameters:
    ---------------
        dpn: Trained policy network (if you trained an Agent class, this would be the Agent.actor object).
        
        decoder: Trained decoder network

        first_ls_vector: The first latent space vector to predict into the future. Must be a torch.Tensor object.

        seconds: Number of seconds to predict in to the future.

        default_frame_skip: During training it's useful to skip some number of frames. We will utilize linear interpolation to get the vectors between the predicted next-step vector.

        interpolate: When true will automatically do the interpolation filling in the missing frames.

        clip: Due to the non-normal distribution of our embedding, there are three elements of a LS vector that is always 0. During policy prediciton, these elements tend to naturally drift (potentially causing frame tearing). This clips those elements to 0.

        is_log_sigma: Some DPN output sigma always positive (either through an ELU activation) or premptively exponentiating the log output.
        
        save_video_path: Where to save mp4 video.

    Returns:
    ---------        
        mp4 video containing the predicted frames.
    '''
    
    def buildVideo(outputs,save_video_path):
        new_images = []
        for frame in outputs:
            decoded = decoder(frame).detach().numpy()*255
            decoded.resize((90,160))
            pil_im = Image.fromarray(decoded).convert('RGB')
            new_images.append(np.array(pil_im))
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(new_images, fps=30)
        clip.write_videofile(save_video_path)   
        
    if interpolate:
        interpolate = lambda f, s: [(1-y)*f + y*s for y in np.linspace(0,1, default_frame_skip)]
    else:
        interpolate = lambda f,s:[f,s]
    if is_log_sigma:
        def get_mu_sigma(frame):
            mu, sigma = dpn(frame)            
            sigma = torch.exp(sigma)
            return mu, sigma
    else:
        
        get_mu_sigma = lambda frame: dpn(frame)

    #calculate how many frames we need to build.
    number_of_passes = round((30/default_frame_skip)*seconds, 0)
    number_of_passes = int(number_of_passes)
    #initialize the current frame
    current = first_ls_vector.detach()
    output = []
    for i in  range(number_of_passes):
        print(first_ls_vector)
        mu, sigma = get_mu_sigma(first_ls_vector)
        dist = Independent(Normal(mu, sigma),1)
        next = dist.sample()
        if clip:
            x = next.numpy()
            x[16] = 0
            x[5] = 0
            x[9] = 0
            next = torch.as_tensor(x)
        interpolated = interpolate(current, next)
        output.extend(interpolated)
        current = next.detach()    
        
    buildVideo(output,save_video_path)    

# Initiate and load DPN
dpnn = DPN(0.001, 20)
PATH = '/Users/natewagner/Documents/ArmyResearchLab/Actor_243.40230742514134_avg_90000_ngames.pt'
dpnn.load_state_dict(torch.load(PATH))

# Initiate and load Decoder
decoder = Decoder(14400,20)
decoder.load_state_dict(torch.load('/Users/natewagner/Documents/ArmyResearchLab/decoder_z20_epch140.pt', map_location=torch.device('cpu')))

# Read in latent space encodings to make predictions
encods = pd.read_csv('/Users/natewagner/Documents/ArmyResearchLab/critter_encodings.csv')
lat_vec = list(encods.iloc[18000][2:])
lat_vec_tens = torch.FloatTensor(lat_vec)

# Make video
generate_latent_future(dpnn, decoder, lat_vec_tens, 15, is_log_sigma = True, save_video_path='/Users/natewagner/Documents/ArmyResearchLab/new_movie4.mp4')

       
