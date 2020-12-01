import cv2
import numpy
import torch
import torch.nn as nn

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

class MakePredictionVideo():
    def __init__(self, PATH_TO_WEIGHTS):
        self.frames = []
        self.PATH_TO_WEIGHTS = PATH_TO_WEIGHTS #'/home/nwagner/aae_models/decoder_z20_epch140.pt'

    def reconstruct(self, vectors):
        """ 
        Use the decoder to reconstruct images from list of LS vectors 
        """
        decoder = self.initialize_decoder()
        decoder.eval()
        for vector in vectors:
            # change type to tensor
            vector = vector.type(torch.cuda.FloatTensor)
            # pass through decoder
            decoded = decoder(vector)
            image = decoded.view(10*9, 160)
            image = image.detach().numpy()
            self.frames.append(image)

    def initialize_decoder(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Running on the GPU")
        else: 
            device = torch.device("cpu")
            print("Running on the CPU")

        dim_z = 20
        decoder = Decoder(14400, dim_z).to(device)
        decoder.load_state_dict(torch.load(self.PATH_TO_WEIGHTS))
        return decoder

    def write_video(self, name):
        """
        Make video from list of frames
        """
        out = cv2.VideoWriter(f'{name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (640,360), isColor=False) # used MJPG due to using OSX 
        for frame in self.frames:
            # resize
            resized = cv2.resize(frame, (640,360))
            out.write(frame)
        out.release()

