import torch
from torch import nn
from .layers import ISTFTLayer, VocosGenerator

class Vocoder(nn.Module):
    def __init__(self, *, n_mels):
        super().__init__()
        
        # Generator
        self.generator = VocosGenerator(
            input_channels = n_mels,
            dim = 512, 
            intermediate_dim = 1536,
            num_layers = 8
        )

        # ISTFT
        self.istft = ISTFTLayer(
            dim = 512, 
            n_fft = 1024, 
            hop_length = 240
        )
    
    def forward(self, x):
        x = self.generator(x)
        x = self.istft(x)
        return x