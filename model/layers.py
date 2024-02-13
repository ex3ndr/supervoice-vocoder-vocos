import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv1d, Conv2d, AvgPool1d, Identity
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from .utils import istft, get_padding, init_weights, melscale_fbanks, hann_window

#
# Inverse Short-Time Fourier Transform (iSTFT) layer
#

class ISTFTLayer(nn.Module):
    def __init__(self, *, dim, n_fft, hop_length):
        super().__init__()
        self.dim = dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out = torch.nn.Linear(dim, n_fft + 2)

    def forward(self, x):

        # Forward pass
        x = self.out(x).transpose(1, 2) # (B, T, F) -> (B, F, T) as it needed by istft
        mag, p = x.chunk(2, dim=1)

        # Compute magnitude
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        
        # Convert phase to complex value
        x = torch.cos(p)
        y = torch.sin(p)
        S = mag * (x + 1j * y)

        # iSTFT
        audio = istft(S, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft)

        # Result
        return audio
    
#
# Spectogram layer
#

class SpectogramLayer(nn.Module):
    def __init__(self, n_mels, n_fft, n_hop_length, sample_rate):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.n_hop_length = n_hop_length
        self.sample_rate = sample_rate
    def forward(self, x):
        # Hann Window
        window = hann_window(self.n_hop_length * 4, x.device)

        # STFT
        stft = torch.stft(x, 
            self.n_fft, 
            hop_length=self.n_hop_length, 
            win_length=self.n_hop_length * 4,
            window=window, 
            center=True,
            return_complex = True
        )

        # Compute magnitudes (|a + ib| = sqrt(a^2 + b^2)) instead of power spectrum (|a + ib|^2 = a^2 + b^2)
        # because magnitude and phase is linear to the input, while power spectrum is quadratic to the input
        # and the magnitude is easier to learn for vocoder
        # power = stft[..., :-1].abs() ** 2
        magnitudes = stft[..., :-1].abs()

        # Mel Log Bank
        mel_filters = melscale_fbanks(self.n_mels, self.n_fft, 0, self.sample_rate / 2, self.sample_rate, x.device)
        mel_spec = (mel_filters @ magnitudes)

        # Log
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        return log_spec

#
# HiFiGAN ResBlock #1
#

class HiFiGANResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.gelu(x)
            xt = c1(xt)
            xt = F.gelu(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


#
# Period Discriminator
#

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.gelu(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


#
# Multi-Scale Discriminator
# 
    
class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.gelu(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            Identity(),
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y = self.meanpools[i-1](y)
            y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

#
# ConvNeXt Block adapted for 1D convolutions
#

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, intermediate_dim, layer_scale_init_value):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = residual + x
        return x
    

#
# Vocos Generator
#
    
class VocosGenerator(nn.Module):
    def __init__(self, *, input_channels, dim, intermediate_dim, num_layers):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = 1 / num_layers
        self.convnext = nn.ModuleList(
            [ ConvNeXtBlock(dim = dim, intermediate_dim = intermediate_dim, layer_scale_init_value = layer_scale_init_value) for _ in range(num_layers) ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(init_weights)

    def forward(self, x):
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x
