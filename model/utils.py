import torch
import torchaudio.transforms as T
import torchaudio.functional as F
from torch import nn

#
# Safe Log
#

def safe_log(x, clip_val = 1e-7):
    return torch.log(torch.clip(x, min=clip_val))

#
# Cached Hann Window
#

hann_window_cache = {}
def hann_window(size, device):
    global hann_window_cache
    key = str(device) + "_" + str(size)
    if key in hann_window_cache:
        return hann_window_cache[key]
    else:
        res = torch.hann_window(size).to(device)
        hann_window_cache[key] = res
        return res

#
# Mel Log Bank
#

melscale_fbank_cache = {}
def melscale_fbanks(n_mels, n_fft, f_min, f_max, sample_rate, device):
    global melscale_fbank_cache
    key = str(n_mels) + "_" + str(n_fft) + "_" + str(f_min) + "_" + str(f_max) + "_" + str(sample_rate) + "_" + str(device)
    if key in melscale_fbank_cache:
        return melscale_fbank_cache[key]
    else:
        res = F.melscale_fbanks(
            n_freqs=int(n_fft // 2 + 1),
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            norm="slaney",
        ).transpose(-1, -2).to(device)
        melscale_fbank_cache[key] = res
        return res

#
# Resampler
#

resampler_cache = {}
def resampler(from_sample_rate, to_sample_rate, device=None):
    global resampler_cache
    if device is None:
        device = "cpu"
    key = str(from_sample_rate) + "_" + str(to_sample_rate) + "_" + str(device)
    if key in resampler_cache:
        return resampler_cache[key]
    else:
        res = T.Resample(from_sample_rate, to_sample_rate).to(device)
        resampler_cache[key] = res
        return res

#
# iSTFT
#
    
def istft(x, *, n_fft, hop_length, win_length):
    return torch.istft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True)

#
# Padding calculation for kernel size and dilation
#

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


#
# Weight Initialization
#

def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

#
# Distributed helpers
#

def rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

def world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1

def is_distributed():
    return world_size() > 1

#
# Average metrics for balancer
# 

def all_reduce(tensor, op):
    if is_distributed():
        return torch.distributed.all_reduce(tensor, op)

def average_metrics(metrics, count=1.):
    if not is_distributed():
        return metrics
    keys, values = zip(*metrics.items())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
    tensor *= count
    all_reduce(tensor)
    averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
    return dict(zip(keys, averaged))


#
# Yield data loader
#

def cycle(dl):
    while True:
        for data in dl:
            yield data    