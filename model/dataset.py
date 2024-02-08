import torch
import torchaudio
import random
from .utils import resampler

class SimpleAudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, *, files, segment_size, sample_rate):
        self.files = files
        self.segment_size = segment_size
        self.sample_rate = sample_rate
    
    def __getitem__(self, index):

        # Load File
        filename = self.files[index]

        # Load audio
        audio = load_mono_audio(filename, self.sample_rate)

        # Pad or trim to target duration
        if audio.shape[0] >= self.segment_size:
            audio_start = random.randint(0, audio.shape[0] - self.segment_size)
            audio = audio[audio_start:audio_start+self.segment_size]
        elif audio.shape[0] < self.segment_size: # Rare or impossible case - just pad with zeros
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.shape[0]))

        return audio

    def __len__(self):
        return len(self.files)
    

def load_mono_audio(src, sample_rate, device=None):

    # Load audio
    audio, sr = torchaudio.load(src)

    # Move to device
    if device is not None:
        audio = audio.to(device)

    # Resample
    if sr != sample_rate:
        audio = resampler(sr, sample_rate, device)(audio)
        sr = sample_rate

    # Convert to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Convert to single dimension
    audio = audio[0]

    return audio