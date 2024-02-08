import torch
import torchaudio
from torch import nn

from .utils import safe_log

#
# Mel-Spectrogram Reconstruction Loss
#

class MelSpecReconstructionLoss(nn.Module):
    def __init__(self, *, sample_rate, n_fft, hop_length, win_length, n_mels):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            win_length=win_length,
            power=1, # Better for loss calculation
            center=True
        )

    def forward(self, a, b) -> torch.Tensor:
        mel_a = safe_log(self.mel_spec(a))
        mel_b = safe_log(self.mel_spec(b))
        loss = torch.nn.functional.l1_loss(mel_a, mel_b)
        return loss

#
# Generator Loss, calculated as the sum of the losses of each discriminator
#

class GeneratorLoss(nn.Module):
    def forward(self, disc_outputs):
        loss = torch.zeros(1, device=disc_outputs[0].device, dtype=disc_outputs[0].dtype)
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(l)
            loss += l

        return loss, 

#
# Discriminator Loss module. Calculates the loss for the discriminator based on real and generated outputs.
#

class DiscriminatorLoss(nn.Module):
    def forward(self, disc_real_outputs, disc_generated_outputs):
        loss = torch.zeros(1, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype)
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss += r_loss + g_loss
            r_losses.append(r_loss)
            g_losses.append(g_loss)
        return loss, r_losses, g_losses

#
# Loss for subscrirminators
#

class FeatureMatchingLoss(nn.Module):
    def forward(self, fmap_r, fmap_g):
        loss = torch.zeros(1, device=fmap_r[0][0].device, dtype=fmap_r[0][0].dtype)
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss