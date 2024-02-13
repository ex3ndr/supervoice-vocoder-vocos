import itertools
import torch
from .losses import MelSpecReconstructionLoss, DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss
from .layers import SpectogramLayer, MultiPeriodDiscriminator, MultiScaleDiscriminator, ISTFTLayer, VocosGenerator

class Trainer:
    def __init__(self, *, accelerator, n_mels, n_fft, sample_rate):

        self.accelerator = accelerator

        # Spectogram
        self.spec = SpectogramLayer(
            n_mels = n_mels, 
            n_fft = n_fft, 
            n_hop_length = sample_rate // 100, # 10ms 
            sample_rate = sample_rate
        )

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

        # Discriminators
        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiScaleDiscriminator()

        # Losses
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(
            sample_rate = 24000, 
            n_fft = 1024, 
            hop_length = 24000 // 100, # 10ms 
            win_length = (24000 // 100) * 4, 
            n_mels = 100
        )

        # Optimizers
        self.steps = 0
        self.optim_g = torch.optim.AdamW(self.generator.parameters(), train_learning_rate, betas=[0.8, 0.999])
        self.optim_d = torch.optim.AdamW(itertools.chain(self.multiperioddisc.parameters(), self.multiresddis.parameters()), train_learning_rate, betas=[0.8, 0.999])

        # Accelerate
        (self.spec, self.generator, self.istft, self.multiperioddisc, self.multiresddisc) = accelerator.prepare(self.spec, self.generator, self.istft, self.multiperioddisc, self.multiresddisc)

    def resynthesize(self, x):
        x = self.spec(x)
        x = self.generator(x)
        x = self.istft(x)
        return x

    def train_step(self, batch):
        audio_hat = self.resynthesize(batch)
        self.train_step_discriminator(batch, audio_hat)
        self.train_step_generator(batch, audio_hat)

    def train_step_discriminator(self, audio_input, audio_hat):

        # Detach audio_hat because we are training discriminator
        audio_hat = audio_hat.detach()

        # Forward pass of a discrimitators
        real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y = audio_input, y_hat = audio_hat)
        real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y = audio_input, y_hat = audio_hat)

        # Calculate loss
        loss_mp, loss_mp_real, _ = self.disc_loss(
            disc_real_outputs = real_score_mp, disc_generated_outputs = gen_score_mp
        )
        loss_mrd, loss_mrd_real, _ = self.disc_loss(
            disc_real_outputs = real_score_mrd, disc_generated_outputs = gen_score_mrd
        )
        loss_mp /= len(loss_mp_real)
        loss_mrd /= len(loss_mrd_real)
        loss = loss_mp + loss_mrd

        # Backward pass
        self.optim_d.zero_grad()
        loss.backward()
        self.optim_d.step()

    def train_step_generator(self, audio_input, audio_hat):
        pass