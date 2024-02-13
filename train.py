# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Core
from glob import glob
from pathlib import Path
import random
import itertools
import shutil

# ML
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed

# Local
from model.vocoder import Vocoder
from model.layers import SpectogramLayer, MultiPeriodDiscriminator, MultiScaleDiscriminator, ISTFTLayer, VocosGenerator
from model.losses import MelSpecReconstructionLoss, DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss
from model.utils import cycle, resampler
from model.dataset import SimpleAudioDataset

#
# Parameters
#

train_experiment = "new_mel_2"
train_project = "supervoice-vocoder"
train_auto_resume = True
train_segment_size = 24000
train_learning_rate = 2e-4
train_batch_size = 32 # Per GPU
train_steps = 1000000
train_loader_workers = 4
train_save_every = 1000
train_log_every = 1
train_evaluate_every = 200
train_evaluate_batches = 10
train_mel_loss_factor = 45
train_mrd_loss_coeff = 0.1

#
# Train
#

def main():
    
    # Prepare accelerator
    accelerator = Accelerator(log_with="wandb")
    device = accelerator.device
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    # Prepare dataset
    accelerator.print("Loading dataset...")

    # Train Files
    train_files = glob("external_datasets/libritts-r-clean-100/*/*/*.wav") + glob("external_datasets/libritts-r-clean-360/*/*/*.wav") + glob("external_datasets/libritts-r-other-500/*/*/*.wav") + glob("external_datasets/ptdb-tug/SPEECH DATA/FEMALE/MIC/*.wav") + glob("external_datasets/ptdb-tug/SPEECH DATA/MALE/MIC/*.wav")
    train_files.sort()
    random.shuffle(train_files)

    # Test Files
    test_files = glob("external_datasets/libritts-r/test-clean/*/*/*.wav") + glob("external_datasets/libritts-r/test-other/*/*/*.wav")
    test_files.sort()
    random.shuffle(test_files)
    test_files = test_files[:train_batch_size * train_evaluate_batches]

    # Dataset
    train_dataset = SimpleAudioDataset(files = train_files, sample_rate=24000, segment_size=train_segment_size)
    test_dataset = SimpleAudioDataset(files = test_files, sample_rate=24000, segment_size=train_segment_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=train_loader_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False, num_workers=train_loader_workers, pin_memory=True)

    # Model
    spec = SpectogramLayer(n_mels = 80, n_fft = 1024, n_hop_length = 16000 // 100, sample_rate = 16000)
    vocoder = Vocoder(n_mels = 80)
    multiperioddisc = MultiPeriodDiscriminator()
    multiresddisc = MultiScaleDiscriminator()
    disc_loss = DiscriminatorLoss()
    gen_loss = GeneratorLoss()
    feat_matching_loss = FeatureMatchingLoss()
    melspec_loss = MelSpecReconstructionLoss(
        sample_rate = 24000, 
        n_fft = 1024, 
        hop_length = 24000 // 100, # 10ms 
        win_length = (24000 // 100) * 4, 
        n_mels = 100
    )

    # Optimizers
    steps = 0
    optim_g = torch.optim.AdamW(vocoder.parameters(), train_learning_rate, betas=[0.8, 0.999])
    optim_d = torch.optim.AdamW(itertools.chain(multiperioddisc.parameters(), multiresddisc.parameters()), train_learning_rate, betas=[0.8, 0.999])
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=train_steps)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=train_steps)

    # Accelerate
    vocoder, multiperioddisc, multiresddisc, optim_g, optim_d, scheduler_g, scheduler_d, train_loader, test_loader = accelerator.prepare(vocoder, multiperioddisc, multiresddisc, optim_g, optim_d, scheduler_g, scheduler_d, train_loader, test_loader)
    train_cycle = cycle(train_loader)
    hps = {
        "segment_size": train_segment_size, 
        "learning_rate": train_learning_rate, 
        "batch_size": train_batch_size, 
        "steps": train_steps, 
    }
    accelerator.init_trackers(train_project, config=hps)
    disc_loss.to(device)
    gen_loss.to(device)
    feat_matching_loss.to(device)
    melspec_loss.to(device)

    # Save/Load
    def save():

        # Save step checkpoint
        fname = str(output_dir / f"{train_experiment}.pt")
        fname_step = str(output_dir / f"{train_experiment}.{steps}.pt")
        torch.save({

            # Model
            'vocoder': accelerator.get_state_dict(vocoder), 
            "mpd": accelerator.get_state_dict(multiperioddisc),
            "msd": accelerator.get_state_dict(multiresddisc),

            # Optimizer
            'steps': steps,
            'optimizer_g': optim_g.state_dict(), 
            'optimizer_d': optim_d.state_dict(), 
            'scheduler_g': scheduler_g.state_dict(),
            'scheduler_d': scheduler_d.state_dict(),

        },  fname_step)

        # Overwrite main checkpoint
        shutil.copyfile(fname_step, fname)

    # Load
    if train_auto_resume:
        if (output_dir / f"{train_experiment}.pt").exists():
            accelerator.print("Resuming training...")

            checkpoint = torch.load(str(output_dir / f"{train_experiment}.pt"), map_location="cpu")

            # Model
            accelerator.unwrap_model(vocoder).load_state_dict(checkpoint['vocoder'])
            accelerator.unwrap_model(multiperioddisc).load_state_dict(checkpoint['mpd'])
            accelerator.unwrap_model(multiresddisc).load_state_dict(checkpoint['msd'])

            # Optimizer
            optim_g.load_state_dict(checkpoint['optimizer_g'])
            optim_d.load_state_dict(checkpoint['optimizer_d'])
            scheduler_g.load_state_dict(checkpoint['scheduler_g'])
            scheduler_d.load_state_dict(checkpoint['scheduler_d'])
            steps = checkpoint['steps']

    # Train step
    def train_step():
        vocoder.train()
        multiresddisc.train()
        multiperioddisc.train()

        # Load batch
        audio = next(train_cycle)

        # Generate
        audio_res = resampler(24000, 16000, audio.device)(audio)
        audio_res_spec = spec(audio_res)
        audio_hat = vocoder(audio_res_spec) # Need to resample from 24k to 16k

        # Adding a channel dimension
        audio_hat = audio_hat.unsqueeze(1)
        audio = audio.unsqueeze(1)

        #
        # Training Discriminators
        #

        # Forward pass of a discrimitators
        real_score_mp, gen_score_mp, _, _ = multiperioddisc(y = audio, y_hat = audio_hat.detach())
        real_score_mrd, gen_score_mrd, _, _ = multiresddisc(y = audio, y_hat = audio_hat.detach())

        # Calculate loss
        loss_mp, loss_mp_real, _ = disc_loss(
            disc_real_outputs = real_score_mp, disc_generated_outputs = gen_score_mp
        )
        loss_mrd, loss_mrd_real, _ = disc_loss(
            disc_real_outputs = real_score_mrd, disc_generated_outputs = gen_score_mrd
        )
        loss_mp /= len(loss_mp_real)
        loss_mrd /= len(loss_mrd_real)
        loss = loss_mp + loss_mrd

        # Backward pass
        optim_d.zero_grad()
        accelerator.backward(loss)
        grad_norm_mpd = accelerator.clip_grad_norm_(multiperioddisc.parameters(), 1000.)
        grad_norm_msd = accelerator.clip_grad_norm_(multiresddisc.parameters(), 1000.)
        optim_d.step()

        #
        # Training Generator
        #

        # Reconstruciton loss
        mel_loss = melspec_loss(audio, audio_hat)

        # Discriminator-based losses
        _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = multiperioddisc(y=audio, y_hat=audio_hat)
        _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = multiresddisc(y=audio, y_hat=audio_hat)
        loss_gen_mp, list_loss_gen_mp = gen_loss(disc_outputs=gen_score_mp)
        loss_gen_mrd, list_loss_gen_mrd = gen_loss(disc_outputs=gen_score_mrd)
        loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
        loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
        loss_fm_mp = feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
        loss_fm_mrd = feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)

        # Backward pass
        loss_gen = (loss_gen_mp + train_mrd_loss_coeff * loss_gen_mrd + loss_fm_mp + train_mrd_loss_coeff * loss_fm_mrd + train_mel_loss_factor * mel_loss)
        optim_g.zero_grad()
        accelerator.backward(loss_gen)
        grad_norm_g = accelerator.clip_grad_norm_(vocoder.parameters(), 1000.)
        optim_g.step()

        #
        # Update learning rate for next batch
        #

        scheduler_d.step()
        scheduler_g.step()

        return loss.item(), loss_gen.item(), mel_loss.item(), loss_gen_mp.item(), loss_gen_mrd.item(), loss_fm_mp.item(), loss_fm_mrd.item(), loss_mp.item(), loss_mrd.item(), grad_norm_mpd, grad_norm_msd, grad_norm_g

    # Train Loop
    accelerator.print("Training started at step", steps)
    while steps < train_steps:

        # Train step
        loss, loss_gen, mel_loss, loss_gen_mp, loss_gen_mrd, loss_fm_mp, loss_fm_mrd, loss_mp, loss_mrd, grad_norm_mpd, grad_norm_msd, grad_norm_g = train_step()

        # Update step
        steps = steps + 1

        # Wait for everyone
        accelerator.wait_for_everyone()

        # Evaluate
        if (steps % train_evaluate_every == 0):
            if accelerator.is_main_process:
                accelerator.print("Evaluating...")
            with torch.inference_mode():      
                vocoder.eval()
                losses = []
                for test_batch in test_loader:
                    audio = test_batch
                    audio_res = resampler(24000, 16000, audio.device)(audio)
                    audio_res_spec = spec(audio_res)
                    audio_hat = vocoder(audio_res_spec)
                    audio_hat = audio_hat.unsqueeze(1)
                    audio = audio.unsqueeze(1)
                    mel_loss = melspec_loss(audio, audio_hat)
                    gathered = accelerator.gather(mel_loss).cpu()
                    if len(gathered.shape) == 0:
                        gathered = gathered.unsqueeze(0)
                    losses += gathered.tolist()
                if accelerator.is_main_process:
                    eval_loss = torch.tensor(losses).mean()
                    accelerator.log({"loss_mel_test": eval_loss}, step=steps)
                    accelerator.print(f"Evaluation Loss: {eval_loss}")

        # Log
        if accelerator.is_main_process and (steps % train_log_every == 0):
            accelerator.print(f"Step {steps}: DLoss: {loss}, GLoss: {loss_gen}, MelLoss: {mel_loss}, GLossMP: {loss_gen_mp}, GLossMRD: {loss_gen_mrd}, FMLossMP: {loss_fm_mp}, FMLossMRD: {loss_fm_mrd}, DLossMP: {loss_mp}, DLossMRD: {loss_mrd}")
            accelerator.log({ 
                'dloss': loss, 
                'gloss': loss_gen, 
                'mel_loss': mel_loss, 
                'gloss_mp': loss_gen_mp, 
                'gloss_mrd': loss_gen_mrd, 
                'fmloss_mp': loss_fm_mp, 
                'fmloss_mrd': loss_fm_mrd, 
                'dloss_mp': loss_mp, 
                'dloss_mrd': loss_mrd,
                'grad_norm_msd': grad_norm_msd,
                'grad_norm_mpd': grad_norm_mpd,
                'grad_norm_g': grad_norm_g
            }, step=steps)

        # Save 
        if accelerator.is_main_process and (steps % train_save_every == 0):
            save()

    # End training
    if accelerator.is_main_process:
        accelerator.print("Finishing training...")
        save()
    accelerator.end_training()
    accelerator.print('✨ Training complete!')


if __name__ == "__main__":
    main()