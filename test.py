# --- Test reconstruction spectrogramme PNG -> WAV ---
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import SpectrogramPNGDataset
from vae import VAE
from utils import load_best_checkpoint

dataset = SpectrogramPNGDataset('dataset/images/classical')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

first_batch = next(iter(dataloader))
input_channels = first_batch.shape[1]
height = first_batch.shape[2]
width = first_batch.shape[3]
latent_dim = 16
model = VAE(input_channels=input_channels, latent_dim=latent_dim, input_size=(height, width))
load_best_checkpoint(model, None)

os.makedirs('out', exist_ok=True)

model.eval()
with torch.no_grad():
    for i, spec in enumerate(dataloader):
        # Reconstruire le spectrogramme
        x_hat, mu, logvar = model(spec)
        recon = x_hat[0].cpu().numpy().squeeze(0)  # [height, width]
        # Sauvegarde image reconstruite
        plt.imsave(f'out/reconstruction_{i}.png', recon, cmap='magma')
        print(f"Image reconstruite sauvegardée: out/reconstruction_{i}.png")
        # Transformation en waveform (Griffin-Lim)
        # On suppose que le spectrogramme est un mel-spectrogramme
        # Il faut inverser le mel-spectrogramme en waveform
        # Ici, on utilise torchaudio.transforms.GriffinLim
        spec_tensor = torch.tensor(recon).unsqueeze(0)  # [1, freq, time]
        freq_bins = spec_tensor.shape[1]
        n_fft = (freq_bins - 1) * 2
        griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft)
        waveform = griffin_lim(spec_tensor)
        # Normalisation
        waveform = waveform / waveform.abs().max()
        # Sauvegarde en WAV
        out_path = f'out/reconstruction_{i}.wav'
        sample_rate = 16000
        torchaudio.save(out_path, waveform, sample_rate)
        print(f"Reconstruit WAV sauvegardé: {out_path}")
        # Un seul exemple pour le test
        break
