# --- Test reconstruction spectrogramme PNG -> WAV ---
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import AudioDataset
from vae import VAE
from utils import load_best_checkpoint
from reconstruct_wav_from_mel import mel_to_wav

dataset = AudioDataset('dataset/genres/classical')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

latent_dim = 16
model = VAE(latent_dim)
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
        # Reconstruction WAV à partir du mel-spectrogramme
        out_path = f'out/reconstruction_{i}.wav'
        sample_rate = 22050  # cohérent avec le dataset
        wav = mel_to_wav(recon, sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=80, out_path=out_path)
        print(f"Reconstruit WAV sauvegardé: {out_path}")
        # Un seul exemple pour le test
        break
