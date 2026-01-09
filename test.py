
# --- Écouter et sauvegarder des reconstructions audio du VAE ---
import torchaudio
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import MaestroDataset
from vae import VAE
from utils import load_best_checkpoint


dataset = MaestroDataset('dataset/genres/classical')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


first_batch = next(iter(dataloader))
input_dim = first_batch[0].numel()
latent_dim = 16
model = VAE(input_dim=input_dim, latent_dim=latent_dim)

load_best_checkpoint(model, None)

# Prendre un batch du dataloader
test_batch = next(iter(dataloader))
test_batch = test_batch.view(test_batch.size(0), -1)

model.eval()
with torch.no_grad():
    x_hat, mu, logvar = model(test_batch)
    # x_hat shape: [batch_size, input_dim]
    for i in range(min(5, x_hat.size(0))):
        audio = x_hat[i].cpu().numpy()
        # Normalisation pour éviter les saturations
        audio = audio / max(abs(audio).max(), 1e-8)
        # Sauvegarde en WAV
        out_path = f'out/reconstruction_{i}.wav'
        torchaudio.save(out_path, torch.tensor(audio).unsqueeze(0), 16000)
        print(f"Reconstruit sauvegardé: {out_path}")
