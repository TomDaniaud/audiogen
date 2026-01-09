import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import SpectrogramPNGDataset
from vae import VAE, vae_loss
from utils import load_best_checkpoint, save_best_checkpoint


dataset = SpectrogramPNGDataset('dataset/images/classical')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

first_batch = next(iter(dataloader))
input_channels = first_batch.shape[1]  # channel (généralement 1)
latent_dim = 16
model = VAE(input_channels=input_channels, latent_dim=latent_dim, input_size=first_batch.shape[2:4])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

loaded_epoch = load_best_checkpoint(model, optimizer)

num_epochs = 100
num_epochs += loaded_epoch
losses = []

for epoch in range(loaded_epoch, num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    for x in dataloader:
        # x shape: [batch_size, 1, height, width]
        x_hat, mu, logvar = model(x)
        loss = vae_loss(x, x_hat, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
    losses.append(avg_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.2f}")
    save_best_checkpoint(model, optimizer, avg_loss, epoch, dir='checkpoints')

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Courbe de loss VAE')
plt.show()
