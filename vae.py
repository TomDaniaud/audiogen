import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim, input_size=(128, 128)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        # Calcul dynamique de la taille après convs
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            out = F.relu(self.conv1(dummy))
            out = F.relu(self.conv2(out))
            out = F.relu(self.conv3(out))
            self.flat_dim = out.numel()
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_channels, output_size=(128, 128)):
        super().__init__()
        # Calcul dynamique de la taille à reconstruire
        self.output_size = output_size
        # Encoder dummy pour obtenir la taille flatten
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *output_size)
            conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
            conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            out = F.relu(conv1(dummy))
            out = F.relu(conv2(out))
            out = F.relu(conv3(out))
            self.flat_dim = out.numel() // out.shape[0]
            self.unflatten_shape = out.shape[1:]  # (C, H, W)
        self.fc = nn.Linear(latent_dim, self.flat_dim)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=self.unflatten_shape)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        z = F.relu(self.fc(z))
        z = self.unflatten(z)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = torch.sigmoid(self.deconv3(z))
        return z


class VAE(nn.Module):
    def __init__(self, input_channels, latent_dim, input_size=(128, 128)):
        super().__init__()
        self.encoder = Encoder(input_channels, latent_dim, input_size=input_size)
        self.decoder = Decoder(latent_dim, input_channels, output_size=input_size)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    # reconstruction
    recon = F.mse_loss(x_hat, x, reduction="mean")

    # KL divergence analytique
    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon + beta * kl