
import torch
import torch.nn as nn

class UNet1D(nn.Module):
	def __init__(self, in_channels, out_channels, num_layers=4, base_channels=64):
		super().__init__()
		self.downs = nn.ModuleList()
		self.ups = nn.ModuleList()
		self.num_layers = num_layers

		# Encoder (downsampling)
		prev_channels = in_channels
		for i in range(num_layers):
			self.downs.append(
				nn.Sequential(
					nn.Conv1d(prev_channels, base_channels * 2**i, kernel_size=4, stride=2, padding=1),
					nn.BatchNorm1d(base_channels * 2**i),
					nn.ReLU()
				)
			)
			prev_channels = base_channels * 2**i

		# Bottleneck
		self.bottleneck = nn.Sequential(
			nn.Conv1d(prev_channels, prev_channels, kernel_size=3, padding=1),
			nn.ReLU()
		)

		# Decoder (upsampling)
		for i in reversed(range(num_layers)):
			self.ups.append(
				nn.Sequential(
					nn.ConvTranspose1d(prev_channels, base_channels * 2**i, kernel_size=4, stride=2, padding=1),
					nn.BatchNorm1d(base_channels * 2**i),
					nn.ReLU()
				)
			)
			prev_channels = base_channels * 2**i

		# Final conv
		self.final = nn.Conv1d(prev_channels, out_channels, kernel_size=1)

	def forward(self, x):
		skips = []
		out = x
		# Encoder
		for down in self.downs:
			out = down(out)
			skips.append(out)
		out = self.bottleneck(out)
		# Decoder
		for up in self.ups:
			skip = skips.pop()
			out = up(out)
			# Crop if needed (for odd sizes)
			if out.shape[-1] > skip.shape[-1]:
				out = out[..., :skip.shape[-1]]
			elif out.shape[-1] < skip.shape[-1]:
				skip = skip[..., :out.shape[-1]]
			out = out + skip  # skip connection
		return self.final(out)


# Exemple de pipeline de diffusion simple (DDPM-like, sans conditionnement)
if __name__ == "__main__":
	latent_dim = 16
	latent_length = 784  # à adapter à ton latent
	batch_size = 8
	num_steps = 50
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = UNet1D(in_channels=latent_dim, out_channels=latent_dim).to(device)
	# Bruit initial
	x = torch.randn(batch_size, latent_dim, latent_length).to(device)

	# Diffusion reverse (simplifié)
	for t in reversed(range(num_steps)):
		# Ici, on simule une étape de débruitage
		noise_pred = model(x)
		# On retire une fraction du bruit prédit (step simplifié)
		x = x - (1.0 / num_steps) * noise_pred

	# x est maintenant un latent "généré" à partir de bruit
	print("Latent généré shape:", x.shape)
	# Pour obtenir l'audio, il faut passer x dans le décodeur du VAE
