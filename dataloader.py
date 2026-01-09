import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

class MaestroDataset(Dataset):
	def __init__(self, root_dir, sample_rate=16000, segment_length=5):
		self.root_dir = root_dir
		self.sample_rate = sample_rate
		self.segment_length = segment_length  # en secondes
		self.audio_files = [
			os.path.join(root_dir, f)
			for f in os.listdir(root_dir)
			if f.endswith('.wav')
		]

	def __len__(self):
		return len(self.audio_files)

	def __getitem__(self, idx):
		audio_path = self.audio_files[idx]
		waveform, sr = torchaudio.load(audio_path)
		# Resample si besoin
		if sr != self.sample_rate:
			waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
		# Mono
		if waveform.shape[0] > 1:
			waveform = waveform.mean(dim=0, keepdim=True)
		# Découpe un segment aléatoire
		num_samples = self.sample_rate * self.segment_length
		if waveform.shape[1] > num_samples:
			start = torch.randint(0, waveform.shape[1] - num_samples, (1,)).item()
			waveform = waveform[:, start:start+num_samples]
		else:
			# Padding si trop court
			pad = num_samples - waveform.shape[1]
			waveform = torch.nn.functional.pad(waveform, (0, pad))
		# Normalisation [-1, 1]
		waveform = waveform / waveform.abs().max()
		return waveform.squeeze(0)

# Exemple d'utilisation
# dataset = MaestroDataset('/chemin/vers/maestro-v3.0.0/wav')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# for batch in dataloader:
#     # batch: [batch_size, sample_rate * segment_length]
#     ... # Utiliser batch pour U-Net ou diffusion
