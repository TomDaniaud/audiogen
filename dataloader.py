# Dataset pour spectrogrammes PNG (images 2D)
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class SpectrogramPNGDataset(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_files = [
			os.path.join(root_dir, f)
			for f in os.listdir(root_dir)
			if f.endswith('.png')
		]

	def __len__(self):
		return len(self.img_files)

	def __getitem__(self, idx):
		img_path = self.img_files[idx]
		img = Image.open(img_path).convert('L')  # 'L' = grayscale
		arr = np.array(img).astype(np.float32) / 255.0  # Normalisation [0, 1]
		tensor = torch.tensor(arr)
		# Ajoute la dimension channel si besoin
		if tensor.dim() == 2:
			tensor = tensor.unsqueeze(0)
		return tensor


# Dataset pour spectrogrammes (images 2D)
class SpectrogramDataset(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.spec_files = [
			os.path.join(root_dir, f)
			for f in os.listdir(root_dir)
			if f.endswith('.pt') or f.endswith('.npy')
		]

	def __len__(self):
		return len(self.spec_files)

	def __getitem__(self, idx):
		spec_path = self.spec_files[idx]
		if spec_path.endswith('.pt'):
			spec = torch.load(spec_path)
		elif spec_path.endswith('.npy'):
			spec = torch.tensor(np.load(spec_path))
		# Normalisation [0, 1]
		spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
		# Ajoute la dimension channel si besoin
		if spec.dim() == 2:
			spec = spec.unsqueeze(0)
		return spec