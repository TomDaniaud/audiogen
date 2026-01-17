# Dataset pour spectrogrammes PNG (images 2D)
import os
import torch
import librosa
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, dir):
        root_dir = os.path.abspath(dir)
        self.files = [
			os.path.join(root_dir, f)
			for f in os.listdir(root_dir)
			if f.endswith('.wav') and os.path.isfile(os.path.join(root_dir, f))
		]

    def __getitem__(self, idx):
        y, sr = librosa.load(self.files[idx], sr=22050)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=80, n_fft=1024, hop_length=256
        )
        log_mel = librosa.power_to_db(mel)
        tensor = torch.tensor(log_mel).unsqueeze(0)  # (1, 80, T)
        # Padding/cropping automatique sur la longueur (T)
        target_length = 2600  # à adapter selon ton projet
        c, h, w = tensor.shape
        if w < target_length:
            pad_w = target_length - w
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, 0))
        elif w > target_length:
            tensor = tensor[:, :, :target_length]
        return tensor

    def __len__(self):
        return len(self.files)
