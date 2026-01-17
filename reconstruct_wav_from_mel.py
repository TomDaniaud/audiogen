# reconstruct_wav_from_mel.py
"""
Script pour reconstruire un fichier WAV à partir d'un mel-spectrogramme (tensor ou numpy array).
"""
import torch
import torchaudio
import numpy as np

def mel_to_wav(mel, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80, out_path=None):
    """
    Reconstruit un wav à partir d'un mel-spectrogramme (tensor ou numpy array).
    Args:
        mel: (n_mels, T) ou (1, n_mels, T) torch.Tensor ou np.ndarray
        sample_rate: int
        n_fft: int
        hop_length: int
        n_mels: int
        out_path: chemin pour sauvegarder le wav (optionnel)
    Returns:
        waveform: torch.Tensor (1, N)
    """
    if isinstance(mel, np.ndarray):
        mel = torch.tensor(mel)
    if mel.dim() == 3:
        mel = mel.squeeze(0)
    assert mel.shape[0] == n_mels, f"n_mels attendu: {n_mels}, trouvé: {mel.shape[0]}"
    # Inverse dB
    mel = torchaudio.functional.DB_to_amplitude(mel, ref=1.0, power=2.0)
    # Inverse Mel -> Linear spectrogram
    mel_inv = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate
    )(mel)
    # Griffin-Lim pour phase
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length)
    waveform = griffin_lim(mel_inv)
    waveform = waveform / waveform.abs().max()  # Normalisation
    if out_path:
        torchaudio.save(out_path, waveform.unsqueeze(0), sample_rate)
    return waveform

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    # Exemple d'utilisation: python reconstruct_wav_from_mel.py mel.npy out.wav
    mel_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "recon.wav"
    mel = np.load(mel_path)
    wav = mel_to_wav(mel, out_path=out_path)
    print(f"WAV reconstruit: {out_path}")
    plt.figure()
    plt.imshow(mel, aspect='auto', origin='lower', cmap='magma')
    plt.title('Mel-spectrogram')
    plt.show()
