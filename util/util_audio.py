import librosa
import torch, torchaudio
import numpy as np

class SoxEffectTransform(torch.nn.Module):
    def __init__(self, effects) :
        super().init()
        self.effects = effects

    def forward(self, tensor: torch.Tensor, sample_rate: int):
        return torchaudio.sox_effects.apply_effects_tensor(tensor, sample_rate, self.effects)


class TorchCQT():
    """ A class to compute pseudo-CQT with Pytorch.
    Written by Keunwoo Choi
    API (+implementations) follows librosa (https://librosa.github.io/librosa/generated/librosa.core.pseudo cqt.html)
    Usage:
    sre, _ = librosa.load (filename)
    src_tensor = torch.tensor (src)
    cqt_calculator = PytorchCqt()
    cqt_calculator(src_tensor)
    """

    def __init__(self, device, sr=22050, hop_length=512, fmin=None, n_bins=84, 
                bins_per_octave=12, tuning=0.0, filter_scale=1,
                norm=1, sparsity=0.01, 
                window='hann', scale=True, 
                pad_mode='reflect'):
        assert scale
        assert window == "hann"

        if fmin is None:
            fmin = librosa.note_to_hz('C1')
        
        if tuning is None:
            tuning = 0.0 # let's make it simple
        cqt_filter_fft = librosa.constantq.__cqt_filter_fft
        fft_basis, n_fft, _ = cqt_filter_fft(sr, fmin, n_bins, bins_per_octave,
                                            filter_scale, norm, sparsity,
                                            hop_length=hop_length, window=window)

        fft_basis = np.abs(fft_basis.astype(dtype=np.float)).todense() # because it was sparse. (n _bins, n_fft)
        self.fft_basis = torch.tensor(fft_basis).float() # .cuda(device) # (n_freq, n_bins)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.scale = scale
        self.window = torch.hann_window(self.n_fft) # .cuda (device)
        self.device = device

    def __call__(self, y):
        return self.forward(y)

    def forward(self, y):
        EPS = 1e-7
        D_torch = torch.stft(y, self.n_fft,
                             hop_length=self.hop_length,
                             window=self.window,
                             return_complex=True).pow(2).sum(-1) # n _freq, time
        
        D_torch = torch.sqrt(D_torch + EPS) # without EPS, backpropagating through QT can yield NaN
        D_torch = D_torch.float()
        # Project onto the pseudo-cqt basis
        C_torch = torch.matmul(self.fft_basis, D_torch)
        # n_bins, time
        C_torch /= torch.tensor(np.sqrt(self.n_fft))
        # because scale is always True
        return C_torch
