import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import numpy as np
import librosa
import pickle
import sys

## Adapted from https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/blob/newfunctions/

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

##################
## other utilities
##################
def trimf(x, params):
    """
    trimf: similar to Matlab definition
    https://www.mathworks.com/help/fuzzy/trimf.html?s_tid=srchtitle

    """
    if len(params) != 3:
        print("trimp requires params to be a list of 3 elements")
        sys.exit(1)
    a = params[0]
    b = params[1]
    c = params[2]
    if a > b or b > c:
        print("trimp(x, [a, b, c]) requires a<=b<=c")
        sys.exit(1)
    y = torch.zeros_like(x, dtype=torch.float32)
    if a < b:
        index = np.logical_and(a < x, x < b).bool()
        y[index] = (x[index] - a) / (b - a)
    if b < c:
        index = np.logical_and(b < x, x < c).bool()
        y[index] = (c - x[index]) / (c - b)
    y[x == b] = 1
    return y

def delta(x):
    """ By default
    input
    -----
    x (batch, Length, dim)

    output
    ------
    output (batch, Length, dim)

    Delta is calculated along Length
    """
    length = x.shape[1]
    x_temp = torch_nn_func.pad(x.unsqueeze(1), (0, 0, 1, 1),
                               'replicate').squeeze(1)
    output = -1 * x_temp[:, 0:length] + x_temp[:, 2:]
    return output


class LFCC(torch_nn.Module):
    """ Based on asvspoof.org baseline Matlab code.
    Difference: with_energy is added to set the first dimension as energy

    """

    def __init__(self, fl, fs, fn, sr, filter_num,
                 with_energy=False, with_emphasis=True,
                 with_delta=True):
        super(LFCC, self).__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr
        self.filter_num = filter_num

        f = (sr / 2) * torch.linspace(0, 1, fn // 2 + 1)
        filter_bands = torch.linspace(min(f), max(f), filter_num + 2)

        filter_bank = torch.zeros([fn // 2 + 1, filter_num])
        for idx in range(filter_num):
            filter_bank[:, idx] = trimf(
                f, [filter_bands[idx],
                    filter_bands[idx + 1],
                    filter_bands[idx + 2]])
        self.lfcc_fb = torch_nn.Parameter(filter_bank, requires_grad=False)
        self.l_dct = LinearDCT(filter_num, 'dct', norm='ortho')
        self.with_energy = with_energy
        self.with_emphasis = with_emphasis
        self.with_delta = with_delta
        self.register_buffer("window", torch.hamming_window(self.fl))
        return

    def forward(self, x):
        """

        input:
        ------
         x: tensor(batch, length), where length is waveform length

        output:
        -------
         lfcc_output: tensor(batch, frame_num, dim_num)
        """
        # pre-emphasis
        if self.with_emphasis:
            x[:, 1:] = x[:, 1:] - 0.97 * x[:, 0:-1]

        # STFT

        x_stft = torch.stft(x, self.fn, self.fs, self.fl,
                            window=self.window,
                            onesided=True, pad_mode="constant",
                            return_complex=False)
        # amplitude
        sp_amp = torch.norm(x_stft, 2, -1).pow(2).permute(0, 2, 1).contiguous()

        # filter bank
        fb_feature = torch.log10(torch.matmul(sp_amp, self.lfcc_fb) +
                                 torch.finfo(torch.float32).eps)

        # DCT
        lfcc = self.l_dct(fb_feature)

        # Add energy
        if self.with_energy:
            power_spec = sp_amp / self.fn
            energy = torch.log10(power_spec.sum(axis=2) +
                                 torch.finfo(torch.float32).eps)
            lfcc[:, :, 0] = energy

        # Add delta coefficients
        if self.with_delta:
            lfcc_delta = delta(lfcc)
            lfcc_delta_delta = delta(lfcc_delta)
            lfcc_output = torch.cat((lfcc, lfcc_delta, lfcc_delta_delta), 2)
        else:
            lfcc_output = lfcc

        # done
        return lfcc_output

######################
### WaveForm utilities
######################

def label_2_float(x, bits):
    """Convert integer numbers to float values

    Note: dtype conversion is not handled
    Args:
    -----
       x: data to be converted Tensor.long or int, any shape.
       bits: number of bits, int

    Return:
    -------
       tensor.float

    """
    return 2 * x / (2 ** bits - 1.) - 1.


def float_2_label(x, bits):
    """Convert float wavs back to integer (quantization)

    Note: dtype conversion is not handled
    Args:
    -----
       x: data to be converted Tensor.float, any shape.
       bits: number of bits, int

    Return:
    -------
       tensor.float

    """
    # assert abs(x).max() <= 1.0
    peak = torch.abs(x).max()
    if peak > 1.0:
        x /= peak
    x = (x + 1.) * (2 ** bits - 1) / 2
    return torch.clamp(x, 0, 2 ** bits - 1)


def mulaw_encode(x, quantization_channels, scale_to_int=True):
    """Adapted from torchaudio
    https://pytorch.org/audio/functional.html mu_law_encoding
    Args:
       x (Tensor): Input tensor, float-valued waveforms in (-1, 1)
       quantization_channels (int): Number of channels
       scale_to_int: Bool
         True: scale mu-law companded to int
         False: return mu-law in (-1, 1)

    Returns:
        Tensor: Input after mu-law encoding
    """
    # mu
    mu = quantization_channels - 1.0

    # no check on the value of x
    if not x.is_floating_point():
        x = x.to(torch.float)
    mu = torch.tensor(mu, dtype=x.dtype, device=x.device)
    x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
    if scale_to_int:
        x_mu = ((x_mu + 1) / 2 * mu + 0.5).to(torch.int64)
    return x_mu


def mulaw_decode(x_mu, quantization_channels, input_int=True):
    """Adapted from torchaudio
    https://pytorch.org/audio/functional.html mu_law_encoding
    Args:
        x_mu (Tensor): Input tensor
        quantization_channels (int): Number of channels
    Returns:
        Tensor: Input after mu-law decoding (float-value waveform (-1, 1))
    """
    mu = quantization_channels - 1.0
    if not x_mu.is_floating_point():
        x_mu = x_mu.to(torch.float)
    mu = torch.tensor(mu, dtype=x_mu.dtype, device=x_mu.device)
    if input_int:
        x = ((x_mu) / mu) * 2 - 1.0
    else:
        x = x_mu
    x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.0) / mu
    return x


######################
### DCT utilities
### https://github.com/zh217/torch-dct
### LICENSE: MIT
###
######################

def dct1(x):
    """
    Discrete Cosine Transform, Type I
    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.rfft(
        torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1)[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I
    Our definition if idct1 is such that idct1(dct1(x)) == x
    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/ scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))
    # print(Vc.shape)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/ scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype,
                     device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.irfft(V, 1, onesided=False)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


class LinearDCT(torch_nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""

    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False  # don't learn this!


if __name__ == "__main__":
    lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
    wav, sr = librosa.load("/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_train/flac/LA_T_3727749.flac", sr=16000)
    # wav = torch.randn(1, 32456)
    wav = torch.Tensor(np.expand_dims(wav, axis=0))
    wav_lfcc = lfcc(wav)
    with open('/dataNVME/neil/ASVspoof2019LAFeatures/train' + '/' + "LA_T_3727749" + "LFCC" + '.pkl', 'rb') as feature_handle:
        ref_lfcc = pickle.load(feature_handle)
    print(ref_lfcc.shape)
    print(ref_lfcc[0:3,0:3])
    print(wav_lfcc.shape)
    print(wav_lfcc[0,0:3,0:3])

