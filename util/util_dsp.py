"""Utilities for signal processing.

Provides mu-law encoding/decoding, waveform quantization helpers, and
DCT/IDCT transforms (Types I and II) with both functional and ``nn.Linear``
backed implementations.

MuLaw code adapted from
https://github.com/fatchord/WaveRNN/blob/master/utils/distribution.py

DCT code adapted from
https://github.com/zh217/torch-dct
"""

from typing import Optional

import numpy as np

import torch
import torch.nn as nn


######################
### WaveForm utilities
######################

def label_2_float(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Convert an N-bit code index to a float value in [-1, 1].

    Note:
        dtype conversion is not handled internally.

    Args:
        x: Code indices. Values should be in [0, 2**bits - 1].
        bits: Number of quantization bits.

    Returns:
        Float tensor with values in [-1, 1], computed as
        ``2 * x / (2**bits - 1) - 1``.
    """
    return 2 * x / (2**bits - 1.) - 1.

def float_2_label(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize a float tensor in [-1, 1] to an N-bit code index.

    If the peak amplitude exceeds 1.0 the signal is rescaled first.
    The returned tensor uses ``torch.float`` even though the values are
    integer code indices.

    Args:
        x: Float tensor with values in [-1, 1].
        bits: Number of quantization bits.

    Returns:
        Quantized tensor with values in [0, 2**bits - 1].
    """
    #assert abs(x).max() <= 1.0
    # scale the peaks
    peak = torch.abs(x).max()
    if peak > 1.0:
        x /= peak
    # quantize
    x = (x + 1.) * (2**bits - 1) / 2
    return torch.clamp(x, 0, 2**bits - 1)

def mulaw_encode(
    x: torch.Tensor,
    quantization_channels: int,
    scale_to_int: bool = True,
) -> torch.Tensor:
    """Apply mu-law companding encoding.

    Adapted from torchaudio
    (https://pytorch.org/audio/functional.html mu_law_encoding).

    Args:
        x: Float-valued waveform tensor with values in (-1, 1).
        quantization_channels: Number of quantization channels.
        scale_to_int: If ``True``, scale the output to integer codes
            (``int64``). If ``False``, return values in (-1, 1).

    Returns:
        Mu-law encoded tensor. ``int64`` when *scale_to_int* is ``True``,
        otherwise ``float``.
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

def mulaw_decode(
    x_mu: torch.Tensor,
    quantization_channels: int,
    input_int: bool = True,
) -> torch.Tensor:
    """Decode a mu-law encoded signal back to a float waveform.

    Adapted from torchaudio
    (https://pytorch.org/audio/functional.html mu_law_encoding).

    Args:
        x_mu: Mu-law encoded tensor.
        quantization_channels: Number of quantization channels.
        input_int: If ``True``, convert *x_mu* from integer codes to float
            before decoding. If ``False``, decode *x_mu* directly.

    Returns:
        Float-valued waveform in (-1, 1).
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

def dct1(x: torch.Tensor) -> torch.Tensor:
    """Compute the Discrete Cosine Transform, Type I.

    Args:
        x: Input signal tensor.

    Returns:
        DCT-I of the signal over the last dimension.
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.rfft(
        torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1)[:, :, 0].view(*x_shape)


def idct1(X: torch.Tensor) -> torch.Tensor:
    """Compute the inverse DCT-I (a scaled DCT-I).

    Satisfies ``idct1(dct1(x)) == x``.

    Args:
        X: Input signal tensor.

    Returns:
        Inverse DCT-I of the signal over the last dimension.
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x: torch.Tensor, norm: Optional[str] = None) -> torch.Tensor:
    """Compute the Discrete Cosine Transform, Type II.

    For the meaning of *norm*, see
    `scipy.fftpack.dct <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html>`_.

    Args:
        x: Input signal tensor.
        norm: Normalization mode, ``None`` or ``'ortho'``.

    Returns:
        DCT-II of the signal over the last dimension.
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

#     Vc = torch.fft.fft(v, dim=1)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi/(2*N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X: torch.Tensor, norm: Optional[str] = None) -> torch.Tensor:
    """Compute the inverse DCT-II (DCT Type III).

    Satisfies ``idct(dct(x)) == x``.

    For the meaning of *norm*, see
    `scipy.fftpack.dct <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html>`_.

    Args:
        X: Input signal tensor.
        norm: Normalization mode, ``None`` or ``'ortho'``.

    Returns:
        Inverse DCT-II of the signal over the last dimension.
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, 
                     device=X.device)[None, :]*np.pi/(2*N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

#     V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

#     v = torch.irfft(V, 1, onesided=False)
    V_complex = torch.complex(V_r, V_i)
    v = torch.fft.ifft(V_complex, dim=1).real
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


class LinearDCT(nn.Linear):
    """DCT implemented as a fixed (non-trainable) linear layer.

    Executes roughly 50x faster than the functional DCT on GPU at the
    cost of storing the full DCT matrix in memory.

    Args:
        in_features: Size of the expected input (last dimension).
        type: Which DCT variant to use. One of ``'dct1'``, ``'idct1'``,
            ``'dct'``, or ``'idct'``.
        norm: Normalization mode forwarded to the underlying DCT function.
            ``None`` or ``'ortho'``.
        bias: If ``True``, add a learnable bias (default ``False``).
    """

    def __init__(
        self,
        in_features: int,
        type: str,
        norm: Optional[str] = None,
        bias: bool = False,
    ) -> None:
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self) -> None:
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
        self.weight.requires_grad = False # don't learn this!
