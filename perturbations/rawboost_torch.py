import torch
import torchaudio
from scipy import signal

def randRange(x1, x2, integer, device):
    low, high = min(x1, x2), max(x1, x2)
    y = torch.empty(1, device=device).uniform_(low, high).item()
    if integer:
        y = int(y)
    return y


def normWav(x: torch.Tensor, always: bool) -> torch.Tensor:
    abs_max = x.abs().amax(dim=1, keepdim=True)
    if always:
        x = x / abs_max.clamp(min=1e-8)
    else:
        mask = (abs_max > 1.0).squeeze(1)
        if mask.any():
#             x[mask] = x[mask] / abs_max[mask].clamp(min=1e-8)
            # safer: avoid in-place assignment
            scale = abs_max.clamp(min=1e-8)
            x = torch.where(mask.unsqueeze(1), x / scale, x)
    return x


def freq_response(b, fs=16000, n_fft=1024):
    """torch estimation of signal.freqz()
    """
    device = b.device
    b_padded = torch.nn.functional.pad(b, (0, n_fft - b.numel()))
    B = torch.fft.fft(b_padded)
    freqs = torch.fft.fftfreq(n_fft, d=1.0/fs).to(device)
    half = n_fft // 2
    return freqs[:half], B[:half]


def genNotchCoeffs(nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
                   minCoeff=10, maxCoeff=100, minG=0, maxG=0, fs=16000,
                   device='cpu', dtype=torch.float32):
    b = torch.tensor([1.0], dtype=dtype, device=device)
    for _ in range(nBands):
        fc = randRange(minF, maxF, 0, device)
        bw = randRange(minBW, maxBW, 0, device)
        c = randRange(minCoeff, maxCoeff, 1, device)
        if c / 2 == int(c / 2):
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000

        # Still uses scipy for filter design
        inter = signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs)
        inter = torch.tensor(inter, dtype=dtype, device=device)
        b = torchaudio.functional.convolve(inter, b)

    G = randRange(minG, maxG, 0, device)
    _, h = freq_response(b, fs=fs)
    temp = h.abs().amax(keepdim=True)
    b = torch.pow(torch.tensor(10.0, dtype=dtype, device=device), G / 20) * b / temp
    return b


def filterFIR(x, b):
    b = b.view(1, 1, -1)
    x = x.unsqueeze(1)
    y = torch.nn.functional.conv1d(x, b, padding=b.shape[-1] // 2)
    return y.squeeze(1)


def LnL_convolutive_noise(x, fs, N_f=5, nBands=5, minF=20, maxF=8000, minBW=100,
                          maxBW=1000, minCoeff=10, maxCoeff=100, minG=0, maxG=0,
                          minBiasLinNonLin=5, maxBiasLinNonLin=20):
    device, dtype = x.device, x.dtype
    y = torch.zeros_like(x)
    for i in range(N_f):
        g_shift = (minBiasLinNonLin, maxBiasLinNonLin) if i == 1 else (0, 0)
        b = genNotchCoeffs(nBands, minF, maxF, minBW, maxBW,
                           minCoeff, maxCoeff,
                           minG - g_shift[0], maxG - g_shift[1],
                           fs, device=device, dtype=dtype)
        filtered_x = filterFIR(torch.pow(x, i + 1), b)
        y = y + filtered_x
    y = y - torch.mean(y)
    y = normWav(y, always=False)
    return y
    

def ISD_additive_noise(x: torch.Tensor, P=10, g_sd=2):
    device, dtype = x.device, x.dtype
    beta = randRange(0, P, 0, device)  # integer beta in [0, P)

    y = x.clone()
    x_len = x.shape[-1]
    n = int(x_len * (beta / 100))

    # Random unique indices (permute and slice)
    perm = torch.randperm(x_len, device=device)[:n]
    # Random factors in range [-1, 1]
    r1 = 2 * torch.rand(n, device=device) - 1
    r2 = 2 * torch.rand(n, device=device) - 1
    f_r = r1 * r2  # Element-wise product
    # Perturbation
    r = g_sd * x[..., perm] * f_r
    y[..., perm] = x[..., perm] + r

    # Normalize (as per your normWav function)
    y = normWav(y, always=False)
    return y


def SSI_additive_noise(x: torch.Tensor, fs: int, SNRmin=10, SNRmax=40,
                       nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
                       minCoeff=10, maxCoeff=100, minG=0, maxG=0):
    device, dtype = x.device, x.dtype
    noise = torch.randn_like(x)
    b = genNotchCoeffs(nBands, minF, maxF, minBW, maxBW,
                       minCoeff, maxCoeff, minG, maxG,
                       fs, device=device, dtype=dtype)
    noise = filterFIR(noise, b)
    noise = normWav(noise, always=True)
    SNR = randRange(SNRmin, SNRmax, 0, device)
    # 5. Scale noise by SNR relative to x
    noise_energy = torch.linalg.norm(noise, ord=2)
    x_energy = torch.linalg.norm(x, ord=2)
    snr_scalar = x_energy / (10.0 ** (0.05 * SNR))
    noise = noise / noise_energy * snr_scalar
    return x + noise


def process_Rawboost_feature(feature, sr, algo):
    """
    feature: torch.Tensor [B, Time] or [B, 1, Time]
    """
    original_shape = feature.shape
    original_dtype = feature.dtype
    device = feature.device

    # Convert to [B, Time]
    if feature.ndim == 3 and feature.shape[1] == 1:
        feature = feature[:, 0, :]

    if algo == 1:
        feature = LnL_convolutive_noise(feature, sr)
    elif algo == 2:
        feature = ISD_additive_noise(feature)
    elif algo == 3:
        feature = SSI_additive_noise(feature, sr)
    elif algo == 4:
        feature = LnL_convolutive_noise(feature, sr)
        feature = ISD_additive_noise(feature) 
        feature = SSI_additive_noise(feature, sr)
    elif algo == 5:
        feature = LnL_convolutive_noise(feature, sr)
        feature = ISD_additive_noise(feature)
    elif algo == 6:
        feature = LnL_convolutive_noise(feature, sr)
        feature = SSI_additive_noise(feature, sr)
    elif algo == 7:
        feature = ISD_additive_noise(feature)
        feature = SSI_additive_noise(feature, sr)
    elif algo == 8:
        feature1 = LnL_convolutive_noise(feature, sr)
        feature2 = ISD_additive_noise(feature)
        feature = normWav(feature1 + feature2, always=False)

    # Restore shape [B, 1, Time] if needed
    if len(original_shape) == 3 and original_shape[1] == 1:
        feature = feature.unsqueeze(1)

    feature = feature.to(device=device, dtype=original_dtype)
    return feature
