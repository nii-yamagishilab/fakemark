import os
import tempfile
import sys
from functools import partial
from torch.nn import functional as F
import torch
import torchaudio
import torchaudio.transforms as T
import glob, os, random
import numpy as np
import librosa
import io
import random
import string
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
import julius
from pathlib import Path
from baselines.wrapper import AudioSealBaseline, TimbreBaseline
from util.utils import load_audio

# Paths to vendored third-party code and weights, relative to the repo root.
# Run setup_third_party.sh once to populate these directories.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_THIRD_PARTY = _REPO_ROOT / "third_party"
_BIGVGAN_PATH = str(_THIRD_PARTY / "BigVGAN")
_BIGVGAN_CKPT_DIR = str(_THIRD_PARTY / "BigVGAN" / "hf_cache")
_WAVTOKENIZER_PATH = str(_THIRD_PARTY / "WavTokenizer")
_WAVTOKENIZER_CKPT = str(_THIRD_PARTY / "WavTokenizer" / "WavTokenizer_small_600_24k_4096.ckpt")
_WAVTOKENIZER_CFG = str(
    _THIRD_PARTY / "WavTokenizer" / "configs"
    / "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
)
_FACODEC_PATH = str(_THIRD_PARTY / "Amphion" / "models" / "codec")
_FACODEC_CKPT_DIR = str(_THIRD_PARTY / "facodec_weights")
_HIFIGAN_SAVEDIR = str(_THIRD_PARTY / "hifigan")

musan_path = "/path/to/your/MUSAN/musan/"
# Optional MUSAN assets can be configured externally when needed.
musan_music_files = glob.glob(os.path.join(musan_path, "music/*/*.wav"))
musan_noise_files = glob.glob(os.path.join(musan_path, "noise/*/*.wav"))
musan_speech_files = glob.glob(os.path.join(musan_path, "speech/*/*.wav"))

def match_length(waveform, target_len):
    """
    Match input to the target length, longer wav will be trimmed randomly,
    shorter ones will be repeated multiple times until longer enough.

    Used in pert_musan()
    """

    L = waveform.shape[-1]
    if L > target_len:
        start = random.randint(0, L - target_len)
        waveform = waveform[..., start:start + target_len]
    else:
        repeat_factor = (target_len + L - 1) // L  # ceil
        waveform = waveform.repeat(1, 1, repeat_factor)[..., :target_len]
    return waveform 

def add_noise_snr(waveform, noise, snr_db):
    """
    Add noise to the waveform with the target SNR.

    Used in pert_musan()
    """

    assert waveform.shape == noise.shape
    signal_power = waveform.pow(2).mean(dim=-1, keepdim=True)
    noise_power = noise.pow(2).mean(dim=-1, keepdim=True)

    snr_linear = 10 ** (snr_db / 10)
    scaling_factor = torch.sqrt(signal_power / (snr_linear * noise_power))
    noise_scaled = noise * scaling_factor
    return waveform + noise_scaled

def pert_musan(
    waveform: torch.Tensor,
    noise_category: str = "noise",
    snr_range=(10, 20),
    snr: int = None,
):
    """ 
    Add noise from MUSAN database to the waveform.

    Arguments
    ---------
    noise_category:
        Types of MUSAN noise, can be "noise", "music", or "speech"
    snr_range:
        Noise will be added with a SNR randomly selected from this range

    """

    target_snr = random.uniform(*snr_range)
    if snr is not None:
        target_snr = snr
    if noise_category == "noise":
        musan_files = musan_noise_files
    elif noise_category == "music":
        musan_files = musan_music_files
    else:
        musan_files = musan_speech_files
    musan_file = random.choice(musan_files)
    noise, sr = load_audio(musan_file)
    if noise.shape[0] > 1:
        noise = noise.mean(dim=0, keepdim=True)
    noise = match_length(noise, waveform.shape[-1])
    noise = noise.squeeze()
    noise = noise.unsqueeze(0)
    pert_waveform = add_noise_snr(
        waveform=waveform,
        noise=noise.to(waveform.device),
        snr_db=target_snr,
    )   
    return pert_waveform

_bigvgan_model = None

def _get_bigvgan():
    """Lazy-load BigVGAN on first use."""
    global _bigvgan_model
    if _bigvgan_model is None:
        if _BIGVGAN_PATH not in sys.path:
            sys.path.append(_BIGVGAN_PATH)
        import bigvgan  # noqa: PLC0415
        _bigvgan_model = bigvgan.BigVGAN.from_pretrained(
            _BIGVGAN_CKPT_DIR, use_cuda_kernel=False
        )
        _bigvgan_model.remove_weight_norm()
        _bigvgan_model = _bigvgan_model.eval().cuda()
    return _bigvgan_model


def pert_bigvgan(waveform, sample_rate=16000):
    if _BIGVGAN_PATH not in sys.path:
        sys.path.append(_BIGVGAN_PATH)
    from meldataset import get_mel_spectrogram  # noqa: PLC0415
    bigvgan_model = _get_bigvgan()
    waveform = torchaudio.functional.resample(
        waveform,
        orig_freq=sample_rate,
        new_freq=24000,
    )
    mel = get_mel_spectrogram(waveform, bigvgan_model.h).cuda()
    with torch.inference_mode():
        wav_gen = bigvgan_model(mel)
    wav_gen = torchaudio.functional.resample(
        wav_gen, orig_freq=24000, new_freq=sample_rate
    )
    return wav_gen.squeeze(0)

def pert_averaging(waveform: torch.Tensor):
    avg_watermark, _ = load_audio(
        str(Path(__file__).resolve().parent / "average_watermark.wav")
    )
    avg_watermark = avg_watermark.to(waveform.device)
    min_len = min(waveform.shape[1], avg_watermark.shape[1])
    waveform = waveform[:, :min_len]
    avg_watermark = avg_watermark[:, :min_len]
    return waveform - avg_watermark


def pert_lowpass(
    waveform: torch.Tensor,
    cutoff_freq: float = 5000,
    sample_rate: int = 16000,
):
    """ 
    Apply a low-pass filter to the waveform.

    If cutoff frequency is not specified, will be randomly chosen from
    [5000, 5500, 6000, 6500, 7000, 7500]
    """

    if cutoff_freq is None:
        cutoff_freq = random.choice([1000, 1500, 2000, 2500, 3000, 3500])
    pert_waveform = julius.lowpass_filter(waveform, cutoff=cutoff_freq / sample_rate)
    return pert_waveform

_audioseal_baseline = None
_timbre_baseline = None

def _get_audioseal_baseline():
    """Lazy-load AudioSealBaseline on first use."""
    global _audioseal_baseline
    if _audioseal_baseline is None:
        _audioseal_baseline = AudioSealBaseline(device="cuda")
    return _audioseal_baseline

def _get_timbre_baseline():
    """Lazy-load TimbreBaseline on first use."""
    global _timbre_baseline
    if _timbre_baseline is None:
        _timbre_baseline = TimbreBaseline(device="cuda", nbit=10)
    return _timbre_baseline

def pert_watermark_overwrite(waveform: torch.Tensor, query=3):
    audioseal = _get_audioseal_baseline()
    timbre = _get_timbre_baseline()
    watermarked_waveform = waveform
    for i in range(query):
        msg_audioseal = torch.randint(0, 2, (1, 16), device=waveform.device)
        watermarked_waveform = audioseal.encode_wav(
            watermarked_waveform, msg_audioseal
        )
        msg_timbre = torch.randint(0, 2, (1, 1, 10), device=waveform.device)
        watermarked_waveform = torchaudio.functional.resample(
            watermarked_waveform, orig_freq=16000, new_freq=22050
        )
        watermarked_waveform = timbre.encode_wav(
            watermarked_waveform, msg_timbre
        )
        watermarked_waveform = torchaudio.functional.resample(
            watermarked_waveform, orig_freq=22050, new_freq=16000
        )
    return watermarked_waveform

def pert_pitchshift(audio, sample_rate, step_range=(-1,1)):
    """Perform pitch shift to the input waveform

    Args:
        waveform:
            The input waveform
        step:
            The steps to shift waveform
    Returns:
        The shifted waveform by torchaudio
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # waveform = waveform.to(device)
    step = random.uniform(*step_range)
    audio = torch.from_numpy(audio) if isinstance(audio, np.ndarray) else audio
    processed = torchaudio.functional.pitch_shift(audio, sample_rate, n_steps=step)

    return processed

def pert_speed(
    waveform: torch.Tensor,
    speed_range: tuple = (0.95, 1.05),
    sample_rate: int = 16000,
):
    """
    Perform time stretch to the waveform.

    Arguments
    ---------
    speed_range:
        A random speed will be chosen from this range, the target waveform will be
        speed up if speed > 1
        slow down if speed < 1
    """

    original_length = waveform.shape[-1]
    speed = torch.FloatTensor(1).uniform_(*speed_range)
    new_sr = int(sample_rate * 1 / speed)
#     pert_waveform = julius.resample.resample_frac(waveform, sample_rate, new_sr)
    pert_waveform = torchaudio.functional.resample(waveform, sample_rate, new_sr)

    pert_length = pert_waveform.shape[-1]
    # adjust output waveform length to be same as original
    if pert_length > original_length:
        pert_waveform = pert_waveform[..., :original_length]
    elif pert_length < original_length:
        pad_amount = original_length - pert_length
        pad = torch.zeros(
            *pert_waveform.shape[:-1], 
            pad_amount, 
            device=pert_waveform.device, 
            dtype=pert_waveform.dtype
        )
        pert_waveform = torch.cat([pert_waveform, pad], dim=-1)
    return pert_waveform

"""WavTokenizer
github: https://github.com/jishengpeng/WavTokenizer
install: Try your own env first; they have requirement.txt
usage:
tested under ../WavTokenizer/
"""

_wavtokenizer_model = None

def _get_wavtokenizer():
    """Lazy-load WavTokenizer on first use."""
    global _wavtokenizer_model
    if _wavtokenizer_model is None:
        if _WAVTOKENIZER_PATH not in sys.path:
            sys.path.append(_WAVTOKENIZER_PATH)
        from decoder.pretrained import WavTokenizer  # noqa: PLC0415
        _wavtokenizer_model = WavTokenizer.from_pretrained0802(
            _WAVTOKENIZER_CFG, _WAVTOKENIZER_CKPT
        )
        _wavtokenizer_model = _wavtokenizer_model.to(torch.device("cpu"))
    return _wavtokenizer_model


def pert_wav_tokenizer(
    waveform,
    original_sr=16000,
):
    """Do encodec attack using WavTokenizer

    Args:
        waveform:
            The input waveform, should be [1, T]
        original_sr:
            Original sampling rate, since the input waveform will be forced to 24kHz
    Returns:
        audio_out:
            Waveform reconstructed by WavTokenizer, [1, T] tensor.
    """
    if _WAVTOKENIZER_PATH not in sys.path:
        sys.path.append(_WAVTOKENIZER_PATH)
    from encoder.utils import convert_audio  # noqa: PLC0415
    wavtokenizer = _get_wavtokenizer()
    waveform = convert_audio(waveform.cpu(), original_sr, 24000, 1)
    bandwidth_id = torch.tensor([0])
    device = torch.device("cpu")
    waveform = waveform.to(device)
    features, discrete_code = wavtokenizer.encode_infer(waveform, bandwidth_id=bandwidth_id)
    audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
    wave_out = torchaudio.functional.resample(audio_out, 24000, original_sr)
    return wave_out.cuda()

_speechtokenizer_model = None
_SPEECHTOKENIZER_CFG = str(_THIRD_PARTY / "SpeechTokenizer" / "config.json")
_SPEECHTOKENIZER_CKPT = str(_THIRD_PARTY / "SpeechTokenizer" / "SpeechTokenizer.pt")

def _get_speechtokenizer():
    """Lazy-load SpeechTokenizer on first use."""
    global _speechtokenizer_model
    if _speechtokenizer_model is None:
        from speechtokenizer import SpeechTokenizer  # noqa: PLC0415
        _speechtokenizer_model = SpeechTokenizer.load_from_checkpoint(
            _SPEECHTOKENIZER_CFG, _SPEECHTOKENIZER_CKPT
        )
        _speechtokenizer_model.cuda().eval()
    return _speechtokenizer_model


def pert_speechtokenizer(waveform, original_sr: int = 16000):
    model = _get_speechtokenizer()
    if original_sr != model.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, original_sr, model.sample_rate
        )
    waveform = waveform.unsqueeze(0)
    with torch.no_grad():
        codes = model.encode(waveform)
        RVQ_1 = codes[:1, :, :]
        RVQ_supplement = codes[1:, :, :]
        recon_wav = model.decode(torch.cat([RVQ_1, RVQ_supplement], axis=0))
    recon_wav = torchaudio.functional.resample(
        recon_wav, model.sample_rate, original_sr
    )
    return recon_wav.squeeze(0)

_vocos_model = None

def _get_vocos():
    """Lazy-load Vocos on first use."""
    global _vocos_model
    if _vocos_model is None:
        from vocos import Vocos  # noqa: PLC0415
        _vocos_model = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        _vocos_model.cuda().eval()
    return _vocos_model


def pert_vocos(waveform, sample_rate=16000):
    vocos = _get_vocos()
    waveform = torchaudio.functional.resample(
        waveform,
        orig_freq=sample_rate,
        new_freq=24000,
    )
    with torch.inference_mode():
        recon_wav = vocos(waveform)
    recon_wav = torchaudio.functional.resample(
        recon_wav, orig_freq=24000, new_freq=sample_rate
    )
    return recon_wav
    
_facodec_encoder = None
_facodec_decoder = None

def _get_facodec():
    """Lazy-load FACodec encoder/decoder on first use."""
    global _facodec_encoder, _facodec_decoder
    if _facodec_encoder is None:
        if _FACODEC_PATH not in sys.path:
            sys.path.append(_FACODEC_PATH)
        from ns3_codec import FACodecEncoder, FACodecDecoder  # noqa: PLC0415
        enc = FACodecEncoder(ngf=32, up_ratios=[2, 4, 5, 5], out_channels=256)
        dec = FACodecDecoder(
            in_channels=256,
            upsample_initial_channel=1024,
            ngf=32,
            up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2,
            vq_num_q_p=1,
            vq_num_q_r=3,
            vq_dim=256,
            codebook_dim=8,
            codebook_size_prosody=10,
            codebook_size_content=10,
            codebook_size_residual=10,
            use_gr_x_timbre=True,
            use_gr_residual_f0=True,
            use_gr_residual_phone=True,
        )
        encoder_ckpt = os.path.join(_FACODEC_CKPT_DIR, "ns3_facodec_encoder.bin")
        decoder_ckpt = os.path.join(_FACODEC_CKPT_DIR, "ns3_facodec_decoder.bin")
        enc.load_state_dict(torch.load(encoder_ckpt))
        dec.load_state_dict(torch.load(decoder_ckpt))
        _facodec_encoder = enc.eval().cuda()
        _facodec_decoder = dec.eval().cuda()
    return _facodec_encoder, _facodec_decoder


def pert_facodec(waveform):
    fa_encoder, fa_decoder = _get_facodec()
    waveform = waveform.unsqueeze(0)
    with torch.no_grad():
        enc_out = fa_encoder(waveform)
        vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)
        recon_wav = fa_decoder.inference(vq_post_emb, spk_embs)
    return recon_wav.squeeze(0)


def pert_quantization(
    waveform, 
    quantization_bit,
):
    """Perform quantization to the given waveform, i.e., mapping continuous
       waveform values to a smaller set of discrete values.

    Args:
        waveform:
            The input waveform, doesn't need specify the shape.
        quantization_bit:
            Size of the targeting quantized value set, larger bit gives relatively
            less rounding errors.
    Returns:
        rescaled_waveform:
            The quantized waveform, should have same shape as the input.
    """

    # Normalize the waveform to the range of the quantization levels
    min_val, max_val = waveform.min(), waveform.max()
    normalized_waveform = (waveform - min_val) / (max_val - min_val)

    # Quantize the normalized waveform
    quantized_waveform = torch.round(normalized_waveform * (quantization_bit - 1))

    # Rescale the quantized waveform back to the original range
    rescaled_waveform = (quantized_waveform / (quantization_bit - 1)) \
                        * (max_val - min_val) + min_val

    return rescaled_waveform

_hifigan_model = None

def _get_hifigan():
    """Lazy-load HiFi-GAN vocoder on first use."""
    global _hifigan_model
    if _hifigan_model is None:
        _hifigan_model = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-libritts-16kHz",
            savedir=_HIFIGAN_SAVEDIR,
            run_opts={"device": "cuda"},
        )
    return _hifigan_model


def pert_hifigan(waveform):
    """Resynthesize the input waveform using a 16kHz version HIFIGAN Vocoder

    Args:
        waveform:
            The input waveform
        dummy:
            Does nothing, just pass a parameter for logging and saving in the main file
    Returns:
        new_waveform:
            The resynthesized waveform with same length as the input one
    """
    hifi_gan = _get_hifigan()
    spectrogram, _ = mel_spectogram(
        audio=waveform.squeeze(), sample_rate=16000,
        hop_length=256, win_length=1024, n_mels=80, n_fft=1024,
        f_min=0.0, f_max=8000.0, power=1, normalized=False,
        min_max_energy_norm=True, norm="slaney", mel_scale="slaney",
        compression=True,
    )
    # Pass to vocoder
    new_waveform = hifi_gan.decode_batch(spectrogram)
    if new_waveform.shape[-1] > waveform.shape[-1]:
        new_waveform = new_waveform[..., :waveform.shape[-1]]
    return new_waveform

def pert_random_trimming(audio: torch.Tensor, max_duration_ms: int = None, sample_rate: int = 16000):
    """
    Randomly trims a 1D or 2D audio tensor to a random segment.
    Same effect as the original function, but depends only on torch.

    Args:
        audio (torch.Tensor): Shape (T,) or (1, T), in range [-1, 1].
        max_duration_ms (int, optional): Max trim length in ms. Defaults to full length.
        sample_rate (int): Audio sample rate in Hz.

    Returns:
        torch.Tensor: Trimmed tensor of shape (1, trimmed_length).
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # (1, T)

    total_len = audio.shape[1]
    total_ms = total_len * 1000 // sample_rate

    if max_duration_ms is None:
        max_duration_ms = total_ms

    min_duration_ms = 1000  # 1 second minimum

    # If audio is too short, return as is
    if total_ms < min_duration_ms:
        return audio

    # Pick random start & duration (in samples)
    start_time_ms = random.randint(0, max(0, total_ms - min_duration_ms))
    duration_ms = random.randint(min_duration_ms, min(max_duration_ms, total_ms - start_time_ms))

    start_sample = int(start_time_ms * sample_rate // 1000)
    duration_samples = int(duration_ms * sample_rate // 1000)

    trimmed = audio[:, start_sample:start_sample + duration_samples]

    return trimmed

def pert_frequency_mask(audio, freq_mask_param):
    if isinstance(audio, torch.Tensor):
        device = audio.device
        audio = audio.detach().cpu().numpy()
        if audio.ndim > 1:
            audio = audio.squeeze()

    # Compute STFT and magnitude-phase separation
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    magnitude, phase = np.abs(stft), np.angle(stft)

    # Apply frequency masking
    db_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    spectrogram = torch.tensor(db_spectrogram, dtype=torch.float32).unsqueeze(0)  # shape: (1, freq, time)
    transform = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
    masked_spectrogram = transform(spectrogram).squeeze(0).numpy()

    # Reconstruct waveform
    masked_magnitude = librosa.db_to_amplitude(masked_spectrogram)
    processed = librosa.istft(masked_magnitude * np.exp(1j * phase), hop_length=512)
    # Normalize to match original peak
    processed = processed / np.max(np.abs(processed))  # peak normalize
    processed = torch.from_numpy(processed).unsqueeze(0)

    return processed.to(device)

def pert_clipping(audio, lower_bound=5, upper_bound=95):
    _lower_bound = np.percentile(audio, lower_bound)
    _upper_bound = np.percentile(audio, lower_bound)
    processed = np.clip(audio, _lower_bound, _upper_bound)
    return processed

def pert_overdrive(audio, gain, colour):
    # https://pytorch.org/audio/main/generated/torchaudio.functional.overdrive.html
    processed =  torchaudio.functional.overdrive(audio, gain=gain, colour=colour)
    return processed 

def pert_eq(audio, sample_rate, min_gain_db, max_gain_db, p):
    # Define the EQ parameters (adjust as needed)
    augmenter = Compose([
        SevenBandParametricEQ(
            min_gain_db=min_gain_db,  # Minimum gain (dB) for each band
            max_gain_db=max_gain_db,   # Maximum gain (dB) for each band
            p=p  # Probability of applying the effect
        )
    ])
    audio_np = audio.cpu().numpy()
    processed = augmenter(audio_np, sample_rate=sample_rate)
    processed = torch.from_numpy(processed)
    return processed.cuda()

def pert_compressor(audio, sample_rate, ratio, threshold_db):

    board = Pedalboard([Compressor(threshold_db=threshold_db, ratio=ratio),])
    audio_np = audio.cpu().numpy()
    processed = board(audio_np, sample_rate)
    processed = torch.from_numpy(processed)
    return processed.cuda()

import noisereduce as nr

def pert_noisegate(audio, sample_rate, threshold):
    # pip install noisereduce
    # Noisereduce is a noise reduction algorithm in python that reduces noise in time-domain signals
    processed = nr.reduce_noise(y = audio.cpu(), sr=sample_rate, n_std_thresh_stationary=threshold, stationary=True)
    processed = torch.from_numpy(processed)
    return processed.cuda()
