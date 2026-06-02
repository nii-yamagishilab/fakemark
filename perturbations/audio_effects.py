import io
import os
import re
import typing as tp
import glob
import random
from functools import partial
import tempfile
import subprocess
import warnings

import julius
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment

from perturbations.encodec_model import CompressionModel
from audiocraft.solvers.compression import CompressionSolver
from perturbations.rawboost_torch import process_Rawboost_feature

def batchwise_random_perturbation(
    waveform: torch.Tensor,
):
    assert waveform.dim() == 3, "Expected waveform of shape [B, 1, T]"
    perturbations = {
        "encodec16": partial(pert_encodec, n_q=16, model=codec_model),
        "speed": partial(pert_speed, speed_range=(0.9, 1.1)),
        "resample": pert_resample,
        "echo": partial(pert_echo, volume_range=(0.1, 0.5)),
        "noise_random": partial(pert_noise_random, noise_std=0.001),
        "noise_pink": partial(pert_noise_pink, noise_std=0.01),
        "lowpass": partial(pert_lowpass, cutoff_freq=5000),
        "highpass": partial(pert_highpass, cutoff_freq=500),
        "bandpass": partial(pert_bandpass, cutoff_freq_low=300, cutoff_freq_high=8000),
        "smooth": partial(pert_smooth, window_size_range=(2, 10)),
        "boost": partial(pert_boost, amount=20),
        "duck": partial(pert_duck, amount=20),
        "aac": pert_aac,
        "mp3": pert_mp3,
        "identity": pert_identity,
    }
    perturb_name, perturb_fn = random.choice(list(perturbations.items()))
    pert_waveform = perturb_fn(waveform)
    return pert_waveform

def pert_rawboost(
    waveform: torch.Tensor,
    algo: int = 5,
    sr: int = 16000,
):
    """
    Perform RawBoost on the input waveform.
    """

    pert_waveform = process_Rawboost_feature(
        feature=waveform,
        sr=sr,
        algo=algo,
    )
    # Normalize to the original amplitude range for stability
#     output_waveform = (
#         pert_waveform
#         / torch.max(torch.abs(pert_waveform))
#         * torch.max(torch.abs(waveform))
#     )
    return pert_waveform

codec_model = CompressionSolver.model_from_checkpoint("//pretrained/facebook/encodec_24khz")
codec_model.train()

def pert_encodec(
    waveform: torch.Tensor,
    n_q: int,
    model: "CompressionModel",
    sample_rate: int = 16000,
):
    """
    Compress and reconstruct the waveform with EnCodec model.
    Adopted from audiocraft/utils/audio_effects.py compress_with_encodec()

    Arguments
    ---------
    n_q:
        number of codebooks, default choices from audiocraft is 4, 8, 16
    """
    orig_len = waveform.shape[-1]
    model.to(waveform.device)
    model.set_num_codebooks(n_q)
    codes, scale = model.encode(
        julius.resample_frac(waveform, old_sr=sample_rate, new_sr=model.sample_rate)
    )
    compressed_waveform = model.decode(codes=codes, scale=scale)
    pert_waveform = julius.resample_frac(
        compressed_waveform, old_sr=model.sample_rate, new_sr=sample_rate
    )
    if pert_waveform.shape[-1] > orig_len:
        pert_waveform = pert_waveform[..., :orig_len]
    elif pert_waveform.shape[-1] < orig_len:
        pad_amount = orig_len - pert_waveform.shape[-1]
        pert_waveform = torch.nn.functional.pad(pert_waveform, (0, pad_amount))
    return pert_waveform

def pert_resample(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    intermediate_freq: int = 32000,
):
    orig_shape = waveform.shape
    temp = julius.resample_frac(
        waveform, old_sr=sample_rate, new_sr=intermediate_freq
    )
    pert_waveform = julius.resample_frac(
        temp, old_sr=intermediate_freq, new_sr=sample_rate
    )
    assert pert_waveform.shape == orig_shape
    return pert_waveform

def pert_speed(
    waveform: torch.Tensor,
    speed_range: tuple = (0.8, 1.2),
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
    pert_waveform = julius.resample.resample_frac(waveform, sample_rate, new_sr)

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

def pert_downsample(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    intermediate_freq: int = 32000,
):
    """
    Add artefacts from resampling operation
    original sampling frequency -> upsample -> intermediate sampling frequency ->
    downsample -> original sampling frequency
    """

    orig_shape = waveform.shape
    temp = julius.resample.resample_frac(waveform, sample_rate, intermediate_freq)
    pert_waveform = julius.resample.resample_frac(temp, intermediate_freq, sample_rate)
    return pert_waveform

def pert_echo(
    waveform: torch.Tensor,
    volume_range: tuple = (0.1, 0.5),
    duration_range: tuple = (0.1, 0.5),
    sample_rate: int = 16000,
):
    """
    Add echo to the waveform

    Arguments
    ---------
    volume_range:
        Volume of echo signal will be randomly chosen from this range
    duration_range:
        Delay of the echo will be randomly chosen from this range
    """

    duration = torch.FloatTensor(1).uniform_(*duration_range)
    volume = torch.FloatTensor(1).uniform_(*volume_range)

    n_samples = int(sample_rate * duration)
    impulse_response = torch.zeros(n_samples).type(waveform.type()).to(waveform.device)

    impulse_response[0] = 1.0  # Direct sound
    impulse_response[
            int(sample_rate * duration) - 1
        ] = volume  # First reflection after 100ms

    impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)
    # bug, unsolvable to me
#     reverbed_signal = julius.fft_conv1d(waveform, impulse_response)
    padded_waveform = torch.nn.functional.pad(
        waveform, (0, impulse_response.shape[-1] - 1)
    )
    reverbed_signal = torch.nn.functional.conv1d(padded_waveform, impulse_response)

    # Normalize to the original amplitude range for stability
    reverbed_signal = (
        reverbed_signal
        / torch.max(torch.abs(reverbed_signal))
        * torch.max(torch.abs(waveform))
    )

    tmp = torch.zeros_like(waveform)
    tmp[..., : reverbed_signal.shape[-1]] = reverbed_signal
    pert_waveform = tmp
    return pert_waveform

def pert_noise_random(
    waveform: torch.Tensor,
    noise_std: float = 0.001,
):
    noise = torch.randn_like(waveform) * noise_std
    pert_waveform = waveform + noise
    return pert_waveform

def pert_noise_pink(
    waveform: torch.Tensor,
    noise_std: float = 0.01,
):
    noise = generate_pink_noise(waveform.shape[-1]) * noise_std
    noise = noise.to(waveform.device)
    pert_waveform = waveform + noise
    return pert_waveform

def generate_pink_noise(length: int) -> torch.Tensor:
    """Generate pink noise using Voss-McCartney algorithm with PyTorch."""
    num_rows = 16
    array = torch.randn(num_rows, length // num_rows + 1)
    reshaped_array = torch.cumsum(array, dim=1)
    reshaped_array = reshaped_array.reshape(-1)
    reshaped_array = reshaped_array[:length]
    # Normalize
    pink_noise = reshaped_array / torch.max(torch.abs(reshaped_array))
    return pink_noise

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

def pert_highpass(
    waveform: torch.Tensor,
    cutoff_freq: float = 500,
    sample_rate: int = 16000,
):
    """
    Apply a high-pass filter to the waveform.

    If the cutin frequency is not specified, will be randomly chosen from
    [100, 300, 500, 700, 900]
    """

    if cutoff_freq is None:
        cutoff_freq = random.choice([500, 1000, 2000, ])
    pert_waveform = julius.highpass_filter(waveform, cutoff=cutoff_freq / sample_rate)
    return pert_waveform

def pert_bandpass(
    waveform: torch.Tensor,
    cutoff_freq_low: float = 300,
    cutoff_freq_high: float = 8000,
    sample_rate: int = 16000,
):
    if cutoff_freq_low is None:
        cutoff_freq_low = random.choice([100, 200, 300, 400, 500, 600])
    if cutoff_freq_high is None:
        cutoff_freq_high = random.choice([5000, 5500, 6000, 6500, 7000, 7500])
    pert_waveform = julius.bandpass_filter(
        waveform,
        cutoff_low=cutoff_freq_low / sample_rate,
        cutoff_high=cutoff_freq_high / sample_rate,
    )
    return pert_waveform

def pert_smooth(
    waveform: torch.Tensor,
    window_size_range: tuple = (2, 10),
):
    window_size = int(torch.FloatTensor(1).uniform_(*window_size_range))
    kernel = torch.ones(1, 1, window_size).type(waveform.type()) / window_size
    kernel = kernel.to(waveform.device)

    smoothed = julius.fft_conv1d(waveform, kernel)
    tmp = torch.zeros_like(waveform)
    tmp[..., : smoothed.shape[-1]] = smoothed
    pert_waveform = tmp
    return pert_waveform

def pert_boost(
    waveform: torch.Tensor,
    amount: float = 20,
):
    pert_waveform = waveform * (1 + amount / 100)
    return pert_waveform

def pert_duck(
    waveform: torch.Tensor,
    amount: float = 20,
):
    pert_waveform = waveform * (1 - amount / 100)
    return pert_waveform

def pert_identity(
    waveform: torch.Tensor,
):
    """
    Return the original waveform.
    """

    return waveform

def apply_compression_skip_grad(tensor: torch.Tensor, compression_fn, **kwargs):
    """Applies a specified compression function to the audio tensor.
    Whire carrying over the grads to the output tensor with skip through estimator
    this is a straight through estimator to make mp3/aac compression differentiable
    see more: Yin et al. 2019 https://arxiv.org/pdf/1903.05662.pdf

    Args:
        tensor (torch.Tensor): The input audio tensor.
        compression_fn (function): The compression function to apply.
        **kwargs: Additional keyword arguments for the compression function.

    Returns:
        torch.Tensor: The output tensor after applying compression and straight through estimator.
    """
    compressed = compression_fn(tensor.detach(), **kwargs)

    # Trim compressed output if needed
    compressed = compressed[:, :, : tensor.size(-1)]

    # Straight through estimator for differentiable compression
    out = tensor + (compressed - tensor).detach()

    # Check that gradients are not broken
    if out.requires_grad:
        assert (
            out.grad_fn
        ), "The computation graph might be broken due to compression augmentation."

    return out

def pert_mp3(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    bitrate: str = "128k",
):
    # to keep wav amplitude same before and after perturbation
    channel_peak = waveform.abs().amax(dim=-1, keepdim=True)
    # perform compression
    pert_waveform = apply_compression_skip_grad(
        waveform, get_mp3, sr=sample_rate, bitrate=bitrate
    )
    # adjust amplitude
    pert_peak = pert_waveform.abs().amax(dim=-1, keepdim=True) + 1e-8
    pert_waveform = pert_waveform / pert_peak * channel_peak
    return pert_waveform

def pert_aac(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    bitrate: str = "128k",
    lowpass_freq: tp.Optional[int] = None,
):
    channel_peak = waveform.abs().amax(dim=-1, keepdim=True)
    pert_waveform = apply_compression_skip_grad(
        waveform, get_aac, sr=sample_rate, bitrate=bitrate, lowpass_freq=lowpass_freq
    )
    pert_peak = pert_waveform.abs().amax(dim=-1, keepdim=True) + 1e-8
    pert_waveform = pert_waveform / pert_peak * channel_peak
    return pert_waveform

def compress(
    wav: torch.Tensor,
    sr: int,
    target_format: tp.Literal["mp3", "ogg", "flac", "aac"] = "mp3",
    bitrate: str = "128k"
) -> tp.Tuple[torch.Tensor, int]:
    """
    Convert audio waveform to a specified lossy format: mp3, ogg, flac, aac

    Args:
        wav (torch.Tensor): Input wav tensor (shape: [channels, time])
        sr (int): Sampling rate
        target_format (str): Compression format
        bitrate (str): Bitrate for compression (e.g., '128k')

    Returns:
        Tuple[torch.Tensor, int]: Compressed audio waveform and sampling rate
    """
    # Parse numeric bitrate
    match = re.search(r"\d+(\.\d+)?", str(bitrate))
    parsed_bitrate = int(match.group()) if match else None
    assert parsed_bitrate, f"Invalid bitrate specified (got {bitrate})"

    try:
        # Convert to AudioSegment (channels first -> channels last)
        audio_np = (wav * 32767).numpy().astype("int16")
        audio_segment = AudioSegment(
            audio_np.T.tobytes(),  # pydub expects bytes
            frame_rate=sr,
            sample_width=2,
            channels=audio_np.shape[0]
        )

        # Export to in-memory buffer
        buffer = io.BytesIO()
        audio_segment.export(buffer, format=target_format, bitrate=bitrate)
        buffer.seek(0)

        # Load back to torch.Tensor using pydub
        compressed_seg = AudioSegment.from_file(buffer, format=target_format)
        compressed_samples = torch.tensor(
            compressed_seg.get_array_of_samples(), dtype=torch.float32
        )

        return compressed_samples, compressed_seg.frame_rate

    except Exception as e:
        warnings.warn(f"Compression failed: {e}. Returning original audio.")
        return wav, sr

def get_mp3(wav_tensor: torch.Tensor, sr: int, bitrate: str = "128k") -> torch.Tensor:
    """Convert a batch of audio files to MP3 format, maintaining the original shape.

    This function takes a batch of audio files represented as a PyTorch tensor, converts
    them to MP3 format using the specified bitrate, and returns the batch in the same
    shape as the input.

    Args:
        wav_tensor (torch.Tensor): Batch of audio files represented as a tensor.
            Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bitrate (str): Bitrate for MP3 conversion, default is '128k'.

    Returns:
        torch.Tensor: Batch of audio files converted to MP3 format, with the same
            shape as the input tensor.
    """
    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape

    # Flatten tensor for conversion and move to CPU
    wav_tensor_flat = wav_tensor.reshape(1, -1).cpu()
    _, len_before_compress = wav_tensor_flat.shape

    # Convert to MP3 format with specified bitrate
    wav_tensor_flat, _ = compress(wav_tensor_flat, sr, bitrate=bitrate)
    wav_tensor_flat = wav_tensor_flat[..., :len_before_compress]

    # Reshape back to original batch format and trim or pad if necessary
    wav_tensor = wav_tensor_flat.reshape(batch_size, channels, -1)
#     compressed_length = wav_tensor.shape[-1]
#     if compressed_length > original_length:
#         wav_tensor = wav_tensor[:, :, :original_length]  # Trim excess frames
#     elif compressed_length < original_length:
#         padding = torch.zeros(
#             batch_size, channels, original_length - compressed_length, device=device
#         )
#         wav_tensor = torch.cat((wav_tensor, padding), dim=-1)  # Pad with zeros

    # Move tensor back to the original device
    return wav_tensor.to(device)

def get_aac(
    wav_tensor: torch.Tensor,
    sr: int,
    bitrate: str = "128k",
    lowpass_freq: tp.Optional[int] = None,
) -> torch.Tensor:
    """Converts a batch of audio tensors to AAC format and then back to tensors.

    This function first saves the input tensor batch as WAV files, then uses FFmpeg to convert
    these WAV files to AAC format. Finally, it loads the AAC files back into tensors.

    Args:
        wav_tensor (torch.Tensor): A batch of audio files represented as a tensor.
                                   Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bitrate (str): Bitrate for AAC conversion, default is '128k'.
        lowpass_freq (Optional[int]): Frequency for a low-pass filter. If None, no filter is applied.

    Returns:
        torch.Tensor: Batch of audio files converted to AAC and back, with the same
                      shape as the input tensor.
    """
    import tempfile
    import subprocess

    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape

    # Parse the bitrate value from the string
    match = re.search(r"\d+(\.\d+)?", bitrate)
    parsed_bitrate = (
        match.group() if match else "128"
    )  # Default to 128 if parsing fails

    # Flatten tensor for conversion and move to CPU
    if torch.isnan(wav_tensor).any() or torch.isinf(wav_tensor).any():
        wav_tensor = torch.nan_to_num(wav_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    wav_tensor_flat = wav_tensor.reshape(1, -1).cpu()

    with tempfile.NamedTemporaryFile(
        suffix=".wav"
    ) as f_in, tempfile.NamedTemporaryFile(suffix=".m4a") as f_out:
        input_path, output_path = f_in.name, f_out.name

        # Save the tensor as a WAV file
        sf.write(input_path, wav_tensor_flat.squeeze(0).numpy(), sr)

        # Prepare FFmpeg command: AAC in M4A container
        command = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-ar", str(sr),
            "-b:a", bitrate,
            "-c:a", "aac",   # AAC encoder
            output_path
        ]
        if lowpass_freq is not None:
            # Optional low-pass filter
            command = ["ffmpeg", "-y", "-i", input_path, "-af", f"lowpass=f={lowpass_freq}",
                       "-ar", str(sr), "-b:a", bitrate, "-c:a", "aac", output_path]

        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Load the AAC audio back into a tensor
#         aac_tensor, _ = torchaudio.load(output_path)
        audio_seg = AudioSegment.from_file(output_path, format="m4a")
        aac_tensor = torch.tensor(
            audio_seg.get_array_of_samples(), 
            dtype=torch.float32,
            device=device,
        )
        
    if audio_seg.channels > 1:
        aac_tensor = aac_tensor.reshape(-1, audio_seg.channels).T
    else:
        aac_tensor = aac_tensor.unsqueeze(0)

    original_length_flat = batch_size * channels * original_length
    compressed_length_flat = aac_tensor.shape[-1]

    # Trim excess frames
    if compressed_length_flat > original_length_flat:
        aac_tensor = aac_tensor[:, :original_length_flat]

    # Pad the shortedn frames
    elif compressed_length_flat < original_length_flat:
        padding = torch.zeros(
            1, original_length_flat - compressed_length_flat, device=device
        )
        aac_tensor = torch.cat((aac_tensor, padding), dim=-1)

    # Reshape and adjust length to match original tensor
    wav_tensor = aac_tensor.reshape(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]

    assert compressed_length == original_length, (
        "AAC-compressed audio does not have the same frames as original one. "
        "One reason can be ffmpeg is not  installed and used as proper backed "
        "for torchaudio, or the AAC encoder is not correct. Run "
        "`torchaudio.utils.ffmpeg_utils.get_audio_encoders()` and make sure we see entry for"
        "AAC in the output."
    )
    return wav_tensor.to(device)
