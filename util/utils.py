"""Shared utility helpers for watermark training and evaluation."""

import random
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

DEFAULT_N_FFT = 1024


def load_audio(
    path: Union[str, Path],
) -> Tuple[torch.Tensor, int]:
    """Load an audio file and return ``(waveform, sample_rate)``.

    Uses ``soundfile`` directly to avoid the ``torchcodec`` dependency
    introduced as the forced default backend in torchaudio ≥ 2.6.

    Args:
        path: Path to an audio file supported by libsndfile (wav, flac, ogg …).

    Returns:
        ``waveform`` of shape ``(channels, samples)`` as a float32 tensor, and
        the native ``sample_rate`` of the file.
    """
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile → (samples, channels); torch convention → (channels, samples)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))
    return waveform, sample_rate
DEFAULT_HOP_LENGTH = 256
DEFAULT_WIN_LENGTH = 1024


class StepCounter:
    """Minimal recoverable global-step container for SpeechBrain checkpoints."""

    def __init__(self, step: int = 0) -> None:
        """Initialize the recoverable step counter.

        Args:
            step: Initial global step value.
        """
        self.step = step

    def state_dict(self) -> dict[str, int]:
        """Return a serialization-friendly state dictionary."""
        return {"step": self.step}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Restore the internal step counter from a state dictionary."""
        self.step = state["step"]


def save_step_counter(counter: StepCounter, path: Any) -> None:
    """Save the global step counter through the SpeechBrain checkpoint API.

    Args:
        counter: Recoverable counter wrapper.
        path: Target checkpoint path provided by SpeechBrain.
    """
    torch.save({"step": counter.step}, path)


def load_step_counter(counter: StepCounter, path: Any, _end_of_epoch: bool) -> None:
    """Restore the global step counter through the SpeechBrain checkpoint API.

    Args:
        counter: Recoverable counter wrapper.
        path: Source checkpoint path provided by SpeechBrain.
        _end_of_epoch: Unused SpeechBrain callback flag.
    """
    state = torch.load(path, map_location="cpu")
    counter.step = state["step"]


def _get_hann_window(device: torch.device) -> torch.Tensor:
    """Build a Hann window on the requested device."""
    return torch.hann_window(DEFAULT_WIN_LENGTH, device=device)


def _apply_frequency_envelope(
    waveform: torch.Tensor,
    mean_envelope: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Filter a watermark waveform by a fixed spectral envelope."""
    stft_waveform = torch.stft(
        waveform.squeeze(1),
        n_fft=DEFAULT_N_FFT,
        hop_length=DEFAULT_HOP_LENGTH,
        win_length=DEFAULT_WIN_LENGTH,
        window=_get_hann_window(waveform.device),
        return_complex=True,
    )
    stft_waveform = stft_waveform * mean_envelope.view(1, -1, 1)
    waveform = torch.istft(
        stft_waveform,
        n_fft=DEFAULT_N_FFT,
        hop_length=DEFAULT_HOP_LENGTH,
        win_length=DEFAULT_WIN_LENGTH,
        window=_get_hann_window(stft_waveform.device),
        length=target_length,
    )
    return waveform.unsqueeze(1)


def _build_watermark_hidden(
    wav: torch.Tensor,
    target: torch.Tensor,
    encoder: Any,
    decoder: Any,
    processor: Any,
    apply_speech_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the common watermark encoder/processor/decoder path."""
    hidden = encoder(wav)
    hidden = processor(hidden, target)

    if apply_speech_mask:
        speech_mask = speech_mask_time(wav)
        hidden_mask = resample_mask_to_hidden(speech_mask, hidden.shape[-1])
        hidden = hidden * hidden_mask
        return decoder(hidden), speech_mask

    return decoder(hidden), None


def get_watermark(
    wav: torch.Tensor,
    target: torch.Tensor,
    encoder: Any,
    decoder: Any,
    processor: Any,
    mean_env: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a shaped watermark and watermarked waveform.

    Args:
        wav: Clean carrier waveform shaped ``(B, 1, T)``.
        target: Collaborator target IDs.
        encoder: Watermark encoder module.
        decoder: Watermark decoder module.
        processor: Label-conditioning processor.
        mean_env: Spectral envelope used to shape the additive watermark.

    Returns:
        Tuple of ``(watermark, watermarked_waveform)``.
    """
    mean_env = mean_env.to(wav.device)
    watermark, speech_mask = _build_watermark_hidden(
        wav,
        target,
        encoder,
        decoder,
        processor,
        apply_speech_mask=True,
    )
    watermark = _apply_frequency_envelope(
        watermark,
        mean_env,
        target_length=wav.shape[-1],
    )
    watermark = watermark - watermark.mean(dim=-1, keepdim=True)
    speech_mask = speech_mask[..., : wav.shape[-1]]
    wav_watermarked = wav + watermark * speech_mask
    return watermark, wav_watermarked


def get_watermark_eval(
    wav: torch.Tensor,
    target: torch.Tensor,
    encoder: Any,
    decoder: Any,
    processor: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a speech-masked watermark for evaluation.

    Args:
        wav: Clean carrier waveform shaped ``(B, 1, T)``.
        target: Collaborator target IDs.
        encoder: Watermark encoder module.
        decoder: Watermark decoder module.
        processor: Label-conditioning processor.

    Returns:
        Tuple of ``(watermark, watermarked_waveform)``.
    """
    watermark, speech_mask = _build_watermark_hidden(
        wav,
        target,
        encoder,
        decoder,
        processor,
        apply_speech_mask=True,
    )
    watermark = watermark[..., : wav.shape[-1]]
    watermark = watermark - watermark.mean(dim=-1, keepdim=True)
    speech_mask = speech_mask[..., : wav.shape[-1]]
    wav_watermarked = wav + watermark * speech_mask
    return watermark, wav_watermarked


def speech_mask_time(
    wav: torch.Tensor,
    sample_rate: int = 16000,
    vad_frame_ms: int = 20,
    vad_hop_ms: int = 10,
    vad_thresh_db: float = -45.0,
    vad_smooth_frames: int = 5,
) -> torch.Tensor:
    """Compute a simple time-domain speech activity mask."""
    assert wav.dim() == 3 and wav.size(1) == 1, "wav must be shaped (B, 1, T)"
    _, _, num_samples = wav.shape

    frame = max(int(sample_rate * vad_frame_ms / 1000), 1)
    hop = max(int(sample_rate * vad_hop_ms / 1000), 1)

    frames = wav.unfold(dimension=-1, size=frame, step=hop)
    eps = torch.finfo(wav.dtype).eps
    rms = frames.pow(2).mean(dim=-1).clamp_min(eps).sqrt()
    decibels = 20.0 * torch.log10(rms)
    mask = (decibels > vad_thresh_db).to(wav.dtype)

    if vad_smooth_frames > 1:
        kernel = (
            torch.ones(
                1,
                1,
                vad_smooth_frames,
                device=wav.device,
                dtype=wav.dtype,
            )
            / vad_smooth_frames
        )
        padding = vad_smooth_frames // 2
        mask = F.conv1d(mask, kernel, padding=padding).clamp(0.0, 1.0)

    return F.interpolate(mask, size=num_samples, mode="nearest").clamp(0.0, 1.0)


def resample_mask_to_hidden(mask_t: torch.Tensor, hidden_len: int) -> torch.Tensor:
    """Resize a time-domain mask to the sequence length used by a hidden layer."""
    assert mask_t.dim() == 3 and mask_t.size(1) == 1, "mask_t must be shaped (B, 1, T)"
    return F.interpolate(mask_t, size=hidden_len, mode="nearest").clamp(0.0, 1.0)


def set_random_seed(seed: int) -> None:
    """Set deterministic random seeds across NumPy and PyTorch.

    Args:
        seed: Shared random seed used across supported libraries.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
