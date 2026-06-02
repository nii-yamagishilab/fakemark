"""Thin wrappers around external watermarking baselines used in evaluation."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
import yaml
from audioseal import AudioSeal

from util.utils import load_audio

logger = logging.getLogger(__name__)

from .timbre_model.conv2_mel_modules import Decoder, Encoder

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_TIMBRE_CHECKPOINT = os.environ.get(
    "TIMBRE_CHECKPOINT",
    "checkpoints/timbre/pert-4bit.pth.tar",
)
DEFAULT_TIMBRE_BASELINE_CHECKPOINT = os.environ.get(
    "TIMBRE_BASELINE_CHECKPOINT",
    "checkpoints/timbre/compressed_none-conv2_ep_20.pth.tar",
)


def _load_audio(file_path: str) -> torch.Tensor:
    """Load audio, downmix to mono, and resample to the expected rate.

    Args:
        file_path: Waveform path on disk.

    Returns:
        Mono waveform tensor shaped ``(1, T)``.
    """
    waveform, sample_rate = load_audio(file_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != DEFAULT_SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=DEFAULT_SAMPLE_RATE,
        )(waveform)
    return waveform


def _ensure_batch_channel(waveform: Any) -> torch.Tensor:
    """Normalize waveform shape to ``(batch, channel, time)``.

    Args:
        waveform: Waveform array or tensor in one of the supported shapes.

    Returns:
        Tensor normalized to ``(B, C, T)`` format.
    """
    if not torch.is_tensor(waveform):
        waveform = torch.tensor(waveform)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    return waveform


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a small YAML config file.

    Args:
        path: YAML file path.

    Returns:
        Parsed YAML mapping.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_timbre_configs(config_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load timbre model and preprocessing configs from disk.

    Args:
        config_dir: Directory containing ``process.yaml`` and ``model.yaml``.

    Returns:
        Tuple of preprocessing and model config dictionaries.
    """
    process_config = _load_yaml(config_dir / "process.yaml")
    model_config = _load_yaml(config_dir / "model.yaml")
    return process_config, model_config


def _build_conv2_generator_detector(
    device: Any,
    nbit: int,
    config_dir: Path,
) -> tuple[Encoder, Decoder]:
    """Create the conv2 timbre generator/detector pair.

    Args:
        device: Torch device or device-like identifier.
        nbit: Watermark message width.
        config_dir: Directory containing the timbre YAML configs.

    Returns:
        Tuple of ``(generator, detector)`` modules.
    """
    process_config, model_config = _load_timbre_configs(config_dir)
    win_dim = process_config["audio"]["win_len"]
    embedding_dim = model_config["dim"]["embedding"]
    nlayers_encoder = model_config["layer"]["nlayers_encoder"]
    nlayers_decoder = model_config["layer"]["nlayers_decoder"]
    attention_heads_encoder = model_config["layer"]["attention_heads_encoder"]
    attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]

    generator = Encoder(
        process_config,
        model_config,
        nbit,
        win_dim,
        embedding_dim,
        nlayers_encoder=nlayers_encoder,
        attention_heads=attention_heads_encoder,
    ).to(device)
    detector = Decoder(
        process_config,
        model_config,
        nbit,
        win_dim,
        embedding_dim,
        nlayers_decoder=nlayers_decoder,
        attention_heads=attention_heads_decoder,
    ).to(device)
    return generator, detector


def _repo_timbre_config_dir() -> Path:
    """Return the config directory bundled with this repository."""
    return Path(__file__).resolve().parent / "timbre-config"


class AudioSealWrapper:
    """Wrapper for AudioSeal watermark encoding and detection."""

    def __init__(
        self,
        device: Any,
        nbits: int,
        ckpt_enc: str = "audioseal_wm_16bits",
        ckpt_dec: str = "audioseal_detector_16bits",
    ) -> None:
        """Initialize AudioSeal generator and detector modules.

        Args:
            device: Torch device or device-like identifier.
            nbits: Watermark message width.
            ckpt_enc: AudioSeal generator checkpoint or model name.
            ckpt_dec: AudioSeal detector checkpoint or model name.
        """
        self.encoder = AudioSeal.load_generator(ckpt_enc, nbits=nbits).to(device)
        self.detector = AudioSeal.load_detector(ckpt_dec, nbits=nbits).to(device)
        self.device = device
        logger.info("AudioSeal initialized")

    def encode(self, file_path: str, msg: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a watermark message into an audio file."""
        signal = _load_audio(file_path).unsqueeze(0).to(self.device)
        msg_tensor = torch.tensor(msg, dtype=torch.int64, device=self.device).unsqueeze(0)
        watermark = self.encoder.get_watermark(
            signal,
            sample_rate=DEFAULT_SAMPLE_RATE,
            message=msg_tensor,
        )
        watermarked_signal = (signal + watermark).squeeze(0)
        return signal, watermarked_signal

    def encode_wav(self, waveform: Any, msg: Any) -> torch.Tensor:
        """Encode a watermark message into an in-memory waveform."""
        signal = _ensure_batch_channel(waveform).to(self.device)
        msg_tensor = torch.tensor(msg, dtype=torch.int64, device=self.device).unsqueeze(0)
        watermark = self.encoder.get_watermark(
            signal,
            sample_rate=DEFAULT_SAMPLE_RATE,
            message=msg_tensor,
        )
        return (signal + watermark).squeeze(0)

    def decode(
        self,
        watermarked_signal: Any,
        msg: Any,
    ) -> tuple[torch.Tensor, float]:
        """Decode AudioSeal watermark bits and a scalar detection score."""
        del msg
        watermarked_signal = _ensure_batch_channel(watermarked_signal).to(self.device)
        self.detector.eval()
        with torch.no_grad():
            _, msg_decoded = self.detector.detect_watermark(
                watermarked_signal,
                sample_rate=DEFAULT_SAMPLE_RATE,
            )
            detect_prob = self._get_detection_probability(watermarked_signal)
        return msg_decoded.squeeze().cpu(), detect_prob[0]

    def _get_detection_probability(self, waveform: torch.Tensor) -> list[float]:
        """Average frame-wise detection probabilities into one scalar score."""
        self.detector.eval()
        with torch.no_grad():
            if waveform.ndim != 3:
                raise ValueError(
                    f"Input tensor 'waveform' must have 3 dimensions, got {waveform.ndim}"
                )
            result, _ = self.detector(waveform, sample_rate=DEFAULT_SAMPLE_RATE)
            detect_probs = torch.gt(result[:, 1, :], 0.5).float().mean(dim=1)
        return detect_probs.cpu().tolist()


class AudioSealBaseline:
    """Minimal wrapper for the default AudioSeal release checkpoints."""

    def __init__(self, device: Any) -> None:
        """Initialize the default AudioSeal baseline checkpoints.

        Args:
            device: Torch device or device-like identifier.
        """
        self.encoder = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
        self.detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
        self.device = device
        logger.info("AudioSeal initialized")

    def encode_wav(self, waveform: Any, msg: Any) -> torch.Tensor:
        """Encode a watermark into an in-memory waveform."""
        signal = _ensure_batch_channel(waveform).to(self.device)
        msg_tensor = torch.tensor(msg, dtype=torch.int64, device=self.device)
        watermark = self.encoder.get_watermark(
            signal,
            sample_rate=DEFAULT_SAMPLE_RATE,
            message=msg_tensor,
        )
        return (signal + watermark).squeeze(0)

    def get_watermark(self, waveform: Any, msg: Any) -> torch.Tensor:
        """Return the raw additive AudioSeal watermark."""
        signal = _ensure_batch_channel(waveform).to(self.device)
        msg_tensor = torch.tensor(msg, dtype=torch.int64, device=self.device)
        return self.encoder.get_watermark(
            signal,
            sample_rate=DEFAULT_SAMPLE_RATE,
            message=msg_tensor,
        )


class TimbreWrapper:
    """Wrapper for the conv2-based Timbre watermark model."""

    def __init__(
        self,
        device: Any,
        nbit: int = 4,
        config_dir: Any = None,
        checkpoint_path: str = DEFAULT_TIMBRE_CHECKPOINT,
    ) -> None:
        """Initialize the conv2-based Timbre watermark wrapper.

        Args:
            device: Torch device or device-like identifier.
            nbit: Watermark message width.
            config_dir: Optional override for the timbre config directory.
            checkpoint_path: Timbre checkpoint path.
        """
        self.device = device
        config_dir = _repo_timbre_config_dir() if config_dir is None else Path(config_dir)
        self.generator, self.detector = _build_conv2_generator_detector(
            device=device,
            nbit=nbit,
            config_dir=config_dir,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        self.generator.load_state_dict(checkpoint["encoder"])
        self.detector.load_state_dict(checkpoint["decoder"], strict=False)
        logger.info("Timbre initialized")
        self.generator.eval()
        self.detector.eval()

    def encode(self, file_path: str, msg: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a binary message into an audio file."""
        signal = _load_audio(file_path).unsqueeze(0).to(self.device)
        rescaled_msg = msg * 2 - 1
        if isinstance(msg, np.ndarray):
            msg_tensor = torch.tensor(
                rescaled_msg,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0).unsqueeze(0)
        else:
            msg_tensor = rescaled_msg
        watermarked_signal = self.generator.test_forward(signal, msg_tensor)[0]
        return signal, watermarked_signal.squeeze(0)

    def decode(
        self,
        watermarked_signal: Any,
        msg: Any,
    ) -> tuple[torch.Tensor, float]:
        """Decode a Timbre watermark and return bit predictions plus accuracy."""
        watermarked_signal = _ensure_batch_channel(watermarked_signal).to(self.device)
        with torch.no_grad():
            msg_decoded = self.detector.test_forward(watermarked_signal).squeeze()
        if isinstance(msg, np.ndarray):
            msg = torch.tensor(msg, dtype=torch.float32, device=self.device)
        bw_acc = self._get_bw_acc(msg_decoded, msg)
        return (msg_decoded > 0).int(), bw_acc.item()

    def _get_bw_acc(self, msg_decoded: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        """Compute bit-wise accuracy for decoded timbre messages."""
        msg = msg * 2 - 1
        if msg.dim() == 1:
            msg = msg.unsqueeze(0)
        return (msg_decoded >= 0).eq(msg >= 0).sum(dim=-1).float() / msg.shape[-1]


class TimbreBaseline:
    """Baseline wrapper for the released Timbre model."""

    def __init__(
        self,
        device: Any,
        nbit: int = 10,
        config_dir: Any = None,
        checkpoint_path: str = DEFAULT_TIMBRE_BASELINE_CHECKPOINT,
    ) -> None:
        """Initialize the released Timbre baseline wrapper.

        Args:
            device: Torch device or device-like identifier.
            nbit: Watermark message width.
            config_dir: Optional override for the timbre config directory.
            checkpoint_path: Timbre baseline checkpoint path.
        """
        self.device = device
        config_dir = _repo_timbre_config_dir() if config_dir is None else Path(config_dir)
        self.generator, self.detector = _build_conv2_generator_detector(
            device=device,
            nbit=nbit,
            config_dir=config_dir,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        self.generator.load_state_dict(checkpoint["encoder"])
        self.detector.load_state_dict(checkpoint["decoder"], strict=False)
        logger.info("Timbre initialized")
        self.generator.eval()
        self.detector.eval()

    def encode_wav(self, waveform: Any, msg: Any) -> torch.Tensor:
        """Encode a binary message into an in-memory waveform."""
        signal = _ensure_batch_channel(waveform).to(self.device)
        msg_tensor = torch.tensor(msg, dtype=torch.float32, device=self.device)
        rescaled_msg = msg_tensor * 2 - 1
        watermarked_signal = self.generator.test_forward(signal, rescaled_msg)[0]
        return watermarked_signal.squeeze(0)
