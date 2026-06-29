#!/usr/bin/env python3
"""Evaluate source-attribution accuracy for all systems across all distortions.

Covers Table 1 (MLAAD v5) and Table 2 (ASVspoof5 + TIMIT-TTS) by switching
the --protocol argument.

Systems
-------
FakeMarkA   Acoustic watermark (SEANet encoder/decoder + MMS-300M collaborator).
            Watermark is embedded with the ground-truth attack label; the
            collaborator classifies the distorted watermarked signal.
FakeMarkT   Timbre watermark (Timbre encoder + MMS-300M collaborator).
            Same embedding/classification pipeline as FakeMarkA.
AudioSeal   AudioSeal watermark baseline.  Predicted class = nearest-Hamming
            attack over a fixed 4-bit message assignment.
Timbre      Conv2-based Timbre watermark baseline.  Same Hamming decoding.
MMS-300M    Standalone MMS-300M (Wav2Vec) classifier; distortion applied to
            the clean (unwatermarked) signal.  Predicted class = argmax.
ResNet34    Standalone ResNet34 + LFB classifier; same clean-signal pipeline.
            Requires a checkpoint that includes trained lmcl.centers (221 keys,
            e.g. exp_ResNet34_STC-2).

Distortions
-----------
none, pitch, speed, noise, stok, facodec, wavtokenizer, hifigan, vocos,
bigvgan, overwrite, avg-audioseal, avg-fakemarka, avg-fakemarkt

Output files
------------
<output>.csv          Aggregate accuracy table: one row per (system, distortion).
<output>_perfile.csv  Per-file detail: file, system, distortion, true_attack,
                      pred_attack, correct, probs.  ``probs`` is the full
                      per-class softmax vector (list of floats) for softmax-based
                      systems, or null for AudioSeal / Timbre.

Watermarked wav saving  (--wav-output-dir)
------------------------------------------
When --wav-output-dir is given, the distorted watermarked signal for each file
is written to disk, organised by system and distortion::

    <wav-output-dir>/FakeMarkA/<distortion>/<stem>_wm.wav
    <wav-output-dir>/FakeMarkT/<distortion>/<stem>_t.wav
    <wav-output-dir>/AudioSeal/<distortion>/<stem>_as.wav
    <wav-output-dir>/Timbre/<distortion>/<stem>_timbre.wav

Checkpoint notes
----------------
* FakeMarkA exp-dir must contain: encoder.ckpt, decoder.ckpt,
  colprocessor.ckpt, collaborator.ckpt (auto-derived from --fakemarka-exp-dir).
* FakeMarkT exp-dir must contain: encoder.ckpt, collaborator.ckpt.
* AudioSeal accepts either a bundled model-card name (audioseal_wm_16bits) or a
  local .pth training checkpoint with xp.cfg + model keys.
* Timbre accepts a .pth.tar file with encoder + decoder keys.
* ResNet34 ssl.ckpt must contain lmcl.centers (221 keys).  Run
  scripts/resnet_calibrate_centers.py if your checkpoint is missing it.

Usage example
-------------
python eval/eval_attribution.py \\
    --protocol /data/protocols/mlaad_v5_test.csv \\
    --root-path /data/ \\
    --system FakeMarkA FakeMarkT AudioSeal Timbre MMS-300M ResNet34 \\
    --distortion none pitch noise vocos bigvgan \\
    --output results/table1.csv \\
    --fakemarka-exp-dir  /ckpts/FakeMarkA \\
    --fakemarkt-exp-dir  /ckpts/FakeMarkT \\
    --audioseal-encoder-ckpt /ckpts/AudioSeal/checkpoint_generator_epoch260.pth \\
    --audioseal-detector-ckpt /ckpts/AudioSeal/checkpoint_detector_epoch260.pth \\
    --timbre-ckpt        /ckpts/Timbre-4bit.pth.tar \\
    --mms-classifier-ckpt /ckpts/MMS_300M.ckpt \\
    --resnet-classifier-ckpt /ckpts/ResNet.ckpt \\
    --wav-output-dir     results/wavs
"""

import argparse
import itertools
import logging
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

# Ensure the project root (parent of this script's directory) is on sys.path so
# that ``import models``, ``import perturbations``, etc. resolve correctly when
# the script is run as ``python eval/eval_attribution.py`` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

from models.W2V import Model as W2VModel
from models.audioseal_modules import ColProcessor
from models.modules import Encoder as TimbreEncoder
from models.ResNet34 import Model as ResNet34Model
from models.seanet import SEANetDecoder, SEANetEncoder
from perturbations.perturbation import (
    pert_bigvgan,
    pert_facodec,
    pert_hifigan,
    pert_musan,
    pert_pitchshift,
    pert_speed,
    pert_speechtokenizer,
    pert_vocos,
    pert_watermark_overwrite,
    pert_wav_tokenizer,
)
from util.attack_labels import id_to_label, model_to_id
from util.utils import get_watermark_eval, load_audio
from baselines.wrapper import AudioSealWrapper, TimbreWrapper

logger = logging.getLogger(__name__)


def _save_wav(path: Path, tensor: torch.Tensor, sample_rate: int) -> None:
    """Save a waveform tensor to disk using soundfile.

    Args:
        path: Destination file path (parent directory must already exist).
        tensor: Waveform tensor shaped ``(channels, T)`` or ``(T,)``.
        sample_rate: Sample rate in Hz.
    """
    audio_np = tensor.cpu().float().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T  # (C, T) → (T, C); soundfile expects (T, C)
    # If already 1-D, soundfile handles it as mono directly.
    sf.write(str(path), audio_np, sample_rate)


SYSTEM_NAMES: List[str] = [
    "FakeMarkA", "FakeMarkT", "AudioSeal", "Timbre", "MMS-300M", "ResNet34"
]

DISTORTION_NAMES: List[str] = [
    "none",
    "pitch", "speed", "noise",
    "stok", "facodec", "wavtokenizer",
    "hifigan", "vocos", "bigvgan",
    "overwrite",
    "avg-audioseal", "avg-fakemarka", "avg-fakemarkt",
]

# Short labels used when printing the results table.
DISTORTION_DISPLAY: Dict[str, str] = {
    "none":          "-",
    "pitch":         "Pitch",
    "speed":         "Speed",
    "noise":         "Noise",
    "stok":          "SpeechTok",
    "facodec":       "FACodec",
    "wavtokenizer":  "WavTok",
    "hifigan":       "HiFi-GAN",
    "vocos":         "Vocos",
    "bigvgan":       "BigVGAN",
    "overwrite":     "Overwrite",
    "avg-audioseal": "Avg-AudioSeal",
    "avg-fakemarka": "Avg-FakeMarkA",
    "avg-fakemarkt": "Avg-FakeMarkT",
}

PerturbationFn = Callable[[torch.Tensor], torch.Tensor]


# ── Shared helpers (mirroring test.py conventions) ────────────────────────────

def get_device(device_arg: Optional[str]) -> torch.device:
    """Resolve the device used for evaluation."""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_waveform(
    path: str, target_sample_rate: int, device: torch.device
) -> torch.Tensor:
    """Load audio file, downmix to mono, resample to target rate.

    Returns:
        Waveform tensor of shape ``(1, T)``.
    """
    waveform, sample_rate = load_audio(path)
    waveform = waveform.mean(dim=0)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=target_sample_rate
        )
    return waveform.unsqueeze(0).to(device)


def ensure_batch_channel(waveform: torch.Tensor) -> torch.Tensor:
    """Normalize waveform shape to ``(batch, channel, time)``."""
    if waveform.dim() == 1:
        return waveform.unsqueeze(0).unsqueeze(0)
    if waveform.dim() == 2:
        return waveform.unsqueeze(0)
    return waveform


def load_protocol(protocol_path: Path, root_path: str) -> pd.DataFrame:
    """Load and normalize the protocol CSV.

    Replaces the ``$ROOT/`` placeholder in the ``Path`` column so that file
    paths can be resolved on the local machine.
    """
    df = pd.read_csv(protocol_path)
    df["Path"] = df["Path"].str.replace("$ROOT/", root_path, regex=False)
    return df


def balance_protocol(
    df: pd.DataFrame, n_per_class: Optional[int]
) -> pd.DataFrame:
    """Subsample to ``n_per_class`` files per attack class, deterministically."""
    if not n_per_class or n_per_class < 0:
        return df
    df = df.copy()
    df["_attack_id"] = df["Attack"].map(model_to_id).fillna(-1).astype(int)
    balanced = (
        df.groupby("_attack_id", group_keys=False)
        .apply(lambda g: g.sample(n=min(n_per_class, len(g)), random_state=42))
    )
    return balanced.drop(columns=["_attack_id"]).reset_index(drop=True)


def classifier_softmax(
    waveform: torch.Tensor, model: torch.nn.Module
) -> torch.Tensor:
    """Run source classifier and return posterior probabilities."""
    logits, _ = model(waveform)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    return torch.nn.functional.softmax(logits.detach().cpu(), dim=1).squeeze(0)


def nearest_attack_id_soft(
    decoded_message: torch.Tensor,
    attack_to_message: Mapping[int, np.ndarray],
) -> int:
    """Return the attack ID whose pre-assigned message has the smallest L1 distance."""
    if torch.is_tensor(decoded_message):
        decoded_message = decoded_message.detach().cpu().numpy()
    if decoded_message.ndim > 1:
        decoded_message = decoded_message.squeeze()
    best_id, best_dist = -1, np.inf
    for attack_id, assigned_message in attack_to_message.items():
        dist = float(np.sum(np.abs(decoded_message - assigned_message)))
        if dist < best_dist:
            best_id, best_dist = attack_id, dist
    return best_id


def build_attack_message_mapping(
    num_attacks: int, seed: int, nbits: int
) -> Dict[int, np.ndarray]:
    """Create a deterministic attack-ID → binary message mapping."""
    all_messages = np.array(
        list(itertools.product([0, 1], repeat=nbits)), dtype=np.int8
    )
    if num_attacks > len(all_messages):
        raise ValueError(
            f"Cannot assign {num_attacks} unique messages with {nbits} bits."
        )
    rng = np.random.default_rng(seed=seed)
    indices = rng.choice(len(all_messages), size=num_attacks, replace=False)
    return {i: all_messages[indices[i]] for i in range(num_attacks)}


# ── Perturbation registry ─────────────────────────────────────────────────────

def make_pert_averaging(avg_watermark_path: str) -> PerturbationFn:
    """Build an averaging removal attack from a pre-computed average watermark file.

    The returned function subtracts the average watermark from the input
    waveform, truncating to the shorter of the two lengths.
    """
    def _fn(waveform: torch.Tensor) -> torch.Tensor:
        avg_wm, _ = load_audio(avg_watermark_path)
        avg_wm = avg_wm.to(waveform.device)
        min_len = min(waveform.shape[1], avg_wm.shape[1])
        return waveform[:, :min_len] - avg_wm[:, :min_len]
    return _fn


def build_perturbations(args: argparse.Namespace) -> Dict[str, PerturbationFn]:
    """Create the full perturbation registry from CLI arguments.

    The three averaging variants are included only when the corresponding
    ``--avg-*-path`` argument is provided.
    """
    registry: Dict[str, PerturbationFn] = {
        "none":         lambda x: x,
        "pitch":        partial(pert_pitchshift, step_range=(-1, 1), sample_rate=args.sample_rate),
        "speed":        partial(pert_speed, speed_range=(0.95, 1.05)),
        "noise":        partial(pert_musan, noise_category="noise", snr=0),
        "stok":         pert_speechtokenizer,
        "facodec":      pert_facodec,
        "wavtokenizer": pert_wav_tokenizer,
        "hifigan":      pert_hifigan,
        "vocos":        pert_vocos,
        "bigvgan":      pert_bigvgan,
        "overwrite":    pert_watermark_overwrite,
    }
    if args.avg_audioseal_path:
        registry["avg-audioseal"] = make_pert_averaging(args.avg_audioseal_path)
    if args.avg_fakemarka_path:
        registry["avg-fakemarka"] = make_pert_averaging(args.avg_fakemarka_path)
    if args.avg_fakemarkt_path:
        registry["avg-fakemarkt"] = make_pert_averaging(args.avg_fakemarkt_path)
    return registry


# ── Model bundles ─────────────────────────────────────────────────────────────

@dataclass
class FakeMarkBundle:
    """FakeMarkA and FakeMarkT watermark encoders plus their collaborator classifiers."""

    fakemarka_classifier: W2VModel
    fakemarkt_classifier: W2VModel
    fakemarka_encoder: SEANetEncoder
    fakemarka_decoder: SEANetDecoder
    fakemarka_processor: ColProcessor
    fakemarkt_encoder: TimbreEncoder


@dataclass
class WatermarkBundle:
    """AudioSeal and Timbre watermark wrappers with deterministic message assignments."""

    audioseal: AudioSealWrapper
    timbre: TimbreWrapper
    attack_to_message: Dict[int, np.ndarray]


@dataclass
class ClassifierBundle:
    """Standalone classifiers (MMS-300M, ResNet34) that operate on raw waveforms."""

    mms: Optional[W2VModel]
    resnet: Optional[ResNet34Model]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_w2v_classifier(
    checkpoint_path: str,
    ssl_checkpoint_path: str,
    model_name: str,
    output_class: int,
    device: torch.device,
) -> W2VModel:
    """Load a Wav2Vec-based source classifier from a saved state dict.

    Uses ``strict=False`` so that transformers-only buffers absent from older
    fairseq-converted checkpoints (e.g. ``masked_spec_embed``) are silently
    kept at their randomly-initialised values, which is correct since they are
    never used during inference.
    """
    model = W2VModel(model_name, ssl_checkpoint_path, output_class)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    benign_missing = {"m_ssl.model.masked_spec_embed"}
    bad_missing = [k for k in missing if k not in benign_missing]
    if bad_missing:
        raise RuntimeError(
            f"Unexpected missing keys when loading {checkpoint_path}:\n"
            + "\n".join(f"  {k}" for k in bad_missing)
        )
    if unexpected:
        raise RuntimeError(
            f"Unexpected extra keys when loading {checkpoint_path}:\n"
            + "\n".join(f"  {k}" for k in unexpected)
        )
    if missing:
        logger.debug("Benign missing key(s) kept at init value: %s", missing)
    return model.eval().to(device)


def load_resnet_classifier(
    checkpoint_path: str,
    output_class: int,
    device: torch.device,
) -> ResNet34Model:
    """Load a ResNet34 + LFB source classifier from a saved state dict.

    The checkpoint must contain all 221 model keys including ``lmcl.centers``
    (shape ``[num_classes, feat_dim]``).  Use the experiment whose ``ssl.ckpt``
    was saved with the LMCL head included in the model — e.g.
    ``exp_ResNet34_STC-2`` (96.1 % val accuracy, 221 keys).  Do *not* use a
    checkpoint where LMCL was a separate ``modules.lmcl`` not in recoverables
    (e.g. ``exp_mms_300m_STC-RESNET-3``), as ``lmcl.centers`` will be absent
    and classification will be random.
    """
    model = ResNet34Model(out_class=output_class)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Only masked_spec_embed (Wav2Vec buffer, irrelevant here) is benign-missing.
    # lmcl.centers MUST be present for meaningful predictions.
    if "lmcl.centers" in missing:
        raise RuntimeError(
            f"lmcl.centers is missing from {checkpoint_path}.\n"
            "This checkpoint was saved without the LMCL head — predictions will "
            "be random.  Use a checkpoint that includes trained class centres "
            "(e.g. exp_ResNet34_STC-2/ssl.ckpt, 221 keys).\n"
            "Alternatively run scripts/resnet_calibrate_centers.py to recompute "
            "centres from a labelled split."
        )
    bad_missing = [k for k in missing if k != "lmcl.centers"]
    if bad_missing:
        raise RuntimeError(
            f"Unexpected missing keys when loading {checkpoint_path}:\n"
            + "\n".join(f"  {k}" for k in bad_missing)
        )
    if unexpected:
        raise RuntimeError(
            f"Unexpected extra keys when loading {checkpoint_path}:\n"
            + "\n".join(f"  {k}" for k in unexpected)
        )
    return model.eval().to(device)


_TIMBRE_ENCODER_KEY_MAP = {
    "msg_processor": "label_embedding",
    "ENc": "content_encoder",
    "EM": "watermark_embedder",
}


def _remap_timbre_encoder_state_dict(
    state_dict: dict,
) -> dict:
    """Remap old Timbre encoder checkpoint keys to the current attribute names.

    Older checkpoints used abbreviated attribute names that were later renamed
    for clarity::

        msg_processor  →  label_embedding
        ENc            →  content_encoder
        EM             →  watermark_embedder

    Any key that does not start with one of the old prefixes is passed through
    unchanged so that future renames do not need a code update here.
    """
    new_sd = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in _TIMBRE_ENCODER_KEY_MAP.items():
            if key == old_prefix or key.startswith(old_prefix + "."):
                new_key = new_prefix + key[len(old_prefix):]
                break
        new_sd[new_key] = value
    return new_sd


def load_fakemark_bundle(
    args: argparse.Namespace, device: torch.device
) -> FakeMarkBundle:
    """Load FakeMarkA / FakeMarkT watermark encoders and collaborator classifiers."""
    fakemarka_classifier = load_w2v_classifier(
        checkpoint_path=args.fakemarka_classifier_ckpt,
        ssl_checkpoint_path=args.ssl_checkpoint,
        model_name=args.ssl_model_name,
        output_class=args.output_class,
        device=device,
    )
    fakemarkt_classifier = load_w2v_classifier(
        checkpoint_path=args.fakemarkt_classifier_ckpt,
        ssl_checkpoint_path=args.ssl_checkpoint,
        model_name=args.ssl_model_name,
        output_class=args.output_class,
        device=device,
    )

    fakemarka_encoder = SEANetEncoder(
        dimension=args.emb_size,
        n_residual_layers=1,
        norm="weight_norm",
        pad_mode="constant",
        lstm=2,
    )
    fakemarka_decoder = SEANetDecoder(
        dimension=args.emb_size,
        n_residual_layers=1,
        norm="weight_norm",
        pad_mode="constant",
        lstm=2,
    )
    fakemarka_processor = ColProcessor(
        hidden_size=args.emb_size,
        num_labels=args.output_class,
    )
    fakemarkt_encoder = TimbreEncoder(
        msg_length=args.output_class,
        layers_CE=args.timbre_layers_ce,
        layers_EM=args.timbre_layers_em,
        hidden_dim=args.timbre_hidden_dim,
    )

    fakemarka_encoder.load_state_dict(
        torch.load(
            Path(args.fakemarka_exp_dir) / "encoder.ckpt",
            map_location="cpu",
            weights_only=True,
        )
    )
    fakemarka_decoder.load_state_dict(
        torch.load(
            Path(args.fakemarka_exp_dir) / "decoder.ckpt",
            map_location="cpu",
            weights_only=True,
        )
    )
    fakemarka_processor.load_state_dict(
        torch.load(
            Path(args.fakemarka_exp_dir) / "colprocessor.ckpt",
            map_location="cpu",
            weights_only=True,
        )
    )
    fakemarkt_encoder.load_state_dict(
        _remap_timbre_encoder_state_dict(
            torch.load(
                Path(args.fakemarkt_exp_dir) / "encoder.ckpt",
                map_location="cpu",
                weights_only=True,
            )
        )
    )

    fakemarka_encoder.eval().to(device)
    fakemarka_decoder.eval().to(device)
    fakemarka_processor.eval().to(device)
    fakemarkt_encoder.eval().to(device)

    return FakeMarkBundle(
        fakemarka_classifier=fakemarka_classifier,
        fakemarkt_classifier=fakemarkt_classifier,
        fakemarka_encoder=fakemarka_encoder,
        fakemarka_decoder=fakemarka_decoder,
        fakemarka_processor=fakemarka_processor,
        fakemarkt_encoder=fakemarkt_encoder,
    )


def load_watermark_bundle(
    args: argparse.Namespace, device: torch.device
) -> WatermarkBundle:
    """Load AudioSeal and Timbre wrappers with their message assignment tables."""
    audioseal = AudioSealWrapper(
        device=device,
        nbits=args.nbits,
        ckpt_enc=args.audioseal_encoder_ckpt,
        ckpt_dec=args.audioseal_detector_ckpt,
    )
    # Use the default checkpoint path when none is provided on the CLI.
    timbre_kwargs: Dict = {"device": device, "nbit": args.nbits}
    if args.timbre_ckpt:
        timbre_kwargs["checkpoint_path"] = args.timbre_ckpt
    timbre = TimbreWrapper(**timbre_kwargs)

    # AudioSeal / Timbre assign one watermark message per real attack class.
    # This is governed by the number of attribution labels (len(id_to_label)),
    # not the collaborator's output dimension (args.output_class), so the message
    # count stays within the nbits budget regardless of the collaborator size.
    attack_to_message = build_attack_message_mapping(
        num_attacks=len(id_to_label),
        seed=args.message_seed,
        nbits=args.nbits,
    )
    return WatermarkBundle(
        audioseal=audioseal,
        timbre=timbre,
        attack_to_message=attack_to_message,
    )


def load_classifier_bundle(
    args: argparse.Namespace, device: torch.device
) -> ClassifierBundle:
    """Load standalone MMS-300M and ResNet34 classifiers.

    Each classifier is only loaded when its checkpoint path is provided.
    If a checkpoint is None the corresponding bundle field is set to None,
    meaning that system must not appear in the requested --system list.
    """
    mms = (
        load_w2v_classifier(
            checkpoint_path=args.mms_classifier_ckpt,
            ssl_checkpoint_path=args.ssl_checkpoint,
            model_name=args.mms_ssl_model_name,
            output_class=args.output_class,
            device=device,
        )
        if args.mms_classifier_ckpt is not None
        else None
    )
    resnet = (
        load_resnet_classifier(
            checkpoint_path=args.resnet_classifier_ckpt,
            output_class=args.output_class,
            device=device,
        )
        if args.resnet_classifier_ckpt is not None
        else None
    )
    return ClassifierBundle(mms=mms, resnet=resnet)


# ── Per-system prediction helpers ─────────────────────────────────────────────

def predict_fakemarka(
    clean_batch: torch.Tensor,
    attack_id: torch.Tensor,
    pert_fn: PerturbationFn,
    bundle: FakeMarkBundle,
) -> tuple:
    """Embed FakeMarkA watermark, apply distortion, return (argmax, probs_list, pert_wav).

    Returns:
        Tuple of (pred_int, probs_list, pert_eval) where pert_eval is shape (1, T).
    """
    _, wm_wav = get_watermark_eval(
        clean_batch,
        attack_id,
        bundle.fakemarka_encoder,
        bundle.fakemarka_decoder,
        bundle.fakemarka_processor,
    )
    wm_eval = wm_wav.squeeze(0)        # (1, T)
    pert_eval = pert_fn(wm_eval)
    probs = classifier_softmax(pert_eval, bundle.fakemarka_classifier)
    return int(probs.argmax().item()), probs.tolist(), pert_eval


def predict_fakemarkt(
    clean_batch: torch.Tensor,
    attack_id: torch.Tensor,
    pert_fn: PerturbationFn,
    bundle: FakeMarkBundle,
) -> tuple:
    """Embed FakeMarkT watermark, apply distortion, return (argmax, probs_list, pert_wav).

    Returns:
        Tuple of (pred_int, probs_list, pert_eval) where pert_eval is shape (1, T).
    """
    wm_wav, _ = bundle.fakemarkt_encoder(clean_batch, attack_id)
    wm_eval = wm_wav.squeeze(0)        # (1, T)
    pert_eval = pert_fn(wm_eval)
    probs = classifier_softmax(pert_eval, bundle.fakemarkt_classifier)
    return int(probs.argmax().item()), probs.tolist(), pert_eval


def predict_audioseal(
    file_path: str,
    attack_id_int: int,
    pert_fn: PerturbationFn,
    bundle: WatermarkBundle,
) -> tuple:
    """Embed AudioSeal watermark, apply distortion, return (argmax, pert_wav).

    Returns:
        Tuple of (pred_int, pert_wav) where pert_wav is shape (1, T).
        probs is always None — AudioSeal uses Hamming-distance decoding, not softmax.
    """
    message = bundle.attack_to_message[attack_id_int]
    _, wm_wav = bundle.audioseal.encode(file_path, message)   # wm_wav: (1, T)
    pert_wav = pert_fn(wm_wav)
    decoded_msg, _ = bundle.audioseal.decode(pert_wav, message)
    pred = nearest_attack_id_soft(decoded_msg, bundle.attack_to_message)
    return pred, pert_wav


def predict_timbre(
    file_path: str,
    attack_id_int: int,
    pert_fn: PerturbationFn,
    bundle: WatermarkBundle,
) -> tuple:
    """Embed Timbre watermark, apply distortion, return (argmax, pert_wav).

    Returns:
        Tuple of (pred_int, pert_wav) where pert_wav is shape (1, T).
        probs is always None — Timbre uses Hamming-distance decoding, not softmax.
    """
    message = bundle.attack_to_message[attack_id_int]
    _, wm_wav = bundle.timbre.encode(file_path, message)      # wm_wav: (1, T)
    pert_wav = pert_fn(wm_wav)
    decoded_msg, _ = bundle.timbre.decode(pert_wav, message)
    pred = nearest_attack_id_soft(decoded_msg, bundle.attack_to_message)
    return pred, pert_wav


def predict_direct(
    clean_eval: torch.Tensor,
    pert_fn: PerturbationFn,
    classifier: torch.nn.Module,
) -> tuple:
    """Apply distortion to clean signal and return (argmax, probs_list).

    Returns:
        Tuple of (pred_int, probs_list).
    """
    pert_eval = pert_fn(clean_eval)
    probs = classifier_softmax(pert_eval, classifier)
    return int(probs.argmax().item()), probs.tolist()


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_system_distortion(
    system: str,
    distortion: str,
    pert_fn: PerturbationFn,
    protocol_df: pd.DataFrame,
    fakemark_bundle: Optional[FakeMarkBundle],
    wm_bundle: Optional[WatermarkBundle],
    clf_bundle: Optional[ClassifierBundle],
    args: argparse.Namespace,
    device: torch.device,
    wav_output_dir: Optional[Path] = None,
) -> tuple:
    """Compute attribution accuracy for one (system, distortion) combination.

    Args:
        system: One of the entries in SYSTEM_NAMES.
        distortion: Distortion name (used to label per-file records).
        pert_fn: Perturbation callable applied to the relevant signal.
        protocol_df: Evaluation protocol with ``Path`` and ``Attack`` columns.
        fakemark_bundle: Required for FakeMarkA / FakeMarkT.
        wm_bundle: Required for AudioSeal / Timbre.
        clf_bundle: Required for MMS-300M / ResNet34.
        args: Parsed CLI namespace for sample rate and device settings.
        device: Torch device for model inference.
        wav_output_dir: When set, distorted watermarked waveforms for FakeMarkA
            and FakeMarkT are saved under
            ``<wav_output_dir>/<system>/<distortion>/<stem>_wm.wav`` and
            ``<wav_output_dir>/<system>/<distortion>/<stem>_t.wav``
            respectively.

    Returns:
        Tuple of (accuracy, per_file_records) where accuracy is a float in
        [0, 1] and per_file_records is a list of dicts with keys:
        ``file``, ``system``, ``distortion``, ``true_attack``, ``pred_attack``,
        ``correct``, ``probs``.  ``probs`` holds the full per-class softmax
        probability vector as a Python list, or ``None`` for systems that use
        Hamming-distance decoding (AudioSeal, Timbre).
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    per_file: List[Dict] = []

    with torch.no_grad():
        for _, row in protocol_df.iterrows():
            file_path = row["Path"]
            attack_name = row["Attack"]
            attack_id_int = model_to_id[attack_name]
            y_true.append(attack_id_int)

            probs_list: Optional[List[float]] = None

            if system in ("FakeMarkA", "FakeMarkT"):
                clean_wav = load_waveform(file_path, args.sample_rate, device)
                clean_batch = ensure_batch_channel(clean_wav)   # (1, 1, T)
                attack_id_t = torch.tensor([attack_id_int], device=device)
                if system == "FakeMarkA":
                    pred, probs_list, pert_eval = predict_fakemarka(
                        clean_batch, attack_id_t, pert_fn, fakemark_bundle
                    )
                    wav_suffix = "_wm.wav"
                else:
                    pred, probs_list, pert_eval = predict_fakemarkt(
                        clean_batch, attack_id_t, pert_fn, fakemark_bundle
                    )
                    wav_suffix = "_t.wav"
                # Optionally save the distorted watermarked waveform.
                if wav_output_dir is not None:
                    stem = Path(file_path).stem
                    save_path = wav_output_dir / system / distortion / f"{stem}{wav_suffix}"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    _save_wav(save_path, pert_eval, args.sample_rate)

            elif system == "AudioSeal":
                pred, pert_wav_wm = predict_audioseal(
                    file_path, attack_id_int, pert_fn, wm_bundle
                )
                if wav_output_dir is not None:
                    stem = Path(file_path).stem
                    save_path = wav_output_dir / system / distortion / f"{stem}_as.wav"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    _save_wav(save_path, pert_wav_wm, args.sample_rate)

            elif system == "Timbre":
                pred, pert_wav_wm = predict_timbre(
                    file_path, attack_id_int, pert_fn, wm_bundle
                )
                if wav_output_dir is not None:
                    stem = Path(file_path).stem
                    save_path = wav_output_dir / system / distortion / f"{stem}_timbre.wav"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    _save_wav(save_path, pert_wav_wm, args.sample_rate)

            else:
                # MMS-300M / ResNet34: distortion applied to the unwatermarked signal.
                clean_wav = load_waveform(file_path, args.sample_rate, device)
                clf = clf_bundle.mms if system == "MMS-300M" else clf_bundle.resnet
                pred, probs_list = predict_direct(clean_wav, pert_fn, clf)

            y_pred.append(pred)
            per_file.append({
                "file": file_path,
                "system": system,
                "distortion": distortion,
                "true_attack": attack_name,
                "pred_attack": id_to_label.get(pred, str(pred)),
                "correct": int(pred == attack_id_int),
                "probs": probs_list,
            })

    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return acc, per_file


# ── Results table ─────────────────────────────────────────────────────────────

def print_results_table(
    results: pd.DataFrame,
    systems: List[str],
    distortions: List[str],
) -> None:
    """Print attribution accuracy matching the Table 1 / Table 2 format.

    Rows are distortions; columns are systems.  Accuracy is shown as a
    percentage with one decimal place.
    """
    col_w = 10
    dist_w = 15
    header = (
        "Distortion".ljust(dist_w)
        + "| "
        + " | ".join(s[:col_w].ljust(col_w) for s in systems)
        + " |"
    )
    print(header)
    print("-" * len(header))

    for dist in distortions:
        label = DISTORTION_DISPLAY.get(dist, dist).ljust(dist_w)
        vals = []
        for sys in systems:
            row = results[(results["distortion"] == dist) & (results["system"] == sys)]
            if row.empty:
                vals.append("N/A".ljust(col_w))
            else:
                vals.append(f"{row['accuracy'].item() * 100:.1f}".rjust(col_w))
        print(label + "| " + " | ".join(vals) + " |")


# ── Argument parsing ──────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Protocol and I/O ---
    parser.add_argument(
        "--protocol", required=True,
        help="Path to evaluation CSV protocol file. Switch between Table 1 and Table 2 "
             "by pointing at the MLAAD v5 or ASVspoof5+TIMIT-TTS protocol respectively.",
    )
    parser.add_argument(
        "--root-path", required=True,
        help="Replacement string for the '$ROOT/' placeholder in protocol CSV paths.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Destination path for the results CSV (e.g. results/table1.csv).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device, e.g. cuda:0 or cpu. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Target sample rate. All audio is resampled to this value (default: 16000).",
    )
    parser.add_argument(
        "--output-class", type=int, default=12,
        help="Collaborator / watermark label-space dimension, matching the trained "
             "source_tracing_num_labels (default: 12).",
    )
    parser.add_argument(
        "--n-per-class", type=int, default=300,
        help="Files per attribution class for balanced evaluation (default: 300, "
             "matching the balanced test protocol). Pass 0 to use all available files.",
    )

    # --- System and distortion selection ---
    parser.add_argument(
        "--system", nargs="+", default=["all"],
        choices=SYSTEM_NAMES + ["all"],
        help="System(s) to evaluate. Pass 'all' to evaluate every system (default: all).",
    )
    parser.add_argument(
        "--distortion", nargs="+", default=["all"],
        choices=DISTORTION_NAMES + ["all"],
        help="Distortion(s) to apply. Pass 'all' to apply every distortion (default: all).",
    )

    # --- FakeMarkA / FakeMarkT ---
    parser.add_argument(
        "--fakemarka-exp-dir", default=None,
        help="Directory containing FakeMarkA encoder.ckpt, decoder.ckpt, colprocessor.ckpt.",
    )
    parser.add_argument(
        "--fakemarka-classifier-ckpt", default=None,
        help="Checkpoint for the FakeMarkA collaborator classifier. "
             "Defaults to collaborator.ckpt inside --fakemarka-exp-dir.",
    )
    parser.add_argument(
        "--fakemarkt-exp-dir", default=None,
        help="Directory containing the FakeMarkT encoder.ckpt.",
    )
    parser.add_argument(
        "--fakemarkt-classifier-ckpt", default=None,
        help="Checkpoint for the FakeMarkT collaborator classifier. "
             "Defaults to collaborator.ckpt inside --fakemarkt-exp-dir.",
    )
    parser.add_argument(
        "--ssl-checkpoint", default="facebook/mms-300m",
        help="Transformers model ID or local model directory for the SSL backbone "
             "used by FakeMark classifiers (default: facebook/mms-300m).",
    )
    parser.add_argument(
        "--ssl-model-name", default="mms_300m",
        help="Named SSL backbone preset for FakeMark classifiers (default: mms_300m).",
    )
    parser.add_argument(
        "--emb-size", type=int, default=128,
        help="FakeMarkA SEANet embedding dimension (default: 128).",
    )
    parser.add_argument("--timbre-layers-ce", type=int, default=4)
    parser.add_argument("--timbre-layers-em", type=int, default=5)
    parser.add_argument("--timbre-hidden-dim", type=int, default=64)

    # --- AudioSeal / Timbre baseline ---
    parser.add_argument(
        "--audioseal-encoder-ckpt", default="audioseal_wm_16bits",
        help="AudioSeal generator checkpoint.  Accepts either a bundled model-card name "
             "(e.g. 'audioseal_wm_16bits') or a local path to a .pth training checkpoint "
             "that contains 'xp.cfg' and 'model' keys "
             "(e.g. .../AudioSeal/checkpoint_generator_epoch260.pth). "
             "Default: audioseal_wm_16bits.",
    )
    parser.add_argument(
        "--audioseal-detector-ckpt", default="audioseal_detector_16bits",
        help="AudioSeal detector checkpoint.  Same format as --audioseal-encoder-ckpt. "
             "Default: audioseal_detector_16bits.",
    )
    parser.add_argument(
        "--timbre-ckpt", default=None,
        help="Timbre baseline checkpoint path (.pth.tar file with 'encoder' and 'decoder' "
             "keys, e.g. .../Timbre-4bit.pth.tar). "
             "Defaults to the $TIMBRE_CHECKPOINT env var.",
    )
    parser.add_argument(
        "--nbits", type=int, default=4,
        help="Watermark message width for AudioSeal and Timbre (default: 4).",
    )
    parser.add_argument(
        "--message-seed", type=int, default=2,
        help="Seed for the deterministic attack-to-message assignment (default: 2).",
    )

    # --- MMS-300M ---
    parser.add_argument(
        "--mms-classifier-ckpt", default=None,
        help="Checkpoint for the MMS-300M standalone classifier.",
    )
    parser.add_argument(
        "--mms-ssl-model-name", default="mms_300m",
        help="Named SSL backbone preset for MMS-300M (default: mms_300m).",
    )

    # --- ResNet34 ---
    parser.add_argument(
        "--resnet-classifier-ckpt", default=None,
        help="Checkpoint for the ResNet34 standalone classifier (LFB + ResNet backbone).",
    )

    # --- Averaging attack paths ---
    parser.add_argument(
        "--avg-audioseal-path", default=None,
        help="Path to the pre-computed AudioSeal average watermark WAV file.",
    )
    parser.add_argument(
        "--avg-fakemarka-path", default=None,
        help="Path to the pre-computed FakeMarkA average watermark WAV file.",
    )
    parser.add_argument(
        "--avg-fakemarkt-path", default=None,
        help="Path to the pre-computed FakeMarkT average watermark WAV file.",
    )

    # --- Wav output ---
    parser.add_argument(
        "--wav-output-dir", default=None,
        help="When set, the distorted watermarked waveform for each FakeMarkA / "
             "FakeMarkT file is saved here, organised as "
             "<dir>/<system>/<distortion>/<stem>_wm.wav (FakeMarkA) or "
             "<dir>/<system>/<distortion>/<stem>_t.wav (FakeMarkT).",
    )

    return parser


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse arguments, run evaluation, print and save the results table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Suppress noisy HTTP-level logs from huggingface_hub / httpx
    for _noisy in ("httpx", "huggingface_hub", "huggingface_hub.file_download"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    args = build_arg_parser().parse_args()

    # Derive classifier checkpoints from exp dirs when not explicitly provided.
    # The collaborator checkpoint is always saved as collaborator.ckpt inside the exp dir.
    if args.fakemarka_classifier_ckpt is None and args.fakemarka_exp_dir is not None:
        args.fakemarka_classifier_ckpt = str(Path(args.fakemarka_exp_dir) / "collaborator.ckpt")
    if args.fakemarkt_classifier_ckpt is None and args.fakemarkt_exp_dir is not None:
        args.fakemarkt_classifier_ckpt = str(Path(args.fakemarkt_exp_dir) / "collaborator.ckpt")

    # Resolve system and distortion lists.
    systems = SYSTEM_NAMES if "all" in args.system else args.system
    distortions = DISTORTION_NAMES if "all" in args.distortion else args.distortion

    # Drop averaging distortions that have no path configured.
    for d_name, d_arg in [
        ("avg-audioseal", args.avg_audioseal_path),
        ("avg-fakemarka", args.avg_fakemarka_path),
        ("avg-fakemarkt", args.avg_fakemarkt_path),
    ]:
        if d_name in distortions and not d_arg:
            logger.warning("%s requested but no path provided; skipping.", d_name)
            distortions = [d for d in distortions if d != d_name]

    device = get_device(args.device)
    logger.info("Device: %s", device)

    # Load and balance protocol.
    protocol_df = load_protocol(Path(args.protocol), args.root_path)
    protocol_df = balance_protocol(protocol_df, args.n_per_class)
    logger.info("Protocol: %d files after balancing", len(protocol_df))

    # Build perturbation registry.
    perturbations = build_perturbations(args)

    # Load only the model bundles needed for the requested systems.
    needs_fakemark = any(s in systems for s in ("FakeMarkA", "FakeMarkT"))
    needs_wm = any(s in systems for s in ("AudioSeal", "Timbre"))
    needs_clf = any(s in systems for s in ("MMS-300M", "ResNet34"))

    fakemark_bundle: Optional[FakeMarkBundle] = None
    wm_bundle: Optional[WatermarkBundle] = None
    clf_bundle: Optional[ClassifierBundle] = None

    if needs_fakemark:
        logger.info("Loading FakeMark models ...")
        fakemark_bundle = load_fakemark_bundle(args, device)
    if needs_wm:
        logger.info("Loading watermark baseline models ...")
        wm_bundle = load_watermark_bundle(args, device)
    if needs_clf:
        logger.info("Loading standalone classifiers ...")
        clf_bundle = load_classifier_bundle(args, device)

    wav_output_dir = Path(args.wav_output_dir) if args.wav_output_dir else None
    if wav_output_dir:
        logger.info("Watermarked waveforms will be saved under %s", wav_output_dir)

    # Run evaluation over all (distortion, system) combinations.
    records = []
    perfile_records = []
    for distortion in distortions:
        pert_fn = perturbations[distortion]
        for system in systems:
            logger.info("Evaluating  system=%-12s  distortion=%s", system, distortion)
            acc, file_rows = evaluate_system_distortion(
                system=system,
                distortion=distortion,
                pert_fn=pert_fn,
                protocol_df=protocol_df,
                fakemark_bundle=fakemark_bundle,
                wm_bundle=wm_bundle,
                clf_bundle=clf_bundle,
                args=args,
                device=device,
                wav_output_dir=wav_output_dir,
            )
            logger.info("  → accuracy = %.4f (%.1f%%)", acc, acc * 100)
            records.append({"system": system, "distortion": distortion, "accuracy": acc})
            perfile_records.extend(file_rows)

    results_df = pd.DataFrame(records)

    # Print formatted table to stdout.
    print()
    print_results_table(results_df, systems, distortions)
    print()

    # Save aggregate results CSV.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)

    # Save per-file results CSV alongside the aggregate CSV.
    perfile_path = output_path.parent / (output_path.stem + "_perfile" + output_path.suffix)
    pd.DataFrame(perfile_records).to_csv(perfile_path, index=False)
    logger.info("Per-file results saved to %s", perfile_path)


if __name__ == "__main__":
    main()
