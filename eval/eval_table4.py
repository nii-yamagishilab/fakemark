#!/usr/bin/env python3
"""Evaluate FakeMark attribution under watermarked vs. non-watermarked conditions (Table 4).

Measures the impact of watermark embedding on attribution accuracy across:
  - Systems:    FakeMarkA, FakeMarkT
  - Conditions: no_watermark (clean unwatermarked signal fed to classifier),
                watermarked (watermark embedded with ground-truth label, then distorted)
  - Datasets:   in-domain (MLAAD v5), cross-domain (ASVspoof5 + TIMIT-TTS)
  - Distortions: none, pitch, speed, noise, stok, facodec, wavtokenizer, hifigan,
                 vocos, bigvgan, overwrite, avg-audioseal, avg-fakemarka, avg-fakemarkt

Usage example
-------------
python eval/eval_table4.py \\
    --protocol-indomain /data/protocols/mlaad_v5_test.csv \\
    --protocol-crossdomain /data/protocols/asvspoof5_timit_test.csv \\
    --root-path /data/ \\
    --output results/table4.csv \\
    --fakemarka-exp-dir checkpoints/fakemarka \\
    --fakemarkt-exp-dir checkpoints/fakemarkt \\
    --ssl-checkpoint facebook/mms-300m \\
    --condition all \\
    --distortion none pitch noise overwrite
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Ensure the project root (parent of this script's directory) is on sys.path so
# that ``import models``, ``import perturbations``, etc. resolve correctly when
# the script is run as ``python eval/eval_table4.py`` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

from models.W2V import Model as W2VModel
from models.audioseal_modules import ColProcessor
from models.modules import Encoder as TimbreEncoder
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
    sf.write(str(path), audio_np, sample_rate)


SYSTEMS: List[str] = ["FakeMarkA", "FakeMarkT"]
CONDITIONS: List[str] = ["no_watermark", "watermarked"]
DATASETS: List[str] = ["indomain", "crossdomain"]

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
    if n_per_class is None:
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


# ── Model bundle ──────────────────────────────────────────────────────────────

@dataclass
class FakeMarkBundle:
    """FakeMarkA and FakeMarkT watermark encoders plus their collaborator classifiers."""

    fakemarka_classifier: W2VModel
    fakemarkt_classifier: W2VModel
    fakemarka_encoder: SEANetEncoder
    fakemarka_decoder: SEANetDecoder
    fakemarka_processor: ColProcessor
    fakemarkt_encoder: TimbreEncoder


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


# ── Per-condition prediction helpers ─────────────────────────────────────────

def predict_no_watermark(
    clean_eval: torch.Tensor,
    pert_fn: PerturbationFn,
    classifier: W2VModel,
) -> tuple:
    """Apply distortion to clean (unwatermarked) signal and return (argmax, probs_list).

    This simulates what the classifier sees when no watermark has been embedded,
    serving as the baseline for Table 4's ``no_watermark`` rows.

    Returns:
        Tuple of (pred_int, probs_list).
    """
    pert_eval = pert_fn(clean_eval)
    probs = classifier_softmax(pert_eval, classifier)
    return int(probs.argmax().item()), probs.tolist()


def predict_watermarked_fakemarka(
    clean_batch: torch.Tensor,
    attack_id: torch.Tensor,
    pert_fn: PerturbationFn,
    bundle: FakeMarkBundle,
) -> tuple:
    """Embed FakeMarkA watermark with the ground-truth label, apply distortion,
    and return (argmax, probs_list, pert_eval).

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


def predict_watermarked_fakemarkt(
    clean_batch: torch.Tensor,
    attack_id: torch.Tensor,
    pert_fn: PerturbationFn,
    bundle: FakeMarkBundle,
) -> tuple:
    """Embed FakeMarkT watermark with the ground-truth label, apply distortion,
    and return (argmax, probs_list, pert_eval).

    Returns:
        Tuple of (pred_int, probs_list, pert_eval) where pert_eval is shape (1, T).
    """
    wm_wav, _ = bundle.fakemarkt_encoder(clean_batch, attack_id)
    wm_eval = wm_wav.squeeze(0)        # (1, T)
    pert_eval = pert_fn(wm_eval)
    probs = classifier_softmax(pert_eval, bundle.fakemarkt_classifier)
    return int(probs.argmax().item()), probs.tolist(), pert_eval


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_one(
    system: str,
    condition: str,
    distortion: str,
    dataset_name: str,
    pert_fn: PerturbationFn,
    protocol_df: pd.DataFrame,
    bundle: FakeMarkBundle,
    args: argparse.Namespace,
    device: torch.device,
    wav_output_dir: Optional[Path] = None,
) -> tuple:
    """Compute attribution accuracy for one (system, condition, distortion) combination.

    Args:
        system: ``"FakeMarkA"`` or ``"FakeMarkT"``.
        condition: ``"no_watermark"`` or ``"watermarked"``.
        distortion: Distortion name (used to label per-file records).
        dataset_name: Dataset name (used to label per-file records).
        pert_fn: Perturbation callable applied to the input signal.
        protocol_df: Evaluation protocol with ``Path`` and ``Attack`` columns.
        bundle: Loaded FakeMark model bundle.
        args: Parsed CLI namespace for sample rate.
        device: Torch device for model inference.
        wav_output_dir: When set, distorted waveforms are saved under
            ``<wav_output_dir>/<system>/<condition>/<distortion>/``.
            FakeMarkA uses suffix ``_wm.wav``, FakeMarkT uses ``_t.wav``.
            For ``no_watermark``, the distorted clean signal is saved as
            ``<stem>_clean.wav``.

    Returns:
        Tuple of (accuracy, per_file_records) where accuracy is a float in
        [0, 1] and per_file_records is a list of dicts with keys:
        ``file``, ``system``, ``condition``, ``dataset``, ``distortion``,
        ``true_attack``, ``pred_attack``, ``correct``, ``probs``.
    """
    classifier = (
        bundle.fakemarka_classifier
        if system == "FakeMarkA"
        else bundle.fakemarkt_classifier
    )

    y_true: List[int] = []
    y_pred: List[int] = []
    per_file: List[Dict] = []

    with torch.no_grad():
        for _, row in protocol_df.iterrows():
            file_path = row["Path"]
            attack_name = row["Attack"]
            attack_id_int = model_to_id[attack_name]
            y_true.append(attack_id_int)

            clean_wav = load_waveform(file_path, args.sample_rate, device)
            clean_batch = ensure_batch_channel(clean_wav)   # (1, 1, T)
            clean_eval = clean_wav.squeeze(0)               # (1, T)

            if condition == "no_watermark":
                # No watermark embedding; distortion is applied to the clean signal.
                pred, probs_list = predict_no_watermark(clean_eval, pert_fn, classifier)
                # Optionally save the distorted clean waveform.
                if wav_output_dir is not None:
                    _pert_clean = pert_fn(clean_eval)
                    stem = Path(file_path).stem
                    save_path = (
                        wav_output_dir / system / condition / distortion
                        / f"{stem}_clean.wav"
                    )
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    _save_wav(save_path, _pert_clean, args.sample_rate)
            else:
                # Watermark embedded using the ground-truth label before distortion.
                attack_id_t = torch.tensor([attack_id_int], device=device)
                if system == "FakeMarkA":
                    pred, probs_list, pert_eval = predict_watermarked_fakemarka(
                        clean_batch, attack_id_t, pert_fn, bundle
                    )
                    wav_suffix = "_wm.wav"
                else:
                    pred, probs_list, pert_eval = predict_watermarked_fakemarkt(
                        clean_batch, attack_id_t, pert_fn, bundle
                    )
                    wav_suffix = "_t.wav"
                # Optionally save the distorted watermarked waveform.
                if wav_output_dir is not None:
                    stem = Path(file_path).stem
                    save_path = (
                        wav_output_dir / system / condition / distortion
                        / f"{stem}{wav_suffix}"
                    )
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    _save_wav(save_path, pert_eval, args.sample_rate)

            y_pred.append(pred)
            per_file.append({
                "file": file_path,
                "system": system,
                "condition": condition,
                "dataset": dataset_name,
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
    results: pd.DataFrame, distortions: List[str]
) -> None:
    """Print Table 4: accuracy for each system × condition × dataset × distortion.

    Columns are ordered as:
        FakeMarkA / no_wm / indomain
        FakeMarkA / no_wm / crossdomain
        FakeMarkA / wm    / indomain
        FakeMarkA / wm    / crossdomain
        FakeMarkT / no_wm / indomain
        ...
    Rows are distortions.
    """
    combos = [
        (sys, cond, ds)
        for sys in SYSTEMS
        for cond in CONDITIONS
        for ds in DATASETS
    ]

    col_w = 9
    dist_w = 15

    # Header: shortened column labels "FMA/no/in", "FMA/no/cr", etc.
    header_parts = []
    for sys, cond, ds in combos:
        s_tag = "FMA" if sys == "FakeMarkA" else "FMT"
        c_tag = "no" if cond == "no_watermark" else "wm"
        d_tag = "in" if ds == "indomain" else "cr"
        header_parts.append(f"{s_tag}/{c_tag}/{d_tag}".ljust(col_w))

    header = "Distortion".ljust(dist_w) + "| " + " | ".join(header_parts) + " |"
    print(header)
    print("-" * len(header))

    for dist in distortions:
        label = DISTORTION_DISPLAY.get(dist, dist).ljust(dist_w)
        vals = []
        for sys, cond, ds in combos:
            row = results[
                (results["system"] == sys)
                & (results["condition"] == cond)
                & (results["dataset"] == ds)
                & (results["distortion"] == dist)
            ]
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
        "--protocol-indomain", required=True,
        help="Path to the in-domain (MLAAD v5) evaluation CSV protocol.",
    )
    parser.add_argument(
        "--protocol-crossdomain", required=True,
        help="Path to the cross-domain (ASVspoof5 + TIMIT-TTS) evaluation CSV protocol.",
    )
    parser.add_argument(
        "--root-path", required=True,
        help="Replacement string for the '$ROOT/' placeholder in protocol CSV paths.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Destination path for the results CSV (e.g. results/table4.csv).",
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
        help="Number of source-tracing classes (default: 12).",
    )
    parser.add_argument(
        "--n-per-class", type=int, default=None,
        help="Files per class for balanced evaluation. Defaults to using all available files.",
    )

    # --- Condition and distortion selection ---
    parser.add_argument(
        "--condition", nargs="+", default=["all"],
        choices=CONDITIONS + ["all"],
        help="Condition(s) to evaluate. Pass 'all' for both conditions (default: all).",
    )
    parser.add_argument(
        "--distortion", nargs="+", default=["all"],
        choices=DISTORTION_NAMES + ["all"],
        help="Distortion(s) to apply. Pass 'all' for all distortions (default: all).",
    )

    # --- FakeMarkA / FakeMarkT ---
    parser.add_argument(
        "--fakemarka-exp-dir", required=True,
        help="Directory containing FakeMarkA encoder.ckpt, decoder.ckpt, colprocessor.ckpt.",
    )
    parser.add_argument(
        "--fakemarka-classifier-ckpt", default=None,
        help="Checkpoint for the FakeMarkA collaborator classifier. "
             "Defaults to collaborator.ckpt inside --fakemarka-exp-dir.",
    )
    parser.add_argument(
        "--fakemarkt-exp-dir", required=True,
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
             "(default: facebook/mms-300m).",
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
        help="When set, distorted waveforms are saved here, organised as "
             "<dir>/<system>/<condition>/<distortion>/. "
             "FakeMarkA uses suffix _wm.wav, FakeMarkT _t.wav, "
             "and no-watermark condition saves <stem>_clean.wav.",
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

    # Resolve condition and distortion lists.
    conditions = CONDITIONS if "all" in args.condition else args.condition
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

    # Load and balance both protocols.
    indomain_df = load_protocol(Path(args.protocol_indomain), args.root_path)
    crossdomain_df = load_protocol(Path(args.protocol_crossdomain), args.root_path)
    indomain_df = balance_protocol(indomain_df, args.n_per_class)
    crossdomain_df = balance_protocol(crossdomain_df, args.n_per_class)
    dataset_dfs: Dict[str, pd.DataFrame] = {
        "indomain": indomain_df,
        "crossdomain": crossdomain_df,
    }
    logger.info(
        "Protocols loaded — in-domain: %d files, cross-domain: %d files",
        len(indomain_df),
        len(crossdomain_df),
    )

    wav_output_dir = Path(args.wav_output_dir) if args.wav_output_dir else None
    if wav_output_dir:
        logger.info("Waveforms will be saved under %s", wav_output_dir)

    # Build perturbation registry and load models.
    perturbations = build_perturbations(args)
    logger.info("Loading FakeMark models ...")
    bundle = load_fakemark_bundle(args, device)

    # Run evaluation over all (distortion, condition, dataset, system) combinations.
    records = []
    perfile_records = []
    for distortion in distortions:
        pert_fn = perturbations[distortion]
        for condition in conditions:
            for dataset_name, protocol_df in dataset_dfs.items():
                for system in SYSTEMS:
                    logger.info(
                        "Evaluating  system=%-10s  condition=%-14s  dataset=%-12s  distortion=%s",
                        system, condition, dataset_name, distortion,
                    )
                    acc, file_rows = evaluate_one(
                        system=system,
                        condition=condition,
                        distortion=distortion,
                        dataset_name=dataset_name,
                        pert_fn=pert_fn,
                        protocol_df=protocol_df,
                        bundle=bundle,
                        args=args,
                        device=device,
                        wav_output_dir=wav_output_dir,
                    )
                    logger.info("  → accuracy = %.4f (%.1f%%)", acc, acc * 100)
                    records.append({
                        "system": system,
                        "condition": condition,
                        "dataset": dataset_name,
                        "distortion": distortion,
                        "accuracy": acc,
                    })
                    perfile_records.extend(file_rows)

    results_df = pd.DataFrame(records)

    # Print formatted table to stdout.
    print()
    print_results_table(results_df, distortions)
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
