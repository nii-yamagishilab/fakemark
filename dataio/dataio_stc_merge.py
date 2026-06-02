"""SpeechBrain data pipeline for merged STChallenge source-tracing labels."""

import random
from typing import Any, Iterator, Mapping

import speechbrain as sb
import torch
import torchaudio
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.sampler import DynamicBatchSampler

from dataio.active_speech import compute_active_speech_level_detailed

TARGET_SAMPLE_RATE = 16000
MODEL_TYPES = [
    "tts_models-multilingual-multi-dataset-xtts_v2",
    "tts_models-multilingual-multi-dataset-xtts_v1.1",
    "griffin_lim",
    "tts_models-en-ljspeech-vits--neon",
    "tts_models-de-thorsten-tacotron2-DDC",
    "tts_models-de-css10-vits-neon",
    "tts_models-lt-cv-vits",
    "tts_models-fr-mai-tacotron2-DDC",
    "facebook-mms-tts-deu",
    "Metavoice-1B",
    "tts_models-it-mai_female-vits",
    "tts_models-en-ljspeech-speedy-speech",
    "tts_models-fr-css10-vits",
    "tts_models-it-mai_male-vits",
    "tts_models-en-ljspeech-tacotron2-DCA",
    "tts_models-en-ljspeech-fast_pitch",
    "Mars5",
    "MeloTTS",
    "vixTTS",
    "suno-bark-small",
    "tts_models-it-mai_female-glow-tts",
    "suno-bark",
    "tts_models-en-ljspeech-tacotron2-DDC",
    "tts_models-en-ljspeech-tacotron2-DDC_ph",
]


def normalize_model_name(name: str) -> str:
    """Map related TTS systems into merged evaluation families.

    Args:
        name: Raw model identifier from the protocol CSV.

    Returns:
        Normalized model family name.
    """
    normalized_name = name.lower()
    if "xtts" in normalized_name:
        return "xtts"
    if name.startswith("suno-bark"):
        return "suno-bark"
    if "tacotron2" in name:
        return "tacotron2"
    if "vits" in name and "xtts" not in normalized_name:
        return "vits"
    if "fast_pitch" in name:
        return "fast_pitch"
    if "griffin_lim" in name:
        return "griffin_lim"
    if "glow-tts" in name:
        return "glow-tts"
    return name


MERGED_TYPES = sorted({normalize_model_name(name) for name in MODEL_TYPES})
MODEL_TO_ID = {
    name: MERGED_TYPES.index(normalize_model_name(name)) for name in MODEL_TYPES
}

# Backward-compatible aliases for older imports.
model_types = MODEL_TYPES
merged_types = MERGED_TYPES
model_to_id = MODEL_TO_ID


def _load_waveform(
    path: str,
    sample_rate: int,
    audio_channel: int,
) -> torch.Tensor:
    """Load, crop, resample, and level-normalize a training waveform.

    Args:
        path: Input waveform path.
        sample_rate: Input sampling rate from the protocol CSV.
        audio_channel: Number of channels in the stored waveform.

    Returns:
        One-dimensional normalized waveform tensor.
    """
    waveform = sb.dataio.dataio.read_audio(path)
    if int(audio_channel) != 1:
        waveform = waveform.mean(dim=1)
    assert waveform.dim() == 1, waveform.dim()

    sample_rate = int(sample_rate)
    original_length = waveform.shape[0]
    if original_length > 10 * sample_rate:
        segment_length = random.randint(6 * sample_rate, 10 * sample_rate)
        max_start = original_length - segment_length
        start_index = random.randint(0, max_start)
        waveform = waveform[start_index : start_index + segment_length]

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE,
        )

    waveform = compute_active_speech_level_detailed(
        waveform.numpy(),
        TARGET_SAMPLE_RATE,
        target_level_dBm0=random.randint(-26, -10),
    )
    return torch.tensor(waveform).float()


def dataio_prepare(
    hparams: Mapping[str, Any]
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Prepare train/validation datasets plus dataloader kwargs.

    Args:
        hparams: Experiment hyperparameter mapping.

    Returns:
        Tuple of ``(train_data, valid_data, train_loader_kwargs,
        valid_loader_kwargs)``.
    """
    data_folder = hparams["data_folder"]
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"ROOT": data_folder},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"ROOT": data_folder},
    )

    @sb.utils.data_pipeline.takes(
        "Path", "Label", "SampleRate", "AudioChannel", "Attack"
    )
    @sb.utils.data_pipeline.provides("wav", "logit")
    def audio_pipeline(
        path: str,
        label: str,
        sample_rate: int,
        audio_channel: int,
        attack: str,
    ) -> Iterator[object]:
        """Load one protocol row and emit waveform and merged class label.

        Args:
            path: Input waveform path.
            label: Unused legacy label column retained for CSV compatibility.
            sample_rate: Input sampling rate from the protocol CSV.
            audio_channel: Number of audio channels for the file.
            attack: Attack/model identifier used to derive the merged label.

        Yields:
            Normalized waveform tensor followed by the integer class ID.
        """
        del label
        yield _load_waveform(path, sample_rate, audio_channel)
        yield MODEL_TO_ID[attack]

    sb.dataio.dataset.add_dynamic_item([train_data, valid_data], audio_pipeline)
    sb.dataio.dataset.set_output_keys(
        datasets=[train_data, valid_data],
        output_keys=["id", "wav", "logit"],
    )

    train_sampler = DynamicBatchSampler(
        train_data,
        length_func=lambda sample: float(sample["Duration"]),
        **hparams["dynamic_batch_sampler_train"],
    )
    train_loader_kwargs = {
        "batch_sampler": train_sampler,
        "collate_fn": PaddedBatch,
        "num_workers": hparams["num_workers"],
        "pin_memory": True,
    }
    valid_loader_kwargs = {
        "batch_size": hparams["batch_size"],
        "collate_fn": PaddedBatch,
        "num_workers": hparams["num_workers"],
        "pin_memory": True,
    }
    return train_data, valid_data, train_loader_kwargs, valid_loader_kwargs
