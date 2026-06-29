<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <h1>FakeMark</h1>
  <p><strong>Deepfake speech attribution via watermarked artifacts</strong></p>
</div>

<div align="center" style="line-height: 1;">
  <a href="./LICENSE"><img alt="Code License"
    src="https://img.shields.io/badge/Code_License-BSD%203--Clause-f5de53?&color=f5de53"/></a>
  <a href="./requirements.txt"><img alt="Python"
    src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white"/></a>
  <a href="https://arxiv.org/abs/2510.12042"><img alt="arXiv"
    src="https://img.shields.io/badge/arXiv-2510.12042-b31b1b?logo=arxiv&logoColor=white"/></a>
</div>

---

## 1. Introduction

FakeMark injects **system-specific artifact patterns** as watermarks rather than predefined bit-string messages. This design lets the decoder attribute the source system by jointly leveraging the injected watermark *and* the intrinsic deepfake artifacts already present in the signal.

Two watermark generator architectures are provided:

- **FakeMarkA** — SEANet-style acoustic generator that conditions watermark injection on a source label.
- **FakeMarkT** — Timbre-style spectral generator with the same label conditioning.

Both share a unified **collaborator** classifier (MMS-300M) that performs attribution from the received watermarked signal.

---

## 2. Table of Contents

- [1. Introduction](#1-introduction)
- [2. Table of Contents](#2-table-of-contents)
- [3. Installation](#3-installation)
- [4. Repository Layout](#4-repository-layout)
- [5. Training](#5-training)
- [6. Evaluation](#6-evaluation)
- [7. Acknowledgments](#7-acknowledgments)
- [8. Citation](#8-citation)

---

## 3. Installation

```bash
conda create -n fakemark python=3.10 -y
conda activate fakemark
conda install -c conda-forge ffmpeg -y
pip install --upgrade pip
pip install -r requirements.txt
# audiocraft is installed separately with --no-deps: audiocraft==1.1.0 hard-pins
# av==11.0.0, which has no prebuilt wheel and fails to build against modern ffmpeg.
# Its required runtime deps are already in requirements.txt, so --no-deps is safe.
pip install --no-deps audiocraft==1.1.0
```

> **Note:** `ffmpeg` must be on your `PATH` (the `conda install` above handles this).
> Training's audio-effect augmentation and `pydub` shell out to the `ffmpeg` binary.

### Checkpoints

Pretrained checkpoints are available for download from:

- 🤗 HuggingFace: [nii-yamagishilab/fakemark](https://huggingface.co/nii-yamagishilab/fakemark)
- 📦 Zenodo: [https://zenodo.org/records/20192327](https://zenodo.org/records/20192327)

Expected checkpoint layout under the repository root:

```
checkpoints/
├── FakeMarkA/
│   ├── encoder.ckpt          # SEANet watermark generator
│   ├── decoder.ckpt          # SEANet decoder
│   ├── colprocessor.ckpt     # ColProcessor conditioning module
│   └── collaborator.ckpt     # MMS-300M collaborator (attribution classifier)
├── FakeMarkT/
│   ├── encoder.ckpt          # Timbre watermark generator
│   └── collaborator.ckpt     # MMS-300M collaborator
├── AudioSeal/
│   ├── checkpoint_generator_epoch260.pth   # AudioSeal generator (retrained)
│   └── checkpoint_detector_epoch260.pth    # AudioSeal detector (retrained)
├── Timbre-4bit.pth.tar       # Timbre generator/detector (retrained)
├── MMS_300M.ckpt             # Standalone MMS-300M classifier
└── ResNet.ckpt               # Standalone ResNet34 + LFB + LMCL classifier
```

---

## 4. Repository Layout

```
FakeMark/
├── baselines/
│   ├── timbre_model/           # Conv2-based Timbre model source
│   ├── timbre-config/          # Timbre process.yaml / model.yaml
│   └── wrapper.py              # AudioSealWrapper, TimbreWrapper
├── config/
│   ├── fakemark_a.yaml         # FakeMarkA training config
│   └── fakemark_t.yaml         # FakeMarkT training config
├── dataio/                     # SpeechBrain data-loading helpers
├── eval/
│   ├── eval_attribution.py     # Tables 1 & 2 — attribution robustness
│   └── eval_table4.py          # Table 4 — watermarked vs. no-watermark ablation
├── models/
│   ├── W2V.py                  # MMS-300M collaborator
│   ├── ResNet34.py             # ResNet34 + LFB standalone classifier
│   ├── audioseal_modules.py    # ColProcessor, SEANet wrappers
│   ├── frontend.py             # LFB frontend
│   ├── loss.py                 # LMCL loss
│   ├── modules.py              # Timbre encoder/decoder
│   └── seanet.py               # SEANet encoder/decoder
├── perturbations/
│   ├── perturbation.py         # All distortion operators
│   └── audio_effects.py        # Low-level audio effects
├── util/
│   ├── attack_labels.py        # id_to_label / model_to_id mappings
│   └── utils.py                # Shared helpers (load_audio, get_watermark_eval, …)
├── main.py                     # Training entrypoint
├── requirements.txt
└── README.md
```

---

## 5. Training

The unified training entrypoint is [`main.py`](main.py).

### FakeMarkA

```bash
python main.py config/fakemark_a.yaml
```

### FakeMarkT

```bash
python main.py config/fakemark_t.yaml
```

### Common CLI Overrides

`main.py` accepts standard SpeechBrain / HyperPyYAML override syntax:

```bash
python main.py config/fakemark_a.yaml \
  --base_path ./experiments \
  --exp_name fakemark_a_run1 \
  --protocol_folder ./protocols \
  --source_tracing_num_labels 12 \
  --lr 1e-4
```

`--base_path` is the only required override. It sets the experiment root and also
determines the audio root: `$ROOT` in every protocol CSV is replaced at runtime by
`<base_path>/Data`, so your audio files should live under `<base_path>/Data/`.

### Dataset Setup

Protocol CSV files are included under [`protocols/`](protocols/):

| File | Split | Rows | Description |
|------|-------|------|-------------|
| `protocols/train.csv` | Train | 10 000 | Training utterances drawn from MLAAD v5 |
| `protocols/valid.csv` | Validation | 1 000 | Validation utterances for checkpoint selection |
| `protocols/test.csv` | Test | 29 100 | Full evaluation set |
| `protocols/train_balanced.csv` | Train | 3 156 | ID-balanced training split |
| `protocols/valid_balanced.csv` | Validation | 1 000 | ID-balanced validation split |
| `protocols/test_balanced.csv` | Test | 13 591 | ID-balanced evaluation set used by the eval examples |

Each CSV follows SpeechBrain format with `$ROOT/`-prefixed audio paths.
Set `--root-path` (evaluation) or configure `data_folder` (training) to the directory
that replaces the `$ROOT/` prefix at runtime. For evaluation, include a trailing
slash (e.g. `--root-path /path/to/Data/`), since the literal `$ROOT/` — slash
included — is what gets replaced.

The full column schema is:

| Column | Description |
|--------|-------------|
| `ID` | Unique utterance identifier |
| `Label` | Binary label (`fake` / `real`) |
| `Duration` | Duration in seconds |
| `SampleRate` | Original sample rate in Hz |
| `Path` | `$ROOT/`-prefixed path to the waveform |
| `Attack` | Source TTS system name (e.g. `MeloTTS`, `vixTTS`) |
| `Speaker` | Speaker ID (if available, else `-`) |
| `Proportion` | Split tag (`train` / `valid` / `test`) |
| `AudioChannel` | Number of audio channels |
| `AudioEncoding` | PCM encoding type |
| `AudioBitSample` | Bit depth |
| `Language` | Language code (e.g. `en`) |

### Default Hyperparameters

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| SSL backbone | `facebook/mms-300m` | Pre-trained feature extractor used as the collaborator frontend |
| Sample rate | `16 000 Hz` | Target waveform sample rate; inputs are resampled if needed |
| Max batch length | `40` | Max total duration of a batch during training |
| Epochs | `100` | Total training epochs over the dataset |
| Learning rate | `1e-4` | Initial learning rate |
| Warmup steps | `2 000` | Linear LR warm-up before the main schedule |
| Message warmup | `0` (FakeMarkA) / `2 000` (FakeMarkT) | Steps before the watermark message loss is activated |
| Validation step | `1 000` | Frequency (in steps) at which validation is run and checkpoints are saved |
| Optimizer step limit | `50 000` | Training stops at this step |
| Number of labels | `12` | Collaborator / watermark label-space dimension (`source_tracing_num_labels`) |

Loss weights: `λ_g_col = 10.0`, `λ_col = 10.0`, `λ_sc = 1.0`, `λ_ssn = 1.0`,
`λ_gen = 1.0` (FakeMarkA) / `7.0` (FakeMarkT), `λ_loud = 10.0`, `λ_l1 = 0.1`,
`λ_feat_match = 1.0`, `λ_l1_spec = 1.0`.

Architecture defaults:

- **FakeMarkA**: `watermark_backend=seanet`, `emb_size=128`, `n_filters=32`, `n_residual_layers=1`
- **FakeMarkT**: `watermark_backend=timbre`, `hidden_dim=64`, `layers_CE=4`, `layers_EM=5`

---

## 6. Evaluation

The maintained evaluation scripts are:

| Script | Purpose |
|--------|---------|
| [`eval/eval_attribution.py`](eval/eval_attribution.py) | Tables 1 & 2 — attribution accuracy across systems and distortions |
| [`eval/eval_table4.py`](eval/eval_table4.py) | Table 4 — watermarked vs. no-watermark ablation |

Evaluation is run on the balanced test split [`protocols/test_balanced.csv`](protocols/test_balanced.csv).
Both scripts subsample it to `--n-per-class` files per attribution class (deterministic, seed 42);
this defaults to **300**, matching the balanced evaluation protocol. Pass `--n-per-class 0` to
evaluate on the full split instead.

### Prerequisites — Third-party Vocoder and Codec Models

Several distortion operators (`vocos`, `bigvgan`, `hifigan`, `stok`, `wavtokenizer`, `facodec`)
depend on external models that are not bundled in this repository.
Run the setup script once after installation to clone the required repos and
download the pretrained weights:

```bash
bash setup_third_party.sh
```

The script sets up the following under `third_party/`:

| Model | Source | Used by distortion |
|-------|--------|--------------------|
| BigVGAN v2 | `nvidia/bigvgan_v2_24khz_100band_256x` | `bigvgan` |
| WavTokenizer | `novateur/WavTokenizer` | `wavtokenizer` |
| FACodec | `amphion/naturalspeech3_facodec` | `facodec` |
| HiFi-GAN | `speechbrain/tts-hifigan-libritts-16kHz` | `hifigan` |
| SpeechTokenizer | `OpenMOSS-Team/SpeechTokenizer` | `stok` |
| Vocos | `charactr/vocos-mel-24khz` | `vocos` |

The `none`, `pitch`, `speed`, and `noise` distortions do not
require any third-party models and can be run immediately after installation.

### Supported Systems

| System | Description |
|--------|-------------|
| `FakeMarkA` | SEANet watermark + MMS-300M collaborator |
| `FakeMarkT` | Timbre watermark + MMS-300M collaborator |
| `MMS-300M` | Standalone MMS-300M classifier (no watermark) |
| `ResNet34` | Standalone ResNet34 + LFB classifier (no watermark) |
| `AudioSeal` | AudioSeal watermark baseline (Hamming nearest-neighbor decoding) |
| `Timbre` | Conv2-based Timbre watermark baseline (Hamming nearest-neighbor decoding) |

### Supported Distortions

| Category | Keys |
|----------|------|
| No distortion | `none` |
| Vocoder resynthesis | `vocos`, `bigvgan`, `hifigan` |
| Neural codec | `stok` (SpeechTokenizer), `wavtokenizer`, `facodec` |
| Signal processing | `pitch`, `speed`, `noise` (MUSAN) |
| Watermark-targeted | `overwrite` |

Use `--distortion all` to run every supported distortion in one pass.

### Attribution Robustness (Tables 1 & 2)

```bash
python eval/eval_attribution.py \
  --protocol protocols/test_balanced.csv \
  --root-path /path/to/Data/ \
  --output results/table1.csv \
  --system FakeMarkA FakeMarkT MMS-300M ResNet34 AudioSeal Timbre \
  --distortion none vocos bigvgan \
  --fakemarka-exp-dir checkpoints/FakeMarkA \
  --fakemarkt-exp-dir checkpoints/FakeMarkT \
  --ssl-checkpoint facebook/mms-300m \
  --mms-classifier-ckpt  checkpoints/MMS_300M.ckpt \
  --resnet-classifier-ckpt checkpoints/ResNet.ckpt \
  --audioseal-encoder-ckpt checkpoints/AudioSeal/checkpoint_generator_epoch260.pth \
  --audioseal-detector-ckpt checkpoints/AudioSeal/checkpoint_detector_epoch260.pth \
  --timbre-ckpt checkpoints/Timbre-4bit.pth.tar \
  --wav-output-dir wavs/
```

**Saving watermarked waveforms** — when `--wav-output-dir` is provided, the
post-distortion watermarked waveform for every (file × system × distortion)
triple is written to disk:

```
wavs/
  FakeMarkA/<distortion>/<stem>_wm.wav
  FakeMarkT/<distortion>/<stem>_t.wav
  AudioSeal/<distortion>/<stem>_as.wav
  Timbre/<distortion>/<stem>_timbre.wav
```

### Watermark Ablation (Table 4)

```bash
python eval/eval_table4.py \
  --protocol-indomain protocols/test_balanced.csv \
  --protocol-crossdomain /path/to/asvspoof5_timit_test.csv \
  --root-path /path/to/Data/ \
  --fakemarka-exp-dir checkpoints/FakeMarkA \
  --fakemarkt-exp-dir checkpoints/FakeMarkT \
  --ssl-checkpoint facebook/mms-300m \
  --output results/table4.csv \
  --distortion none vocos bigvgan \
  --wav-output-dir wavs/
```

`--protocol-indomain`, `--protocol-crossdomain`, and `--root-path` are all
required. The in-domain protocol is the MLAAD v5 balanced test split; the
cross-domain protocol is the ASVspoof5 + TIMIT-TTS split.

### Output Files

Both scripts write two CSV files per run.

**Aggregate CSV** (`--output <path>`):

```
system,distortion,accuracy
FakeMarkA,none,1.0
FakeMarkA,vocos,0.94
FakeMarkT,none,0.97
...
```

**Per-file CSV** (auto-saved as `<stem>_perfile.csv`):

```
file,system,distortion,true_attack,pred_attack,correct,probs
/data/clip01.wav,FakeMarkA,none,suno-bark,Bark,1,"[0.01, 0.95, ...]"
/data/clip02.wav,FakeMarkA,vocos,vits,VITS (variants),0,"[0.40, 0.10, ...]"
...
```

| Column | Description |
|--------|-------------|
| `file` | Absolute path to the evaluated audio file |
| `system` | Watermarking system |
| `distortion` | Applied distortion |
| `true_attack` | Ground-truth attack name from the protocol CSV |
| `pred_attack` | Predicted attack (canonical name from `id_to_label`) |
| `correct` | `1` if prediction matched ground truth, else `0` |
| `probs` | Softmax probability vector over all classes (Python list string); `None` for Hamming-decoded systems (AudioSeal, Timbre) |

For `eval_table4.py` the per-file CSV additionally contains `condition`
(`watermarked` / `no_watermark`) and `dataset` columns.

---

## 7. Acknowledgments
This project is based on results obtained from a JST AIP Acceleration Research project (JPMJCR24U3). It is  partially supported by JSPS MEXT KAKENHI Grant (25K24398) and JST PRESTO Grant (JPMJPR23P9).

This project was carried out using the TSUBAME4.0 supercomputer at the Institute of Science Tokyo.

Code and scripts in this project is built on the following open-source projects:

- [SpeechBrain](https://github.com/speechbrain/speechbrain)
- [AudioSeal](https://github.com/facebookresearch/audioseal)
- [Timbre Watermarking](https://github.com/TimbreWatermarking/TimbreWatermarking)

The code in this repository is released under the BSD 3-Clause License.
See [`LICENSE`](LICENSE).

---

## 8. Citation

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{fakemark,
  title     = {FakeMark: Deepfake Speech Attribution With Watermarked Artifacts},
  author    = {Ge, Wanying and Wang, Xin and Yamagishi, Junichi},
  booktitle = {The Authenticity \& Provenance in the Age of AI Workshop},
  year      = {2026}
}
```
