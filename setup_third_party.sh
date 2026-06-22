#!/usr/bin/env bash
# setup_third_party.sh
#
# Downloads and sets up the external models required by --distortion:
#   bigvgan      nvidia/bigvgan_v2_24khz_100band_256x
#   wavtokenizer novateur/WavTokenizer  (code + checkpoint)
#   facodec      amphion/naturalspeech3_facodec (code from Amphion + weights)
#   hifigan      speechbrain/tts-hifigan-libritts-16kHz
#   stok         OpenMOSS-Team/SpeechTokenizer (config + weights)
#   vocos        charactr/vocos-mel-24khz (pip package + cached weights)
#
# Usage:
#   bash setup_third_party.sh          # runs in the repo root
#
# Requirements (already in the FAKEMARK conda env):
#   git, python, huggingface_hub, speechbrain

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="$REPO_ROOT/third_party"
PYTHON="${PYTHON:-python}"   # override with PYTHON=/path/to/python if needed

echo "==> Repo root : $REPO_ROOT"
echo "==> third_party: $THIRD_PARTY"
echo ""

mkdir -p "$THIRD_PARTY"

# ──────────────────────────────────────────────────────────────────────────────
# 1. BigVGAN
# ──────────────────────────────────────────────────────────────────────────────
echo "────────────────────────────────────"
echo "[1/6] BigVGAN"
echo "────────────────────────────────────"

BIGVGAN_DIR="$THIRD_PARTY/BigVGAN"

if [ ! -d "$BIGVGAN_DIR/.git" ]; then
    git clone --depth 1 https://github.com/NVIDIA/BigVGAN.git "$BIGVGAN_DIR"
    echo "  Cloned BigVGAN."
else
    echo "  BigVGAN already cloned, skipping."
fi

# Patch _from_pretrained so it tolerates missing proxies/resume_download kwargs
# that older BigVGAN releases required but newer huggingface_hub no longer passes.
# Uses a line-by-line approach so the patch is idempotent (safe to re-run).
$PYTHON - <<PYEOF
path = "$BIGVGAN_DIR/bigvgan.py"
with open(path) as f:
    lines = f.readlines()

patched = []
changed = False
for line in lines:
    stripped = line.lstrip()
    # Add '= None' default to proxies parameter only if it has no default yet
    if stripped.startswith("proxies:") and "=" not in line:
        line = line.rstrip("\n").rstrip().rstrip(",") + " = None,\n"
        changed = True
    # Add '= False' default to resume_download parameter only if it has no default yet
    elif stripped.startswith("resume_download:") and "=" not in line:
        line = line.rstrip("\n").rstrip().rstrip(",") + " = False,\n"
        changed = True
    patched.append(line)

if changed:
    with open(path, "w") as f:
        f.writelines(patched)
    print("  bigvgan.py: patched proxies/resume_download to have defaults.")
else:
    print("  bigvgan.py: no patch needed (already has defaults or different signature).")
PYEOF

# Pre-download BigVGAN weights into third_party/BigVGAN so the model loads offline
BIGVGAN_CKPT_DIR="$BIGVGAN_DIR/hf_cache"
if [ ! -f "$BIGVGAN_CKPT_DIR/config.json" ]; then
    echo "  Downloading BigVGAN weights from HuggingFace..."
    $PYTHON - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="nvidia/bigvgan_v2_24khz_100band_256x",
    local_dir="$BIGVGAN_CKPT_DIR",
    ignore_patterns=["*.msgpack", "flax_model*"],
)
print("  BigVGAN weights downloaded.")
PYEOF
else
    echo "  BigVGAN weights already present."
fi

echo ""

# ──────────────────────────────────────────────────────────────────────────────
# 2. WavTokenizer (code + checkpoint)
# ──────────────────────────────────────────────────────────────────────────────
echo "────────────────────────────────────"
echo "[2/6] WavTokenizer"
echo "────────────────────────────────────"

WAVTOK_DIR="$THIRD_PARTY/WavTokenizer"

if [ ! -d "$WAVTOK_DIR/.git" ]; then
    git clone --depth 1 https://github.com/jishengpeng/WavTokenizer.git "$WAVTOK_DIR"
    echo "  Cloned WavTokenizer."
else
    echo "  WavTokenizer already cloned, skipping."
fi

WAVTOK_CKPT="$WAVTOK_DIR/WavTokenizer_small_600_24k_4096.ckpt"
if [ ! -f "$WAVTOK_CKPT" ]; then
    echo "  Downloading WavTokenizer checkpoint..."
    $PYTHON - <<PYEOF
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="novateur/WavTokenizer",
    filename="WavTokenizer_small_600_24k_4096.ckpt",
    local_dir="$WAVTOK_DIR",
)
print("  WavTokenizer checkpoint downloaded.")
PYEOF
else
    echo "  WavTokenizer checkpoint already present."
fi

echo ""

# ──────────────────────────────────────────────────────────────────────────────
# 3. FACodec  (ns3_codec source from Amphion + pretrained weights)
# ──────────────────────────────────────────────────────────────────────────────
echo "────────────────────────────────────"
echo "[3/6] FACodec"
echo "────────────────────────────────────"

AMPHION_DIR="$THIRD_PARTY/Amphion"
NS3_DIR="$AMPHION_DIR/models/codec/ns3_codec"

if [ ! -d "$NS3_DIR" ]; then
    echo "  Sparse-cloning ns3_codec from Amphion..."
    mkdir -p "$AMPHION_DIR"
    # Sparse checkout — only pulls models/codec/ns3_codec, not the full repo
    git -C "$AMPHION_DIR" init -q
    git -C "$AMPHION_DIR" remote add origin https://github.com/open-mmlab/Amphion.git 2>/dev/null || true
    git -C "$AMPHION_DIR" config core.sparseCheckout true
    echo "models/codec/ns3_codec/" > "$AMPHION_DIR/.git/info/sparse-checkout"
    git -C "$AMPHION_DIR" fetch --depth 1 origin main
    git -C "$AMPHION_DIR" checkout FETCH_HEAD
    echo "  ns3_codec source ready."
else
    echo "  ns3_codec already present."
fi

FACODEC_CKPT_DIR="$THIRD_PARTY/facodec_weights"
if [ ! -f "$FACODEC_CKPT_DIR/ns3_facodec_encoder.bin" ]; then
    echo "  Downloading FACodec weights from HuggingFace..."
    $PYTHON - <<PYEOF
from huggingface_hub import hf_hub_download
for fname in ["ns3_facodec_encoder.bin", "ns3_facodec_decoder.bin"]:
    hf_hub_download(
        repo_id="amphion/naturalspeech3_facodec",
        filename=fname,
        local_dir="$FACODEC_CKPT_DIR",
    )
print("  FACodec weights downloaded.")
PYEOF
else
    echo "  FACodec weights already present."
fi

echo ""

# ──────────────────────────────────────────────────────────────────────────────
# 4. HiFi-GAN  (speechbrain auto-downloads; just pre-cache it here)
# ──────────────────────────────────────────────────────────────────────────────
echo "────────────────────────────────────"
echo "[4/6] HiFi-GAN"
echo "────────────────────────────────────"

HIFIGAN_DIR="$THIRD_PARTY/hifigan"
if [ ! -f "$HIFIGAN_DIR/hyperparams.yaml" ]; then
    echo "  Downloading HiFi-GAN from SpeechBrain HuggingFace hub..."
    $PYTHON - <<PYEOF
from speechbrain.inference.vocoders import HIFIGAN
HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-libritts-16kHz",
    savedir="$HIFIGAN_DIR",
    run_opts={"device": "cpu"},
)
print("  HiFi-GAN downloaded.")
PYEOF
else
    echo "  HiFi-GAN already present."
fi

echo ""

# ──────────────────────────────────────────────────────────────────────────────
# 5. SpeechTokenizer  (config + pretrained weights)
# ──────────────────────────────────────────────────────────────────────────────
echo "────────────────────────────────────"
echo "[5/6] SpeechTokenizer"
echo "────────────────────────────────────"

STOK_DIR="$THIRD_PARTY/SpeechTokenizer"
if [ ! -f "$STOK_DIR/SpeechTokenizer.pt" ]; then
    echo "  Downloading SpeechTokenizer config + weights from HuggingFace..."
    $PYTHON - <<PYEOF
from huggingface_hub import hf_hub_download
# Files live in the speechtokenizer_hubert_avg/ subfolder of the repo.
# We flatten them into STOK_DIR so the paths in perturbation.py are clean.
for fname in ["speechtokenizer_hubert_avg/config.json",
              "speechtokenizer_hubert_avg/SpeechTokenizer.pt"]:
    hf_hub_download(
        repo_id="OpenMOSS-Team/SpeechTokenizer",
        filename=fname,
        local_dir="$STOK_DIR",
    )
# Move out of the subfolder so paths match _SPEECHTOKENIZER_CFG / _CKPT
import shutil, os
src = "$STOK_DIR/speechtokenizer_hubert_avg"
dst = "$STOK_DIR"
for f in os.listdir(src):
    shutil.move(os.path.join(src, f), os.path.join(dst, f))
os.rmdir(src)
print("  SpeechTokenizer downloaded.")
PYEOF
    # Also install the missing beartype dependency needed by speechtokenizer
    $PYTHON -m pip install --quiet beartype
    echo "  beartype installed."
else
    echo "  SpeechTokenizer already present."
fi

echo ""

# ──────────────────────────────────────────────────────────────────────────────
# 6. Vocos  (pip package; pre-cache the pretrained weights here)
# ──────────────────────────────────────────────────────────────────────────────
echo "────────────────────────────────────"
echo "[6/6] Vocos"
echo "────────────────────────────────────"

VOCOS_DIR="$THIRD_PARTY/vocos"
VOCOS_MARKER="$VOCOS_DIR/.cached"
if [ ! -f "$VOCOS_MARKER" ]; then
    echo "  Pre-caching Vocos weights (charactr/vocos-mel-24khz)..."
    # pert_vocos() loads the model via Vocos.from_pretrained, which pulls the
    # weights into the HuggingFace cache. Instantiating it once here primes that
    # cache so the distortion runs offline afterwards.
    $PYTHON - <<PYEOF
from vocos import Vocos
Vocos.from_pretrained("charactr/vocos-mel-24khz")
print("  Vocos weights cached.")
PYEOF
    mkdir -p "$VOCOS_DIR"
    touch "$VOCOS_MARKER"
else
    echo "  Vocos already cached."
fi

echo ""
echo "════════════════════════════════════"
echo "All third-party models are ready."
echo "  third_party/"
echo "  ├── BigVGAN/          (+ hf_cache/)"
echo "  ├── WavTokenizer/     (+ checkpoint)"
echo "  ├── Amphion/models/codec/ns3_codec/"
echo "  ├── facodec_weights/"
echo "  ├── hifigan/"
echo "  ├── SpeechTokenizer/"
echo "  └── vocos/            (weights in HuggingFace cache)"
echo "════════════════════════════════════"
