import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from models.frontend import LFB
from models.loss import LMCL_loss

# ---------------- Residual Block ---------------- #
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

# ---------------- TS Pooling ---------------- #
class TSPooling(nn.Module):
    def forward(self, x):
        # x: [B, C, T]
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        return torch.cat((mean, std), dim=1)  # [B, 2C]


# ---------------- Full Model ---------------- #
class Model(nn.Module):
    def __init__(self, out_class=12, sample_rate=16000, n_mels=80, embed_dim=128, layers=[3,4,6,3]):
        super().__init__()
        self.in_channels = 64

        self.lfb = LFB(400, 160, 400, 16000, 80, is_cmvn=True)

        # ----- ResNet Backbone ----- #
        input_dim = n_mels * 3  # LFB + Δ + ΔΔ = 240
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(ResidualBlock, 64,  layers[0])
        self.layer2 = self._make_layer(ResidualBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, layers[3], stride=2)

        # ----- Temporal Stats Pooling ----- #
        self.pooling = TSPooling()

        # ----- Embedding Projection ----- #
        self.fc = nn.Linear(512*2, embed_dim)

        self.lmcl = LMCL_loss(out_class, embed_dim)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, wav):
        # wav: [B, T] (raw waveform)
        # ----- Feature Extraction ----- #
        with torch.no_grad():
            feats = self.lfb(wav)  # [B, n_mels, T]
            delta = torchaudio.functional.compute_deltas(feats)
            delta2 = torchaudio.functional.compute_deltas(delta)
            feats = torch.cat([feats, delta, delta2], dim=1)  # [B, 240, T]

        # ----- ResNet Backbone ----- #
        x = self.relu(self.bn1(self.conv1(feats)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # ----- TS Pooling + FC ----- #
        x = self.pooling(x)   # [B, 1024]
        emb = self.fc(x)      # [B, 128]
        pred = self.lmcl(emb)
        return pred, emb      # (logits, embedding) — matches W2VModel interface
