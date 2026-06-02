"""Model components used by the timbre watermarking configuration."""

import torch
import torch.nn.functional as F

from dataio.frequency import fixed_STFT
from models.blocks import Conv2Encoder, WatermarkEmbedder
from util.utils import speech_mask_time


class Encoder(torch.nn.Module):
    """Embed a class-conditioned watermark into the STFT magnitude domain.

    Args:
        msg_length: Number of supported watermark labels.
        block: Residual block style used by the convolutional submodules.
        layers_CE: Number of layers in the content encoder.
        layers_EM: Number of layers in the watermark embedder.
        hidden_dim: Hidden channel dimension shared by the submodules.
        n_fft: FFT size used by the fixed STFT frontend.
        win_length: Window length used by the fixed STFT frontend.
        hop_length: Hop length used by the fixed STFT frontend.
    """

    def __init__(
        self,
        msg_length,
        block="skip",
        layers_CE=3,
        layers_EM=4,
        hidden_dim=64,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
    ):
        super().__init__()
        self.msg_length = msg_length
        self.block = block
        self.layers_CE = layers_CE
        self.layers_EM = layers_EM
        self.hidden_dim = hidden_dim
        self.embedding_input_dim = hidden_dim + 2
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        frequency_bins = int((n_fft / 2) + 1)
        self.label_embedding = torch.nn.Embedding(msg_length, frequency_bins)
        self.content_encoder = Conv2Encoder(
            input_channel=1,
            hidden_dim=self.hidden_dim,
            block=self.block,
            n_layers=self.layers_CE,
        )
        self.watermark_embedder = WatermarkEmbedder(
            input_channel=self.embedding_input_dim,
            hidden_dim=self.hidden_dim,
            block=self.block,
            n_layers=self.layers_EM,
        )
        self.stft = fixed_STFT(self.n_fft, self.hop_length, self.win_length)

    def forward(self, waveform, labels):
        """Apply the watermark to a batch of waveforms.

        Args:
            waveform: Input audio shaped ``(batch, 1, time)``.
            labels: Integer conditioning labels shaped ``(batch,)``.

        Returns:
            Tuple containing:
                - watermarked waveform shaped ``(batch, 1, time)``
                - isolated watermark waveform shaped ``(batch, 1, time)``
        """
        waveform_length = waveform.shape[-1]
        magnitude, phase = self.stft.transform(waveform)

        speech_mask = speech_mask_time(waveform)
        speech_mask = F.interpolate(
            speech_mask, size=magnitude.shape[-1], mode="nearest"
        )
        speech_mask = speech_mask.unsqueeze(2).repeat(1, 1, magnitude.shape[1], 1)

        encoded_magnitude = self.content_encoder(magnitude.unsqueeze(1))
        label_embedding = self.label_embedding(labels)
        if label_embedding.dim() == 1:
            label_embedding = label_embedding.unsqueeze(0)

        label_embedding = label_embedding.unsqueeze(1).unsqueeze(-1).repeat(
            1, 1, 1, encoded_magnitude.shape[3]
        )
        label_embedding = label_embedding * speech_mask

        fused_features = torch.cat(
            (encoded_magnitude, magnitude.unsqueeze(1), label_embedding),
            dim=1,
        )
        watermark_embedding = self.watermark_embedder(fused_features)
        watermarked_magnitude = magnitude + watermark_embedding.squeeze(1)

        self.stft.num_samples = waveform_length
        watermarked_waveform = self.stft.inverse(
            watermarked_magnitude.squeeze(1), phase.squeeze(1)
        )
        watermark = self.stft.inverse(
            watermark_embedding.squeeze(1), phase.squeeze(1)
        )
        return watermarked_waveform, watermark
