"""Wav2Vec2-based source-tracing classifier.

Classes:
    SSLModel: Loads a pre-trained ``transformers`` SSL model and extracts
        frame-level representations from raw audio.
    Model: Wraps :class:`SSLModel` with an adaptive-pooling + linear
        projection head for closed-set source classification.
"""

import logging
from pathlib import Path

import torch

try:
    from transformers import AutoModel
except ImportError as exc:  # pragma: no cover - depends on user environment.
    AutoModel = None
    TRANSFORMERS_IMPORT_ERROR = exc
else:
    TRANSFORMERS_IMPORT_ERROR = None

from models.W2V_configs import get_default_ssl_source, get_expected_hidden_size


logger = logging.getLogger(__name__)


class SSLModel(torch.nn.Module):
    """Self-supervised Wav2Vec 2.0 feature extractor.

    Args:
        model_name: Named SSL backbone preset used for metadata and default
            source resolution.
        checkpoint_path: ``transformers`` model ID or a local Hugging Face
            checkpoint directory. When omitted or empty, the maintained default
            source for ``model_name`` is used.
    """

    def __init__(self, model_name: str, checkpoint_path: str):
        super().__init__()
        self.model_name = model_name
        self.pretrained_source = self._resolve_pretrained_source(checkpoint_path)
        self.model = self._load_backbone(self.pretrained_source)
        self.out_dim: int = int(getattr(self.model.config, "hidden_size"))

        expected_hidden_size = get_expected_hidden_size(model_name)
        if self.out_dim != expected_hidden_size:
            logger.warning(
                "SSL hidden size mismatch for '%s': expected %s, got %s from %s.",
                model_name,
                expected_hidden_size,
                self.out_dim,
                self.pretrained_source,
            )

    def _resolve_pretrained_source(self, checkpoint_path: str) -> str:
        """Resolve the ``transformers`` model source for the SSL backbone."""
        if checkpoint_path:
            return checkpoint_path
        return get_default_ssl_source(self.model_name)

    def _load_backbone(self, pretrained_source: str) -> torch.nn.Module:
        """Load SSL backbone weights from a transformers-compatible source."""
        if AutoModel is None:
            raise ImportError(
                "transformers is required for SSL backbone loading. Install the "
                "project dependencies with `pip install -r requirements.txt`."
            ) from TRANSFORMERS_IMPORT_ERROR

        source_path = Path(pretrained_source)
        if source_path.is_file():
            raise ValueError(
                "Fairseq-style single-file SSL checkpoints are no longer "
                "supported. Please pass a transformers model ID such as "
                "'facebook/mms-300m' or a local Hugging Face model directory."
            )

        model = AutoModel.from_pretrained(pretrained_source)
        model.gradient_checkpointing_disable()
        if hasattr(model, "feature_extractor"):
            model.feature_extractor._requires_grad = False
        logger.info("Loaded SSL backbone from transformers source: %s", pretrained_source)
        return model

    def extract_feat(self, input_data: torch.Tensor) -> torch.Tensor:
        """Extract frame-level features from raw waveforms.

        Args:
            input_data: Raw waveform tensor of shape ``(batch, T)`` or
                ``(batch, T, 1)``.

        Returns:
            Frame-level features of shape ``(batch, frames, hidden_dim)``.
        """
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)

        if input_data.ndim == 3:
            if input_data.size(1) == 1:
                input_tmp = input_data.squeeze(1)
            elif input_data.size(-1) == 1:
                input_tmp = input_data.squeeze(-1)
            else:
                raise ValueError(
                    "Expected waveform shape (batch, time), (batch, 1, time), "
                    "or (batch, time, 1)."
                )
        else:
            input_tmp = input_data
        outputs = self.model(input_values=input_tmp)
        emb = outputs.last_hidden_state
        return emb


class Model(torch.nn.Module):
    """Source-tracing classifier built on an SSL backbone.

    Args:
        model_name: SSL architecture key (forwarded to :class:`SSLModel`).
        checkpoint_path: Transformers model ID or local model directory for the
            SSL backbone.
        v_out_class: Number of output source classes.
    """

    def __init__(self, model_name: str, checkpoint_path: str, v_out_class: int):
        super().__init__()
        self.v_out_class = v_out_class

        self.m_ssl = SSLModel(model_name, checkpoint_path)

        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.proj_fc = torch.nn.Linear(
            in_features=self.m_ssl.out_dim,
            out_features=self.v_out_class,
        )

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run classification on a batch of waveforms.

        Args:
            wav: Input waveform of shape ``(batch, T)`` or ``(batch, 1, T)``.

        Returns:
            Tuple of:
                - ``pred``: class logits of shape ``(batch, v_out_class)``.
                - ``pooled_emb``: pooled embedding of shape ``(batch, hidden_dim)``.
        """
        if wav.dim() == 3 and wav.size(1) == 1:
            wav = wav.squeeze(1)
        # [batch, frame, hidden_dim]
        emb = self.m_ssl.extract_feat(wav)
        # [batch, hidden_dim, frame]
        emb = emb.transpose(1, 2)
        # [batch, hidden_dim, 1]
        pooled_emb = self.adap_pool1d(emb)
        # [batch, hidden_dim]
        pooled_emb = pooled_emb.view(emb.size(0), -1)
        # [batch, v_out_class]
        pred = self.proj_fc(pooled_emb)
        return pred, pooled_emb
