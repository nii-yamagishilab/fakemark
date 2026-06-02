"""Legacy timbre-side SSL classifier backed by ``transformers``.

Classes:
    * SSLModel: Loads a pre-trained SSL backbone and extracts frame-level
      representations from raw audio input.
    * Model: A wrapper model that uses SSLModel for feature extraction and adds
      a projection layer for classification.
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

logger = logging.getLogger(__name__)


class SSLModel(torch.nn.Module):
    """Legacy wrapper around a transformers SSL backbone."""

    def __init__(self, cp_path: str, ssl_orig_output_dim: int):
        """Initialize the SSL backbone used by legacy timbre modules.

        Args:
          cp_path: ``transformers`` model ID or local Hugging Face model
            directory.
          ssl_orig_output_dim: Expected dimension of the SSL model output
            feature.
        """
        super().__init__()
        if AutoModel is None:
            raise ImportError(
                "transformers is required for SSL backbone loading. Install the "
                "project dependencies with `pip install -r requirements.txt`."
            ) from TRANSFORMERS_IMPORT_ERROR

        source_path = Path(cp_path)
        if source_path.is_file():
            raise ValueError(
                "Fairseq-style single-file SSL checkpoints are no longer "
                "supported. Please pass a transformers model ID such as "
                "'facebook/mms-300m' or a local Hugging Face model directory."
            )
        self.model = AutoModel.from_pretrained(cp_path)
        self.out_dim = int(getattr(self.model.config, "hidden_size"))
        if self.out_dim != ssl_orig_output_dim:
            logger.warning(
                "Legacy timbre SSL hidden size mismatch: expected %s, got %s "
                "from %s.",
                ssl_orig_output_dim,
                self.out_dim,
                cp_path,
            )
    
    def extract_feat(self, input_data: torch.Tensor) -> torch.Tensor:
        """Extract frame-level SSL features.

        Args:
          input_data: Tensor waveform shaped ``(batch, T)`` or
            ``(batch, 1, T)``.

        Return:
          Tensor feature map shaped ``(batch, frame, hidden_dim)``.
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

        emb = self.model(input_values=input_tmp).last_hidden_state
        return emb

class Model(torch.nn.Module):
    """ Model definition
    Args:
        ssl_orig_output_dim: same in SSLModel(cp_path, ssl_orig_output_dim)
        ssl_path: cp_path in SSLModel(cp_path, ssl_orig_output_dim)
    """
    def __init__(self, ssl_orig_output_dim: int, ssl_path: str, v_out_class: int):
        super().__init__()
        self.v_out_class = v_out_class
        self.m_ssl = SSLModel(ssl_path, ssl_orig_output_dim)

        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.proj_fc = torch.nn.Linear(
            in_features=self.m_ssl.out_dim,
            out_features=self.v_out_class,
        )

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if wav.dim() == 3 and wav.size(1) == 1:
            wav = wav.squeeze(1)
        emb = self.m_ssl.extract_feat(wav)
        emb = emb.transpose(1, 2)
        pooled_emb = self.adap_pool1d(emb)
        pooled_emb = pooled_emb.view(emb.size(0), -1)
        pred = self.proj_fc(pooled_emb)
        return pred, pooled_emb
