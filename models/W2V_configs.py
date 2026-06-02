"""Configuration helpers for transformers-based SSL backbones.

This module centralizes the maintained SSL backbone presets used by the
FakeMark collaborator. Each preset provides:

- a public model identifier that can be resolved by ``transformers``
- the expected output feature dimension of the backbone
"""

from typing import Dict


SSL_BACKBONES: Dict[str, Dict[str, object]] = {
    "w2v_small": {
        "source": "facebook/wav2vec2-base",
        "hidden_size": 768,
    },
    "w2v_large": {
        "source": "facebook/wav2vec2-large",
        "hidden_size": 1024,
    },
    "mms_300m": {
        "source": "facebook/mms-300m",
        "hidden_size": 1024,
    },
    "mms_1b": {
        "source": "facebook/mms-1b",
        "hidden_size": 1280,
    },
    "xlsr_1b": {
        "source": "facebook/wav2vec2-xls-r-1b",
        "hidden_size": 1280,
    },
    "xlsr_2b": {
        "source": "facebook/wav2vec2-xls-r-2b",
        "hidden_size": 1920,
    },
}

# Backward-compatible aliases for older codepaths.
global_input_dims = {
    name: int(spec["hidden_size"]) for name, spec in SSL_BACKBONES.items()
}
global_sources = {
    name: str(spec["source"]) for name, spec in SSL_BACKBONES.items()
}


def get_default_ssl_source(model_name: str) -> str:
    """Return the default ``transformers`` source for a named SSL backbone.

    Args:
        model_name: Named FakeMark SSL backbone preset.

    Returns:
        Hugging Face model ID for the requested backbone.

    Raises:
        KeyError: If ``model_name`` is not a maintained backbone preset.
    """
    return global_sources[model_name]


def get_expected_hidden_size(model_name: str) -> int:
    """Return the expected hidden size for a named SSL backbone.

    Args:
        model_name: Named FakeMark SSL backbone preset.

    Returns:
        Output feature dimension used by the collaborator head.

    Raises:
        KeyError: If ``model_name`` is not a maintained backbone preset.
    """
    return global_input_dims[model_name]
