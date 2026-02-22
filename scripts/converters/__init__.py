"""
Audio codec converters for GGUF format.
"""

from .base import BaseConverter
from .mimi import MimiConverter
from .dac import DacConverter
from .wavtokenizer import WavTokenizerConverter

# Registry of supported models
_CONVERTER_REGISTRY = {
    'mimi': MimiConverter,
    'dac': DacConverter,
    'wavtokenizer': WavTokenizerConverter,
}


def get_converter_for_model(model_type: str) -> type:
    """Get the converter class for a model type."""
    model_type = model_type.lower()
    if model_type not in _CONVERTER_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported: {', '.join(list_supported_models())}"
        )
    return _CONVERTER_REGISTRY[model_type]


def list_supported_models() -> list:
    """List all supported model types."""
    return list(_CONVERTER_REGISTRY.keys())