"""
Audio codec converters for GGUF format.
"""

from .base import BaseConverter
from .mimi import MimiConverter
from .dac import DacConverter
from .wavtokenizer import WavTokenizerConverter
from .qwen3_tts_tokenizer import Qwen3TTSTokenizerConverter

# Registry of supported models
_CONVERTER_REGISTRY = {
    'mimi': MimiConverter,
    'dac': DacConverter,
    'wavtokenizer': WavTokenizerConverter,
    'qwen3_tts_tokenizer': Qwen3TTSTokenizerConverter,
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
