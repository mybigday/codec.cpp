"""
Audio codec converters for GGUF format.
"""

from .base import BaseConverter
from .mimi import MimiConverter
from .dac import DacConverter
from .wavtokenizer import WavTokenizerConverter
from .qwen3_tts_tokenizer import Qwen3TTSTokenizerConverter
from .chatterbox import ChatterboxS3TConverter, ChatterboxS3GConverter
from .soprano import SopranoConverter
from .nemo_nano_codec import NemoNanoCodecConverter
from .neucodec import NeuCodecConverter, DistillNeuCodecConverter
from .xcodec2 import XCodec2Converter
from .snac import SnacConverter
from .moss_audio import MossAudioConverter
from .xy_tokenizer import XYTokenizerConverter
from .csm import CsmConverter
from .moshi import MoshiConverter
from .bluemagpie import BlueMagpieConverter
from .pocket_tts import PocketTTSConverter

# Registry of supported models
_CONVERTER_REGISTRY = {
    'mimi': MimiConverter,
    'dac': DacConverter,
    'wavtokenizer': WavTokenizerConverter,
    'qwen3_tts_tokenizer': Qwen3TTSTokenizerConverter,
    'chatterbox_s3t': ChatterboxS3TConverter,
    'chatterbox_s3g': ChatterboxS3GConverter,
    'soprano': SopranoConverter,
    'nemo_nano_codec': NemoNanoCodecConverter,
    'neucodec': NeuCodecConverter,
    'distill_neucodec': DistillNeuCodecConverter,
    'xcodec2': XCodec2Converter,
    'snac': SnacConverter,
    'moss_audio': MossAudioConverter,
    'moss_audio_nano': MossAudioConverter,
    'xy_tokenizer': XYTokenizerConverter,
    'csm': CsmConverter,
    'moshi': MoshiConverter,
    'bluemagpie': BlueMagpieConverter,
    'pocket_tts': PocketTTSConverter,
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
