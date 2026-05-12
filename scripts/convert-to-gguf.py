#!/usr/bin/env python3
"""
Unified GGUF converter for audio codec models.

Supports Mimi, DAC, WavTokenizer, and Qwen3-TTS-Tokenizer models from HuggingFace or local checkpoints.

Usage:
    # From HuggingFace
    python convert-to-gguf.py --model-id kyutai/mimi --output mimi.gguf
    
    # From local checkpoint
    python convert-to-gguf.py --checkpoint-path ./mimi-checkpoint --output mimi.gguf
    
    # With quantization
    python convert-to-gguf.py --model-id kyutai/mimi --output mimi-q4.gguf --quantization Q4_K_M
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from converters import get_converter_for_model, list_supported_models


# LM source `architectures[0]` -> codec converter to pair with.  Used when
# auto-detection sees an LM-only config (no codec); the runner then sets
# model_type to the codec, and lm_source to the input itself.
#
# MOSS-TTSD v1.0 / MOSS-TTS pair with `OpenMOSS-Team/MOSS-Audio-Tokenizer`
# (per their processor's default `codec_path`), not XY-Tokenizer like
# v0.5 / v0.7.  Different codec lineage in the same family.
LM_SOURCE_TO_CODEC: dict = {
    "MossTTSDForCausalLM":             "xy_tokenizer",         # MOSS-TTSD v0.5 / v0.7
    "AsteroidTTSModel":                "xy_tokenizer",         # MOSS-TTSD v0 (early prototype)
    "MossTTSDelayModel":               "moss_audio",           # MOSS-TTSD v1.0 / MOSS-TTS
    "Qwen3TTSForConditionalGeneration":"qwen3_tts_tokenizer",  # Qwen3-TTS-12Hz-* (residual_depth_ar)
    "Lfm2AudioForConditionalGeneration":"mimi",                # LFM2-Audio-1.5B uses kyutai/mimi externally
    # MossTTSNanoForCausalLM uses a depth-AR (local_transformer) adaptor;
    # pending codec_lm M3 (residual_depth_ar).
}

# When the user points at an LM source directly (e.g. `--model-id
# fnlp/MOSS-TTSD-v0.5`), we still need to load the codec from somewhere.
# Each LM family pins a specific codec repo (per the upstream release);
# this is the default we fetch unless the user passes --codec-source.
LM_SOURCE_DEFAULT_CODEC_REPO: dict = {
    "MossTTSDForCausalLM":              "fnlp/XY_Tokenizer_TTSD_V0_hf",      # v0.5 / v0.7 README pin
    "AsteroidTTSModel":                 "fnlp/XY_Tokenizer_TTSD_V0_hf",
    "MossTTSDelayModel":                "OpenMOSS-Team/MOSS-Audio-Tokenizer", # v1.0 / MOSS-TTS processor default
    "Qwen3TTSForConditionalGeneration": "Qwen/Qwen3-TTS-Tokenizer-12Hz",     # Qwen3-TTS-12Hz-* official codec pairing
    "Lfm2AudioForConditionalGeneration":"kyutai/mimi",                       # LFM2-Audio README: "Audio tokenizer: Mimi"
}

# Codec converters that accept an `lm_source=...` kwarg.  Currently both
# MOSS-Audio-Tokenizer and XY-Tokenizer pair with MOSS-TTS-family LMs;
# more codecs will join when M3+ adds residual_depth_ar models.
LM_SOURCE_CAPABLE_CONVERTERS = {"xy_tokenizer", "moss_audio", "qwen3_tts_tokenizer", "mimi"}


def detect_model_type_from_config(config_path: Path) -> str:
    """Detect model type from config.json."""
    import json

    with open(config_path, 'r') as f:
        config = json.load(f)

    # LM-source auto-detection: if architectures[0] matches a known LM
    # family, return the *codec* it pairs with.  Caller then sets
    # lm_source to the same input.
    for a in (config.get("architectures") or []):
        if a in LM_SOURCE_TO_CODEC:
            return LM_SOURCE_TO_CODEC[a]

    model_type = config.get("model_type", "").lower()
    
    if "csm" == model_type:
        return "csm"
    if model_type == "moshi":
        return "moshi"
    elif "mimi" in model_type:
        return "mimi"
    elif "dac" in model_type or "descript" in model_type:
        return "dac"
    elif "wavtokenizer" in model_type:
        return "wavtokenizer"
    elif "qwen3" in model_type or "qwen" in model_type:
        return "qwen3_tts_tokenizer"
    elif "chatterbox_s3t" in model_type or "s3t" == model_type:
        return "chatterbox_s3t"
    elif "chatterbox_s3g" in model_type or "s3g" == model_type:
        return "chatterbox_s3g"
    elif "soprano" in model_type:
        return "soprano"
    elif "nemo" in model_type or "nano" in model_type:
        return "nemo_nano_codec"
    elif "neucodec" in model_type:
        if "distill" in model_type:
            return "distill_neucodec"
        return "neucodec"
    elif "xcodec2" in model_type or "x-codec2" in model_type:
        return "xcodec2"
    elif "bigcodec" in model_type:
        # HKUSTAudio/xcodec2 declares model_type "xcodec2" via its custom config,
        # but earlier snapshots used "bigcodec".
        return "xcodec2"
    elif "snac" in model_type:
        return "snac"
    elif "moss-audio" in model_type or "moss_audio" in model_type or "mossaudio" in model_type:
        return "moss_audio"
    elif "xy_tokenizer" in model_type or "xy-tokenizer" in model_type:
        return "xy_tokenizer"
    else:
        # Try to infer from architecture or other fields
        arch = config.get("architectures", [""])[0].lower() if config.get("architectures") else ""
        if "csmforconditionalgeneration" in arch:
            return "csm"
        elif "mimi" in arch:
            return "mimi"
        elif "dac" in arch:
            return "dac"
        elif "wavtokenizer" in arch:
            return "wavtokenizer"
        elif "qwen3" in arch or "qwen" in arch:
            return "qwen3_tts_tokenizer"
        elif "chatterbox_s3t" in arch or arch == "s3t":
            return "chatterbox_s3t"
        elif "chatterbox_s3g" in arch or arch == "s3g":
            return "chatterbox_s3g"
        elif "soprano" in arch:
            return "soprano"
        elif "nemo" in arch or "nano" in arch:
            return "nemo_nano_codec"
        elif "neucodec" in arch:
            if "distill" in arch:
                return "distill_neucodec"
            return "neucodec"
        elif "xcodec2" in arch or "bigcodec" in arch:
            return "xcodec2"
        elif "snac" in arch:
            return "snac"
        elif "moss" in arch and "audio" in arch:
            return "moss_audio"
        elif "xy_tokenizer" in arch or "xytokenizer" in arch:
            return "xy_tokenizer"

    raise ValueError(f"Unknown model type: {model_type}. Cannot auto-detect.")


def infer_model_type_from_filename(filename: str) -> str | None:
    """Infer model type from checkpoint filename."""
    name_lower = filename.lower()
    if 'wavtokenizer' in name_lower:
        return 'wavtokenizer'
    elif 'mimi' in name_lower:
        return 'mimi'
    elif 'dac' in name_lower:
        return 'dac'
    elif 'qwen' in name_lower:
        return 'qwen3_tts_tokenizer'
    elif 'chatterbox-s3t' in name_lower or 'chatterbox_s3t' in name_lower or name_lower == 's3t':
        return 'chatterbox_s3t'
    elif 'chatterbox-s3g' in name_lower or 'chatterbox_s3g' in name_lower or name_lower == 's3g':
        return 'chatterbox_s3g'
    elif 'soprano' in name_lower:
        return 'soprano'
    elif 'nemo' in name_lower or 'nano-codec' in name_lower:
        return 'nemo_nano_codec'
    elif 'neucodec' in name_lower:
        if 'distill' in name_lower:
            return 'distill_neucodec'
        return 'neucodec'
    elif 'xcodec2' in name_lower or 'x-codec2' in name_lower or 'x_codec2' in name_lower:
        return 'xcodec2'
    elif 'snac' in name_lower:
        return 'snac'
    elif 'moss' in name_lower and 'audio' in name_lower:
        return 'moss_audio'
    elif 'xy_tokenizer' in name_lower or 'xy-tokenizer' in name_lower:
        return 'xy_tokenizer'
    return None


def _is_lm_source_config(config_path: Path) -> bool:
    """True iff config.json's architectures[0] is in LM_SOURCE_TO_CODEC.
    Lets convert-to-gguf default --lm-source to the input when the user
    points at a MOSS-TTSD-style LM checkpoint directly."""
    import json
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
    except Exception:
        return False
    for a in (cfg.get("architectures") or []):
        if a in LM_SOURCE_TO_CODEC:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert audio codec models to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert from HuggingFace
  python convert-to-gguf.py --model-id kyutai/mimi --output mimi.gguf
  
  # Convert from local checkpoint file (auto-detect)
  python convert-to-gguf.py --checkpoint-path ./model.ckpt --output model.gguf
  
  # Convert from local checkpoint directory
  python convert-to-gguf.py --checkpoint-path ./checkpoints/mimi --output mimi.gguf
  
  # With quantization
  python convert-to-gguf.py --model-id kyutai/mimi --output mimi-q4.gguf --quantization Q4_K_M
  
Supported models: mimi, dac, wavtokenizer, qwen3_tts_tokenizer, chatterbox_s3t, chatterbox_s3g, soprano, nemo_nano_codec, neucodec, distill_neucodec
        """
    )
    
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--model-id",
        type=str,
        help="HuggingFace model ID (e.g., kyutai/mimi)"
    )
    source_group.add_argument(
        "--checkpoint-path",
        type=str,
        help="Local checkpoint (directory or single .ckpt/.pth/.bin file)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output GGUF file path"
    )
    
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        default="F16",
        choices=["F32", "F16", "Q8_0", "Q4_K_M", "Q5_K_M"],
        help="Quantization type (default: F16)"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=list_supported_models(),
        help="Force model type (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--quantize-codebook",
        action="store_true",
        help="Also quantize codebook embeddings (default: keep F16 for quality)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # WavTokenizer-specific options
    parser.add_argument(
        "--config-path",
        type=str,
        default="checkpoints/config.yaml",
        help="Config path for WavTokenizer (default: checkpoints/config.yaml)"
    )
    
    parser.add_argument(
        "--wavtokenizer-source",
        type=str,
        help="Path to WavTokenizer source repository"
    )

    parser.add_argument(
        "--lm-source",
        type=str,
        default=None,
        help="Optional path or HF repo id for an LM-side adaptor checkpoint "
             "to bundle into the same GGUF (e.g. fnlp/MOSS-TTSD-v0.5, "
             "OpenMOSS-Team/MOSS-TTSD-v1.0). The codec converter writes "
             "codec.* tensors and metadata; lm.* adaptor weights + "
             "codec.lm.* metadata are appended via the right per-arch "
             "handler in scripts/converters/lm_adaptor/."
    )

    parser.add_argument(
        "--codec-source",
        type=str,
        default=None,
        help="Override the codec source when --model-id / --checkpoint-path "
             "points at an LM checkpoint instead of a codec.  Defaults to "
             "the LM family's pinned codec (fnlp/XY_Tokenizer_TTSD_V0_hf for "
             "MOSS-TTSD-v0.5/v0.7, OpenMOSS-Team/MOSS-Audio-Tokenizer for "
             "MOSS-TTSD-v1.0 / MOSS-TTS)."
    )

    args = parser.parse_args()
    
    # Determine input path and model type
    checkpoint_path: Path | None = None
    
    if args.model_id:
        # HuggingFace mode
        checkpoint_path = None
        if args.model_type:
            model_type = args.model_type
        else:
            # Try to detect from HF model
            try:
                from huggingface_hub import hf_hub_download
                import json
                config_path = hf_hub_download(args.model_id, "config.json")
                # If the input is an LM source (e.g. MOSS-TTSD-v0.5), the
                # caller pointed at the LM, not the codec.  Auto-set
                # --lm-source = that input, redirect --model-id to the
                # LM family's default codec repo.
                if _is_lm_source_config(Path(config_path)):
                    with open(config_path) as f:
                        cfg = json.load(f)
                    arch = (cfg.get("architectures") or [""])[0]
                    model_type = LM_SOURCE_TO_CODEC[arch]
                    if args.lm_source is None:
                        args.lm_source = args.model_id
                    if args.codec_source is None:
                        args.codec_source = LM_SOURCE_DEFAULT_CODEC_REPO.get(arch)
                        if args.codec_source is None:
                            raise ValueError(
                                f"no default codec known for LM arch {arch!r}; "
                                f"pass --codec-source explicitly"
                            )
                    args.model_id = args.codec_source  # codec converter loads this
                    if args.verbose:
                        print(f"Detected LM source ({arch}); "
                              f"lm_source={args.lm_source} codec={args.codec_source} "
                              f"model_type={model_type}")
                else:
                    model_type = detect_model_type_from_config(Path(config_path))
                    if args.verbose:
                        print(f"Auto-detected model type: {model_type}")
            except Exception as e:
                raise ValueError(f"Cannot auto-detect model type for {args.model_id}. Please specify --model-type.") from e
    else:
        # Local checkpoint mode
        input_path = Path(args.checkpoint_path)
        checkpoint_path = input_path
        
        if args.model_type:
            model_type = args.model_type
        elif input_path.is_file() and input_path.suffix in ('.ckpt', '.pth', '.bin'):
            # Single file - infer from filename
            model_type = infer_model_type_from_filename(input_path.name)
            if model_type is None:
                raise ValueError(f"Cannot auto-detect model type from filename {input_path.name}. Please specify --model-type")
            if args.verbose:
                print(f"Auto-detected model type from filename: {model_type}")
        elif input_path.is_dir():
            # Directory - try to find config.json
            config_path = input_path / "config.json"
            if config_path.exists():
                if _is_lm_source_config(config_path):
                    import json
                    with open(config_path) as f:
                        cfg = json.load(f)
                    arch = (cfg.get("architectures") or [""])[0]
                    model_type = LM_SOURCE_TO_CODEC[arch]
                    if args.lm_source is None:
                        args.lm_source = str(input_path)
                    if args.codec_source is None:
                        # Local LM checkpoint without an explicit codec —
                        # fall back to fetching the LM family's pinned codec
                        # from HF (still local-first via HF cache).
                        args.codec_source = LM_SOURCE_DEFAULT_CODEC_REPO.get(arch)
                        if args.codec_source is None:
                            raise ValueError(
                                f"no default codec known for LM arch {arch!r}; "
                                f"pass --codec-source explicitly"
                            )
                    if args.verbose:
                        print(f"Detected LM source ({arch}); "
                              f"lm_source={args.lm_source} "
                              f"codec={args.codec_source} model_type={model_type}")
                    # Re-route the codec load through HF instead of the
                    # local LM checkpoint dir.
                    checkpoint_path = None
                    args.model_id = args.codec_source
                else:
                    model_type = detect_model_type_from_config(config_path)
                    if args.verbose:
                        print(f"Auto-detected model type from config: {model_type}")
            else:
                # Try to infer from directory name
                model_type = infer_model_type_from_filename(input_path.name)
                if model_type is None:
                    raise FileNotFoundError(f"No config.json found in {input_path} and cannot auto-detect")
                if args.verbose:
                    print(f"Auto-detected model type from directory name: {model_type}")
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")
    
    # Get converter class
    converter_class = get_converter_for_model(model_type)
    
    # Create converter instance
    converter_kwargs = {
        "quantization": args.quantization,
        "quantize_codebook": args.quantize_codebook,
        "verbose": args.verbose
    }
    
    # Add model-specific options
    if model_type == "wavtokenizer":
        converter_kwargs["config_path"] = args.config_path
        if args.wavtokenizer_source:
            converter_kwargs["wavtokenizer_source"] = args.wavtokenizer_source

    if args.lm_source is not None:
        if model_type not in LM_SOURCE_CAPABLE_CONVERTERS:
            raise ValueError(
                f"--lm-source is not supported with --model-type {model_type!r}. "
                f"Supported codecs: {sorted(LM_SOURCE_CAPABLE_CONVERTERS)}. "
                f"Other codec converters can opt in by adding an "
                f"`lm_source` __init__ arg and calling `dump_lm_into` after "
                f"writing their codec section (see XYTokenizerConverter / "
                f"MossAudioConverter for reference)."
            )
        converter_kwargs["lm_source"] = args.lm_source

    converter = converter_class(**converter_kwargs)
    
    # Load model
    if args.verbose:
        print(f"Loading {model_type} model...")
    
    if args.model_id:
        converter.load_from_huggingface(args.model_id)
    else:
        # checkpoint_path is guaranteed to be not None here
        converter.load_from_checkpoint(checkpoint_path)
    
    # Convert and save
    if args.verbose:
        print(f"Converting to GGUF with quantization={args.quantization}...")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    converter.convert_and_save(output_path)
    
    print(f"✓ Successfully converted to {output_path}")
    
    # Print file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
