#!/usr/bin/env python3
"""
Unified GGUF converter for audio codec models.

Supports Mimi, DAC, WavTokenizer, and Qwen3-TTS-Tokenizer models from HuggingFace or local checkpoints.

Usage:
    # From HuggingFace
    python convert-to-gguf.py --model-id kyutai/mimi --output mimi.gguf
    
    # From local checkpoint
    python convert-to-gguf.py --input-dir ./mimi-checkpoint --output mimi.gguf
    
    # With quantization
    python convert-to-gguf.py --model-id kyutai/mimi --output mimi-q4.gguf --quantization Q4_K_M
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from converters import get_converter_for_model, list_supported_models


def detect_model_type_from_config(config_path: Path) -> str:
    """Detect model type from config.json."""
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_type = config.get("model_type", "").lower()
    
    if "mimi" in model_type:
        return "mimi"
    elif "dac" in model_type or "descript" in model_type:
        return "dac"
    elif "wavtokenizer" in model_type:
        return "wavtokenizer"
    elif "qwen3" in model_type or "qwen" in model_type:
        return "qwen3_tts_tokenizer"
    else:
        # Try to infer from architecture or other fields
        arch = config.get("architectures", [""])[0].lower() if config.get("architectures") else ""
        if "mimi" in arch:
            return "mimi"
        elif "dac" in arch:
            return "dac"
        elif "wavtokenizer" in arch:
            return "wavtokenizer"
        elif "qwen3" in arch or "qwen" in arch:
            return "qwen3_tts_tokenizer"
    
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
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert audio codec models to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert from HuggingFace
  python convert-to-gguf.py --model-id kyutai/mimi --output mimi.gguf
  
  # Convert from local checkpoint file (auto-detect)
  python convert-to-gguf.py --input-dir ./model.ckpt --output model.gguf
  
  # Convert from local checkpoint directory
  python convert-to-gguf.py --input-dir ./checkpoints/mimi --output mimi.gguf
  
  # With quantization
  python convert-to-gguf.py --model-id kyutai/mimi --output mimi-q4.gguf --quantization Q4_K_M
  
Supported models: mimi, dac, wavtokenizer, qwen3_tts_tokenizer
        """
    )
    
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--model-id",
        type=str,
        help="HuggingFace model ID (e.g., kyutai/mimi)"
    )
    source_group.add_argument(
        "--input-dir",
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
                model_type = detect_model_type_from_config(Path(config_path))
                if args.verbose:
                    print(f"Auto-detected model type: {model_type}")
            except Exception as e:
                raise ValueError(f"Cannot auto-detect model type for {args.model_id}. Please specify --model-type.") from e
    else:
        # Local checkpoint mode
        input_path = Path(args.input_dir)
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
