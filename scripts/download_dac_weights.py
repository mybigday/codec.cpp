#!/usr/bin/env python3
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> int:
    parser = argparse.ArgumentParser(description="Download DAC files from Hugging Face")
    parser.add_argument("--repo-id", default="ibm-research/DAC.speech.v1.0", help="HF repository id")
    parser.add_argument(
        "--weight-file",
        default="weights_24khz_3kbps_v1.0.pth",
        help="Weight filename in the repository",
    )
    parser.add_argument(
        "--config-file",
        default="config.json",
        help="Config filename in the repository",
    )
    parser.add_argument(
        "--out-dir",
        default="checkpoints/dac_hf",
        help="Output directory for downloaded files",
    )
    parser.add_argument("--revision", default=None, help="Optional branch/tag/commit revision")
    parser.add_argument("--local-files-only", action="store_true", help="Only use local HF cache")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"repo_id: {args.repo_id}")
    print(f"out_dir: {out_dir}")

    config_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.config_file,
        local_dir=str(out_dir),
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    weight_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.weight_file,
        local_dir=str(out_dir),
        revision=args.revision,
        local_files_only=args.local_files_only,
    )

    print(f"downloaded config : {config_path}")
    print(f"downloaded weights: {weight_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
