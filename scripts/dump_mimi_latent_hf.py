#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import MimiModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump HF Mimi RVQ latent (post-proj, pre-decoder-transformer)")
    parser.add_argument("--model-dir", default="/home/node/.openclaw/workspace/checkpoints/mimi")
    parser.add_argument("--codes", default="/tmp/mimi_codes.npy")
    parser.add_argument("--out", default="/tmp/mimi_latent_hf.bin")
    args = parser.parse_args()

    codes_path = Path(args.codes)
    if not codes_path.exists():
        raise FileNotFoundError(f"codes not found: {codes_path}")

    codes_np = np.load(str(codes_path))
    if codes_np.ndim != 2:
        raise ValueError(f"expected codes shape (n_q, n_frames), got {codes_np.shape}")

    codes = torch.from_numpy(codes_np.astype(np.int64, copy=False)).unsqueeze(0)

    model = MimiModel.from_pretrained(args.model_dir, local_files_only=True)
    model.eval()

    q = model.quantizer
    n_sem = int(q.num_semantic_quantizers)

    with torch.no_grad():
        sem_codes = codes[:, :n_sem, :]
        latent = q.semantic_residual_vector_quantizer.decode(sem_codes)

        if codes.shape[1] > n_sem:
            acu_codes = codes[:, n_sem:, :]
            latent = latent + q.acoustic_residual_vector_quantizer.decode(acu_codes)

        if latent.ndim != 3:
            raise RuntimeError(f"unexpected latent rank: {latent.ndim}")

        latent_ct = latent[0].detach().to(torch.float32).cpu().numpy()

    if latent_ct.shape[0] != 512:
        print(f"warning: expected hidden size 512, got {latent_ct.shape[0]}")

    out_path = Path(args.out)
    out_path.write_bytes(latent_ct.astype(np.float32, copy=False).tobytes())
    print(f"HF latent dumped: shape={latent_ct.shape}, file={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
