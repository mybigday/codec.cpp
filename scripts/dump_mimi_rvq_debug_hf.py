#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import MimiModel


def decode_preproj(rvq, codes: torch.Tensor, codebook_dim: int) -> torch.Tensor:
    if codes.ndim != 3:
        raise ValueError(f"expected codes rank=3 [B,K,T], got shape={tuple(codes.shape)}")
    bsz, n_q, n_frames = codes.shape
    if n_q == 0:
        return torch.zeros((bsz, codebook_dim, n_frames), dtype=torch.float32, device=codes.device)

    quantized = torch.zeros((bsz, codebook_dim, n_frames), dtype=torch.float32, device=codes.device)
    for qi in range(n_q):
        layer = rvq.layers[qi]
        indices = codes[:, qi, :]
        quantized = quantized + layer.decode(indices)
    return quantized


def dump_tensor(path: str, x: torch.Tensor) -> tuple[int, int]:
    x = x.detach().to(torch.float32).cpu()
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"expected tensor shape (1, C, T), got {tuple(x.shape)}")
    xt = x[0].contiguous().numpy().astype(np.float32, copy=False)
    Path(path).write_bytes(xt.tobytes())
    return int(xt.shape[0]), int(xt.shape[1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump Mimi HF RVQ debug checkpoints")
    parser.add_argument("--model-dir", default="/home/node/.openclaw/workspace/checkpoints/mimi")
    parser.add_argument("--codes", default="/tmp/mimi_codes.npy")
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
    codebook_dim = int(model.config.codebook_dim)

    with torch.no_grad():
        sem_codes = codes[:, :n_sem, :]
        acu_codes = codes[:, n_sem:, :]

        sem_decoded = decode_preproj(q.semantic_residual_vector_quantizer, sem_codes, codebook_dim)
        acu_decoded = decode_preproj(q.acoustic_residual_vector_quantizer, acu_codes, codebook_dim)
        sem_acu_sum = sem_decoded + acu_decoded

        latent_final = torch.zeros(
            (codes.shape[0], int(model.config.hidden_size), codes.shape[2]),
            dtype=torch.float32,
            device=codes.device,
        )
        if q.semantic_residual_vector_quantizer.output_proj is not None:
            latent_final = latent_final + q.semantic_residual_vector_quantizer.output_proj(sem_decoded)
        if q.acoustic_residual_vector_quantizer.output_proj is not None and acu_codes.shape[1] > 0:
            latent_final = latent_final + q.acoustic_residual_vector_quantizer.output_proj(acu_decoded)

    c, t = dump_tensor("/tmp/mimi_debug_sem_decoded_hf.bin", sem_decoded)
    print(f"DEBUG: dumped sem_decoded_hf: shape={c}x{t}, file=/tmp/mimi_debug_sem_decoded_hf.bin")

    c, t = dump_tensor("/tmp/mimi_debug_acu_decoded_hf.bin", acu_decoded)
    print(f"DEBUG: dumped acu_decoded_hf: shape={c}x{t}, file=/tmp/mimi_debug_acu_decoded_hf.bin")

    c, t = dump_tensor("/tmp/mimi_debug_sem_acu_sum_hf.bin", sem_acu_sum)
    print(f"DEBUG: dumped sem_acu_sum_hf: shape={c}x{t}, file=/tmp/mimi_debug_sem_acu_sum_hf.bin")

    c, t = dump_tensor("/tmp/mimi_debug_latent_final_hf.bin", latent_final)
    print(f"DEBUG: dumped latent_final_hf: shape={c}x{t}, file=/tmp/mimi_debug_latent_final_hf.bin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
