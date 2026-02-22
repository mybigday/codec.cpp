#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import MimiModel


DECODER_STAGE_BY_LAYER = {
    0: "z_dec_l0_conv",
    1: "z_dec_l1_elu",
    2: "z_dec_l2_convtr",
    3: "z_dec_l3_resblock",
    4: "z_dec_l4_elu",
    5: "z_dec_l5_convtr",
    6: "z_dec_l6_resblock",
    7: "z_dec_l7_elu",
    8: "z_dec_l8_convtr",
    9: "z_dec_l9_resblock",
    10: "z_dec_l10_elu",
    11: "z_dec_l11_convtr",
    12: "z_dec_l12_resblock",
    13: "z_dec_l13_elu",
}


def dump_ct(path: str, x: torch.Tensor) -> tuple[int, int]:
    x = x.detach().to(torch.float32).cpu()
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"expected tensor shape (1, C, T), got {tuple(x.shape)}")
    xt = x[0].contiguous().numpy().astype(np.float32, copy=False)
    Path(path).write_bytes(xt.tobytes())
    return int(xt.shape[0]), int(xt.shape[1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump HF Mimi post-latent checkpoints")
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

    decoder_stage_out: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        stage_name = DECODER_STAGE_BY_LAYER[layer_idx]

        def _hook(_module, _inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            decoder_stage_out[stage_name] = out.detach().to(torch.float32).cpu()

        return _hook

    for layer_idx, layer in enumerate(model.decoder.layers):
        if layer_idx in DECODER_STAGE_BY_LAYER:
            hooks.append(layer.register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        z_before_upsample = model.quantizer.decode(codes)
        z_after_upsample = model.upsample(z_before_upsample)
        dtr_out = model.decoder_transformer(z_after_upsample.transpose(1, 2), return_dict=True)
        z_after_transformer = dtr_out.last_hidden_state.transpose(1, 2)
        y_pre_tanh = model.decoder(z_after_transformer)

    for hook in hooks:
        hook.remove()

    meta_lines = []
    c, t = dump_ct("/tmp/mimi_dbg_z_before_upsample_hf.bin", z_before_upsample)
    meta_lines.append(f"z_before_upsample shape={c}x{t} file=/tmp/mimi_dbg_z_before_upsample_hf.bin")
    print(f"DEBUG: dumped z_before_upsample_hf: shape={c}x{t}, file=/tmp/mimi_dbg_z_before_upsample_hf.bin")

    c, t = dump_ct("/tmp/mimi_dbg_z_after_upsample_hf.bin", z_after_upsample)
    meta_lines.append(f"z_after_upsample shape={c}x{t} file=/tmp/mimi_dbg_z_after_upsample_hf.bin")
    print(f"DEBUG: dumped z_after_upsample_hf: shape={c}x{t}, file=/tmp/mimi_dbg_z_after_upsample_hf.bin")

    c, t = dump_ct("/tmp/mimi_dbg_z_after_transformer_hf.bin", z_after_transformer)
    meta_lines.append(f"z_after_transformer shape={c}x{t} file=/tmp/mimi_dbg_z_after_transformer_hf.bin")
    print(f"DEBUG: dumped z_after_transformer_hf: shape={c}x{t}, file=/tmp/mimi_dbg_z_after_transformer_hf.bin")

    for stage_name in DECODER_STAGE_BY_LAYER.values():
        if stage_name not in decoder_stage_out:
            raise RuntimeError(f"missing decoder stage hook output: {stage_name}")
        out_path = f"/tmp/mimi_dbg_{stage_name}_hf.bin"
        c, t = dump_ct(out_path, decoder_stage_out[stage_name])
        meta_lines.append(f"{stage_name} shape={c}x{t} file={out_path}")
        print(f"DEBUG: dumped {stage_name}_hf: shape={c}x{t}, file={out_path}")

    c, t = dump_ct("/tmp/mimi_dbg_y_pre_tanh_hf.bin", y_pre_tanh)
    meta_lines.append(f"y_pre_tanh shape={c}x{t} file=/tmp/mimi_dbg_y_pre_tanh_hf.bin")
    print(f"DEBUG: dumped y_pre_tanh_hf: shape={c}x{t}, file=/tmp/mimi_dbg_y_pre_tanh_hf.bin")

    Path("/tmp/mimi_dbg_post_latent_hf_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    print("DEBUG: wrote metadata: /tmp/mimi_dbg_post_latent_hf_meta.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
