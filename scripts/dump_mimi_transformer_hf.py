#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path
import types

import numpy as np
import torch
from transformers import MimiModel
from transformers.models.mimi.modeling_mimi import apply_rotary_pos_emb, repeat_kv


STAGES = ["ln1", "q", "k", "v", "q_rope", "k_rope", "attn_scores", "attn_ctx", "attn", "resid1", "ln2", "mlp_fc1", "mlp_act", "mlp_fc2", "resid2"]


def load_ct(path: Path, channels: int) -> torch.Tensor:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"empty file: {path}")
    if arr.size % channels != 0:
        raise ValueError(f"invalid size for {path}: {arr.size} not divisible by {channels}")
    ct = arr.reshape(channels, arr.size // channels)
    return torch.from_numpy(ct).unsqueeze(0)


def dump_btc_as_ct(path: Path, x_btc: torch.Tensor) -> tuple[int, int]:
    x = x_btc.detach().to(torch.float32).cpu()
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"expected tensor shape (1, T, C), got {tuple(x.shape)}")
    ct = x[0].transpose(0, 1).contiguous().numpy().astype(np.float32, copy=False)
    path.write_bytes(ct.tobytes())
    return int(ct.shape[0]), int(ct.shape[1])


def dump_btt_as_tt(path: Path, x_btt: torch.Tensor) -> tuple[int, int]:
    x = x_btt.detach().to(torch.float32).cpu()
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != x.shape[2]:
        raise ValueError(f"expected tensor shape (1, T, T), got {tuple(x.shape)}")
    tt = x[0].contiguous().numpy().astype(np.float32, copy=False)
    path.write_bytes(tt.tobytes())
    return int(tt.shape[0]), int(tt.shape[1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump HF Mimi transformer internals by layer")
    parser.add_argument("--model-dir", default="/home/node/.openclaw/workspace/checkpoints/mimi")
    parser.add_argument("--in", dest="inp", default="/tmp/mimi_dbg_z_after_upsample_hf.bin")
    parser.add_argument("--channels", type=int, default=512)
    args = parser.parse_args()

    in_path = Path(args.inp)
    if not in_path.exists():
        raise FileNotFoundError(f"input not found: {in_path}")

    model = MimiModel.from_pretrained(args.model_dir, local_files_only=True)
    model.eval()

    z_ct = load_ct(in_path, args.channels)
    with torch.no_grad():
        z = z_ct
        if z.shape[-1] == 110:
            z = model.upsample(z)
            print("DEBUG: input had T=110; applied model.upsample -> T=220")
        z_btc = z.transpose(1, 2).contiguous().to(torch.float32)

    layers = model.decoder_transformer.layers
    n_layers = len(layers)
    debug_tf = os.getenv("CODEC_MIMI_DEBUG_TRANSFORMER", "")
    dump_all_layers = debug_tf == "all"
    dump_layers = list(range(n_layers)) if dump_all_layers else [0]

    stage_store: dict[tuple[int, str], torch.Tensor] = {}
    hooks = []

    def save_stage(layer_idx: int, stage: str, value: torch.Tensor) -> None:
        if layer_idx not in dump_layers:
            return
        stage_store[(layer_idx, stage)] = value.detach().to(torch.float32).cpu()

    for li, layer in enumerate(layers):
        hooks.append(layer.input_layernorm.register_forward_hook(lambda m, inp, out, li=li: save_stage(li, "ln1", out)))
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda m, inp, li=li: save_stage(li, "resid1", inp[0])))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(lambda m, inp, out, li=li: save_stage(li, "ln2", out)))
        def make_mlp_forward_debug(layer_idx: int):
            def mlp_forward_debug(self, hidden_states: torch.Tensor):
                fc1_out = self.fc1(hidden_states)
                save_stage(layer_idx, "mlp_fc1", fc1_out)
                act_out = self.activation_fn(fc1_out)
                save_stage(layer_idx, "mlp_act", act_out)
                fc2_out = self.fc2(act_out)
                save_stage(layer_idx, "mlp_fc2", fc2_out)
                return fc2_out

            return mlp_forward_debug

        layer.mlp.forward = types.MethodType(make_mlp_forward_debug(li), layer.mlp)

        def layer_out_hook(_m, _inp, out, li=li):
            y = out[0] if isinstance(out, tuple) else out
            save_stage(li, "resid2", y)

        hooks.append(layer.register_forward_hook(layer_out_hook))

        attn = layer.self_attn

        def make_attn_forward_debug(layer_idx: int):
            def attn_forward_debug(
                self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.LongTensor | None = None,
                past_key_values=None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: torch.LongTensor | None = None,
                **kwargs,
            ):
                bsz, q_len, _ = hidden_states.size()

                q_lin = self.q_proj(hidden_states)
                k_lin = self.k_proj(hidden_states)
                v_lin = self.v_proj(hidden_states)
                save_stage(layer_idx, "q", q_lin)
                save_stage(layer_idx, "k", k_lin)
                save_stage(layer_idx, "v", v_lin)

                query_states = q_lin.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = k_lin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = v_lin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                cos, sin = self.rotary_emb(value_states, position_ids)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                save_stage(layer_idx, "q_rope", query_states.transpose(1, 2).contiguous().view(bsz, q_len, -1))
                save_stage(layer_idx, "k_rope", key_states.transpose(1, 2).contiguous().view(bsz, q_len, -1))

                if past_key_values is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                causal_mask = attention_mask
                if attention_mask is not None:
                    causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

                scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
                if causal_mask is not None:
                    scores = scores + causal_mask
                elif q_len > 1:
                    k_len = key_states.shape[-2]
                    causal = torch.ones((q_len, k_len), dtype=torch.bool, device=scores.device).tril()
                    scores = scores.masked_fill(~causal.view(1, 1, q_len, k_len), torch.finfo(scores.dtype).min)

                attn_weights = torch.softmax(scores.to(torch.float32), dim=-1).to(query_states.dtype)
                save_stage(layer_idx, "attn_scores", attn_weights.mean(dim=1))

                attn_ctx = torch.matmul(attn_weights, value_states)
                attn_ctx_btc = attn_ctx.transpose(1, 2).contiguous().view(bsz, q_len, -1)
                save_stage(layer_idx, "attn_ctx", attn_ctx_btc)

                attn_output = self.o_proj(attn_ctx_btc)
                save_stage(layer_idx, "attn", attn_output)

                return attn_output, (attn_weights if output_attentions else None)

            return attn_forward_debug

        attn.forward = types.MethodType(make_attn_forward_debug(li), attn)

    with torch.no_grad():
        _ = model.decoder_transformer(z_btc, return_dict=True)

    for h in hooks:
        h.remove()

    meta_lines = [
        f"num_layers={n_layers}",
        f"channels={args.channels}",
        f"input_file={in_path}",
        "stages=" + ",".join(STAGES),
    ]

    for li in dump_layers:
        for stage in STAGES:
            key = (li, stage)
            if key not in stage_store:
                raise RuntimeError(f"missing captured stage: {stage}_l{li}")
            out_path = Path(f"/tmp/mimi_dbg_z_{stage}_l{li}_hf.bin")
            if stage == "attn_scores":
                r, c = dump_btt_as_tt(out_path, stage_store[key])
                meta_lines.append(f"z_{stage}_l{li} shape={r}x{c} file={out_path}")
                print(f"DEBUG: dumped z_{stage}_l{li}_hf: shape={r}x{c}, file={out_path}")
            else:
                c, t = dump_btc_as_ct(out_path, stage_store[key])
                meta_lines.append(f"z_{stage}_l{li} shape={c}x{t} file={out_path}")
                print(f"DEBUG: dumped z_{stage}_l{li}_hf: shape={c}x{t}, file={out_path}")

    meta_path = Path("/tmp/mimi_dbg_transformer_hf_meta.txt")
    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    print(f"DEBUG: wrote metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
