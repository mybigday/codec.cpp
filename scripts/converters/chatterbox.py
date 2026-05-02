"""Chatterbox S3T/S3G converters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
from safetensors import safe_open
import torch

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

from .base import BaseConverter
from utils.gguf_writer import GGUFWriter


def _load_safetensors(path: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            out[key] = f.get_tensor(key)
    return out


def _load_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _find_s3gen_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = [
        checkpoint_dir / "s3gen_meanflow.safetensors",
        checkpoint_dir / "s3gen.safetensors",
        checkpoint_dir / "model.safetensors",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    safetensors_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if len(safetensors_files) == 1:
        return safetensors_files[0]

    raise FileNotFoundError(
        f"Missing S3 weights in {checkpoint_dir}. Expected s3gen_meanflow.safetensors, "
        "s3gen.safetensors, model.safetensors, or a single *.safetensors file."
    )


def _materialize_weight_norm(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
    g = np.asarray(weight_g, dtype=np.float32)
    v = np.asarray(weight_v, dtype=np.float32)
    axes = tuple(range(1, v.ndim))
    norm = np.linalg.norm(v, axis=axes, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return (v * (g / norm)).astype(np.float32, copy=False)


def _materialize_state_dict(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Bake `parametrizations.weight.original{0,1}` into a single effective weight."""
    out: Dict[str, np.ndarray] = {}
    pending: Dict[str, Dict[str, np.ndarray]] = {}
    for key, value in state.items():
        if key.endswith(".parametrizations.weight.original0"):
            base = key[: -len(".parametrizations.weight.original0")]
            pending.setdefault(base, {})["g"] = value
        elif key.endswith(".parametrizations.weight.original1"):
            base = key[: -len(".parametrizations.weight.original1")]
            pending.setdefault(base, {})["v"] = value
        else:
            out[key] = value
    for base, gv in pending.items():
        if "g" not in gv or "v" not in gv:
            raise RuntimeError(f"incomplete weight_norm parametrization at {base}")
        out[base + ".weight"] = _materialize_weight_norm(gv["g"], gv["v"])
    return out


def _load_optional_conds(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    conds = torch.load(path, map_location="cpu")
    if not isinstance(conds, dict):
        raise ValueError(f"Expected dict in {path}, got {type(conds)}")
    return conds


def _as_numpy_float32(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.astype(np.float32, copy=False)


def _as_numpy_int32_1d(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.astype(np.int32, copy=False).reshape(-1)


# --- S3G state_dict → stable GGUF tensor name map -------------------------

_S3G_FLOW_NUM_DOWN_BLOCKS = 6
_S3G_FLOW_NUM_UP_BLOCKS = 4
_S3G_CFM_NUM_DOWN_BLOCKS = 1
_S3G_CFM_NUM_MID_BLOCKS = 12
_S3G_CFM_NUM_UP_BLOCKS = 1
_S3G_CFM_TRANSFORMERS_PER_BLOCK = 4
_S3G_HIFT_F0_NUM_LAYERS = 5
_S3G_HIFT_NUM_UPS = 3


def _take(state: Dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in state:
        raise KeyError(f"missing S3G tensor: {key}")
    return state.pop(key)


def _emit_flow_attn_block(
    out: list[tuple[str, np.ndarray]],
    state: Dict[str, np.ndarray],
    src_prefix: str,
    dst_prefix: str,
) -> None:
    # ConformerEncoderLayer with rel_pos self-attention + FFN.
    a = src_prefix + ".self_attn"
    f = src_prefix + ".feed_forward"
    out.append((dst_prefix + ".norm_mha.w", _take(state, src_prefix + ".norm_mha.weight")))
    out.append((dst_prefix + ".norm_mha.b", _take(state, src_prefix + ".norm_mha.bias")))
    out.append((dst_prefix + ".norm_ff.w",  _take(state, src_prefix + ".norm_ff.weight")))
    out.append((dst_prefix + ".norm_ff.b",  _take(state, src_prefix + ".norm_ff.bias")))
    out.append((dst_prefix + ".attn.q.w",   _take(state, a + ".linear_q.weight")))
    out.append((dst_prefix + ".attn.q.b",   _take(state, a + ".linear_q.bias")))
    out.append((dst_prefix + ".attn.k.w",   _take(state, a + ".linear_k.weight")))
    out.append((dst_prefix + ".attn.k.b",   _take(state, a + ".linear_k.bias")))
    out.append((dst_prefix + ".attn.v.w",   _take(state, a + ".linear_v.weight")))
    out.append((dst_prefix + ".attn.v.b",   _take(state, a + ".linear_v.bias")))
    out.append((dst_prefix + ".attn.o.w",   _take(state, a + ".linear_out.weight")))
    out.append((dst_prefix + ".attn.o.b",   _take(state, a + ".linear_out.bias")))
    out.append((dst_prefix + ".attn.pos.w", _take(state, a + ".linear_pos.weight")))
    out.append((dst_prefix + ".attn.pbu",   _take(state, a + ".pos_bias_u")))
    out.append((dst_prefix + ".attn.pbv",   _take(state, a + ".pos_bias_v")))
    out.append((dst_prefix + ".ff.w1.w",    _take(state, f + ".w_1.weight")))
    out.append((dst_prefix + ".ff.w1.b",    _take(state, f + ".w_1.bias")))
    out.append((dst_prefix + ".ff.w2.w",    _take(state, f + ".w_2.weight")))
    out.append((dst_prefix + ".ff.w2.b",    _take(state, f + ".w_2.bias")))


def _emit_cfm_resnet(
    out: list[tuple[str, np.ndarray]],
    state: Dict[str, np.ndarray],
    src_prefix: str,
    dst_prefix: str,
) -> None:
    out.append((dst_prefix + ".b1.cv.w", _take(state, src_prefix + ".block1.block.0.weight")))
    out.append((dst_prefix + ".b1.cv.b", _take(state, src_prefix + ".block1.block.0.bias")))
    out.append((dst_prefix + ".b1.ln.w", _take(state, src_prefix + ".block1.block.2.weight")))
    out.append((dst_prefix + ".b1.ln.b", _take(state, src_prefix + ".block1.block.2.bias")))
    out.append((dst_prefix + ".b2.cv.w", _take(state, src_prefix + ".block2.block.0.weight")))
    out.append((dst_prefix + ".b2.cv.b", _take(state, src_prefix + ".block2.block.0.bias")))
    out.append((dst_prefix + ".b2.ln.w", _take(state, src_prefix + ".block2.block.2.weight")))
    out.append((dst_prefix + ".b2.ln.b", _take(state, src_prefix + ".block2.block.2.bias")))
    out.append((dst_prefix + ".mlp.w",   _take(state, src_prefix + ".mlp.1.weight")))
    out.append((dst_prefix + ".mlp.b",   _take(state, src_prefix + ".mlp.1.bias")))
    out.append((dst_prefix + ".res.w",   _take(state, src_prefix + ".res_conv.weight")))
    out.append((dst_prefix + ".res.b",   _take(state, src_prefix + ".res_conv.bias")))


def _emit_cfm_transformer(
    out: list[tuple[str, np.ndarray]],
    state: Dict[str, np.ndarray],
    src_prefix: str,
    dst_prefix: str,
) -> None:
    # BasicTransformerBlock: norm1 → self-attn → norm3 → GELU FFN.
    a = src_prefix + ".attn1"
    out.append((dst_prefix + ".norm1.w", _take(state, src_prefix + ".norm1.weight")))
    out.append((dst_prefix + ".norm1.b", _take(state, src_prefix + ".norm1.bias")))
    out.append((dst_prefix + ".norm3.w", _take(state, src_prefix + ".norm3.weight")))
    out.append((dst_prefix + ".norm3.b", _take(state, src_prefix + ".norm3.bias")))
    out.append((dst_prefix + ".attn.q.w", _take(state, a + ".to_q.weight")))
    out.append((dst_prefix + ".attn.k.w", _take(state, a + ".to_k.weight")))
    out.append((dst_prefix + ".attn.v.w", _take(state, a + ".to_v.weight")))
    out.append((dst_prefix + ".attn.o.w", _take(state, a + ".to_out.0.weight")))
    out.append((dst_prefix + ".attn.o.b", _take(state, a + ".to_out.0.bias")))
    out.append((dst_prefix + ".ff.w1.w",  _take(state, src_prefix + ".ff.net.0.proj.weight")))
    out.append((dst_prefix + ".ff.w1.b",  _take(state, src_prefix + ".ff.net.0.proj.bias")))
    out.append((dst_prefix + ".ff.w2.w",  _take(state, src_prefix + ".ff.net.2.weight")))
    out.append((dst_prefix + ".ff.w2.b",  _take(state, src_prefix + ".ff.net.2.bias")))


def _emit_resblock(
    out: list[tuple[str, np.ndarray]],
    state: Dict[str, np.ndarray],
    src_prefix: str,
    dst_prefix: str,
) -> None:
    # HiFi-GAN ResBlock: 3× (snake-act → conv1 → snake-act → conv2 → residual).
    for k in range(3):
        out.append((dst_prefix + f".cv1.{k}.w", _take(state, src_prefix + f".convs1.{k}.weight")))
        out.append((dst_prefix + f".cv1.{k}.b", _take(state, src_prefix + f".convs1.{k}.bias")))
        out.append((dst_prefix + f".cv2.{k}.w", _take(state, src_prefix + f".convs2.{k}.weight")))
        out.append((dst_prefix + f".cv2.{k}.b", _take(state, src_prefix + f".convs2.{k}.bias")))
        out.append((dst_prefix + f".a1.{k}",    _take(state, src_prefix + f".activations1.{k}.alpha")))
        out.append((dst_prefix + f".a2.{k}",    _take(state, src_prefix + f".activations2.{k}.alpha")))


def _build_s3g_tensor_map(
    state: Dict[str, np.ndarray],
    *,
    meanflow: bool,
) -> list[tuple[str, np.ndarray]]:
    state = dict(state)  # take pop ownership
    out: list[tuple[str, np.ndarray]] = []

    # Drop tokenizer/speaker_encoder weights — not used by the builtin-conds decode path.
    for key in list(state.keys()):
        if key.startswith("tokenizer.") or key.startswith("speaker_encoder."):
            del state[key]

    # --- flow shared
    out.append(("s3g.flow.input_emb.w", _take(state, "flow.input_embedding.weight")))
    out.append(("s3g.flow.spk_aff.w",   _take(state, "flow.spk_embed_affine_layer.weight")))
    out.append(("s3g.flow.spk_aff.b",   _take(state, "flow.spk_embed_affine_layer.bias")))
    out.append(("s3g.flow.proj.w",      _take(state, "flow.encoder_proj.weight")))
    out.append(("s3g.flow.proj.b",      _take(state, "flow.encoder_proj.bias")))

    # --- flow encoder (UpsampleConformerEncoder)
    out.append(("s3g.flow.enc.embed.lin.w", _take(state, "flow.encoder.embed.out.0.weight")))
    out.append(("s3g.flow.enc.embed.lin.b", _take(state, "flow.encoder.embed.out.0.bias")))
    out.append(("s3g.flow.enc.embed.ln.w",  _take(state, "flow.encoder.embed.out.1.weight")))
    out.append(("s3g.flow.enc.embed.ln.b",  _take(state, "flow.encoder.embed.out.1.bias")))
    out.append(("s3g.flow.enc.up_embed.lin.w", _take(state, "flow.encoder.up_embed.out.0.weight")))
    out.append(("s3g.flow.enc.up_embed.lin.b", _take(state, "flow.encoder.up_embed.out.0.bias")))
    out.append(("s3g.flow.enc.up_embed.ln.w",  _take(state, "flow.encoder.up_embed.out.1.weight")))
    out.append(("s3g.flow.enc.up_embed.ln.b",  _take(state, "flow.encoder.up_embed.out.1.bias")))
    out.append(("s3g.flow.enc.after_norm.w", _take(state, "flow.encoder.after_norm.weight")))
    out.append(("s3g.flow.enc.after_norm.b", _take(state, "flow.encoder.after_norm.bias")))
    out.append(("s3g.flow.enc.pre.cv1.w", _take(state, "flow.encoder.pre_lookahead_layer.conv1.weight")))
    out.append(("s3g.flow.enc.pre.cv1.b", _take(state, "flow.encoder.pre_lookahead_layer.conv1.bias")))
    out.append(("s3g.flow.enc.pre.cv2.w", _take(state, "flow.encoder.pre_lookahead_layer.conv2.weight")))
    out.append(("s3g.flow.enc.pre.cv2.b", _take(state, "flow.encoder.pre_lookahead_layer.conv2.bias")))
    out.append(("s3g.flow.enc.up.w", _take(state, "flow.encoder.up_layer.conv.weight")))
    out.append(("s3g.flow.enc.up.b", _take(state, "flow.encoder.up_layer.conv.bias")))
    for li in range(_S3G_FLOW_NUM_DOWN_BLOCKS):
        _emit_flow_attn_block(out, state, f"flow.encoder.encoders.{li}", f"s3g.flow.enc.blk.{li}")
    for li in range(_S3G_FLOW_NUM_UP_BLOCKS):
        _emit_flow_attn_block(out, state, f"flow.encoder.up_encoders.{li}", f"s3g.flow.enc.up_blk.{li}")

    # --- CFM estimator (ConditionalDecoder)
    out.append(("s3g.cfm.t.l1.w", _take(state, "flow.decoder.estimator.time_mlp.linear_1.weight")))
    out.append(("s3g.cfm.t.l1.b", _take(state, "flow.decoder.estimator.time_mlp.linear_1.bias")))
    out.append(("s3g.cfm.t.l2.w", _take(state, "flow.decoder.estimator.time_mlp.linear_2.weight")))
    out.append(("s3g.cfm.t.l2.b", _take(state, "flow.decoder.estimator.time_mlp.linear_2.bias")))
    if meanflow:
        out.append(("s3g.cfm.t_mix.w", _take(state, "flow.decoder.estimator.time_embed_mixer.weight")))

    def _emit_cfm_section(group: str, n_blocks: int, has_trailing: bool) -> None:
        for bi in range(n_blocks):
            src_b = f"flow.decoder.estimator.{group}.{bi}"
            dst_b = f"s3g.cfm.{ {'down_blocks':'dn','mid_blocks':'md','up_blocks':'up'}[group] }.{bi}"
            _emit_cfm_resnet(out, state, src_b + ".0", dst_b + ".r")
            for ti in range(_S3G_CFM_TRANSFORMERS_PER_BLOCK):
                _emit_cfm_transformer(out, state, f"{src_b}.1.{ti}", f"{dst_b}.t.{ti}")
            if has_trailing:
                out.append((dst_b + ".x.w", _take(state, src_b + ".2.weight")))
                out.append((dst_b + ".x.b", _take(state, src_b + ".2.bias")))

    _emit_cfm_section("down_blocks", _S3G_CFM_NUM_DOWN_BLOCKS, has_trailing=True)
    _emit_cfm_section("mid_blocks",  _S3G_CFM_NUM_MID_BLOCKS,  has_trailing=False)
    _emit_cfm_section("up_blocks",   _S3G_CFM_NUM_UP_BLOCKS,   has_trailing=True)

    out.append(("s3g.cfm.final.cv.w", _take(state, "flow.decoder.estimator.final_block.block.0.weight")))
    out.append(("s3g.cfm.final.cv.b", _take(state, "flow.decoder.estimator.final_block.block.0.bias")))
    out.append(("s3g.cfm.final.ln.w", _take(state, "flow.decoder.estimator.final_block.block.2.weight")))
    out.append(("s3g.cfm.final.ln.b", _take(state, "flow.decoder.estimator.final_block.block.2.bias")))
    out.append(("s3g.cfm.proj.w",     _take(state, "flow.decoder.estimator.final_proj.weight")))
    out.append(("s3g.cfm.proj.b",     _take(state, "flow.decoder.estimator.final_proj.bias")))

    # --- HiFTGenerator (mel2wav)
    for li in range(_S3G_HIFT_F0_NUM_LAYERS):
        # condnet has format Sequential(Conv, ELU, Conv, ELU, ...) so even indices are Convs.
        src_idx = li * 2
        out.append((f"s3g.hift.f0.cn.{li}.w", _take(state, f"mel2wav.f0_predictor.condnet.{src_idx}.weight")))
        out.append((f"s3g.hift.f0.cn.{li}.b", _take(state, f"mel2wav.f0_predictor.condnet.{src_idx}.bias")))
    out.append(("s3g.hift.f0.cls.w", _take(state, "mel2wav.f0_predictor.classifier.weight")))
    out.append(("s3g.hift.f0.cls.b", _take(state, "mel2wav.f0_predictor.classifier.bias")))
    out.append(("s3g.hift.src.lin.w", _take(state, "mel2wav.m_source.l_linear.weight")))
    out.append(("s3g.hift.src.lin.b", _take(state, "mel2wav.m_source.l_linear.bias")))
    out.append(("s3g.hift.conv_pre.w", _take(state, "mel2wav.conv_pre.weight")))
    out.append(("s3g.hift.conv_pre.b", _take(state, "mel2wav.conv_pre.bias")))
    out.append(("s3g.hift.conv_post.w", _take(state, "mel2wav.conv_post.weight")))
    out.append(("s3g.hift.conv_post.b", _take(state, "mel2wav.conv_post.bias")))
    for ui in range(_S3G_HIFT_NUM_UPS):
        out.append((f"s3g.hift.up.{ui}.w", _take(state, f"mel2wav.ups.{ui}.weight")))
        out.append((f"s3g.hift.up.{ui}.b", _take(state, f"mel2wav.ups.{ui}.bias")))
        out.append((f"s3g.hift.src_dn.{ui}.w", _take(state, f"mel2wav.source_downs.{ui}.weight")))
        out.append((f"s3g.hift.src_dn.{ui}.b", _take(state, f"mel2wav.source_downs.{ui}.bias")))
        _emit_resblock(out, state, f"mel2wav.source_resblocks.{ui}", f"s3g.hift.src_rb.{ui}")
        for ki in range(3):  # 3 parallel resblocks per upsample stage
            _emit_resblock(out, state, f"mel2wav.resblocks.{ui * 3 + ki}", f"s3g.hift.rb.{ui * 3 + ki}")

    # Sanity: warn if we left tensors on the floor (other than known runtime-recomputed buffers).
    leftovers = sorted(state.keys())
    if leftovers:
        raise RuntimeError(f"unmapped S3G tensors after conversion: {leftovers[:20]} (+{len(leftovers)-20} more)" if len(leftovers) > 20 else f"unmapped S3G tensors after conversion: {leftovers}")
    return out


class _BaseChatterboxConverter(BaseConverter):
    model_name = "Chatterbox"

    def _write_common_metadata(self, writer: GGUFWriter, cfg: Dict[str, Any]) -> None:
        writer.add_name(self.model_name)
        writer.add_uint32("codec.sample_rate", int(cfg["sample_rate"]))
        if int(cfg.get("encode_sample_rate", 0)) > 0:
            writer.add_uint32("codec.encode_sample_rate", int(cfg["encode_sample_rate"]))
        writer.add_uint32("codec.hop_size", int(cfg["hop_size"]))
        writer.add_uint32("codec.n_q", int(cfg["n_q"]))
        writer.add_uint32("codec.codebook_size", int(cfg["codebook_size"]))
        writer.add_bool("codec.has_encoder", bool(cfg["has_encoder"]))
        writer.add_bool("codec.has_decoder", bool(cfg["has_decoder"]))
        if int(cfg.get("n_fft", -1)) > 0:
            writer.add_uint32("codec.n_fft", int(cfg["n_fft"]))
        if int(cfg.get("win_length", -1)) > 0:
            writer.add_uint32("codec.win_length", int(cfg["win_length"]))
        if int(cfg.get("n_mels", -1)) > 0:
            writer.add_uint32("codec.n_mels", int(cfg["n_mels"]))
        if int(cfg.get("token_rate_hz", 0)) > 0:
            writer.add_uint32("codec.token_rate_hz", int(cfg["token_rate_hz"]))

    def _add_tensor(self, writer: GGUFWriter, name: str, arr: np.ndarray, st_dtype: str | None = None) -> None:
        arr = np.asarray(arr)
        if arr.ndim == 0:
            return
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32, copy=False)
        super()._add_tensor(writer, name, arr, st_dtype=st_dtype)


class ChatterboxS3TConverter(_BaseChatterboxConverter):
    model_name = "Chatterbox-S3T"

    @property
    def model_type(self) -> str:
        return "chatterbox_s3t"

    @property
    def architecture(self) -> str:
        return "chatterbox_s3t"

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        cfg = {
            "sample_rate": 24000,
            "encode_sample_rate": 16000,
            "hop_size": 960,
            "n_q": 1,
            "codebook_size": 6561,
            "n_fft": 400,
            "win_length": 400,
            "n_mels": 128,
            "token_rate_hz": 25,
            "audio_state": 1280,
            "audio_head": 20,
            "audio_layer": 6,
            "fsmn_kernel_size": 31,
            "rope_theta": 10000.0,
            "has_encoder": True,
            "has_decoder": False,
        }
        cfg.update(_load_optional_json(checkpoint_dir / "config.json"))

        weights_path = _find_s3gen_checkpoint(checkpoint_dir)
        state = _load_safetensors(weights_path)
        self.state_dict = {k: v for k, v in state.items() if k.startswith("tokenizer.")}
        if not self.state_dict:
            raise RuntimeError(f"{weights_path} does not contain tokenizer.* weights")
        self.config = cfg
        self.log(
            f"Loaded Chatterbox-S3T from {weights_path} "
            f"({len(self.state_dict)} tokenizer tensors)"
        )

    def load_from_huggingface(self, model_id: str) -> None:
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub is required to load from HuggingFace")
        ckpt_dir = Path(
            snapshot_download(
                model_id,
                allow_patterns=["*.safetensors", "*.json", "*.pt", "*.txt", "*.model"],
            )
        )
        self.load_from_checkpoint(ckpt_dir)

    def convert_and_save(self, output_path: Path) -> None:
        if self.state_dict is None or self.config is None:
            raise RuntimeError("No model loaded. Call load_from_checkpoint/load_from_huggingface first.")

        writer = GGUFWriter(output_path, self.architecture)
        self._reset_quant_stats()
        self._write_common_metadata(writer, self.config)
        writer.add_uint32("chatterbox_s3t.audio_state", int(self.config["audio_state"]))
        writer.add_uint32("chatterbox_s3t.audio_head", int(self.config["audio_head"]))
        writer.add_uint32("chatterbox_s3t.audio_layer", int(self.config["audio_layer"]))
        writer.add_uint32("chatterbox_s3t.fsmn_kernel_size", int(self.config["fsmn_kernel_size"]))
        writer.add_float32("chatterbox_s3t.rope_theta", float(self.config["rope_theta"]))

        def add(name: str, key: str) -> None:
            if key not in self.state_dict:
                raise KeyError(f"missing tokenizer tensor: {key}")
            self._add_tensor(writer, name, self.state_dict[key])

        add("s3t.mel_filters", "tokenizer._mel_filters")
        if "tokenizer.window" in self.state_dict:
            add("s3t.window", "tokenizer.window")
        add("s3t.enc.conv1.w", "tokenizer.encoder.conv1.weight")
        add("s3t.enc.conv1.b", "tokenizer.encoder.conv1.bias")
        add("s3t.enc.conv2.w", "tokenizer.encoder.conv2.weight")
        add("s3t.enc.conv2.b", "tokenizer.encoder.conv2.bias")

        n_layers = int(self.config["audio_layer"])
        for li in range(n_layers):
            p = f"tokenizer.encoder.blocks.{li}"
            d = f"s3t.enc.blk.{li}"
            add(f"{d}.attn_ln.w", f"{p}.attn_ln.weight")
            add(f"{d}.attn_ln.b", f"{p}.attn_ln.bias")
            add(f"{d}.attn.q.w", f"{p}.attn.query.weight")
            add(f"{d}.attn.q.b", f"{p}.attn.query.bias")
            add(f"{d}.attn.k.w", f"{p}.attn.key.weight")
            add(f"{d}.attn.v.w", f"{p}.attn.value.weight")
            add(f"{d}.attn.v.b", f"{p}.attn.value.bias")
            add(f"{d}.attn.o.w", f"{p}.attn.out.weight")
            add(f"{d}.attn.o.b", f"{p}.attn.out.bias")
            add(f"{d}.attn.fsmn.w", f"{p}.attn.fsmn_block.weight")
            add(f"{d}.mlp_ln.w", f"{p}.mlp_ln.weight")
            add(f"{d}.mlp_ln.b", f"{p}.mlp_ln.bias")
            add(f"{d}.mlp.fc1.w", f"{p}.mlp.0.weight")
            add(f"{d}.mlp.fc1.b", f"{p}.mlp.0.bias")
            add(f"{d}.mlp.fc2.w", f"{p}.mlp.2.weight")
            add(f"{d}.mlp.fc2.b", f"{p}.mlp.2.bias")

        add("s3t.q.proj.w", "tokenizer.quantizer._codebook.project_down.weight")
        add("s3t.q.proj.b", "tokenizer.quantizer._codebook.project_down.bias")

        writer.write()
        self._warn_if_no_quantized()


class ChatterboxS3GConverter(_BaseChatterboxConverter):
    model_name = "Chatterbox-S3G"

    @property
    def model_type(self) -> str:
        return "chatterbox_s3g"

    @property
    def architecture(self) -> str:
        return "chatterbox_s3g"

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        cfg = {
            "sample_rate": 24000,
            "hop_size": 960,
            "n_q": 1,
            "codebook_size": 6561,
            "token_rate_hz": 25,
            "meanflow": False,
            "has_encoder": False,
            "has_decoder": True,
        }
        cfg.update(_load_optional_json(checkpoint_dir / "config.json"))
        weights_path = _find_s3gen_checkpoint(checkpoint_dir)
        cfg["meanflow"] = bool(cfg.get("meanflow", False) or ("meanflow" in weights_path.name))
        self.state_dict = _load_safetensors(weights_path)
        self.conds = _load_optional_conds(checkpoint_dir / "conds.pt")
        self.config = cfg
        self.log(
            f"Loaded Chatterbox-S3G checkpoint from {weights_path} "
            f"({len(self.state_dict)} tensors, builtin_conds={self.conds is not None})"
        )

    def load_from_huggingface(self, model_id: str) -> None:
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub is required to load from HuggingFace")
        ckpt_dir = Path(
            snapshot_download(
                model_id,
                allow_patterns=["*.safetensors", "*.json", "*.pt", "*.txt", "*.model"],
            )
        )
        self.load_from_checkpoint(ckpt_dir)

    def convert_and_save(self, output_path: Path) -> None:
        if self.state_dict is None or self.config is None:
            raise RuntimeError("No model loaded. Call load_from_checkpoint/load_from_huggingface first.")

        meanflow = bool(self.config.get("meanflow", False))
        writer = GGUFWriter(output_path, self.architecture)
        self._reset_quant_stats()
        self._write_common_metadata(writer, self.config)
        writer.add_bool("chatterbox_s3g.meanflow", meanflow)

        if self.conds is not None:
            gen_conds = self.conds.get("gen")
            if not isinstance(gen_conds, dict):
                raise ValueError("conds.pt missing dict entry 'gen'")
            prompt_token = _as_numpy_int32_1d(gen_conds["prompt_token"])
            prompt_token_len = _as_numpy_int32_1d(gen_conds["prompt_token_len"])
            prompt_feat = _as_numpy_float32(gen_conds["prompt_feat"])
            embedding = _as_numpy_float32(gen_conds["embedding"])

            if prompt_token_len.size != 1:
                raise ValueError("S3G builtin prompt_token_len must be scalar/batch-1")
            if prompt_feat.ndim != 3 or prompt_feat.shape[0] != 1:
                raise ValueError(f"S3G builtin prompt_feat must be [1, T, 80], got {prompt_feat.shape}")
            if embedding.ndim != 2 or embedding.shape[0] != 1:
                raise ValueError(f"S3G builtin embedding must be [1, D], got {embedding.shape}")
            if int(prompt_token_len[0]) > prompt_token.size:
                raise ValueError("S3G builtin prompt_token_len exceeds prompt_token size")

            writer.add_bool("chatterbox_s3g.has_builtin_conditioning", True)
            writer.add_uint32("chatterbox_s3g.cond.prompt_token_len", int(prompt_token_len[0]))
            writer.add_uint32("chatterbox_s3g.cond.prompt_feat_frames", int(prompt_feat.shape[1]))
            writer.add_uint32("chatterbox_s3g.cond.prompt_feat_dim", int(prompt_feat.shape[2]))
            writer.add_uint32("chatterbox_s3g.cond.embedding_dim", int(embedding.shape[1]))
            writer.add_array("chatterbox_s3g.cond.prompt_token", prompt_token.tolist())
            self._add_tensor(writer, "s3g.cond.prompt_feat", prompt_feat, st_dtype="F32")
            self._add_tensor(writer, "s3g.cond.embedding", embedding, st_dtype="F32")
        else:
            writer.add_bool("chatterbox_s3g.has_builtin_conditioning", False)

        flat = _materialize_state_dict(self.state_dict)
        emit = _build_s3g_tensor_map(flat, meanflow=meanflow)
        for gguf_name, arr in emit:
            self._add_tensor(writer, gguf_name, arr)

        writer.write()
        self._warn_if_no_quantized()
