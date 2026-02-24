"""Qwen3-TTS-Tokenizer converter."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from safetensors import safe_open

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

from .base import BaseConverter
from .mimi import (
    build_weight_transforms as mimi_build_weight_transforms,
    map_tensor_names as mimi_map_tensor_names,
    shorten_tensor_name as mimi_shorten_tensor_name,
    transform_tensor_for_codec as mimi_transform_tensor_for_codec,
)
from utils.gguf_writer import GGUFWriter
from utils import quantization as quant_utils

MAX_TENSOR_NAME = 63
MAX_QWEN_Q = 64


def _load_safetensors(path: Path) -> Dict[str, np.ndarray]:
    state_dict: Dict[str, np.ndarray] = {}
    with safe_open(path, framework="numpy") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def _as_int_list(value: Any) -> List[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if value is None:
        return []
    return [int(value)]


def _decoder_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    dec = config.get("decoder_config")
    if isinstance(dec, dict):
        return dec
    return config


def _encoder_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    enc = config.get("encoder_config")
    if isinstance(enc, dict):
        return enc
    return config


RE_PT_LAYER = re.compile(r"^decoder\.pre_transformer\.layers\.(\d+)\.(.+)$")
RE_UPSAMPLE = re.compile(r"^decoder\.upsample\.(\d+)\.(\d+)\.(.+)$")
RE_DEC_BLOCK = re.compile(r"^decoder\.decoder\.(\d+)\.block\.(\d+)\.(.+)$")
RE_DEC_SIMPLE = re.compile(r"^decoder\.decoder\.(\d+)\.(.+)$")


def _transform_conv_weight(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"conv weight must be rank-3: {arr.shape}")
    # Keep PyTorch layout. GGUF writer reverses shape so ggml sees:
    # - Conv1d: torch [out, in, k] -> ggml [k, in, out]
    # - ConvTranspose1d: torch [in, out, k] -> ggml [k, out, in]
    return np.ascontiguousarray(arr)


def _transform_linear_weight(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"linear weight must be rank-2: {arr.shape}")
    # Keep PyTorch layout [out, in]. GGUF writer reverses dims so ggml sees [in, out].
    return np.ascontiguousarray(arr)


def _snakebeta_alpha(alpha: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float32)
    return np.exp(alpha)


def _snakebeta_inv_beta(beta: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, dtype=np.float32)
    return 1.0 / (np.exp(beta) + 1e-9)


class Qwen3TTSTokenizerConverter(BaseConverter):
    @property
    def model_type(self) -> str:
        return "qwen3_tts_tokenizer"

    @property
    def architecture(self) -> str:
        return "qwen3_tts_tokenizer"

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        st_path = checkpoint_dir / "model.safetensors"
        cfg_path = checkpoint_dir / "config.json"
        if not st_path.exists():
            raise FileNotFoundError(f"Missing model.safetensors in {checkpoint_dir}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config.json in {checkpoint_dir}")
        self.state_dict = _load_safetensors(st_path)
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.log(f"Loaded Qwen3-TTS-Tokenizer checkpoint from {checkpoint_dir} ({len(self.state_dict)} tensors)")

    def load_from_huggingface(self, model_id: str) -> None:
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub is required to load from HuggingFace")
        ckpt_dir = Path(snapshot_download(model_id))
        self.load_from_checkpoint(ckpt_dir)

    def _write_encoder_weights(self, writer: GGUFWriter, used: set[str]) -> None:
        if self.state_dict is None:
            raise RuntimeError("state_dict not loaded")
        encoder_sd = {k[len("encoder."):]: v for k, v in self.state_dict.items() if k.startswith("encoder.")}
        keys = sorted(encoder_sd.keys())
        weight_transforms = mimi_build_weight_transforms(keys)
        for key in keys:
            arr = mimi_transform_tensor_for_codec(key, encoder_sd[key], weight_transforms)
            for mapped in mimi_map_tensor_names(key):
                short = mimi_shorten_tensor_name(mapped, used)
                writer.add_tensor(short, arr, self._get_target_dtype(short, arr))

        # Add normalized RVQ codebook embeddings (semantic + acoustic) for encoder.
        def add_codebook_embeds(prefix: str, short_prefix: str) -> None:
            for qi in range(MAX_QWEN_Q):
                es_key = f"{prefix}layers.{qi}.codebook.embed_sum"
                cu_key = f"{prefix}layers.{qi}.codebook.cluster_usage"
                if es_key not in encoder_sd or cu_key not in encoder_sd:
                    continue
                embed_sum = np.asarray(encoder_sd[es_key], dtype=np.float32)
                usage = np.asarray(encoder_sd[cu_key], dtype=np.float32)
                denom = np.maximum(usage[:, None], 1e-5)
                embed = embed_sum / denom
                if embed.ndim != 2:
                    raise ValueError(f"Unexpected encoder codebook shape: {es_key} {embed.shape}")
                name = f"{short_prefix}layers.{qi}.cb.embed"
                writer.add_tensor(name, embed, self._get_target_dtype(name, embed))
                used.add(name)

        add_codebook_embeds("quantizer.semantic_residual_vector_quantizer.", "q.s.")
        add_codebook_embeds("quantizer.acoustic_residual_vector_quantizer.", "q.a.")

    def _write_decoder_weights(self, writer: GGUFWriter, used: set[str]) -> None:
        if self.state_dict is None:
            raise RuntimeError("state_dict not loaded")

        def add_tensor(name: str, arr: np.ndarray) -> None:
            if len(name) > MAX_TENSOR_NAME:
                raise ValueError(f"Qwen tensor name too long: {name}")
            writer.add_tensor(name, arr, self._get_target_dtype(name, arr))
            used.add(name)

        def map_decoder_key(key: str) -> Optional[Tuple[str, str]]:
            if key.startswith("decoder.pre_transformer.layers."):
                m = RE_PT_LAYER.match(key)
                if not m:
                    return None
                li = int(m.group(1))
                rest = m.group(2)
                prefix = f"q3t.dec.pt.l{li}."
                mapping = {
                    "input_layernorm.weight": ("inln.w", "norm"),
                    "post_attention_layernorm.weight": ("paln.w", "norm"),
                    "self_attn.q_proj.weight": ("attn.q.w", "linear"),
                    "self_attn.q_proj.bias": ("attn.q.b", "bias"),
                    "self_attn.k_proj.weight": ("attn.k.w", "linear"),
                    "self_attn.k_proj.bias": ("attn.k.b", "bias"),
                    "self_attn.v_proj.weight": ("attn.v.w", "linear"),
                    "self_attn.v_proj.bias": ("attn.v.b", "bias"),
                    "self_attn.o_proj.weight": ("attn.o.w", "linear"),
                    "self_attn.o_proj.bias": ("attn.o.b", "bias"),
                    "mlp.gate_proj.weight": ("mlp.gate.w", "linear"),
                    "mlp.up_proj.weight": ("mlp.up.w", "linear"),
                    "mlp.down_proj.weight": ("mlp.down.w", "linear"),
                    "self_attn_layer_scale.scale": ("sa.scale", "scale"),
                    "mlp_layer_scale.scale": ("mlp.scale", "scale"),
                }
                if rest in mapping:
                    suffix, kind = mapping[rest]
                    return prefix + suffix, kind
                return None

            if key.startswith("decoder.pre_transformer."):
                rest = key[len("decoder.pre_transformer.") :]
                mapping = {
                    "norm.weight": ("q3t.dec.pt.norm.w", "norm"),
                    "input_proj.weight": ("q3t.dec.pt.in.w", "linear"),
                    "input_proj.bias": ("q3t.dec.pt.in.b", "bias"),
                    "output_proj.weight": ("q3t.dec.pt.out.w", "linear"),
                    "output_proj.bias": ("q3t.dec.pt.out.b", "bias"),
                }
                if rest in mapping:
                    return mapping[rest]
                return None

            if key.startswith("decoder.quantizer."):
                if key == "decoder.quantizer.rvq_first.output_proj.weight":
                    return "q3t.dec.q.s.op.w", "linear_raw"
                if key == "decoder.quantizer.rvq_rest.output_proj.weight":
                    return "q3t.dec.q.a.op.w", "linear_raw"
                return None

            if key.startswith("decoder.pre_conv.conv."):
                suffix = key[len("decoder.pre_conv.conv.") :]
                if suffix == "weight":
                    return "q3t.dec.pre.conv.w", "conv"
                if suffix == "bias":
                    return "q3t.dec.pre.conv.b", "bias"
                return None

            if key.startswith("decoder.upsample."):
                m = RE_UPSAMPLE.match(key)
                if not m:
                    return None
                ui = int(m.group(1))
                block = int(m.group(2))
                rest = m.group(3)
                if block == 0:
                    if rest == "conv.weight":
                        return f"q3t.dec.up{ui}.tr.w", "convtr"
                    if rest == "conv.bias":
                        return f"q3t.dec.up{ui}.tr.b", "bias"
                if block == 1:
                    if rest == "dwconv.conv.weight":
                        return f"q3t.dec.up{ui}.cnx.dw.w", "conv"
                    if rest == "dwconv.conv.bias":
                        return f"q3t.dec.up{ui}.cnx.dw.b", "bias"
                    if rest == "norm.weight":
                        return f"q3t.dec.up{ui}.cnx.norm.w", "norm"
                    if rest == "norm.bias":
                        return f"q3t.dec.up{ui}.cnx.norm.b", "bias"
                    if rest == "pwconv1.weight":
                        return f"q3t.dec.up{ui}.cnx.pw1.w", "linear"
                    if rest == "pwconv1.bias":
                        return f"q3t.dec.up{ui}.cnx.pw1.b", "bias"
                    if rest == "pwconv2.weight":
                        return f"q3t.dec.up{ui}.cnx.pw2.w", "linear"
                    if rest == "pwconv2.bias":
                        return f"q3t.dec.up{ui}.cnx.pw2.b", "bias"
                    if rest == "gamma":
                        return f"q3t.dec.up{ui}.cnx.gamma", "gamma"
                return None

            if key.startswith("decoder.decoder."):
                m = RE_DEC_BLOCK.match(key)
                if m:
                    bi_raw = int(m.group(1))
                    if bi_raw <= 0:
                        return None
                    bi = bi_raw - 1
                    idx = int(m.group(2))
                    rest = m.group(3)
                    if idx == 0:
                        if rest == "alpha":
                            return f"q3t.dec.b{bi}.s0.a", "snake_alpha"
                        if rest == "beta":
                            return f"q3t.dec.b{bi}.s0.binv", "snake_beta"
                    if idx == 1:
                        if rest == "conv.weight":
                            return f"q3t.dec.b{bi}.tr.w", "convtr"
                        if rest == "conv.bias":
                            return f"q3t.dec.b{bi}.tr.b", "bias"
                    if idx in (2, 3, 4):
                        ri = idx - 2
                        if rest == "act1.alpha":
                            return f"q3t.dec.b{bi}.r{ri}.s1.a", "snake_alpha"
                        if rest == "act1.beta":
                            return f"q3t.dec.b{bi}.r{ri}.s1.binv", "snake_beta"
                        if rest == "conv1.conv.weight":
                            return f"q3t.dec.b{bi}.r{ri}.c1.w", "conv"
                        if rest == "conv1.conv.bias":
                            return f"q3t.dec.b{bi}.r{ri}.c1.b", "bias"
                        if rest == "act2.alpha":
                            return f"q3t.dec.b{bi}.r{ri}.s2.a", "snake_alpha"
                        if rest == "act2.beta":
                            return f"q3t.dec.b{bi}.r{ri}.s2.binv", "snake_beta"
                        if rest == "conv2.conv.weight":
                            return f"q3t.dec.b{bi}.r{ri}.c2.w", "conv"
                        if rest == "conv2.conv.bias":
                            return f"q3t.dec.b{bi}.r{ri}.c2.b", "bias"
                    return None

                m = RE_DEC_SIMPLE.match(key)
                if m:
                    idx = int(m.group(1))
                    rest = m.group(2)
                    if rest == "conv.weight":
                        if idx == 0:
                            return "q3t.dec.d0.w", "conv"
                        if idx == 6:
                            return "q3t.dec.final.w", "conv"
                        return None
                    if rest == "conv.bias":
                        if idx == 0:
                            return "q3t.dec.d0.b", "bias"
                        if idx == 6:
                            return "q3t.dec.final.b", "bias"
                        return None
                    if rest == "alpha":
                        if idx == 5:
                            return "q3t.dec.final.s.a", "snake_alpha"
                        return None
                    if rest == "beta":
                        if idx == 5:
                            return "q3t.dec.final.s.binv", "snake_beta"
                        return None
                return None

            return None

        decoder_keys = [k for k in self.state_dict.keys() if k.startswith("decoder.")]
        for key in sorted(decoder_keys):
            if ".embedding_sum" in key or ".cluster_usage" in key:
                continue
            mapped = map_decoder_key(key)
            if mapped is None:
                self.log(f"Skipping unmapped decoder tensor: {key}")
                continue
            name, kind = mapped
            arr = np.asarray(self.state_dict[key])
            if kind == "conv" or kind == "convtr":
                arr = _transform_conv_weight(arr)
            elif kind == "linear":
                if arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = arr[..., 0]
                arr = _transform_linear_weight(arr)
            elif kind == "linear_raw":
                if arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = arr[..., 0]
            elif kind == "snake_alpha":
                arr = _snakebeta_alpha(arr)
            elif kind == "snake_beta":
                arr = _snakebeta_inv_beta(arr)
            add_tensor(name, arr)

        # Add normalized codebook embeddings for quantizers (rvq_first + rvq_rest).
        n_sem = 1
        for group, offset in (("rvq_first", 0), ("rvq_rest", n_sem)):
            for qi in range(MAX_QWEN_Q):
                es_key = f"decoder.quantizer.{group}.vq.layers.{qi}._codebook.embedding_sum"
                cu_key = f"decoder.quantizer.{group}.vq.layers.{qi}._codebook.cluster_usage"
                if es_key not in self.state_dict or cu_key not in self.state_dict:
                    continue
                embed_sum = np.asarray(self.state_dict[es_key], dtype=np.float32)
                usage = np.asarray(self.state_dict[cu_key], dtype=np.float32)
                denom = np.maximum(usage[:, None], 1e-5)
                embed = embed_sum / denom
                if embed.ndim != 2:
                    raise ValueError(f"Unexpected codebook shape: {es_key} {embed.shape}")
                embed = np.ascontiguousarray(embed)
                name = f"q3t.dec.q.l{qi + offset}.codebook"
                add_tensor(name, embed)

    def _get_target_dtype(self, name: str, arr: np.ndarray) -> np.dtype:
        if arr.dtype == np.float16:
            return np.float16
        if any(
            token in name
            for token in (
                ".codebook",
                ".norm.",
                ".inln.",
                ".paln.",
                ".scale",
                ".gamma",
                ".a",
                ".binv",
            )
        ):
            return np.float16 if self.quantization != "F32" else np.float32
        if self.quantization == "F32":
            return np.float32
        if self.quantization == "F16":
            return np.float16
        if arr.ndim >= 2 and self.should_quantize_tensor(name, arr):
            if self.quantization == "Q8_0":
                return quant_utils.Q8_0
            if self.quantization == "Q4_K_M":
                return quant_utils.Q4_K_M
            if self.quantization == "Q5_K_M":
                return quant_utils.Q5_K_M
        return np.float16

    def convert_and_save(self, output_path: Path) -> None:
        if self.state_dict is None or self.config is None:
            raise RuntimeError("No model loaded. Call load_from_checkpoint/load_from_huggingface first.")

        writer = GGUFWriter(output_path, self.architecture)
        writer.add_name("Qwen3-TTS-Tokenizer")

        dec = _decoder_cfg(self.config)
        enc = _encoder_cfg(self.config)

        sr = int(self.config.get("output_sample_rate", self.config.get("input_sample_rate", 24000)))
        hop_size = int(self.config.get("decode_upsample_rate", 0))
        if hop_size <= 0:
            frame_rate = float(self.config.get("frame_rate", dec.get("frame_rate", 12.0)))
            hop_size = int(round(sr / frame_rate)) if frame_rate > 0 else int(dec.get("hop_size", 0))

        dec_n_q = int(dec.get("num_quantizers", 0))
        dec_codebook_size = int(dec.get("codebook_size", 2048))
        dec_codebook_dim = int(dec.get("codebook_dim", 1024))
        dec_latent_dim = int(dec.get("latent_dim", dec.get("output_dim", 1024)))

        writer.add_uint32("codec.sample_rate", sr)
        writer.add_uint32("codec.hop_size", hop_size)
        writer.add_uint32("codec.n_q", dec_n_q)
        writer.add_uint32("codec.codebook_size", dec_codebook_size)
        writer.add_uint32("codec.codebook_dim", dec_codebook_dim)
        writer.add_uint32("codec.latent_dim", dec_latent_dim)
        writer.add_bool("codec.has_encoder", True)
        writer.add_bool("codec.has_decoder", True)

        # Encoder metadata (Mimi-compatible) for codec_mimi_encode.
        writer.add_uint32("qwen3.encoder.codebook_size", int(enc.get("codebook_size", dec_codebook_size)))
        writer.add_uint32("qwen3.encoder.codebook_dim", int(enc.get("codebook_dim", dec_codebook_dim)))
        writer.add_uint32("qwen3.encoder.n_q", int(enc.get("num_quantizers", dec_n_q)))
        writer.add_uint32("qwen3.encoder.hidden_size", int(enc.get("hidden_size", dec_latent_dim)))
        writer.add_uint32("qwen3.encoder.num_hidden_layers", int(enc.get("num_hidden_layers", enc.get("num_layers", 8))))
        writer.add_uint32("qwen3.encoder.num_attention_heads", int(enc.get("num_attention_heads", 8)))
        writer.add_uint32("qwen3.encoder.head_dim", int(enc.get("head_dim", 64)))
        writer.add_uint32("qwen3.encoder.intermediate_size", int(enc.get("intermediate_size", 2048)))
        writer.add_float32("qwen3.encoder.rope_theta", float(enc.get("rope_theta", 10000.0)))
        writer.add_float32("qwen3.encoder.rope_scaling_factor", float(enc.get("rope_scaling_factor", 1.0)))

        # Decoder metadata.
        writer.add_uint32("qwen3.decoder.hidden_size", int(dec.get("hidden_size", 1024)))
        writer.add_uint32("qwen3.decoder.num_hidden_layers", int(dec.get("num_hidden_layers", 8)))
        writer.add_uint32("qwen3.decoder.num_attention_heads", int(dec.get("num_attention_heads", 16)))
        writer.add_uint32("qwen3.decoder.num_key_value_heads", int(dec.get("num_key_value_heads", dec.get("num_attention_heads", 16))))
        writer.add_uint32("qwen3.decoder.head_dim", int(dec.get("head_dim", dec.get("hidden_size", 1024) // max(1, int(dec.get("num_attention_heads", 16))))))
        writer.add_uint32("qwen3.decoder.intermediate_size", int(dec.get("intermediate_size", 3072)))
        writer.add_float32("qwen3.decoder.rope_theta", float(dec.get("rope_theta", 10000.0)))
        writer.add_uint32("qwen3.decoder.sliding_window", int(dec.get("sliding_window", 0)))
        writer.add_uint32("qwen3.decoder.latent_dim", int(dec.get("latent_dim", dec_latent_dim)))
        writer.add_uint32("qwen3.decoder.decoder_dim", int(dec.get("decoder_dim", 1536)))
        writer.add_uint32("qwen3.decoder.codebook_dim", dec_codebook_dim)
        writer.add_uint32("qwen3.decoder.n_q", dec_n_q)

        for name, values in (
            ("qwen3.decoder.upsample_rates", _as_int_list(dec.get("upsample_rates"))),
            ("qwen3.decoder.upsampling_ratios", _as_int_list(dec.get("upsampling_ratios"))),
        ):
            if values:
                writer.add_array(name, values)

        used: set[str] = set()
        self._write_encoder_weights(writer, used)
        self._write_decoder_weights(writer, used)

        writer.write_all()
        self.log(f"Wrote Qwen3-TTS-Tokenizer GGUF to {output_path}")
