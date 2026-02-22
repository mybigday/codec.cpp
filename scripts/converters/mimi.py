"""Mimi model converter."""

import json
import os
import re
import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
from safetensors import safe_open

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

from .base import BaseConverter
from utils.gguf_writer import GGUFWriter
from utils import quantization as quant_utils

MAX_TENSOR_NAME = 63

RE_QUANTIZE_ENCODER_CONV = re.compile(r"^enc\.l\d+(?:\.block\.[13])?\.conv\.w$")
RE_QUANTIZE_ENCODER_TRANS = re.compile(r"^etr\.l\d+\..*\.w$")
RE_NEVER_Q_BIAS = re.compile(r"\.b$")
RE_NEVER_Q_LAYER_NORM = re.compile(r"(?:_ln|inln|paln)\.w$")
RE_NEVER_Q_CODEBOOK = re.compile(r"^q\..*\.embed$")
RE_NEVER_Q_CODEBOOK_CB = re.compile(r"^q\..*\.cb\.embed$")


def _is_mimi_encoder_conv1d_weight_key(key: str) -> bool:
    if key == "downsample.conv.weight":
        return True
    return bool(re.match(r"^encoder\.layers\.\d+(?:\.block\.[13])?\.conv\.weight$", key))


def _is_mimi_encoder_transformer_linear_weight_key(key: str) -> bool:
    return bool(
        re.match(
            r"^encoder_transformer\.layers\.\d+\."
            r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:fc1|fc2))\.weight$",
            key,
        )
    )


def _is_mimi_convtranspose1d_weight_key(key: str) -> bool:
    return key == "upsample.conv.weight"


def build_weight_transforms(keys: List[str]) -> Dict[str, str]:
    weight_transforms: Dict[str, str] = {}
    for key in keys:
        if _is_mimi_convtranspose1d_weight_key(key):
            weight_transforms[key] = "transpose_2_1_0"
        elif _is_mimi_encoder_conv1d_weight_key(key):
            weight_transforms[key] = "transpose_2_1_0"
        elif _is_mimi_encoder_transformer_linear_weight_key(key):
            weight_transforms[key] = "transpose_1_0"
    return weight_transforms


def _normalize_rvq_codebook_embed_layout(key: str, arr: np.ndarray) -> np.ndarray:
    if not key.endswith(".codebook.embed") and not key.endswith(".cb.embed"):
        return arr
    if arr.ndim != 2:
        raise ValueError(f"RVQ codebook embed must be rank-2: {key} shape={arr.shape}")
    # Keep HF logical layout [codebook_size, codebook_dim].
    # GGUF writer handles ggml dimension order when serializing.
    return arr


def transform_tensor_for_codec(key: str, arr: np.ndarray, weight_transforms: Dict[str, str]) -> np.ndarray:
    transform_info = weight_transforms.get(key)
    if transform_info is None:
        return _normalize_rvq_codebook_embed_layout(key, arr)

    transform_op = transform_info
    if transform_op == "keep":
        return _normalize_rvq_codebook_embed_layout(key, arr)
    if transform_op == "transpose_1_0":
        if arr.ndim != 2:
            raise ValueError(f"Linear weight must be rank-2: {key} shape={arr.shape}")
        arr = np.transpose(arr, (1, 0)).copy()
        return _normalize_rvq_codebook_embed_layout(key, arr)
    if transform_op == "transpose_2_1_0":
        if arr.ndim != 3:
            raise ValueError(f"Conv/ConvTranspose1d weight must be rank-3: {key} shape={arr.shape}")
        arr = np.transpose(arr, (2, 1, 0)).copy()
        return _normalize_rvq_codebook_embed_layout(key, arr)
    raise ValueError(f"unknown transform op: {transform_op} for {key}")


def shorten_tensor_name(name: str, used: set[str]) -> str:
    if len(name) <= MAX_TENSOR_NAME and name not in used:
        used.add(name)
        return name

    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    prefix_budget = MAX_TENSOR_NAME - 1 - len(digest)
    short = f"{name[:prefix_budget]}.{digest}"
    i = 1
    while short in used:
        suffix = f".{digest}{i}"
        prefix_budget = MAX_TENSOR_NAME - len(suffix)
        short = f"{name[:prefix_budget]}{suffix}"
        i += 1
    used.add(short)
    return short


def map_tensor_name_primary(key: str) -> str:
    out = key
    replacements = (
        ("decoder_transformer.layers.", "dtr.l"),
        ("encoder_transformer.layers.", "etr.l"),
        ("decoder.layers.", "dec.l"),
        ("encoder.layers.", "enc.l"),
        ("quantizer.acoustic_residual_vector_quantizer.", "q.a."),
        ("quantizer.semantic_residual_vector_quantizer.", "q.s."),
        ("codebook.embed_sum", "cb.es"),
        ("codebook.cluster_usage", "cb.cu"),
        ("codebook.initialized", "cb.init"),
        ("input_layernorm.", "inln."),
        ("post_attention_layernorm.", "paln."),
        ("self_attn_layer_scale.", "sa_ls."),
        ("mlp_layer_scale.", "mlp_ls."),
        ("self_attn.", "attn."),
        ("input_proj.weight", "ip.w"),
        ("output_proj.weight", "op.w"),
        ("downsample.conv.weight", "dn.cv.w"),
        ("upsample.conv.weight", "up.cv.w"),
        (".weight", ".w"),
        (".bias", ".b"),
    )
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def map_tensor_name_aliases(key: str) -> List[str]:
    aliases: List[str] = []

    if key.startswith("decoder_transformer.layers."):
        alias = key.replace("decoder_transformer.layers.", "dec.transformer.blocks.", 1)
        alias = alias.replace(".weight", ".w").replace(".bias", ".b")
        aliases.append(alias)

    m = re.match(r"^decoder\.layers\.(3|6|9|12)\.block\.(1|3)\.conv\.(weight|bias)$", key)
    if m:
        layer_to_res = {"3": 0, "6": 1, "9": 2, "12": 3}
        res_idx = layer_to_res[m.group(1)]
        conv_idx = "1" if m.group(2) == "1" else "2"
        suffix = "w" if m.group(3) == "weight" else "b"
        aliases.append(f"up.r{res_idx}.c{conv_idx}.{suffix}")

    if key.startswith("decoder.in_proj."):
        aliases.append(
            key.replace("decoder.in_proj.", "dec.in_proj.", 1).replace(".weight", ".w").replace(".bias", ".b")
        )
    if key.startswith("decoder.out_proj."):
        aliases.append(
            key.replace("decoder.out_proj.", "dec.out_proj.", 1).replace(".weight", ".w").replace(".bias", ".b")
        )
    if key.startswith("decoder.conv."):
        aliases.append(key.replace(".weight", ".w").replace(".bias", ".b"))

    out: List[str] = []
    seen: set[str] = set()
    for name in aliases:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def map_tensor_names(key: str) -> List[str]:
    primary = map_tensor_name_primary(key)
    aliases = map_tensor_name_aliases(key)
    out = [primary]
    for alias in aliases:
        if alias != primary:
            out.append(alias)
    return out


class MimiConverter(BaseConverter):
    @property
    def model_type(self) -> str:
        return "mimi"

    @property
    def architecture(self) -> str:
        return "mimi"

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        st_path = checkpoint_dir / "model.safetensors"
        cfg_path = checkpoint_dir / "config.json"

        if not st_path.is_file():
            raise FileNotFoundError(f"missing safetensors: {st_path}")
        if not cfg_path.is_file():
            raise FileNotFoundError(f"missing config: {cfg_path}")

        self.config = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.state_dict = {}
        with safe_open(str(st_path), framework="np", device="cpu") as handle:
            for key in handle.keys():
                self.state_dict[key] = handle.get_tensor(key)

        self.log(f"Loaded Mimi checkpoint from {checkpoint_dir} ({len(self.state_dict)} tensors)")

    def load_from_huggingface(self, model_id: str) -> None:
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub is required for load_from_huggingface")

        path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors", "config.json"])
        self.load_from_checkpoint(Path(path))

    def should_quantize_tensor(self, name: str, arr: np.ndarray) -> bool:
        if self.quantization not in ("Q4_K_M", "Q5_K_M", "Q8_0"):
            return False

        if RE_NEVER_Q_BIAS.search(name) or RE_NEVER_Q_LAYER_NORM.search(name):
            return False
        if (RE_NEVER_Q_CODEBOOK.search(name) or RE_NEVER_Q_CODEBOOK_CB.search(name)) and not self.quantize_codebook:
            return False

        is_candidate = (
            bool(RE_QUANTIZE_ENCODER_CONV.match(name))
            or bool(RE_QUANTIZE_ENCODER_TRANS.match(name))
            or name.endswith(".w")
        )
        if not is_candidate or arr.ndim < 2:
            return False

        ne0 = int(arr.shape[0])
        if self.quantization in ("Q4_K_M", "Q5_K_M"):
            return (ne0 % quant_utils.QK_K) == 0
        if self.quantization == "Q8_0":
            return (ne0 % quant_utils.QK8_0) == 0
        return False

    def _add_codebook_embed_tensors(self, writer: GGUFWriter, used_names: set[str]) -> int:
        if self.state_dict is None:
            raise RuntimeError("state_dict not loaded")

        normalize = os.environ.get("MIMI_CB_NORMALIZE", "1") != "0"
        n_added = 0

        for prefix in (
            "quantizer.semantic_residual_vector_quantizer.layers.",
            "quantizer.acoustic_residual_vector_quantizer.layers.",
        ):
            for qi in range(64):
                es_key = f"{prefix}{qi}.codebook.embed_sum"
                cu_key = f"{prefix}{qi}.codebook.cluster_usage"
                if es_key not in self.state_dict or cu_key not in self.state_dict:
                    continue

                embed_sum = np.asarray(self.state_dict[es_key], dtype=np.float32)
                usage = np.asarray(self.state_dict[cu_key], dtype=np.float32)
                if normalize:
                    denom = np.maximum(usage[:, None], 1e-6)
                    embed = embed_sum / denom
                else:
                    embed = embed_sum

                embed_key = es_key.replace(".embed_sum", ".embed")
                embed = _normalize_rvq_codebook_embed_layout(embed_key, embed)
                mapped = map_tensor_name_primary(embed_key)
                short = shorten_tensor_name(mapped, used_names)
                writer.add_tensor(short, embed)
                n_added += 1

        return n_added

    def _add_kernel_tensors(self, writer: GGUFWriter, hop_size: int) -> None:
        if hop_size <= 0:
            raise ValueError(f"invalid hop_size: {hop_size}")
        kernel = np.full((hop_size, 1, 1), 1.0 / float(hop_size), dtype=np.float16)
        writer.add_tensor("mimi.decode.kernel", kernel)
        writer.add_tensor("mimi.encode.kernel", kernel)

    def convert_and_save(self, output_path: Path) -> None:
        if self.state_dict is None or self.config is None:
            raise RuntimeError("No model loaded. Call load_from_checkpoint/load_from_huggingface first.")

        keys = sorted(self.state_dict.keys())
        weight_transforms = build_weight_transforms(keys)

        writer = GGUFWriter(output_path, self.architecture)
        writer.add_name("Mimi")
        sr = int(self.config.get("sampling_rate", 24000))
        hop_size = int(round(sr / float(self.config.get("frame_rate", 12.5))))

        writer.add_uint32("codec.sample_rate", sr)
        writer.add_uint32("codec.hop_size", hop_size)
        writer.add_uint32("codec.n_q", int(self.config.get("num_semantic_quantizers", 1)) + 31)
        writer.add_uint32("codec.num_semantic_quantizers", int(self.config.get("num_semantic_quantizers", 1)))
        writer.add_uint32("codec.codebook_size", int(self.config.get("codebook_size", 2048)))
        writer.add_uint32("codec.codebook_dim", int(self.config.get("codebook_dim", 256)))
        writer.add_uint32("codec.latent_dim", int(self.config.get("hidden_size", 512)))
        writer.add_uint32("codec.num_hidden_layers", int(self.config.get("num_hidden_layers", 8)))
        writer.add_uint32("codec.num_attention_heads", int(self.config.get("num_attention_heads", 8)))
        writer.add_uint32("codec.head_dim", int(self.config.get("head_dim", 64)))
        writer.add_uint32("codec.intermediate_size", int(self.config.get("intermediate_size", 2048)))
        writer.add_bool("codec.has_encoder", True)
        writer.add_bool("codec.has_decoder", True)

        used_names: set[str] = set()
        for key in keys:
            arr = transform_tensor_for_codec(key, self.state_dict[key], weight_transforms)
            for mapped in map_tensor_names(key):
                short = shorten_tensor_name(mapped, used_names)
                writer.add_tensor(short, arr, self._get_target_dtype(short, arr))

        self._add_codebook_embed_tensors(writer, used_names)
        self._add_kernel_tensors(writer, hop_size)
        writer.write()
        self.log(f"Wrote Mimi GGUF to {output_path}")
