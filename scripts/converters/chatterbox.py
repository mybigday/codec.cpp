"""Chatterbox S3T/S3G converters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

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


def _short_tensor_name(prefix: str, key: str, used: set[str]) -> str:
    name = f"{prefix}{key}"
    if len(name) <= 63 and name not in used:
        used.add(name)
        return name

    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    name = f"{prefix}{digest}"
    suffix = 0
    while name in used:
        suffix += 1
        name = f"{prefix}{digest[:12]}{suffix:04d}"
    used.add(name)
    return name


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

        writer = GGUFWriter(output_path, self.architecture)
        self._reset_quant_stats()
        self._write_common_metadata(writer, self.config)
        writer.add_bool("chatterbox_s3g.meanflow", bool(self.config.get("meanflow", False)))

        gen_conds = None
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

        # Decode runtime is not implemented yet. Preserve the real checkpoint tensors under
        # stable names so S3G conversion is no longer blocked on re-downloading or re-parsing.
        used: set[str] = set()
        for key in sorted(self.state_dict.keys()):
            if key.startswith("tokenizer."):
                continue
            arr = np.asarray(self.state_dict[key])
            if arr.ndim == 0:
                continue
            name = _short_tensor_name("s3g.", key, used)
            self._add_tensor(writer, name, arr)

        writer.write()
        self._warn_if_no_quantized()
