"""NVIDIA NeMo Nano Codec converter."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, Tuple
import math
import tarfile
import tempfile

import numpy as np

try:
    import torch
except ImportError:
    torch = None

import yaml

from .base import BaseConverter
from utils.gguf_writer import GGUFWriter


def _to_numpy(tensor):
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def _apply_weight_norm(weight_v: np.ndarray, weight_g: np.ndarray) -> np.ndarray:
    if weight_v.ndim < 2:
        raise ValueError(f"weight_norm expects ndim >= 2, got {weight_v.ndim}")
    out_channels = weight_v.shape[0]
    v_flat = weight_v.reshape(out_channels, -1)
    norm = np.linalg.norm(v_flat, axis=1, keepdims=True)
    scale = weight_g.reshape(out_channels) / (norm.reshape(out_channels) + 1e-12)
    reshape = (out_channels,) + (1,) * (weight_v.ndim - 1)
    return weight_v * scale.reshape(reshape)


def _reorder_conv_weight_for_ggml(weight: np.ndarray) -> np.ndarray:
    """No-op for NeMo conv weights.

    PyTorch conv weights are [C0, C1, K] with K as the last (fastest) dimension.
    GGUF reverses shapes on write, so ggml sees [K, C1, C0] while preserving the
    data buffer order. This already matches ggml's expectation (ne0/K fastest).
    """
    return weight


def _load_nemo_archive(nemo_path: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    if torch is None:
        raise RuntimeError("torch is required for NeMo checkpoint conversion")

    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(nemo_path, "r") as tar:
            tar.extractall(tmpdir)

        cfg_path = Path(tmpdir) / "model_config.yaml"
        ckpt_path = Path(tmpdir) / "model_weights.ckpt"

        if not cfg_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError("nemo archive missing model_config.yaml or model_weights.ckpt")

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError("unsupported checkpoint format inside .nemo")

        state_np = OrderedDict((k, _to_numpy(v)) for k, v in state.items())
        return cfg, state_np


def _map_key(key: str) -> str | None:
    if key == "audio_encoder.pre_conv.conv.weight":
        return "nemo.enc.pre.w"
    if key == "audio_encoder.pre_conv.conv.bias":
        return "nemo.enc.pre.b"
    if key == "audio_encoder.post_conv.conv.weight":
        return "nemo.enc.post.w"
    if key == "audio_encoder.post_conv.conv.bias":
        return "nemo.enc.post.b"

    if key.startswith("audio_encoder.down_sample_conv_layers."):
        rest = key[len("audio_encoder.down_sample_conv_layers.") :]
        parts = rest.split(".")
        if len(parts) < 3:
            return None
        layer = parts[0]
        param = parts[-1]
        if param == "weight":
            return f"nemo.enc.down.{layer}.w"
        if param == "bias":
            return f"nemo.enc.down.{layer}.b"

    if key.startswith("audio_encoder.res_layers."):
        rest = key[len("audio_encoder.res_layers.") :]
        parts = rest.split(".")
        if len(parts) < 8:
            return None
        layer = parts[0]
        block = parts[2]
        res = parts[4]
        which = parts[5]  # input_conv or skip_conv
        param = parts[-1]
        if which == "input_conv":
            base = f"nemo.enc.res.l{layer}.b{block}.r{res}.in"
        elif which == "skip_conv":
            base = f"nemo.enc.res.l{layer}.b{block}.r{res}.sk"
        else:
            return None
        if param == "weight":
            return base + ".w"
        if param == "bias":
            return base + ".b"

    if key == "audio_decoder.pre_conv.conv.weight":
        return "nemo.dec.pre.w"
    if key == "audio_decoder.pre_conv.conv.bias":
        return "nemo.dec.pre.b"
    if key == "audio_decoder.post_conv.conv.weight":
        return "nemo.dec.post.w"
    if key == "audio_decoder.post_conv.conv.bias":
        return "nemo.dec.post.b"

    if key.startswith("audio_decoder.activations.") and key.endswith("activation.snake_act.alpha"):
        idx = key.split(".")[2]
        return f"nemo.dec.act.{idx}.a"

    if key == "audio_decoder.post_activation.activation.snake_act.alpha":
        return "nemo.dec.post.a"

    if key.startswith("audio_decoder.up_sample_conv_layers."):
        rest = key[len("audio_decoder.up_sample_conv_layers.") :]
        parts = rest.split(".")
        if len(parts) < 3:
            return None
        layer = parts[0]
        param = parts[-1]
        if param == "weight":
            return f"nemo.dec.up.{layer}.w"
        if param == "bias":
            return f"nemo.dec.up.{layer}.b"

    if key.startswith("audio_decoder.res_layers."):
        rest = key[len("audio_decoder.res_layers.") :]
        parts = rest.split(".")
        if len(parts) < 8:
            return None
        layer = parts[0]
        block = parts[2]
        res = parts[4]
        if parts[5] in ("input_conv", "skip_conv"):
            which = parts[5]
            param = parts[-1]
            base = "nemo.dec.res.l" + layer + ".b" + block + ".r" + res
            base += ".in" if which == "input_conv" else ".sk"
            if param == "weight":
                return base + ".w"
            if param == "bias":
                return base + ".b"
        if parts[5] in ("input_activation", "skip_activation") and parts[-1] == "alpha":
            which = parts[5]
            base = "nemo.dec.res.l" + layer + ".b" + block + ".r" + res
            base += ".in" if which == "input_activation" else ".sk"
            return base + ".a"

    return None


class NemoNanoCodecConverter(BaseConverter):
    @property
    def model_type(self) -> str:
        return "nemo_nano_codec"

    @property
    def architecture(self) -> str:
        return "nemo_nano_codec"

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        if torch is None:
            raise RuntimeError("torch is required for NeMo conversion")

        checkpoint_dir = Path(checkpoint_dir)

        if checkpoint_dir.is_file() and checkpoint_dir.suffix == ".nemo":
            cfg, state = _load_nemo_archive(checkpoint_dir)
            self.config = cfg
            self.state_dict = state
            return

        # handle directories and .ckpt
        if checkpoint_dir.is_file() and checkpoint_dir.suffix in (".ckpt", ".pth"):
            cfg_path = checkpoint_dir.parent / "model_config.yaml"
            if not cfg_path.exists():
                raise FileNotFoundError(f"missing model_config.yaml near {checkpoint_dir}")
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            state = torch.load(checkpoint_dir, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise RuntimeError("unsupported checkpoint format")
            self.config = cfg
            self.state_dict = OrderedDict((k, _to_numpy(v)) for k, v in state.items())
            return

        if checkpoint_dir.is_dir():
            nemo_files = list(checkpoint_dir.glob("*.nemo"))
            if nemo_files:
                cfg, state = _load_nemo_archive(nemo_files[0])
                self.config = cfg
                self.state_dict = state
                return

            cfg_path = checkpoint_dir / "model_config.yaml"
            ckpt_path = checkpoint_dir / "model_weights.ckpt"
            if not cfg_path.exists() or not ckpt_path.exists():
                raise FileNotFoundError(f"missing model_config.yaml or model_weights.ckpt in {checkpoint_dir}")
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise RuntimeError("unsupported checkpoint format")
            self.config = cfg
            self.state_dict = OrderedDict((k, _to_numpy(v)) for k, v in state.items())
            return

        raise FileNotFoundError(f"input path not found: {checkpoint_dir}")

    def load_from_huggingface(self, model_id: str) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise RuntimeError("huggingface_hub is required for NeMo conversion") from exc

        nemo_file = hf_hub_download(repo_id=model_id, filename="nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo")
        self.load_from_checkpoint(Path(nemo_file))

    def convert_and_save(self, output_path: Path) -> None:
        if self.state_dict is None or self.config is None:
            raise RuntimeError("model must be loaded before conversion")

        cfg = self.config
        writer = GGUFWriter(output_path, self.architecture)
        writer.add_uint32("codec.sample_rate", int(cfg.get("sample_rate", 22050)))
        writer.add_uint32("codec.hop_size", int(cfg.get("samples_per_frame", 1764)))

        vq_cfg = cfg.get("vector_quantizer", {})
        num_groups = int(vq_cfg.get("num_groups", 4))
        num_levels = list(map(int, vq_cfg.get("num_levels_per_group", [9, 8, 8, 7])))
        codebook_dim = len(num_levels)
        codebook_size = int(np.prod(num_levels))
        writer.add_uint32("codec.n_q", num_groups)
        writer.add_uint32("codec.codebook_size", codebook_size)
        writer.add_uint32("codec.codebook_dim", codebook_dim)
        writer.add_uint32("codec.latent_dim", num_groups * codebook_dim)
        writer.add_bool("codec.has_encoder", True)
        writer.add_bool("codec.has_decoder", True)

        state = self.state_dict
        mapped: Dict[str, np.ndarray] = {}

        # Apply weight norm and map keys
        for key, value in state.items():
            if key.endswith(".weight_g"):
                v_key = key[: -len(".weight_g")] + ".weight_v"
                if v_key not in state:
                    continue
                weight_v = _to_numpy(state[v_key]).astype(np.float32, copy=False)
                weight_g = _to_numpy(value).astype(np.float32, copy=False)
                weight = _apply_weight_norm(weight_v, weight_g)
                out_name = _map_key(key[: -len(".weight_g")] + ".weight")
                if out_name is None:
                    continue
                mapped[out_name] = weight
                continue
            if key.endswith(".weight_v"):
                # handled by weight_g
                continue
            if key.endswith(".bias"):
                out_name = _map_key(key)
                if out_name is None:
                    continue
                mapped[out_name] = _to_numpy(value)
                continue
            if key.endswith(".alpha"):
                out_name = _map_key(key)
                if out_name is None:
                    continue
                mapped[out_name] = _to_numpy(value)

        # Expand grouped conv transpose weights to dense
        for layer in range(5):
            w_name = f"nemo.dec.up.{layer}.w"
            if w_name not in mapped:
                continue
            w = mapped[w_name]
            if w.ndim != 3:
                raise ValueError(f"unexpected upsample weight rank: {w_name} shape={w.shape}")
            in_ch, out_per_group, k = w.shape
            if out_per_group != 1:
                continue
            out_ch = in_ch // 2
            dense = np.zeros((in_ch, out_ch, k), dtype=w.dtype)
            for in_idx in range(in_ch):
                out_idx = in_idx // 2
                dense[in_idx, out_idx, :] = w[in_idx, 0, :]
            mapped[w_name] = dense

        # FSQ constants and codebooks
        num_levels_arr = np.asarray(num_levels, dtype=np.float32)
        scale = (num_levels_arr // 2).astype(np.float32)
        out_scale = (num_levels_arr - 1.0) / 2.0
        out_scale *= (1.0 - 1e-3)
        out_offset = np.where((num_levels_arr.astype(np.int32) % 2) == 0, 0.5, 0.0).astype(np.float32)
        in_shift = np.tan(out_offset / out_scale).astype(np.float32)
        dim_base_index = np.cumprod(np.concatenate([[1], num_levels_arr[:-1]])).astype(np.float32)

        mapped["nemo.fsq.scale"] = scale
        mapped["nemo.fsq.out_scale"] = out_scale
        mapped["nemo.fsq.out_offset"] = out_offset
        mapped["nemo.fsq.in_shift"] = in_shift
        mapped["nemo.fsq.dim_base"] = dim_base_index

        # Build codebook for each group
        codebook_dim = len(num_levels)
        codebook = np.zeros((codebook_size, codebook_dim), dtype=np.float32)
        bases = dim_base_index.astype(np.int64)
        levels = np.asarray(num_levels, dtype=np.int64)
        for idx in range(codebook_size):
            codes_nonneg = [(idx // bases[d]) % levels[d] for d in range(codebook_dim)]
            codes = (np.asarray(codes_nonneg, dtype=np.float32) - scale) / scale
            codebook[idx, :] = codes
        for g in range(num_groups):
            mapped[f"nemo.fsq.codebook.{g}"] = codebook

        self._reset_quant_stats()
        used = set()
        for name, arr in mapped.items():
            # Normalize snake alpha tensors to 1D
            if name.endswith(".a") and arr.ndim > 1:
                arr = np.asarray(arr).reshape(-1)
            if name.endswith(".w") and not name.startswith("nemo.fsq."):
                arr = _reorder_conv_weight_for_ggml(np.asarray(arr))
            # Convert codebook to ggml layout (codebook_size, codebook_dim)
            if name.startswith("nemo.fsq.codebook."):
                arr = np.asarray(arr)
                if arr.shape[0] != codebook_size or arr.shape[1] != codebook_dim:
                    raise ValueError(f"codebook shape mismatch: {arr.shape}")
                st = "F16" if not self.quantize_codebook else None
                self._add_tensor(writer, name, arr, st_dtype=st)
                continue
            self._add_tensor(writer, name, np.asarray(arr))

        self._warn_if_no_quantized()
        writer.write()
        self.log(f"Wrote NeMo Nano Codec GGUF to {output_path}")
