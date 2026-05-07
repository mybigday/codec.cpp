"""OpenMOSS-Team/MOSS-Audio-Tokenizer (Nano + full) converter.

Pure-Transformer codec ("CAT" / "Causal Audio Tokenizer with Transformer").
- 48 kHz multi-channel input.  Channels are interleaved into a single mono-equiv
  stream (`(B, 2, T) → (B, 1, T*number_channels)`) before the encoder.
- Encoder: alternating PatchedPretransform (parameter-free reshape) and
  ProjectedTransformer (Linear in_proj → causal MHA + RoPE + LayerScale + GELU
  FFN, Linear out_proj).
- Quantizer: top-level WNConv1d in/out_proj into rvq_dim, then `num_quantizers`
  LFQ levels (each with its own WNConv1d in/out_proj + codebook).  Codes are
  picked via L2-normalised cosine NN; the converter pre-bakes the L2-normalised
  codebook so the runtime can skip the per-codebook normalisation.
- Decoder: mirror of encoder.

Both the Nano (22 M) and the full 1.6 B variant share the exact same module
hierarchy — only the per-block layer counts and dims differ — so this single
converter handles either by reading `config.json`.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .base import BaseConverter
from utils.gguf_writer import GGUFWriter


def _to_numpy(t):
    if torch is not None and isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def _apply_weight_norm(weight_v: np.ndarray, weight_g: np.ndarray, dim: int = 0) -> np.ndarray:
    if weight_v.ndim < 2:
        raise ValueError(f"weight_norm expects ndim >= 2, got {weight_v.ndim}")
    axes = tuple(i for i in range(weight_v.ndim) if i != dim)
    norm = np.linalg.norm(weight_v, axis=axes, keepdims=True)
    if weight_g.shape != norm.shape:
        weight_g = weight_g.reshape(norm.shape)
    return weight_v * (weight_g / (norm + 1e-12))


def _load_state_dict(path: Path) -> Dict[str, np.ndarray]:
    """Load a (possibly sharded) safetensors checkpoint or a `pytorch_model.bin`.

    The MOSS Nano ships a single `model-00001-of-00001.safetensors`; the full
    variant uses `model-{i}-of-2.safetensors` plus an index json.  Both are
    handled transparently here.
    """
    path = Path(path)
    if path.is_dir():
        st_files = sorted(path.glob("model-*.safetensors"))
        if st_files:
            from safetensors.numpy import load_file as st_load
            sd: Dict[str, np.ndarray] = {}
            for f in st_files:
                sd.update(st_load(str(f)))
            return sd
        bin_path = path / "pytorch_model.bin"
        if bin_path.is_file() and torch is not None:
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
            return OrderedDict((k, _to_numpy(v)) for k, v in state.items())
        raise FileNotFoundError(f"no MOSS-Audio-Tokenizer checkpoint files in {path}")
    if path.suffix == ".safetensors":
        from safetensors.numpy import load_file as st_load
        return dict(st_load(str(path)))
    if torch is None:
        raise RuntimeError("torch required to load .bin checkpoints")
    state = torch.load(path, map_location="cpu", weights_only=True)
    return OrderedDict((k, _to_numpy(v)) for k, v in state.items())


class MossAudioConverter(BaseConverter):
    @property
    def model_type(self) -> str:
        return "moss_audio"

    @property
    def architecture(self) -> str:
        return "moss_audio_tokenizer"

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        path = Path(checkpoint_dir)
        cfg_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"missing config.json next to {path}")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        # Validate the architecture is one we support.  The full MOSS-Audio
        # Tokenizer adds a single global stack of Transformer + RVQ, but the
        # per-module schema is identical, so the same loader handles both.
        for mod in cfg["encoder_kwargs"] + cfg["decoder_kwargs"]:
            if mod["module_type"] not in {"PatchedPretransform", "Transformer"}:
                raise NotImplementedError(f"unsupported MOSS-Audio module_type: {mod['module_type']}")
            if mod["module_type"] == "Transformer":
                if mod.get("gating", "none") != "none":
                    raise NotImplementedError("MOSS-Audio gating != 'none' not implemented")
                if mod.get("positional_embedding") not in {"rope"}:
                    raise NotImplementedError(
                        f"MOSS-Audio positional_embedding={mod.get('positional_embedding')} not supported"
                    )
                if mod.get("norm", "layer_norm") != "layer_norm":
                    raise NotImplementedError("MOSS-Audio norm != 'layer_norm' not supported")
                if not mod.get("causal", False):
                    raise NotImplementedError("MOSS-Audio non-causal Transformer not supported")
        self.state_dict = _load_state_dict(path)
        self.config = cfg

    def load_from_huggingface(self, model_id: str) -> None:
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.safetensors", "*.json", "*.py"],
        )
        self.load_from_checkpoint(Path(local))

    def convert_and_save(self, output_path: Path) -> None:
        if self.state_dict is None or self.config is None:
            raise RuntimeError("No model loaded.")
        sd = self.state_dict
        cfg = self.config

        writer = GGUFWriter(output_path, self.architecture)
        self._reset_quant_stats()

        n_channels = int(cfg.get("number_channels", 1))
        interleave = bool(cfg.get("enable_channel_interleave", True))
        sr = int(cfg["sampling_rate"])
        downsample = int(cfg["downsample_rate"])

        # codes_rate_per_channel = `effective_input_rate / downsample_rate`,
        # where effective_input_rate = sr * number_channels when interleaving.
        # For Nano: 48 000 * 2 / 3840 = 25 codes/sec (stereo).
        n_q = int(cfg["quantizer_kwargs"]["num_quantizers"])
        cb_size = int(cfg["quantizer_kwargs"]["codebook_size"])
        cb_dim = int(cfg["quantizer_kwargs"]["codebook_dim"])
        rvq_dim = int(cfg["quantizer_kwargs"].get("rvq_dim", cfg["quantizer_kwargs"]["input_dim"]))
        latent_dim = int(cfg["code_dim"])
        quantizer_type = cfg.get("quantizer_type", cfg["quantizer_kwargs"].get("quantizer_type", "rvq"))

        writer.add_name(cfg.get("name", "MOSS-Audio-Tokenizer"))
        writer.add_uint32("codec.sample_rate", sr)
        writer.add_uint32("codec.encode_sample_rate", sr)
        writer.add_uint32("codec.hop_size", downsample)
        writer.add_uint32("codec.n_q", n_q)
        writer.add_uint32("codec.codebook_size", cb_size)
        writer.add_uint32("codec.codebook_dim", cb_dim)
        writer.add_uint32("codec.latent_dim", latent_dim)
        writer.add_bool("codec.has_encoder", True)
        writer.add_bool("codec.has_decoder", True)

        writer.add_uint32("moss.number_channels", n_channels)
        writer.add_bool("moss.channel_interleave", interleave)
        writer.add_uint32("moss.rvq_dim", rvq_dim)
        writer.add_string("moss.quantizer_type", str(quantizer_type))
        writer.add_float32("moss.context_duration", float(cfg.get("causal_transformer_context_duration", 10.0)))

        def _save_module_kwargs(side: str, module_list: List[Dict[str, Any]]) -> None:
            # Walk the encoder/decoder list and bake every parameter the runtime
            # needs to rebuild the same module sequence: patch sizes, projection
            # shapes, layer counts, head dims, RoPE base, and per-block context
            # duration (so we can derive the windowed-attention context in
            # tokens at runtime).
            module_types: List[int] = []
            patch_sizes: List[int] = []
            in_dims: List[int] = []
            out_dims: List[int] = []
            d_models: List[int] = []
            n_heads: List[int] = []
            n_layers: List[int] = []
            ffn_dims: List[int] = []
            context_durations: List[float] = []
            max_periods: List[float] = []
            layer_scales: List[float] = []
            for mod in module_list:
                if mod["module_type"] == "PatchedPretransform":
                    module_types.append(0)
                    patch_sizes.append(int(mod["patch_size"]))
                    in_dims.append(0)
                    out_dims.append(0)
                    d_models.append(0)
                    n_heads.append(0)
                    n_layers.append(0)
                    ffn_dims.append(0)
                    context_durations.append(0.0)
                    max_periods.append(0.0)
                    layer_scales.append(0.0)
                else:
                    module_types.append(1)
                    patch_sizes.append(0)
                    in_dims.append(int(mod["input_dimension"]))
                    out_dims.append(int(mod["output_dimension"]))
                    d_models.append(int(mod["d_model"]))
                    n_heads.append(int(mod["num_heads"]))
                    n_layers.append(int(mod["num_layers"]))
                    ffn_dims.append(int(mod["dim_feedforward"]))
                    context_durations.append(float(mod.get("context_duration", 10.0)))
                    max_periods.append(float(mod.get("max_period", 10000.0)))
                    layer_scales.append(float(mod.get("layer_scale", 0.0) or 0.0))
            base = f"moss.{side}"
            writer.add_uint32(base + ".n_modules", len(module_list))
            writer.add_array(base + ".module_types", module_types)
            writer.add_array(base + ".patch_sizes", patch_sizes)
            writer.add_array(base + ".in_dims", in_dims)
            writer.add_array(base + ".out_dims", out_dims)
            writer.add_array(base + ".d_models", d_models)
            writer.add_array(base + ".n_heads", n_heads)
            writer.add_array(base + ".n_layers", n_layers)
            writer.add_array(base + ".ffn_dims", ffn_dims)
            writer.add_array(base + ".context_durations", context_durations)
            writer.add_array(base + ".max_periods", max_periods)
            writer.add_array(base + ".layer_scales", layer_scales)

        _save_module_kwargs("enc", cfg["encoder_kwargs"])
        _save_module_kwargs("dec", cfg["decoder_kwargs"])

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def _t(name: str) -> np.ndarray:
            arr = sd.get(name)
            if arr is None:
                raise KeyError(f"missing tensor: {name}")
            return np.asarray(arr)

        def add_linear(prefix: str, out_name: str, has_bias: bool = False) -> None:
            self._add_tensor(writer, out_name + ".w", _t(prefix + ".weight"))
            if has_bias and (prefix + ".bias") in sd:
                self._add_tensor(writer, out_name + ".b", _t(prefix + ".bias"), "F32")

        def add_layernorm(prefix: str, out_name: str) -> None:
            self._add_tensor(writer, out_name + ".w", _t(prefix + ".weight"), "F32")
            self._add_tensor(writer, out_name + ".b", _t(prefix + ".bias"), "F32")

        def add_wn_conv(prefix: str, out_name: str, dim: int = 0) -> None:
            wv = _t(prefix + ".parametrizations.weight.original1")
            wg = _t(prefix + ".parametrizations.weight.original0")
            w = _apply_weight_norm(wv, wg, dim=dim)
            self._add_tensor(writer, out_name + ".w", w)
            if (prefix + ".bias") in sd:
                self._add_tensor(writer, out_name + ".b", _t(prefix + ".bias"), "F32")

        def _first_present(*names: str) -> str:
            for n in names:
                if n in sd:
                    return n
            raise KeyError(f"none of {names} found in state_dict")

        def add_transformer_layer(prefix: str, out_name: str) -> None:
            add_layernorm(prefix + ".norm1", out_name + ".norm1")
            add_layernorm(prefix + ".norm2", out_name + ".norm2")
            # Nano stores `self_attn.in_proj.weight`; full uses
            # `self_attn.in_projs.0.weight` (a ModuleList wrapper supporting
            # weights-per-step, even when only index 0 is populated).
            qkv_name = _first_present(
                prefix + ".self_attn.in_proj.weight",
                prefix + ".self_attn.in_projs.0.weight",
            )
            out_name_in = _first_present(
                prefix + ".self_attn.out_proj.weight",
                prefix + ".self_attn.out_projs.0.weight",
            )
            self._add_tensor(writer, out_name + ".attn.qkv.w", _t(qkv_name))
            self._add_tensor(writer, out_name + ".attn.out.w", _t(out_name_in))
            # Nano: `ffn.0` / `ffn.2` (sequential gelu sandwich); full:
            # `linear1` / `linear2` (no Sequential wrapper).
            fc1_name = _first_present(
                prefix + ".ffn.0.weight",
                prefix + ".linear1.weight",
            )
            fc2_name = _first_present(
                prefix + ".ffn.2.weight",
                prefix + ".linear2.weight",
            )
            self._add_tensor(writer, out_name + ".ffn.fc1.w", _t(fc1_name))
            self._add_tensor(writer, out_name + ".ffn.fc2.w", _t(fc2_name))
            self._add_tensor(writer, out_name + ".ls1",
                             _t(prefix + ".layer_scale_1.scale"), "F32")
            self._add_tensor(writer, out_name + ".ls2",
                             _t(prefix + ".layer_scale_2.scale"), "F32")

        # ------------------------------------------------------------------
        # Encoder + decoder transformers
        # ------------------------------------------------------------------
        for side, modules in [("enc", cfg["encoder_kwargs"]), ("dec", cfg["decoder_kwargs"])]:
            for mi, mod in enumerate(modules):
                if mod["module_type"] != "Transformer":
                    continue
                base_in = ("encoder" if side == "enc" else "decoder") + f".{mi}"
                base_out = f"moss.{side}.b{mi}"
                # Full MOSS-Audio-Tokenizer uses nn.Identity for input/output
                # projections when `input_dim == d_model` / `d_model ==
                # output_dim`, so the corresponding weight is absent from the
                # state-dict.  Emit only when the source weight exists; the
                # runtime treats missing projection tensors as identity.
                if (base_in + ".input_proj.weight") in sd:
                    add_linear(base_in + ".input_proj", base_out + ".input_proj")
                if (base_in + ".output_proj.weight") in sd:
                    add_linear(base_in + ".output_proj", base_out + ".output_proj")
                for li in range(int(mod["num_layers"])):
                    add_transformer_layer(
                        base_in + f".transformer.layers.{li}",
                        base_out + f".l{li}",
                    )

        # ------------------------------------------------------------------
        # Residual quantizer
        # ------------------------------------------------------------------
        add_wn_conv("quantizer.input_proj",  "moss.q.input_proj")
        add_wn_conv("quantizer.output_proj", "moss.q.output_proj")
        for qi in range(n_q):
            base = f"quantizer.quantizers.{qi}"
            o = f"moss.q.{qi}"
            add_wn_conv(base + ".in_proj",  o + ".in_proj")
            add_wn_conv(base + ".out_proj", o + ".out_proj")
            cb = _t(base + ".codebook.weight").astype(np.float32)  # (codebook_size, codebook_dim)
            cb_norm = cb / (np.linalg.norm(cb, axis=1, keepdims=True) + 1e-12)
            self._add_tensor(writer, o + ".codebook", cb, "F16")
            self._add_tensor(writer, o + ".codebook_norm", cb_norm, "F16")

        self._warn_if_no_quantized()
        writer.write()
        self.log(f"Wrote MOSS-Audio-Tokenizer GGUF to {output_path}")
