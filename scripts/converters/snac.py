"""SNAC (hubertsiuzdak/snac_24khz) converter.

Architecture: BigVGAN-style depthwise SNAC for 24 kHz audio.
- Encoder: WNConv1d(1→48) → 4 EncoderBlocks (strides [2, 4, 8, 8]) → depthwise WNConv1d.
- Quantizer: 3-level Residual VQ at strides [4, 2, 1] of the latent (L2-norm cosine NN).
- Decoder: depthwise + pointwise → 4 DecoderBlocks (rates [8, 8, 4, 2]) → Tanh head.

The 24 kHz checkpoint has `attn_window_size=null`, so no LocalMHA in encoder/decoder.
NoiseBlock is run as identity (deterministic decode); the parity test mirrors that.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, List

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
    """Reverse of `torch.nn.utils.parametrizations.weight_norm`. Only `dim=0` is
    used here since both `Conv1d` and `ConvTranspose1d` weight_norm modules in
    SNAC default to dim=0."""
    if weight_v.ndim < 2:
        raise ValueError(f"weight_norm expects ndim >= 2, got {weight_v.ndim}")
    axes = tuple(i for i in range(weight_v.ndim) if i != dim)
    norm = np.linalg.norm(weight_v, axis=axes, keepdims=True)
    if weight_g.shape != norm.shape:
        weight_g = weight_g.reshape(norm.shape)
    return weight_v * (weight_g / (norm + 1e-12))


class SnacConverter(BaseConverter):
    @property
    def model_type(self) -> str:
        return "snac"

    @property
    def architecture(self) -> str:
        return "snac"

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        if torch is None:
            raise RuntimeError("torch is required for SNAC checkpoint conversion")
        path = Path(checkpoint_dir)
        if path.is_dir():
            cfg_path = path / "config.json"
            ckpt_path = path / "pytorch_model.bin"
        else:
            ckpt_path = path
            cfg_path = path.parent / "config.json"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"missing SNAC checkpoint: {ckpt_path}")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.state_dict = OrderedDict((k, _to_numpy(v)) for k, v in state.items())
        self.config = {
            "sample_rate":     int(cfg["sampling_rate"]),
            "encoder_dim":     int(cfg["encoder_dim"]),
            "encoder_rates":   list(cfg["encoder_rates"]),
            "decoder_dim":     int(cfg["decoder_dim"]),
            "decoder_rates":   list(cfg["decoder_rates"]),
            "attn_window":     cfg.get("attn_window_size"),
            "codebook_size":   int(cfg["codebook_size"]),
            "codebook_dim":    int(cfg["codebook_dim"]),
            "vq_strides":      list(cfg["vq_strides"]),
            "noise":           bool(cfg.get("noise", True)),
            "depthwise":       bool(cfg.get("depthwise", True)),
            "latent_dim":      int(cfg["encoder_dim"]) * (2 ** len(cfg["encoder_rates"])),
        }

    def load_from_huggingface(self, model_id: str) -> None:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(repo_id=model_id, filename="config.json")
        ckpt_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        # Cache the directory for load_from_checkpoint compat
        local = Path(cfg_path).parent
        # snapshot_download keeps the cfg + bin alongside each other; reuse.
        self.load_from_checkpoint(local)
        # `local` may contain config.json but not the .bin if huggingface_hub
        # uses content-addressed paths.  Override directly to avoid surprise:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.state_dict = OrderedDict((k, _to_numpy(v)) for k, v in state.items())
        self.config["latent_dim"] = int(cfg["encoder_dim"]) * (2 ** len(cfg["encoder_rates"]))

    def convert_and_save(self, output_path: Path) -> None:
        if self.state_dict is None or self.config is None:
            raise RuntimeError("No model loaded. Call load_from_checkpoint/load_from_huggingface first.")
        if self.config["attn_window"] is not None:
            raise NotImplementedError("attn_window != null SNAC variants not supported yet (no LocalMHA path)")
        if not self.config["depthwise"]:
            raise NotImplementedError("only depthwise=True SNAC variants are wired up")

        sd = self.state_dict
        cfg = self.config
        writer = GGUFWriter(output_path, self.architecture)
        self._reset_quant_stats()

        # Hop length = product of encoder rates; the model also pads input to
        # `hop * lcm(vq_strides[0], 1)` (no attention here so lcm = vq_strides[0]).
        hop = int(np.prod(cfg["encoder_rates"]))
        pad_to = hop * int(cfg["vq_strides"][0])

        writer.add_name("SNAC-24khz")
        writer.add_uint32("codec.sample_rate", int(cfg["sample_rate"]))
        writer.add_uint32("codec.encode_sample_rate", int(cfg["sample_rate"]))
        writer.add_uint32("codec.hop_size", hop)
        writer.add_uint32("codec.pad_to", pad_to)
        writer.add_uint32("codec.n_q", len(cfg["vq_strides"]))
        writer.add_uint32("codec.codebook_size", int(cfg["codebook_size"]))
        writer.add_uint32("codec.codebook_dim", int(cfg["codebook_dim"]))
        writer.add_uint32("codec.latent_dim", int(cfg["latent_dim"]))
        writer.add_bool("codec.has_encoder", True)
        writer.add_bool("codec.has_decoder", True)

        writer.add_array("snac.encoder_rates", list(cfg["encoder_rates"]))
        writer.add_array("snac.decoder_rates", list(cfg["decoder_rates"]))
        writer.add_array("snac.vq_strides", list(cfg["vq_strides"]))
        writer.add_uint32("snac.encoder_dim", int(cfg["encoder_dim"]))
        writer.add_uint32("snac.decoder_dim", int(cfg["decoder_dim"]))
        writer.add_bool("snac.depthwise", bool(cfg["depthwise"]))
        writer.add_bool("snac.noise", bool(cfg["noise"]))

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def _t(name: str) -> np.ndarray:
            arr = sd.get(name)
            if arr is None:
                raise KeyError(f"missing tensor: {name}")
            return np.asarray(arr)

        def add_wn_conv(prefix: str, out_name: str) -> None:
            wv = _t(prefix + ".parametrizations.weight.original1")
            wg = _t(prefix + ".parametrizations.weight.original0")
            w = _apply_weight_norm(wv, wg, dim=0)
            self._add_tensor(writer, out_name + ".w", w)
            if prefix + ".bias" in sd:
                self._add_tensor(writer, out_name + ".b", _t(prefix + ".bias"), "F32")

        def add_alpha(prefix: str, out_name: str) -> None:
            # Snake1d alpha shape (1, c, 1) — flatten to (c,) for codec_op_snake.
            alpha = _t(prefix + ".alpha").reshape(-1).astype(np.float32)
            self._add_tensor(writer, out_name + ".alpha", alpha, "F32")

        def add_residual_unit(prefix: str, out_name: str) -> None:
            # ResidualUnit: Snake → conv (depthwise k=7 dilated) → Snake → conv (k=1)
            #   block.0: Snake1d  → alpha
            #   block.1: WNConv1d (depthwise k=7)
            #   block.2: Snake1d  → alpha
            #   block.3: WNConv1d (k=1)
            add_alpha(prefix + ".block.0", out_name + ".act1")
            add_wn_conv(prefix + ".block.1",  out_name + ".conv1")
            add_alpha(prefix + ".block.2", out_name + ".act2")
            add_wn_conv(prefix + ".block.3",  out_name + ".conv2")

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        # encoder.block.0 is the initial conv (1 → 48). Then 4 EncoderBlocks at
        # encoder.block.{1..4}, each containing 3 ResidualUnits + Snake +
        # downsample WNConv1d. encoder.block.5 is the final depthwise WNConv1d.
        add_wn_conv("encoder.block.0", "snac.enc.conv0")

        for bi, stride in enumerate(cfg["encoder_rates"], start=1):
            base = f"encoder.block.{bi}.block"
            o = f"snac.enc.b{bi}"
            for ri in range(3):
                add_residual_unit(f"{base}.{ri}", f"{o}.r{ri}")
            add_alpha(f"{base}.3",   f"{o}.act")
            add_wn_conv(f"{base}.4", f"{o}.down")

        add_wn_conv("encoder.block.5", "snac.enc.conv_final")

        # ------------------------------------------------------------------
        # Quantizer (3 levels)
        # ------------------------------------------------------------------
        for qi in range(len(cfg["vq_strides"])):
            base = f"quantizer.quantizers.{qi}"
            add_wn_conv(base + ".in_proj",  f"snac.q.{qi}.in_proj")
            add_wn_conv(base + ".out_proj", f"snac.q.{qi}.out_proj")

            cb = _t(base + ".codebook.weight").astype(np.float32)  # (codebook_size, codebook_dim)
            # Bake the L2-normalized codebook so the runtime can skip the per-
            # codebook normalize step.  SNAC's `decode_latents` runs both
            # encodings and codebook through `F.normalize` before the
            # cosine-NN argmax.
            cb_norm = cb / (np.linalg.norm(cb, axis=1, keepdims=True) + 1e-12)
            self._add_tensor(writer, f"snac.q.{qi}.codebook", cb, "F16")
            self._add_tensor(writer, f"snac.q.{qi}.codebook_norm", cb_norm, "F16")

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        # decoder.model.0: depthwise WNConv1d(latent, latent, k=7, groups=latent)
        # decoder.model.1: pointwise WNConv1d(latent, decoder_dim, k=1)
        add_wn_conv("decoder.model.0", "snac.dec.conv_in_dw")
        add_wn_conv("decoder.model.1", "snac.dec.conv_in_pw")

        # decoder.model.{2..5}: 4 DecoderBlocks
        for bi, stride in enumerate(cfg["decoder_rates"], start=2):
            base = f"decoder.model.{bi}.block"
            o = f"snac.dec.b{bi - 2}"
            add_alpha(base + ".0",  o + ".act")
            # block.1: WNConvTranspose1d
            add_wn_conv(base + ".1", o + ".convtr")
            # block.2: NoiseBlock (linear is WNConv1d k=1, no bias)
            add_wn_conv(base + ".2.linear", o + ".noise")
            # block.{3..5}: 3 ResidualUnits
            for ri in range(3):
                add_residual_unit(f"{base}.{3 + ri}", f"{o}.r{ri}")

        # decoder.model.6: Snake1d
        # decoder.model.7: final WNConv1d (k=7, → 1 channel)
        add_alpha("decoder.model.6", "snac.dec.act_final")
        add_wn_conv("decoder.model.7", "snac.dec.conv_final")

        self._warn_if_no_quantized()
        writer.write()
        self.log(f"Wrote SNAC GGUF to {output_path}")
