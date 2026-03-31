"""Soprano (Vocos) decoder converter."""

from collections import OrderedDict
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .base import BaseConverter
from utils.gguf_writer import GGUFWriter


def _to_numpy(tensor):
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def _map_key(key: str) -> str | None:
    if key == "decoder.embed.weight":
        return "sop.decode.embed.w"
    if key == "decoder.embed.bias":
        return "sop.decode.embed.b"
    if key == "decoder.norm.weight":
        return "sop.decode.norm.w"
    if key == "decoder.norm.bias":
        return "sop.decode.norm.b"
    if key.startswith("decoder.convnext."):
        rest = key[len("decoder.convnext.") :]
        parts = rest.split(".")
        if len(parts) < 2:
            return None
        layer = parts[0]
        suffix = ".".join(parts[1:])
        mapping = {
            "dwconv.weight": "dw.w",
            "dwconv.bias": "dw.b",
            "norm.weight": "ln.w",
            "norm.bias": "ln.b",
            "pwconv1.weight": "pw1.w",
            "pwconv1.bias": "pw1.b",
            "pwconv2.weight": "pw2.w",
            "pwconv2.bias": "pw2.b",
            "gamma": "gamma",
        }
        if suffix not in mapping:
            return None
        return f"sop.decode.cnx.{layer}.{mapping[suffix]}"
    if key == "decoder.final_layer_norm.weight":
        return "sop.decode.fln.w"
    if key == "decoder.final_layer_norm.bias":
        return "sop.decode.fln.b"
    if key == "head.out.weight":
        return "sop.decode.head.out.w"
    if key == "head.out.bias":
        return "sop.decode.head.out.b"
    if key == "head.istft.window":
        return "sop.decode.istft.window"
    return None


class SopranoConverter(BaseConverter):
    @property
    def model_type(self) -> str:
        return "soprano"

    @property
    def architecture(self) -> str:
        return "soprano"

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        if torch is None:
            raise RuntimeError("torch is required for Soprano checkpoint conversion")

        checkpoint_dir = Path(checkpoint_dir)
        if checkpoint_dir.is_dir():
            model_file = checkpoint_dir / "decoder.pth"
        else:
            model_file = checkpoint_dir

        if not model_file.is_file():
            raise FileNotFoundError(f"missing Soprano decoder checkpoint: {model_file}")

        state = torch.load(model_file, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported checkpoint format at {model_file}")

        self.state_dict = OrderedDict((k, _to_numpy(v)) for k, v in state.items())

        # Fixed config for Soprano 1.1 decoder
        self.config = {
            "sample_rate": 32000,
            "hop_size": 512,
            "n_fft": 2048,
            "win_length": 2048,
            "latent_dim": 512,
            "decoder_dim": 768,
            "intermediate_dim": 2304,
            "num_layers": 8,
            "upscale": 4,
            "dw_kernel": 3,
            "has_encoder": False,
            "has_decoder": True,
        }

    def load_from_huggingface(self, model_id: str) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise RuntimeError("huggingface_hub is required for Soprano HF conversion") from exc

        model_file = hf_hub_download(repo_id=model_id, filename="decoder.pth")
        self.load_from_checkpoint(Path(model_file))

    def convert_and_save(self, output_path: Path) -> None:
        if self.state_dict is None or self.config is None:
            raise RuntimeError("No model loaded. Call load_from_checkpoint/load_from_huggingface first.")

        writer = GGUFWriter(output_path, self.architecture)
        self._reset_quant_stats()
        writer.add_name("Soprano-Decoder")
        writer.add_uint32("codec.sample_rate", int(self.config["sample_rate"]))
        writer.add_uint32("codec.hop_size", int(self.config["hop_size"]))
        writer.add_uint32("codec.n_fft", int(self.config["n_fft"]))
        writer.add_uint32("codec.win_length", int(self.config["win_length"]))
        writer.add_uint32("codec.latent_dim", int(self.config["latent_dim"]))
        writer.add_bool("codec.has_encoder", bool(self.config["has_encoder"]))
        writer.add_bool("codec.has_decoder", bool(self.config["has_decoder"]))

        writer.add_uint32("soprano.decoder_dim", int(self.config["decoder_dim"]))
        writer.add_uint32("soprano.intermediate_dim", int(self.config["intermediate_dim"]))
        writer.add_uint32("soprano.num_layers", int(self.config["num_layers"]))
        writer.add_uint32("soprano.upscale", int(self.config["upscale"]))
        writer.add_uint32("soprano.dw_kernel", int(self.config["dw_kernel"]))

        for raw_key, value in self.state_dict.items():
            out = _map_key(raw_key)
            if out is None:
                continue
            arr = np.asarray(value)
            # NOTE: GGUF writer reverses NumPy shapes into ggml (ne0, ne1, ...).
            # Torch Conv1d and Linear weights are already in the correct logical
            # order once written, so do not transpose here.
            self._add_tensor(writer, out, arr, self._get_target_dtype(out, arr))

        self._warn_if_no_quantized()
        writer.write()
        self.log(f"Wrote Soprano GGUF to {output_path}")
