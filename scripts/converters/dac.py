"""DAC model converter."""

import hashlib
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None

from .base import BaseConverter
from utils.gguf_writer import GGUFWriter

MAX_TENSOR_NAME = 63


def normalize_key(key: str) -> str:
    for prefix in ("module.", "model.", "generator."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    if key.startswith("dac."):
        key = key[len("dac.") :]
    return key


def remap_transformers_key_to_runtime_key(key: str) -> str:
    if key.startswith("encoder.conv1."):
        return "encoder.block.0." + key[len("encoder.conv1.") :]

    match = re.match(r"^encoder\.block\.(\d+)\.(.+)$", key)
    if match:
        idx = int(match.group(1)) + 1
        return f"encoder.block.{idx}.block.{match.group(2)}"

    if key.startswith("encoder.snake1."):
        return "encoder.block.5." + key[len("encoder.snake1.") :]
    if key.startswith("encoder.conv2."):
        return "encoder.block.6." + key[len("encoder.conv2.") :]

    if key.startswith("decoder.conv1."):
        return "decoder.model.0." + key[len("decoder.conv1.") :]

    match = re.match(r"^decoder\.block\.(\d+)\.(.+)$", key)
    if match:
        idx = int(match.group(1)) + 1
        return f"decoder.model.{idx}.block.{match.group(2)}"

    if key.startswith("decoder.snake1."):
        return "decoder.model.5." + key[len("decoder.snake1.") :]
    if key.startswith("decoder.conv2."):
        return "decoder.model.6." + key[len("decoder.conv2.") :]

    return key


def map_key(key: str) -> str | None:
    key = normalize_key(key)
    key = remap_transformers_key_to_runtime_key(key)

    if key.startswith("encoder."):
        return "enc." + key[len("encoder.") :]
    if key.startswith("quantizer.quantizers."):
        return "vq.q" + key[len("quantizer.quantizers.") :]
    if key.startswith("decoder.model."):
        # Keep full path for decoder: dec.model.N.*
        return "dec." + key[len("decoder.") :]
    if key.startswith("decoder."):
        # Legacy naming - remap to dec.model.*
        return "dec.model." + key[len("decoder.") :]
    return None


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


def to_numpy(tensor):
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def _construct_state_dict_key(module_name, param_name):
    return f"{module_name}.{param_name}" if module_name else param_name


def _is_depthwise_conv(module):
    return module.groups == module.in_channels and module.out_channels == module.in_channels


def build_conv_transforms(model):
    conv_transforms = {}

    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("transformers/torch required. Use: ./.audio-env/bin/python ...") from exc

    for module_name, module in model.named_modules():
        transform = None
        if isinstance(module, nn.Conv1d):
            transform = "transpose_2_0_1" if _is_depthwise_conv(module) else "keep"
        elif isinstance(module, nn.ConvTranspose1d):
            transform = "transpose_2_0_1" if _is_depthwise_conv(module) else "transpose_2_1_0"
        if transform is None:
            continue

        for param_name, _ in module.named_parameters(recurse=False):
            if param_name != "weight":
                continue
            full_key = _construct_state_dict_key(module_name, param_name)
            conv_transforms[full_key] = transform
            conv_transforms[normalize_key(full_key)] = transform

    return conv_transforms


def transform_tensor_for_codec(key, arr, conv_transforms):
    transform = conv_transforms.get(key)
    if transform is None:
        return arr

    if arr.ndim != 3:
        raise ValueError(f"conv weight must be rank-3: {key} shape={arr.shape}")

    if transform == "keep":
        return arr
    if transform == "transpose_2_1_0":
        return arr.transpose((2, 1, 0)).copy()
    if transform == "transpose_2_0_1":
        return arr.transpose((2, 0, 1)).copy()

    raise ValueError(f"unknown transform op: {transform} for {key}")


def load_model(model_ref):
    if AutoModel is None:
        raise RuntimeError("transformers is required for DAC conversion")

    auto_exc = None
    try:
        return AutoModel.from_pretrained(model_ref, trust_remote_code=True)
    except Exception as exc:
        auto_exc = exc

    try:
        from transformers import DacModel

        return DacModel.from_pretrained(model_ref)
    except Exception as dac_exc:
        raise RuntimeError(
            "failed to load DAC model via transformers.\n"
            f"AutoModel error: {auto_exc}\n"
            f"DacModel error: {dac_exc}"
        ) from dac_exc


class DacConverter(BaseConverter):
    def __init__(self, *args, model_name: str = "DAC.speech.v1.0", **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.conv_transforms: Dict[str, str] = {}

    @property
    def model_type(self) -> str:
        return "dac"

    @property
    def architecture(self) -> str:
        return "dac"

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        st_candidates = sorted(checkpoint_dir.glob("*.safetensors"))
        if st_candidates:
            if safe_open is None:
                raise RuntimeError("safetensors is required for DAC .safetensors checkpoint conversion")
            model_file = st_candidates[0]
            state = OrderedDict()
            with safe_open(str(model_file), framework="np", device="cpu") as handle:
                for key in handle.keys():
                    state[key] = handle.get_tensor(key)
            self.state_dict = state
        else:
            if torch is None:
                raise RuntimeError("torch is required for DAC checkpoint conversion")

            model_file = checkpoint_dir / "pytorch_model.bin"
            if not model_file.is_file():
                candidates = (
                    sorted(checkpoint_dir.glob("*.bin"))
                    + sorted(checkpoint_dir.glob("*.pt"))
                    + sorted(checkpoint_dir.glob("*.pth"))
                )
                if not candidates:
                    raise FileNotFoundError(f"No DAC checkpoint file found in {checkpoint_dir}")
                model_file = candidates[0]

            state = torch.load(model_file, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            if not isinstance(state, dict):
                raise RuntimeError(f"Unsupported checkpoint format at {model_file}")

            self.state_dict = OrderedDict((k, to_numpy(v)) for k, v in state.items())

        self.config = {
            "sample_rate": 24000,
            "hop_size": 512,
            "n_q": 4,
            "codebook_size": 1024,
            "latent_dim": 1024,
            "codebook_dim": 8,
            "has_encoder": True,
            "has_decoder": True,
        }
        self.conv_transforms = {}

    def load_from_huggingface(self, model_id: str) -> None:
        model = load_model(model_id)
        model.eval()

        self.state_dict = OrderedDict((k, to_numpy(v)) for k, v in model.state_dict().items())
        self.conv_transforms = build_conv_transforms(model)
        self.config = {
            "sample_rate": 24000,
            "hop_size": 512,
            "n_q": 4,
            "codebook_size": 1024,
            "latent_dim": 1024,
            "codebook_dim": 8,
            "has_encoder": True,
            "has_decoder": True,
        }

    def convert_and_save(self, output_path: Path) -> None:
        if self.state_dict is None or self.config is None:
            raise RuntimeError("No model loaded. Call load_from_checkpoint/load_from_huggingface first.")

        writer = GGUFWriter(output_path, self.architecture)
        writer.add_name(self.model_name)
        writer.add_uint32("codec.sample_rate", int(self.config["sample_rate"]))
        writer.add_uint32("codec.hop_size", int(self.config["hop_size"]))
        writer.add_uint32("codec.n_q", int(self.config["n_q"]))
        writer.add_uint32("codec.codebook_size", int(self.config["codebook_size"]))
        writer.add_uint32("codec.latent_dim", int(self.config["latent_dim"]))
        writer.add_uint32("codec.codebook_dim", int(self.config["codebook_dim"]))
        writer.add_bool("codec.has_encoder", bool(self.config["has_encoder"]))
        writer.add_bool("codec.has_decoder", bool(self.config["has_decoder"]))

        mapped = OrderedDict()
        for raw_key, value in self.state_dict.items():
            key = normalize_key(raw_key)
            out = map_key(key)
            if out is None:
                continue

            arr = np.asarray(value)
            arr = transform_tensor_for_codec(key, arr, self.conv_transforms)
            mapped[key] = (out, arr)

        used = set()
        for _src_key, (mapped_name, arr) in mapped.items():
            out_name = shorten_tensor_name(mapped_name, used)
            writer.add_tensor(out_name, arr, self._get_target_dtype(out_name, arr))

        writer.write()
        self.log(f"Wrote DAC GGUF to {output_path}")
