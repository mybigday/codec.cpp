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
            # Keep torch layout [in, out, k]. GGUF stores reversed dims so ggml sees [k, out, in].
            # Depthwise conv_transpose is not supported by ggml; keep layout to avoid extra transposes.
            transform = "keep"
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


def _is_descript_dac_state_dict(state_dict) -> bool:
    return any(k.endswith(".weight_g") for k in state_dict.keys()) and any(k.endswith(".weight_v") for k in state_dict.keys())


def _materialize_weight_norm(weight_g, weight_v) -> np.ndarray:
    g = np.asarray(to_numpy(weight_g), dtype=np.float32)
    v = np.asarray(to_numpy(weight_v), dtype=np.float32)
    axes = tuple(range(1, v.ndim))
    norm = np.linalg.norm(v, axis=axes, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return (v * (g / norm)).astype(np.float32, copy=False)


def materialize_descript_dac_weight_norm(state_dict) -> OrderedDict:
    out = OrderedDict()
    keys = set(state_dict.keys())

    for key, value in state_dict.items():
        if key.endswith(".weight_g") or key.endswith(".weight_v"):
            continue
        out[key] = to_numpy(value)

    for key, value in state_dict.items():
        if not key.endswith(".weight_g"):
            continue
        base = key[: -len(".weight_g")]
        v_key = base + ".weight_v"
        if v_key not in keys:
            raise RuntimeError(f"missing DAC weight_norm tensor: {v_key}")
        out[base + ".weight"] = _materialize_weight_norm(value, state_dict[v_key])

    return out


def map_descript_dac_key(key: str) -> str | None:
    key = normalize_key(key)

    match = re.match(r"^encoder\.block\.0\.(weight|bias)$", key)
    if match:
        return "enc.block.0." + match.group(1)

    match = re.match(r"^encoder\.block\.([1-4])\.block\.([0-2])\.block\.(0|1|2|3)\.(alpha|weight|bias)$", key)
    if match:
        block_i = int(match.group(1))
        res_i = int(match.group(2)) + 1
        part_i = int(match.group(3))
        param = match.group(4)
        part = {
            0: "snake1.alpha",
            1: f"conv1.{param}",
            2: "snake2.alpha",
            3: f"conv2.{param}",
        }[part_i]
        if (part_i in (0, 2) and param != "alpha") or (part_i in (1, 3) and param == "alpha"):
            return None
        return f"enc.block.{block_i}.block.res_unit{res_i}.{part}"

    match = re.match(r"^encoder\.block\.([1-4])\.block\.3\.alpha$", key)
    if match:
        return f"enc.block.{match.group(1)}.block.snake1.alpha"

    match = re.match(r"^encoder\.block\.([1-4])\.block\.4\.(weight|bias)$", key)
    if match:
        return f"enc.block.{match.group(1)}.block.conv1.{match.group(2)}"

    match = re.match(r"^encoder\.block\.5\.alpha$", key)
    if match:
        return "enc.block.5.alpha"

    match = re.match(r"^encoder\.block\.6\.(weight|bias)$", key)
    if match:
        return "enc.block.6." + match.group(1)

    match = re.match(r"^decoder\.model\.0\.(weight|bias)$", key)
    if match:
        return "dec.model.0." + match.group(1)

    match = re.match(r"^decoder\.model\.([1-4])\.block\.0\.alpha$", key)
    if match:
        return f"dec.model.{match.group(1)}.block.snake1.alpha"

    match = re.match(r"^decoder\.model\.([1-4])\.block\.1\.(weight|bias)$", key)
    if match:
        return f"dec.model.{match.group(1)}.block.conv_t1.{match.group(2)}"

    match = re.match(r"^decoder\.model\.([1-4])\.block\.([2-4])\.block\.(0|1|2|3)\.(alpha|weight|bias)$", key)
    if match:
        block_i = int(match.group(1))
        res_i = int(match.group(2)) - 1
        part_i = int(match.group(3))
        param = match.group(4)
        part = {
            0: "snake1.alpha",
            1: f"conv1.{param}",
            2: "snake2.alpha",
            3: f"conv2.{param}",
        }[part_i]
        if (part_i in (0, 2) and param != "alpha") or (part_i in (1, 3) and param == "alpha"):
            return None
        return f"dec.model.{block_i}.block.res_unit{res_i}.{part}"

    match = re.match(r"^decoder\.model\.5\.alpha$", key)
    if match:
        return "dec.model.5.alpha"

    match = re.match(r"^decoder\.model\.6\.(weight|bias)$", key)
    if match:
        return "dec.model.6." + match.group(1)

    if key.startswith("quantizer.quantizers."):
        return "vq.q" + key[len("quantizer.quantizers.") :]

    return None


def config_from_descript_metadata(metadata, state_dict) -> Dict[str, int | bool]:
    kwargs = {}
    if isinstance(metadata, dict):
        kwargs = metadata.get("kwargs") or {}
    encoder_rates = kwargs.get("encoder_rates") or [2, 4, 5, 8]
    n_q = kwargs.get("n_codebooks")
    if n_q is None:
        n_q = len({
            re.match(r"^quantizer\.quantizers\.(\d+)\.codebook\.weight$", k).group(1)
            for k in state_dict.keys()
            if re.match(r"^quantizer\.quantizers\.(\d+)\.codebook\.weight$", k)
        })
    return {
        "sample_rate": int(kwargs.get("sample_rate", 24000)),
        "hop_size": int(np.prod(encoder_rates)),
        "n_q": int(n_q),
        "codebook_size": int(kwargs.get("codebook_size", 1024)),
        "latent_dim": int(kwargs.get("latent_dim", 1024)),
        "codebook_dim": int(kwargs.get("codebook_dim", 8)),
        "has_encoder": True,
        "has_decoder": True,
    }


def transform_descript_dac_tensor_for_codec(key: str, arr: np.ndarray) -> np.ndarray:
    if key.endswith(".weight") and arr.ndim == 3:
        return arr.transpose((2, 1, 0)).copy()
    return arr


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
        self.uses_descript_dac_layout = False

    @property
    def model_type(self) -> str:
        return "dac"

    @property
    def architecture(self) -> str:
        return "dac"

    def should_quantize_tensor(self, name: str, arr: np.ndarray) -> bool:
        if self.quantization not in ("Q4_K_M", "Q5_K_M", "Q8_0"):
            return False

        if arr.ndim < 2:
            return False

        if name.endswith(".bias") or name.endswith(".b"):
            return False

        # Never quantize layer norm / adanorm weights even when 2D — block
        # quantization on tensors clustered around 1.0/0.0 produces audio noise.
        import re
        if re.search(r"(?:_ln|inln|paln|norm|ln)\.(?:w|weight|scale\.weight|shift\.weight)$", name):
            return False

        if self.uses_descript_dac_layout and name.endswith(".out_proj.weight"):
            return False

        if (".codebook." in name or name.endswith(".embed")) and not self.quantize_codebook:
            return False

        is_weight = name.endswith(".weight") or name.endswith(".w")
        if not is_weight:
            return False

        ne0 = int(arr.shape[-1])
        if self.quantization in ("Q4_K_M", "Q5_K_M"):
            return (ne0 % 256) == 0
        if self.quantization == "Q8_0":
            return (ne0 % 32) == 0
        return False

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        if torch is None:
            raise RuntimeError("torch is required for DAC checkpoint conversion")

        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "config.json"
        model_file = checkpoint_dir / "pytorch_model.bin"

        if checkpoint_dir.is_file() and checkpoint_dir.suffix in (".safetensors", ".bin", ".pt", ".pth"):
            model_file = checkpoint_dir

        elif not model_file.is_file():
            candidates = (
                sorted(checkpoint_dir.glob("*.safetensors"))
                + sorted(checkpoint_dir.glob("*.bin"))
                + sorted(checkpoint_dir.glob("*.pt"))
                + sorted(checkpoint_dir.glob("*.pth"))
            )
            if not candidates:
                raise FileNotFoundError(f"No DAC checkpoint file found in {checkpoint_dir}")
            model_file = candidates[0]

        checkpoint_metadata = None
        if model_file.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(model_file)
        else:
            state = torch.load(model_file, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            checkpoint_metadata = state.get("metadata")
            state = state["state_dict"]

        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported checkpoint format at {model_file}")

        is_descript_dac = _is_descript_dac_state_dict(state)
        if is_descript_dac:
            self.state_dict = materialize_descript_dac_weight_norm(state)
            self.config = config_from_descript_metadata(checkpoint_metadata, state)
            self.uses_descript_dac_layout = True
        else:
            self.state_dict = OrderedDict((k, to_numpy(v)) for k, v in state.items())
            self.uses_descript_dac_layout = False

        if not is_descript_dac and config_path.is_file():
            import json

            with config_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.config = {
                "sample_rate": int(cfg.get("sampling_rate", cfg.get("sample_rate", 24000))),
                "hop_size": int(cfg.get("hop_length", cfg.get("hop_size", 320))),
                "n_q": int(cfg.get("n_codebooks", cfg.get("n_q", 12))),
                "codebook_size": int(cfg.get("codebook_size", 1024)),
                "latent_dim": int(cfg.get("hidden_size", cfg.get("latent_dim", 1024))),
                "codebook_dim": int(cfg.get("codebook_dim", 8)),
                "has_encoder": True,
                "has_decoder": True,
            }
        elif not is_descript_dac:
            self.config = {
                "sample_rate": 24000,
                "hop_size": 320,
                "n_q": 12,
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
        self.uses_descript_dac_layout = False
        cfg = model.config
        self.config = {
            "sample_rate": int(getattr(cfg, "sampling_rate", getattr(cfg, "sample_rate", 24000))),
            "hop_size": int(getattr(cfg, "hop_length", getattr(cfg, "hop_size", 320))),
            "n_q": int(getattr(cfg, "n_codebooks", getattr(cfg, "n_q", 12))),
            "codebook_size": int(getattr(cfg, "codebook_size", 1024)),
            "latent_dim": int(getattr(cfg, "hidden_size", getattr(cfg, "latent_dim", 1024))),
            "codebook_dim": int(getattr(cfg, "codebook_dim", 8)),
            "has_encoder": True,
            "has_decoder": True,
        }

    def convert_and_save(self, output_path: Path) -> None:
        if self.state_dict is None or self.config is None:
            raise RuntimeError("No model loaded. Call load_from_checkpoint/load_from_huggingface first.")

        writer = GGUFWriter(output_path, self.architecture)
        self._reset_quant_stats()
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
            out = map_descript_dac_key(key) if self.uses_descript_dac_layout else map_key(key)
            if out is None:
                continue

            arr = np.asarray(value)
            if self.uses_descript_dac_layout:
                arr = transform_descript_dac_tensor_for_codec(key, arr)
            else:
                arr = transform_tensor_for_codec(key, arr, self.conv_transforms)
            mapped[key] = (out, arr)

        used = set()
        for _src_key, (mapped_name, arr) in mapped.items():
            out_name = shorten_tensor_name(mapped_name, used)
            self._add_tensor(writer, out_name, arr)

        self._warn_if_no_quantized()
        writer.write()
        self.log(f"Wrote DAC GGUF to {output_path}")
