"""WavTokenizer converter."""

import hashlib
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .base import BaseConverter
from utils.gguf_writer import GGUFWriter

MAX_TENSOR_NAME = 63

EXCLUDED_SUBSTRINGS = ("discriminator", "disc", "loss")
EXCLUDED_PREFIXES = ("dac",)


def unwrap_state_dict(blob):
    if not isinstance(blob, dict):
        return blob
    return blob.get("state_dict", blob.get("model", blob))


def normalize_key(key):
    for prefix in ("module.", "generator.", "model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def should_exclude(key):
    low = key.lower()
    if any(sub in low for sub in EXCLUDED_SUBSTRINGS):
        return True
    return any(low.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def remap_key(key):
    if key.startswith("feature_extractor.encodec.encoder."):
        return "enc." + key[len("feature_extractor.encodec.encoder.") :]
    if key.startswith("feature_extractor.encodec.quantizer."):
        return "vq." + key[len("feature_extractor.encodec.quantizer.") :]
    if key.startswith("feature_extractor.encodec.decoder."):
        return "dec.feature_extractor." + key[len("feature_extractor.encodec.decoder.") :]
    if key.startswith("feature_extractor."):
        return "dec.feature_extractor." + key[len("feature_extractor.") :]
    if key.startswith("backbone."):
        return "dec.backbone." + key[len("backbone.") :]
    if key.startswith("head."):
        return "dec.head." + key[len("head.") :]
    return None


def compress_name(name):
    replacements = (
        ("_orig_mod.", ""),
        (".residual_unit.", ".ru."),
        (".snake1d.", ".s1."),
        (".snake_beta", ".sb"),
        (".snake_gamma", ".sg"),
        (".weight_g", ".wg"),
        (".weight_v", ".wv"),
        (".kernel_size", ".ks"),
        (".upsample", ".up"),
        (".downsample", ".dn"),
        # Additional compression for WavTokenizer
        ("feature_extractor.", "feat."),
        (".convnext.", ".cnx."),
        ("backbone.", "bb."),
        (".final_layer_norm.", ".fln."),
    )
    out = name
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def shorten_tensor_name(name, used_names):
    candidate = compress_name(name)
    if len(candidate) <= MAX_TENSOR_NAME and candidate not in used_names:
        used_names.add(candidate)
        return candidate

    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    prefix_budget = MAX_TENSOR_NAME - 1 - len(digest)
    short = f"{candidate[:prefix_budget]}.{digest}"
    i = 1
    while short in used_names:
        suffix = f".{digest}{i}"
        prefix_budget = MAX_TENSOR_NAME - len(suffix)
        short = f"{candidate[:prefix_budget]}{suffix}"
        i += 1
    used_names.add(short)
    return short


def to_numpy(tensor):
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def apply_weight_norm(weight_v: np.ndarray, weight_g: np.ndarray) -> np.ndarray:
    if weight_v.ndim < 2:
        raise ValueError(f"weight_norm expects ndim >= 2, got {weight_v.ndim}")
    out_channels = weight_v.shape[0]
    v_flat = weight_v.reshape(out_channels, -1)
    norm = np.linalg.norm(v_flat, axis=1)
    norm = np.maximum(norm, 1e-12)
    scale = weight_g.reshape(out_channels) / norm
    reshape = (out_channels,) + (1,) * (weight_v.ndim - 1)
    return weight_v * scale.reshape(reshape)


def maybe_transpose_lstm_weight(key: str, arr: np.ndarray) -> np.ndarray:
    if ".lstm.weight_ih_" in key or ".lstm.weight_hh_" in key:
        if arr.ndim != 2:
            raise ValueError(f"lstm weight must be rank-2: {key} shape={arr.shape}")
        return arr.T.copy()
    return arr


def _construct_state_dict_key(module_name, param_name):
    return f"{module_name}.{param_name}" if module_name else param_name


def _is_depthwise_conv(module):
    return module.groups == module.in_channels and module.out_channels == module.in_channels


def build_conv_transforms(model):
    conv_transforms = {}

    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("torch is required for module-type conv transforms") from exc

    for module_name, module in model.named_modules():
        transform = None
        if isinstance(module, nn.Conv1d):
            transform = "keep"
        elif isinstance(module, nn.ConvTranspose1d):
            transform = "transpose_2_0_1" if _is_depthwise_conv(module) else "transpose_2_1_0"
        if transform is None:
            continue

        for param_name, _ in module.named_parameters(recurse=False):
            if param_name not in ("weight", "weight_v", "weight_orig"):
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
        return np.transpose(arr, (2, 1, 0)).copy()
    if transform == "transpose_2_0_1":
        return np.transpose(arr, (2, 0, 1)).copy()

    raise ValueError(f"unknown transform op: {transform} for {key}")


def resolve_wavtokenizer_source_path(explicit_path):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    env_path = Path(__file__).resolve().parents[3] / "WavTokenizer-source"
    candidates.append(env_path)

    for path in candidates:
        if path.is_dir():
            return path
    return None


def load_wavtokenizer_model_for_transforms(config_path, wavtokenizer_source):
    source_path = resolve_wavtokenizer_source_path(wavtokenizer_source)
    if source_path is None:
        raise RuntimeError(
            "failed to locate WavTokenizer-source. "
            "Provide wavtokenizer_source path (repo root containing decoder/ and encoder/)."
        )

    source_str = str(source_path)
    if source_str not in sys.path:
        sys.path.insert(0, source_str)

    try:
        from decoder.pretrained import WavTokenizer
    except Exception as exc:
        raise RuntimeError(
            "failed to import WavTokenizer python package from WavTokenizer-source. "
            f"source={source_path}, import error: {exc}"
        ) from exc

    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise RuntimeError(f"config not found: {cfg_path}")

    try:
        model = WavTokenizer.from_hparams0802(str(cfg_path))
    except Exception as exc:
        raise RuntimeError(
            "failed to instantiate WavTokenizer from config for conv transforms. "
            f"config={cfg_path}, source={source_path}, error: {exc}"
        ) from exc

    model.eval()
    return model


def collect_weights(source_name, ckpt_path, allow_unmapped, merged_state):
    if torch is None:
        raise RuntimeError("torch is required for WavTokenizer conversion")

    data = torch.load(ckpt_path, map_location="cpu")
    state = unwrap_state_dict(data)
    if not isinstance(state, dict):
        raise RuntimeError(f"{ckpt_path} did not contain a state dict")

    for raw_key, value in state.items():
        key = normalize_key(raw_key)
        if should_exclude(key):
            continue

        mapped = remap_key(key)
        if mapped is None:
            if not allow_unmapped:
                continue
            mapped = "unmapped." + key

        if key in merged_state:
            continue
        merged_state[key] = (mapped, value)


def add_kernel_tensors(gguf_writer, hop_size):
    if hop_size <= 0:
        raise ValueError(f"invalid hop_size: {hop_size}")
    kernel = np.full((hop_size, 1, 1), 1.0 / float(hop_size), dtype=np.float16)
    gguf_writer.add_tensor("wt.decode.kernel", kernel)
    gguf_writer.add_tensor("wt.encode.kernel", kernel)


class WavTokenizerConverter(BaseConverter):
    def __init__(
        self,
        *args,
        allow_unmapped: bool = False,
        config_path: str = "checkpoints/config.yaml",
        wavtokenizer_source: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.allow_unmapped = allow_unmapped
        self.config_path = config_path
        self.wavtokenizer_source = wavtokenizer_source
        self.conv_transforms: Dict[str, str] = {}

    @property
    def model_type(self) -> str:
        return "wavtokenizer"

    @property
    def architecture(self) -> str:
        return "wavtokenizer_large"

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        checkpoint_path = Path(checkpoint_dir)
        
        # Support single merged .ckpt file (e.g., wavtokenizer_large_speech_320_v2.ckpt)
        if checkpoint_path.is_file() and checkpoint_path.suffix in (".ckpt", ".pth", ".bin"):
            # Single merged checkpoint file
            if torch is None:
                raise RuntimeError("torch is required for loading .ckpt files")
            
            self.log(f"Loading merged checkpoint: {checkpoint_path}")
            data = torch.load(checkpoint_path, map_location="cpu")
            state = unwrap_state_dict(data)
            
            if not isinstance(state, dict):
                raise RuntimeError(f"Unexpected checkpoint format in {checkpoint_path}")
            
            merged_state = {}
            for raw_key, value in state.items():
                key = normalize_key(raw_key)
                if should_exclude(key):
                    continue
                
                mapped = remap_key(key)
                if mapped is None:
                    if self.allow_unmapped:
                        mapped = "unmapped." + key
                    else:
                        continue
                
                if key in merged_state:
                    continue
                merged_state[key] = (mapped, value)
            
            self.state_dict = merged_state
            self.conv_transforms = {}
            self.config = {
                "sample_rate": 24000,
                "hop_size": 320,
                "has_encoder": True,
                "has_decoder": True,
                "merged_ckpt": True,
            }
            return
        
        # Original logic for separate encoder/decoder
        checkpoint_dir = checkpoint_path
        encoder = checkpoint_dir / "encoder.ckpt"
        decoder = checkpoint_dir / "decoder.ckpt"

        if not encoder.is_file() and not decoder.is_file():
            # Try to find any .ckpt file as merged checkpoint
            search = list(checkpoint_dir.glob("*.ckpt")) + list(checkpoint_dir.glob("*.pth"))
            if search:
                try:
                    self.load_from_checkpoint(search[0])
                    return
                except Exception as e:
                    raise FileNotFoundError(
                        f"No checkpoint found in {checkpoint_dir}, and couldn't load {search[0]}: {e}"
                    )
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_dir}. Expected encoder.ckpt and/or decoder.ckpt"
            )

        model_for_transforms = load_wavtokenizer_model_for_transforms(self.config_path, self.wavtokenizer_source)
        self.conv_transforms = build_conv_transforms(model_for_transforms)

        merged_state = {}
        if encoder.is_file():
            collect_weights("encoder", str(encoder), self.allow_unmapped, merged_state)
        if decoder.is_file():
            collect_weights("decoder", str(decoder), self.allow_unmapped, merged_state)

        self.state_dict = merged_state
        self.config = {
            "sample_rate": 24000,
            "hop_size": 320,
            "has_encoder": encoder.is_file(),
            "has_decoder": decoder.is_file(),
            "merged_ckpt": False,
        }

    def load_from_huggingface(self, model_id: str) -> None:
        raise NotImplementedError(
            "WavTokenizer HF loading is not implemented in legacy converter. Use load_from_checkpoint()."
        )

    def convert_and_save(self, output_path: Path) -> None:
        if self.state_dict is None or self.config is None:
            raise RuntimeError("No model loaded. Call load_from_checkpoint/load_from_huggingface first.")

        writer = GGUFWriter(output_path, self.architecture)
        writer.add_name("WavTokenizer")
        writer.add_uint32("codec.sample_rate", int(self.config["sample_rate"]))
        writer.add_uint32("codec.hop_size", int(self.config["hop_size"]))
        writer.add_bool("codec.has_encoder", bool(self.config["has_encoder"]))
        writer.add_bool("codec.has_decoder", bool(self.config["has_decoder"]))

        used_names = set()
        handled = set()
        for src_key in sorted(self.state_dict):
            if src_key in handled:
                continue

            mapped, value = self.state_dict[src_key]

            if src_key.endswith(".weight_g"):
                v_key = src_key[: -len(".weight_g")] + ".weight_v"
                if v_key in self.state_dict:
                    mapped_v, value_v = self.state_dict[v_key]
                    weight_v = to_numpy(value_v).astype(np.float32, copy=False)
                    weight_g = to_numpy(value).astype(np.float32, copy=False)
                    arr = apply_weight_norm(weight_v, weight_g)
                    arr = transform_tensor_for_codec(v_key, arr, self.conv_transforms)
                    out_mapped = mapped_v.replace(".weight_v", ".weight")
                    out_name = shorten_tensor_name(out_mapped, used_names)
                    print(f"add_tensor {v_key}/{src_key} -> {out_name} {arr.shape} {self._get_target_dtype(out_name, arr)}")
                    writer.add_tensor(out_name, arr, self._get_target_dtype(out_name, arr))
                    handled.add(src_key)
                    handled.add(v_key)
                    continue

            if src_key.endswith(".weight_v"):
                g_key = src_key[: -len(".weight_v")] + ".weight_g"
                if g_key in self.state_dict:
                    continue

            arr = to_numpy(value)
            arr = maybe_transpose_lstm_weight(src_key, arr)
            arr = transform_tensor_for_codec(src_key, arr, self.conv_transforms)
            out_name = shorten_tensor_name(mapped, used_names)
            print(f"add_tensor {src_key} -> {out_name} {arr.shape} {self._get_target_dtype(out_name, arr)}")
            writer.add_tensor(out_name, arr, self._get_target_dtype(out_name, arr))

        add_kernel_tensors(writer, int(self.config["hop_size"]))
        writer.write()
        self.log(f"Wrote WavTokenizer GGUF to {output_path}")
