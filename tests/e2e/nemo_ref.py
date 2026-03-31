from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple
import tarfile
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def _load_nemo_checkpoint(path: Path) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    path = Path(path)
    if path.is_file() and path.suffix == ".nemo":
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(path, "r") as tar:
                tar.extractall(tmpdir)
            cfg_path = Path(tmpdir) / "model_config.yaml"
            ckpt_path = Path(tmpdir) / "model_weights.ckpt"
            if not cfg_path.exists() or not ckpt_path.exists():
                raise FileNotFoundError("nemo archive missing model_config.yaml or model_weights.ckpt")
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            state = torch.load(ckpt_path, map_location="cpu")
    else:
        cfg_path = path / "model_config.yaml"
        ckpt_path = path / "model_weights.ckpt"
        if not cfg_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(f"missing model_config.yaml or model_weights.ckpt in {path}")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("unsupported NeMo checkpoint format")

    state_t = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state_t[k] = v.detach().to(torch.float32)
        else:
            state_t[k] = torch.as_tensor(v, dtype=torch.float32)
    return cfg, state_t


def _apply_weight_norm(weight_v: torch.Tensor, weight_g: torch.Tensor) -> torch.Tensor:
    out_channels = weight_v.shape[0]
    v_flat = weight_v.reshape(out_channels, -1)
    norm = torch.linalg.norm(v_flat, dim=1, keepdim=True)
    scale = weight_g.reshape(out_channels) / (norm.reshape(out_channels) + 1e-12)
    reshape = (out_channels,) + (1,) * (weight_v.ndim - 1)
    return weight_v * scale.reshape(reshape)


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
        if len(parts) >= 3:
            layer = parts[0]
            param = parts[-1]
            if param == "weight":
                return f"nemo.enc.down.{layer}.w"
            if param == "bias":
                return f"nemo.enc.down.{layer}.b"

    if key.startswith("audio_encoder.res_layers."):
        rest = key[len("audio_encoder.res_layers.") :]
        parts = rest.split(".")
        if len(parts) >= 8:
            layer = parts[0]
            block = parts[2]
            res = parts[4]
            which = parts[5]
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
        if len(parts) >= 3:
            layer = parts[0]
            param = parts[-1]
            if param == "weight":
                return f"nemo.dec.up.{layer}.w"
            if param == "bias":
                return f"nemo.dec.up.{layer}.b"

    if key.startswith("audio_decoder.res_layers."):
        rest = key[len("audio_decoder.res_layers.") :]
        parts = rest.split(".")
        if len(parts) >= 8:
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


def _snake(x: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    a = torch.clamp(alpha, min=eps).view(1, -1, 1)
    ax = a * x
    s2 = torch.sin(ax) ** 2
    return x + (s2 / a)


def _half_snake(x: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    c = x.shape[1]
    c_half = c // 2
    x_left = x[:, :c_half, :]
    x_right = x[:, c_half:, :]
    x_left = _snake(x_left, alpha, eps)
    x_right = F.leaky_relu(x_right, 0.01)
    return torch.cat([x_left, x_right], dim=1)


def _conv1d_replicate(x, w, b, stride, dilation, padding):
    if padding > 0:
        x = F.pad(x, (padding, padding), mode="replicate")
    return F.conv1d(x, w, b, stride=stride, dilation=dilation)


def _conv1d_causal(x, w, b, stride, dilation):
    k = w.shape[-1]
    kernel_eff = (k - 1) * dilation + 1
    pad_left = kernel_eff - stride
    if pad_left > 0:
        x = F.pad(x, (pad_left, 0), mode="constant", value=0.0)
    return F.conv1d(x, w, b, stride=stride, dilation=dilation)


def _convtr1d_causal(x, w, b, stride, dilation):
    y = F.conv_transpose1d(x, w, b, stride=stride, dilation=dilation)
    kernel = w.shape[-1]
    crop_right = max(0, kernel - stride)
    if crop_right > 0:
        y = y[:, :, :-crop_right]
    return y


@dataclass
class NemoNanoCodecRef:
    cfg: Dict[str, Any]
    weights: Dict[str, torch.Tensor]
    num_groups: int
    num_levels: list[int]
    codebook_dim: int
    codebook_size: int
    sample_rate: int
    hop_size: int
    device: torch.device

    @classmethod
    def from_checkpoint(cls, nemo_path: Path, device: str = "cpu") -> "NemoNanoCodecRef":
        cfg, state = _load_nemo_checkpoint(nemo_path)
        vq = cfg["vector_quantizer"]
        num_groups = int(vq["num_groups"])
        num_levels = [int(x) for x in vq["num_levels_per_group"]]
        codebook_dim = len(num_levels)
        codebook_size = int(np.prod(np.asarray(num_levels)))
        sample_rate = int(cfg.get("sample_rate", 22050))
        hop_size = int(cfg.get("samples_per_frame", 1764))

        mapped: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if key.endswith(".weight_g"):
                v_key = key[: -len(".weight_g")] + ".weight_v"
                if v_key not in state:
                    continue
                weight = _apply_weight_norm(state[v_key], value)
                out_name = _map_key(key[: -len(".weight_g")] + ".weight")
                if out_name is not None:
                    mapped[out_name] = weight
                continue
            if key.endswith(".weight_v"):
                continue
            if key.endswith(".bias"):
                out_name = _map_key(key)
                if out_name is not None:
                    mapped[out_name] = value
                continue
            if key.endswith(".alpha"):
                out_name = _map_key(key)
                if out_name is not None:
                    mapped[out_name] = value.reshape(-1)

        # Expand grouped conv transpose weights to dense
        for layer in range(5):
            w_name = f"nemo.dec.up.{layer}.w"
            if w_name not in mapped:
                continue
            w = mapped[w_name]
            if w.ndim != 3:
                continue
            in_ch, out_per_group, k = w.shape
            if out_per_group != 1:
                continue
            out_ch = in_ch // 2
            dense = torch.zeros((in_ch, out_ch, k), dtype=w.dtype)
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

        mapped["nemo.fsq.scale"] = torch.from_numpy(scale)
        mapped["nemo.fsq.out_scale"] = torch.from_numpy(out_scale)
        mapped["nemo.fsq.out_offset"] = torch.from_numpy(out_offset)
        mapped["nemo.fsq.in_shift"] = torch.from_numpy(in_shift)
        mapped["nemo.fsq.dim_base"] = torch.from_numpy(dim_base_index)

        codebook = np.zeros((codebook_size, codebook_dim), dtype=np.float32)
        bases = dim_base_index.astype(np.int64)
        levels = np.asarray(num_levels, dtype=np.int64)
        for idx in range(codebook_size):
            codes_nonneg = [(idx // bases[d]) % levels[d] for d in range(codebook_dim)]
            codes = (np.asarray(codes_nonneg, dtype=np.float32) - scale) / scale
            codebook[idx, :] = codes
        for g in range(num_groups):
            mapped[f"nemo.fsq.codebook.{g}"] = torch.from_numpy(codebook)

        dev = torch.device(device)
        mapped = {k: v.to(dev) for k, v in mapped.items()}
        return cls(cfg, mapped, num_groups, num_levels, codebook_dim, codebook_size, sample_rate, hop_size, dev)

    def _w(self, name: str) -> torch.Tensor:
        return self.weights[name]

    def encode(self, audio: torch.Tensor, dump_dir: Path | None = None) -> torch.Tensor:
        if audio.ndim == 3:
            x = audio.to(self.device)
        elif audio.ndim == 2:
            x = audio.unsqueeze(1).to(self.device)
        else:
            x = audio.view(1, 1, -1).to(self.device)
        x = x.to(torch.float32)
        if dump_dir is not None:
            _dump_npy(dump_dir / "nemo.encode.pcm.npy", x.squeeze(0))
            w = self._w("nemo.enc.pre.w").permute(2, 1, 0).contiguous()
            _dump_npy(dump_dir / "nemo.enc.pre.w.npy", w)
            _dump_npy(dump_dir / "nemo.enc.pre.b.npy", self._w("nemo.enc.pre.b").view(-1))
            x_pad = F.pad(x, (3, 3), mode="replicate")
            _dump_npy(dump_dir / "nemo.enc.pre.pad.npy", x_pad.squeeze(0))

        x = _conv1d_replicate(x, self._w("nemo.enc.pre.w"), self._w("nemo.enc.pre.b"), 1, 1, 3)
        if dump_dir is not None:
            _dump_npy(dump_dir / "nemo.enc.pre.out.npy", x.squeeze(0))

        in_channels = 24
        down_rates = [2, 3, 6, 7, 7]
        dilations = [1, 3, 5]

        for li in range(5):
            res_sum = None
            for bi in range(3):
                k = 7 if bi == 1 else (11 if bi == 2 else 3)
                x_block = x
                for ri in range(3):
                    h = F.leaky_relu(x_block, 0.01)
                    pad_in = (k * dilations[ri] - dilations[ri]) // 2
                    pad_sk = k // 2
                    h = _conv1d_replicate(
                        h,
                        self._w(f"nemo.enc.res.l{li}.b{bi}.r{ri}.in.w"),
                        self._w(f"nemo.enc.res.l{li}.b{bi}.r{ri}.in.b"),
                        1,
                        dilations[ri],
                        pad_in,
                    )
                    h = F.leaky_relu(h, 0.01)
                    h = _conv1d_replicate(
                        h,
                        self._w(f"nemo.enc.res.l{li}.b{bi}.r{ri}.sk.w"),
                        self._w(f"nemo.enc.res.l{li}.b{bi}.r{ri}.sk.b"),
                        1,
                        1,
                        pad_sk,
                    )
                    x_block = x_block + h
                    if dump_dir is not None:
                        name = f"nemo.enc.l{li}.b{bi}.r{ri}.out.npy"
                        _dump_npy(dump_dir / name, x_block.squeeze(0))
                res_sum = x_block if res_sum is None else res_sum + x_block

            x = res_sum / 3.0
            x = F.leaky_relu(x, 0.01)

            out_channels = in_channels * 2
            stride = down_rates[li]
            kernel = 2 * stride
            padding = (kernel - stride + 1) // 2
            x = _conv1d_replicate(
                x,
                self._w(f"nemo.enc.down.{li}.w"),
                self._w(f"nemo.enc.down.{li}.b"),
                stride,
                1,
                padding,
            )
            if dump_dir is not None:
                name = f"nemo.enc.down.{li}.out.npy"
                _dump_npy(dump_dir / name, x.squeeze(0))
            in_channels = out_channels

        x = F.leaky_relu(x, 0.01)
        x = _conv1d_replicate(x, self._w("nemo.enc.post.w"), self._w("nemo.enc.post.b"), 1, 1, 3)
        if dump_dir is not None:
            _dump_npy(dump_dir / "nemo.enc.post.out.npy", x.squeeze(0))

        # FSQ encode
        t = x.shape[-1]
        x_tc = x.transpose(1, 2).contiguous()  # [B, T, C]
        x_tc = x_tc.view(-1, t, self.num_groups, self.codebook_dim)
        x_tc = x_tc.squeeze(0)

        scale = self._w("nemo.fsq.scale")
        out_scale = self._w("nemo.fsq.out_scale")
        out_offset = self._w("nemo.fsq.out_offset")
        in_shift = self._w("nemo.fsq.in_shift")
        dim_base = self._w("nemo.fsq.dim_base")

        tokens = []
        for g in range(self.num_groups):
            x_g = x_tc[:, g, :]
            x_add = x_g + in_shift
            x_tanh = torch.tanh(x_add)
            x_mul = x_tanh * out_scale
            x_comp = x_mul - out_offset
            x_round = torch.round(x_comp)
            x_norm = x_round / scale
            x_nonneg = x_norm * scale + scale
            x_idx = x_nonneg * dim_base
            idx = torch.sum(x_idx, dim=1)
            tokens.append(idx.to(torch.int32))

        tok = torch.stack(tokens, dim=0)  # [n_q, t]
        return tok

    def decode(self, codes: torch.Tensor, dump_dir: Path | None = None) -> torch.Tensor:
        if not isinstance(codes, torch.Tensor):
            codes = torch.as_tensor(codes, dtype=torch.int32, device=self.device)
        codes = codes.to(self.device)
        if codes.ndim == 3:
            if codes.shape[0] == 1:
                codes = codes[0]
            elif codes.shape[1] == 1:
                codes = codes[:, 0, :]
            else:
                codes = codes.reshape(-1, codes.shape[-1])
        if codes.ndim != 2:
            raise RuntimeError(f"unsupported codes shape: {tuple(codes.shape)}")
        if codes.shape[0] != self.num_groups and codes.shape[1] == self.num_groups:
            codes = codes.T

        t = codes.shape[1]
        chunks = []
        for g in range(self.num_groups):
            cb = self._w(f"nemo.fsq.codebook.{g}")  # [codebook_size, codebook_dim]
            idx = codes[g].clamp(0, self.codebook_size - 1).to(torch.int64)
            emb = cb.index_select(0, idx)  # [t, codebook_dim]
            chunks.append(emb)
        x = torch.cat(chunks, dim=1)  # [t, c]
        if dump_dir is not None:
            _dump_npy(dump_dir / "nemo.dec.embed.out.npy", x.transpose(0, 1))
        x = x.transpose(0, 1).unsqueeze(0)  # [1, c, t]

        x = _conv1d_causal(x, self._w("nemo.dec.pre.w"), self._w("nemo.dec.pre.b"), 1, 1)
        if dump_dir is not None:
            _dump_npy(dump_dir / "nemo.dec.pre.out.npy", x.squeeze(0))

        up_rates = [7, 7, 6, 3, 2]
        in_channels = 864
        dilations = [1, 3, 5]
        for li in range(5):
            x = _half_snake(x, self._w(f"nemo.dec.act.{li}.a"))

            out_channels = in_channels // 2
            stride = up_rates[li]
            x = _convtr1d_causal(
                x,
                self._w(f"nemo.dec.up.{li}.w"),
                self._w(f"nemo.dec.up.{li}.b"),
                stride,
                1,
            )
            if dump_dir is not None:
                name = f"nemo.dec.up.{li}.out.npy"
                _dump_npy(dump_dir / name, x.squeeze(0))
            in_channels = out_channels

            res_sum = None
            for bi in range(3):
                k = 7 if bi == 1 else (11 if bi == 2 else 3)
                x_block = x
                for ri in range(3):
                    x_act = _half_snake(x_block, self._w(f"nemo.dec.res.l{li}.b{bi}.r{ri}.in.a"))
                    h = _conv1d_causal(
                        x_act,
                        self._w(f"nemo.dec.res.l{li}.b{bi}.r{ri}.in.w"),
                        self._w(f"nemo.dec.res.l{li}.b{bi}.r{ri}.in.b"),
                        1,
                        dilations[ri],
                    )
                    h_act = _half_snake(h, self._w(f"nemo.dec.res.l{li}.b{bi}.r{ri}.sk.a"))
                    h = _conv1d_causal(
                        h_act,
                        self._w(f"nemo.dec.res.l{li}.b{bi}.r{ri}.sk.w"),
                        self._w(f"nemo.dec.res.l{li}.b{bi}.r{ri}.sk.b"),
                        1,
                        1,
                    )
                    x_block = x_block + h
                    if dump_dir is not None:
                        name = f"nemo.dec.l{li}.b{bi}.r{ri}.out.npy"
                        _dump_npy(dump_dir / name, x_block.squeeze(0))
                res_sum = x_block if res_sum is None else res_sum + x_block
            x = res_sum / 3.0

        x = _half_snake(x, self._w("nemo.dec.post.a"))
        if dump_dir is not None:
            _dump_npy(dump_dir / "nemo.dec.post.act.npy", x.squeeze(0))
        x = _conv1d_causal(x, self._w("nemo.dec.post.w"), self._w("nemo.dec.post.b"), 1, 1)
        if dump_dir is not None:
            _dump_npy(dump_dir / "nemo.dec.post.out.npy", x.squeeze(0))
        x = torch.clamp(x, -1.0, 1.0)
        if dump_dir is not None:
            _dump_npy(dump_dir / "nemo.decode.out.npy", x.squeeze(0))
        return x.squeeze(0).squeeze(0).to(torch.float32)


def _dump_npy(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.detach().cpu().to(torch.float32).numpy()
    np.save(path, arr)
