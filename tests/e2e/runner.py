#!/usr/bin/env python3
"""Unified E2E runner with direct HF model calls inside this file."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.json")
DEFAULT_INPUT_AUDIO = REPO_ROOT / "input_audio/reference_10_2.mp3"
DEFAULT_SAMPLE_RATE = 24000
PRIMARY_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_THRESHOLDS = {
    "corr_min": 0.99,
    "mse_max": 0.0001,
}


@dataclass
class ModelResult:
    name: str
    mode: str
    status: str
    duration_sec: float
    log_path: str
    reason: str = ""
    return_code: int | None = None
    metrics: dict[str, Any] | None = None


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_tools() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")


def run_and_log(command: list[str], log_path: Path, prefix: str, cwd: Path | None = None) -> int:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(command)}\n")
        proc = subprocess.Popen(
            command,
            cwd=str(cwd or REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{prefix}] {line}", end="")
            log_file.write(line)
        ret = proc.wait()
        log_file.write(f"[exit={ret}]\n\n")
    return ret


def ffmpeg_to_mono_wav(src: Path, dst: Path, sample_rate: int, log_path: Path, model_name: str) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(dst),
    ]
    ret = run_and_log(cmd, log_path, model_name)
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed for {src}")


def read_wav_float_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        ch = int(wf.getnchannels())
        sw = int(wf.getsampwidth())
        n = int(wf.getnframes())
        payload = wf.readframes(n)
    if sw != 2:
        raise RuntimeError(f"{path}: expected PCM16 WAV, sample_width={sw}")
    pcm = np.frombuffer(payload, dtype=np.int16)
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
    wav = np.clip(pcm.astype(np.float32) / 32768.0, -1.0, 1.0)
    return sr, wav


def save_wav_pcm16(path: Path, waveform: np.ndarray, sample_rate: int) -> int:
    mono = np.asarray(waveform, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.reshape(-1)
    mono = np.clip(mono, -1.0, 1.0)
    pcm = np.round(mono * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return int(mono.size)


def read_wav_mono_i16(path: Path) -> tuple[int, array]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        payload = wf.readframes(n)
    if ch != 1 or sw != 2:
        raise RuntimeError(f"{path}: expected mono PCM16 WAV, got channels={ch} sample_width={sw}")
    pcm = array("h")
    pcm.frombytes(payload)
    return sr, pcm


def compare_wav(ref_wav: Path, out_wav: Path) -> dict[str, Any]:
    sr_ref, a = read_wav_mono_i16(ref_wav)
    sr_out, b = read_wav_mono_i16(out_wav)

    n_ref = len(a)
    n_out = len(b)
    n = min(n_ref, n_out)
    if n <= 0:
        raise RuntimeError("empty waveform")

    sum_err = 0.0
    sum_aa = 0.0
    sum_bb = 0.0
    sum_ab = 0.0
    for i in range(n):
        x = a[i] / 32768.0
        y = b[i] / 32768.0
        d = x - y
        sum_err += d * d
        sum_aa += x * x
        sum_bb += y * y
        sum_ab += x * y

    mse = sum_err / n
    corr = sum_ab / math.sqrt(max(sum_aa * sum_bb, 1e-20))
    return {
        "sample_rate_ref": sr_ref,
        "sample_rate_out": sr_out,
        "n_samples_ref": n_ref,
        "n_samples_out": n_out,
        "length_diff": abs(n_ref - n_out),
        "mse": mse,
        "corr": corr,
    }


def merge_thresholds(model_cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(DEFAULT_THRESHOLDS)
    out.update(model_cfg.get("thresholds", {}))
    return out


def assert_metric(metrics: dict[str, Any], thresholds: dict[str, Any], hop_size: int | None = None) -> list[str]:
    failures: list[str] = []

    if int(metrics["sample_rate_ref"]) != int(metrics["sample_rate_out"]):
        failures.append(
            f"sample_rate mismatch ref={metrics['sample_rate_ref']} out={metrics['sample_rate_out']}"
        )

    corr_min = thresholds.get("corr_min")
    if corr_min is not None and float(metrics["corr"]) < float(corr_min):
        failures.append(f"corr {metrics['corr']:.6f} < {float(corr_min):.6f}")

    mse_max = thresholds.get("mse_max")
    if mse_max is not None and float(metrics["mse"]) > float(mse_max):
        failures.append(f"mse {metrics['mse']:.8f} > {float(mse_max):.8f}")

    len_diff_max = thresholds.get("length_diff_max")
    if len_diff_max is None and hop_size is not None:
        len_diff_max = hop_size
    if len_diff_max is not None and int(metrics["length_diff"]) > int(len_diff_max):
        failures.append(f"length_diff {metrics['length_diff']} > {int(len_diff_max)}")

    return failures


def parse_model_class(class_spec: str) -> tuple[str, str]:
    spec = class_spec.strip()
    if not spec:
        raise RuntimeError("empty class spec")

    if spec.lower() == "dac":
        return "dac", ""

    if spec.lower() == "wavtokenizer":
        return "wavtokenizer", ""

    if ":" not in spec:
        raise RuntimeError(
            f"invalid class spec '{class_spec}': expected 'transformers:ClassName', 'dac', or 'wavtokenizer'"
        )

    module_name, class_name = spec.split(":", 1)
    module_name = module_name.strip()
    class_name = class_name.strip()
    if module_name != "transformers" or not class_name:
        raise RuntimeError(
            f"invalid class spec '{class_spec}': expected 'transformers:ClassName', 'dac', or 'wavtokenizer'"
        )
    return module_name, class_name


def preferred_python_executable() -> str:
    if PRIMARY_PYTHON.is_file():
        return str(PRIMARY_PYTHON)
    return sys.executable


def resolve_model_local_path(model_cfg: dict[str, Any]) -> Path:
    local_path = model_cfg.get("local_path")
    if not local_path:
        return REPO_ROOT / "models" / model_cfg["name"]
    path = Path(local_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


class NativeDacAdapter:
    def __init__(self, model: Any):
        self.model = model

    def eval(self):
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self

    def encode(self, *args, **kwargs):
        import torch

        audio = kwargs.pop("input_values", None)
        if audio is None:
            audio = kwargs.pop("audio_values", None)
        if audio is None and args:
            audio = args[0]
        if audio is None:
            raise TypeError("missing audio input")

        n_quantizers = kwargs.pop("n_quantizers", None)
        if kwargs:
            raise TypeError(f"unexpected kwargs: {sorted(kwargs.keys())}")

        x = audio
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32, copy=False))
        if x.ndim == 2:
            x = x.unsqueeze(1)

        encode_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = [((x,), {})]
        if n_quantizers is not None:
            encode_calls.insert(0, ((x,), {"n_quantizers": int(n_quantizers)}))

        out = call_with_fallback(self.model.encode, encode_calls, "dac.native.encode")
        if isinstance(out, tuple):
            # Common DAC output: (latents, codes, ...)
            if len(out) < 2:
                raise RuntimeError("dac native encode returned tuple without codes")
            codes = out[1]
        elif hasattr(out, "audio_codes"):
            codes = out.audio_codes
        else:
            raise RuntimeError("dac native encode output does not expose codes")
        return SimpleNamespace(audio_codes=codes)

    def decode(self, *args, **kwargs):
        audio_codes = kwargs.pop("audio_codes", None)
        if audio_codes is None and args:
            audio_codes = args[0]
        if audio_codes is None:
            raise TypeError("missing audio codes")
        if kwargs:
            raise TypeError(f"unexpected kwargs: {sorted(kwargs.keys())}")

        if hasattr(self.model, "quantizer") and hasattr(self.model.quantizer, "from_codes"):
            latent = self.model.quantizer.from_codes(audio_codes)
            if isinstance(latent, tuple):
                latent = latent[0]
            decoded = call_with_fallback(
                self.model.decode,
                [((latent,), {})],
                "dac.native.decode(latent)",
            )
        else:
            decoded = call_with_fallback(
                self.model.decode,
                [((audio_codes,), {})],
                "dac.native.decode(codes)",
            )
        return SimpleNamespace(audio_values=decoded)


class NativeWavTokenizerAdapter:
    def __init__(self, model: Any, default_bandwidth_id: int):
        self.model = model
        self.default_bandwidth_id = int(default_bandwidth_id)

    def eval(self):
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self

    def encode(self, *args, **kwargs):
        import torch

        audio = kwargs.pop("input_values", None)
        if audio is None:
            audio = kwargs.pop("audio_values", None)
        if audio is None and args:
            audio = args[0]
        if audio is None:
            raise TypeError("missing audio input")

        bandwidth_id = kwargs.pop("bandwidth_id", None)
        if kwargs:
            raise TypeError(f"unexpected kwargs: {sorted(kwargs.keys())}")
        if bandwidth_id is None:
            bandwidth_id = torch.tensor([self.default_bandwidth_id], dtype=torch.long)

        x = audio
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32, copy=False))
        if x.ndim == 3 and x.shape[1] == 1:
            x = x[:, 0, :]
        if x.ndim == 1:
            x = x.view(1, -1)

        if hasattr(self.model, "encode_infer"):
            return call_with_fallback(
                self.model.encode_infer,
                [
                    ((x,), {"bandwidth_id": bandwidth_id}),
                    ((x,), {}),
                ],
                "wavtokenizer.native.encode_infer",
            )

        return call_with_fallback(
            self.model.encode,
            [
                ((x,), {"bandwidth_id": bandwidth_id}),
                ((x,), {}),
            ],
            "wavtokenizer.native.encode",
        )

    def decode(self, *args, **kwargs):
        if not args:
            raise TypeError("missing features/tokens for decode")
        return call_with_fallback(
            self.model.decode,
            [
                (args, kwargs),
                (args, {}),
            ],
            "wavtokenizer.native.decode",
        )


def load_native_dac_model(model_cfg: dict[str, Any]):
    try:
        dac = importlib.import_module("dac")
    except Exception as exc:
        raise RuntimeError(
            "dac model requires descript-audio-codec package in the active Python env"
        ) from exc

    local_path = resolve_model_local_path(model_cfg)
    repo_id = model_cfg.get("hf_repo_id")
    errors: list[str] = []

    if local_path.exists():
        load_fn = getattr(getattr(dac, "DAC", None), "load", None)
        if callable(load_fn):
            candidate_paths: list[Path] = []
            if local_path.is_file():
                candidate_paths.append(local_path)
            elif local_path.is_dir():
                candidate_paths.extend(sorted(local_path.rglob("*.pt")))
                candidate_paths.extend(sorted(local_path.rglob("*.ckpt")))

            for ckpt in candidate_paths:
                try:
                    return NativeDacAdapter(load_fn(str(ckpt))).eval()
                except Exception as exc:
                    errors.append(f"DAC.load({ckpt}) failed: {exc}")

    from_pretrained_fn = getattr(getattr(dac, "DAC", None), "from_pretrained", None)
    if callable(from_pretrained_fn) and repo_id:
        try:
            return NativeDacAdapter(from_pretrained_fn(repo_id)).eval()
        except Exception as exc:
            errors.append(f"DAC.from_pretrained({repo_id}) failed: {exc}")

    raise RuntimeError(
        "failed to load dac model via descript-audio-codec; "
        f"local_path={local_path}, hf_repo_id={repo_id}, errors={errors}"
    )


def ensure_wavtokenizer_source() -> Path:
    """Clone WavTokenizer official repo to models/wavtokenizer-src/ if not exists."""
    src_path = REPO_ROOT / "models" / "wavtokenizer-src"
    
    if src_path.exists() and (src_path / "decoder" / "pretrained.py").is_file():
        return src_path
    
    # Clone the official repo
    print(f"[wavtokenizer] Cloning official source to {src_path}...")
    src_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "git", "clone",
        "--depth", "1",
        "https://github.com/jishengpeng/WavTokenizer.git",
        str(src_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone WavTokenizer repo: {result.stderr}")
    
    if not (src_path / "decoder" / "pretrained.py").is_file():
        raise RuntimeError(f"WavTokenizer source cloned but decoder/pretrained.py not found in {src_path}")
    
    print(f"[wavtokenizer] Source cloned to {src_path}")
    return src_path


def load_native_wavtokenizer_model(model_cfg: dict[str, Any]):
    """Load WavTokenizer model using official source code from models/wavtokenizer-src/."""
    # 1. Ensure official source is cloned
    src_path = ensure_wavtokenizer_source()
    
    # 2. Model weights/checkpoints are in local_path (downloaded from HF)
    local_path = resolve_model_local_path(model_cfg)
    
    # 3. Add source path to sys.path for importing
    src_path_str = str(src_path)
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)
    
    # Also need the repo root for other imports like encoder/decoder
    repo_path_str = str(src_path)
    if repo_path_str not in sys.path:
        sys.path.insert(0, repo_path_str)

    try:
        module = importlib.import_module("decoder.pretrained")
        wavtokenizer_cls = getattr(module, "WavTokenizer")
    except Exception as exc:
        raise RuntimeError(
            f"failed to import WavTokenizer from source path={src_path}"
        ) from exc

    # Find config file (in local_path which contains HF downloaded files)
    config_candidates: list[Path] = []
    explicit_config = model_cfg.get("config_path")
    if explicit_config:
        config_path = Path(explicit_config)
        config_candidates.append(config_path if config_path.is_absolute() else local_path / config_path)
    config_candidates.extend(
        [
            local_path / "checkpoints" / "config.yaml",
            local_path / "config.yaml",
        ]
    )
    config_path = next((p for p in config_candidates if p.is_file()), None)
    if config_path is None:
        raise RuntimeError(
            f"wavtokenizer config not found under local_path={local_path}; "
            "set model config_path or place checkpoints/config.yaml"
        )

    # Find checkpoint file (in local_path)
    ckpt_candidates: list[Path] = []
    explicit_ckpt = model_cfg.get("checkpoint")
    if explicit_ckpt:
        ckpt_path = Path(explicit_ckpt)
        ckpt_candidates.append(ckpt_path if ckpt_path.is_absolute() else local_path / ckpt_path)
    ckpt_candidates.extend(sorted((local_path / "checkpoints").glob("*.ckpt")))
    ckpt_candidates.extend(sorted(local_path.glob("*.ckpt")))
    checkpoint_path = next((p for p in ckpt_candidates if p.is_file()), None)

    load_errors: list[str] = []
    model = None

    # Use from_pretrained0802(config_path, checkpoint_path) as specified
    if checkpoint_path is not None and hasattr(wavtokenizer_cls, "from_pretrained0802"):
        try:
            model = wavtokenizer_cls.from_pretrained0802(str(config_path), str(checkpoint_path))
        except Exception as exc:
            load_errors.append(f"from_pretrained0802({config_path}, {checkpoint_path}) failed: {exc}")

    if model is None and hasattr(wavtokenizer_cls, "from_hparams0802"):
        try:
            model = wavtokenizer_cls.from_hparams0802(str(config_path))
        except Exception as exc:
            load_errors.append(f"from_hparams0802({config_path}) failed: {exc}")

    if model is None:
        raise RuntimeError(
            "failed to construct WavTokenizer model; "
            f"src_path={src_path}, local_path={local_path}, config={config_path}, checkpoint={checkpoint_path}, "
            f"errors={load_errors}"
        )

    return NativeWavTokenizerAdapter(model, int(model_cfg.get("bandwidth_id", 0))).eval()


def load_hf_model(model_cfg: dict[str, Any]):
    module_name, class_name = parse_model_class(model_cfg["class"])
    if module_name == "dac":
        return load_native_dac_model(model_cfg)
    if module_name == "wavtokenizer":
        return load_native_wavtokenizer_model(model_cfg)

    cache_dir = str(REPO_ROOT / "models" / "hf")
    kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
    }
    hf_file = model_cfg.get("hf_file")
    if hf_file:
        kwargs["filename"] = hf_file
    if module_name == "transformers":
        module = importlib.import_module("transformers")
        model_cls = getattr(module, class_name)
        model = model_cls.from_pretrained(model_cfg["hf_repo_id"], **kwargs)
    else:
        raise RuntimeError(f"unsupported class family '{model_cfg['class']}'")

    model.eval()
    return model


def to_torch_audio(wav: np.ndarray, channels: bool = True):
    import torch

    x = torch.from_numpy(wav.astype(np.float32, copy=False))
    if channels:
        return x.view(1, 1, -1)
    return x.view(1, -1)


def normalize_codes_2d(codes) -> np.ndarray:
    import torch

    if isinstance(codes, np.ndarray):
        t = codes
    elif isinstance(codes, torch.Tensor):
        t = codes.detach().cpu().numpy()
    else:
        t = np.asarray(codes)

    if t.ndim == 3:
        # Common HF layouts: (B, Q, T) or (Q, B, T)
        if t.shape[0] == 1:
            t = t[0]
        elif t.shape[1] == 1:
            t = t[:, 0, :]
        else:
            t = t.reshape(-1, t.shape[-1])
    elif t.ndim == 2:
        pass
    elif t.ndim > 3:
        t = t.reshape(-1, t.shape[-1])
    else:
        raise RuntimeError(f"unsupported code tensor shape: {tuple(t.shape)}")

    if t.ndim != 2:
        raise RuntimeError(f"failed to normalize codes to 2D, got shape={tuple(t.shape)}")

    return t.astype(np.int32, copy=False)


def extract_audio_values(decoded) -> np.ndarray:
    import torch

    if hasattr(decoded, "audio_values"):
        y = decoded.audio_values
    else:
        y = decoded

    if isinstance(y, torch.Tensor):
        t = y.detach().to(torch.float32).cpu()
        if t.ndim == 3 and t.shape[1] == 1:
            t = t[0, 0]
        elif t.ndim == 2:
            t = t[0]
        elif t.ndim == 1:
            pass
        else:
            t = t.reshape(-1)
        return t.numpy().astype(np.float32, copy=False)

    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32, copy=False)


def call_with_fallback(func, call_args: list[tuple[tuple[Any, ...], dict[str, Any]]], op_name: str):
    last_exc: Exception | None = None
    for args, kwargs in call_args:
        try:
            return func(*args, **kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise RuntimeError(f"{op_name} signature mismatch: {last_exc}") from last_exc
    raise RuntimeError(f"{op_name} could not be called")


def class_family(class_spec: str) -> str:
    module_name, class_name = parse_model_class(class_spec)
    if module_name == "dac":
        return "dac"
    if module_name == "wavtokenizer":
        return "wavtokenizer"
    return class_name.lower().replace("model", "")


def encode_decode_hf(
    model_cfg: dict[str, Any],
    input_wav: Path,
    codes_out: Path,
    ref_out: Path,
    sample_rate: int,
    log_path: Path,
    model: Any,
) -> None:
    import torch

    sr, mono = read_wav_float_mono(input_wav)
    if sr != sample_rate:
        raise RuntimeError(f"expected {sample_rate} Hz input wav, got {sr}")

    class_alias = class_family(model_cfg["class"])

    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"[hf] loading model: {model_cfg['hf_repo_id']} ({model_cfg['class']})\n")

    with torch.no_grad():
        if class_alias == "mimi":
            audio = to_torch_audio(mono, channels=True)
            encoded = call_with_fallback(
                model.encode,
                [
                    ((), {"input_values": audio}),
                    ((), {"audio_values": audio}),
                    ((audio,), {}),
                ],
                "mimi.encode",
            )
            codes_raw = encoded.audio_codes
            decoded = call_with_fallback(
                model.decode,
                [
                    ((), {"audio_codes": codes_raw}),
                    ((codes_raw,), {}),
                ],
                "mimi.decode",
            )

        elif class_alias == "dac":
            audio = to_torch_audio(mono, channels=True)
            n_q = model_cfg.get("n_q")
            encode_calls = [
                ((), {"input_values": audio}),
                ((), {"audio_values": audio}),
                ((audio,), {}),
            ]
            if n_q is not None:
                n_q_int = int(n_q)
                encode_calls = [
                    (args, {**kwargs, "n_quantizers": n_q_int}) for args, kwargs in encode_calls
                ] + encode_calls
            encoded = call_with_fallback(model.encode, encode_calls, "dac.encode")
            codes_raw = encoded.audio_codes
            decoded = call_with_fallback(
                model.decode,
                [
                    ((), {"audio_codes": codes_raw}),
                    ((codes_raw,), {}),
                ],
                "dac.decode",
            )

        elif class_alias == "wavtokenizer":
            audio_bt = to_torch_audio(mono, channels=False)
            bandwidth_id = int(model_cfg.get("bandwidth_id", 0))
            bw_tensor = torch.tensor([bandwidth_id], dtype=torch.long)

            encode_out = None
            encode_last_exc: Exception | None = None
            for args, kwargs in [
                ((), {"input_values": audio_bt}),
                ((), {"audio_values": audio_bt}),
                ((audio_bt,), {}),
                ((audio_bt,), {"bandwidth_id": bw_tensor}),
            ]:
                try:
                    encode_out = model.encode(*args, **kwargs)
                    break
                except TypeError as exc:
                    encode_last_exc = exc
            if encode_out is None:
                raise RuntimeError(f"wavtokenizer.encode signature mismatch: {encode_last_exc}")

            if isinstance(encode_out, tuple) and len(encode_out) == 2:
                feats, tokens = encode_out
                codes_raw = tokens
                decode_out = call_with_fallback(
                    model.decode,
                    [
                        ((feats,), {"bandwidth_id": bw_tensor}),
                        ((feats,), {}),
                    ],
                    "wavtokenizer.decode(feats)",
                )
                decoded = decode_out
            elif hasattr(encode_out, "audio_codes"):
                codes_raw = encode_out.audio_codes
                decoded = call_with_fallback(
                    model.decode,
                    [
                        ((), {"audio_codes": codes_raw}),
                        ((codes_raw,), {}),
                    ],
                    "wavtokenizer.decode(audio_codes)",
                )
            else:
                raise RuntimeError("wavtokenizer encode output does not expose codes")

        else:
            raise RuntimeError(f"unsupported class '{model_cfg['class']}'")

    codes_2d = normalize_codes_2d(codes_raw)
    np.save(codes_out, codes_2d)

    decoded_audio = extract_audio_values(decoded)
    save_wav_pcm16(ref_out, decoded_audio, sample_rate)


def download_hf_snapshot(model_cfg: dict[str, Any], log_path: Path, model_name: str) -> Path:
    """Download HF model snapshot to local cache using huggingface_hub"""
    from huggingface_hub import snapshot_download
    
    repo_id = model_cfg["hf_repo_id"]
    local_path = REPO_ROOT / model_cfg.get("local_path", f"models/{model_name}")
    
    # Use HF cache dir
    cache_dir = REPO_ROOT / "models" / "hf"
    
    print(f"[{model_name}] Downloading HF snapshot: {repo_id}")
    
    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
        )
        print(f"[{model_name}] Downloaded to: {downloaded_path}")
        return Path(downloaded_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download HF snapshot: {e}")


def auto_convert_gguf(model_cfg: dict[str, Any], output_path: Path, log_path: Path, model_name: str) -> Path:
    """Auto-convert HF model to GGUF using scripts/convert-to-gguf.py"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_spec = model_cfg.get("class", "")
    
    # For transformers models, use direct HF conversion
    if class_spec.startswith("transformers:") and model_cfg.get("converter") is None:
        cmd = [
            preferred_python_executable(),
            str(REPO_ROOT / "scripts" / "convert-to-gguf.py"),
            "--model-id", model_cfg["hf_repo_id"],
            "--output", str(output_path),
        ]
    else:
        # For non-transformers models (dac, wavtokenizer), download first then convert
        local_path = download_hf_snapshot(model_cfg, log_path, model_name)
        
        converter = model_cfg.get("converter", model_name)
        cmd = [
            preferred_python_executable(),
            str(REPO_ROOT / "scripts" / "convert-to-gguf.py"),
            "--input-dir", str(local_path),
            "--output", str(output_path),
            "--model-type", converter,
        ]

    quantization = model_cfg.get("quantization")
    if quantization:
        cmd.extend(["--quantization", quantization])

    ret = run_and_log(cmd, log_path, model_name)
    if ret != 0:
        raise RuntimeError(f"auto-convert failed (exit={ret})")

    if not output_path.is_file():
        raise RuntimeError(f"auto-convert did not produce: {output_path}")

    return output_path


def resolve_gguf_path(model_cfg: dict[str, Any]) -> Path:
    explicit = model_cfg.get("gguf")
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()

    name = model_cfg["name"]
    candidates = [
        REPO_ROOT / "models" / name / f"{name}.gguf",
        REPO_ROOT / "models" / f"{name}.gguf",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return candidates[0].resolve()


def run_decode(gguf_path: Path, codes_npy: Path, out_wav: Path, n_q: int, log_path: Path, model_name: str) -> None:
    cmd = [
        str(REPO_ROOT / "build/codec-cli"),
        "decode",
        "--model",
        str(gguf_path),
        "--codes",
        str(codes_npy),
        "--out",
        str(out_wav),
    ]
    if n_q > 0:
        cmd.extend(["--nq", str(n_q)])

    ret = run_and_log(cmd, log_path, model_name)
    if ret != 0:
        raise RuntimeError(f"codec-cli decode failed (exit={ret})")


def run_model(report_dir: Path, model_cfg: dict[str, Any], input_audio: Path, sample_rate: int) -> ModelResult:
    name = model_cfg["name"]
    start = time.monotonic()
    log_path = report_dir / f"{name}.log"
    hf_model = load_hf_model(model_cfg)
    mode = "full" if hf_model is not None else "conversion-only"

    report_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    tmpdir = Path(tempfile.mkdtemp(prefix=f"codec-e2e-{name}-"))
    try:
        if not input_audio.is_file():
            raise RuntimeError(f"input audio not found: {input_audio}")

        expected_gguf = resolve_gguf_path(model_cfg)
        if model_cfg.get("auto_convert", True):
            print(f"[{name}] Always regenerating GGUF from HF...")
            gguf_path = auto_convert_gguf(model_cfg, expected_gguf, log_path, name)
        else:
            gguf_path = expected_gguf
            if not gguf_path.is_file():
                raise RuntimeError(f"GGUF model not found and auto_convert disabled: {gguf_path}")

        if mode == "conversion-only":
            duration = time.monotonic() - start
            return ModelResult(
                name=name,
                mode=mode,
                status="passed",
                duration_sec=duration,
                log_path=str(log_path),
            )

        input_wav = tmpdir / "input.wav"
        hf_codes = tmpdir / "hf_codes.npy"
        hf_ref_wav = tmpdir / "hf_reference.wav"
        cpp_out_wav = tmpdir / "cpp_decode.wav"
        if hf_model is None:
            raise RuntimeError("internal error: full mode selected without an HF model")

        ffmpeg_to_mono_wav(input_audio, input_wav, sample_rate, log_path, name)
        encode_decode_hf(model_cfg, input_wav, hf_codes, hf_ref_wav, sample_rate, log_path, hf_model)

        n_q = int(model_cfg.get("n_q", 0))
        run_decode(gguf_path, hf_codes, cpp_out_wav, n_q, log_path, name)

        metrics = compare_wav(hf_ref_wav, cpp_out_wav)
        thresholds = merge_thresholds(model_cfg)
        hop_size = thresholds.get("length_diff_hop_size")
        failures = assert_metric(metrics, thresholds, hop_size=int(hop_size) if hop_size is not None else None)

        metrics["thresholds"] = thresholds
        metrics["ok"] = len(failures) == 0
        metrics["failures"] = failures

        metrics_path = report_dir / f"{name}.metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            f.write("\n")

        duration = time.monotonic() - start
        if failures:
            return ModelResult(
                name=name,
                mode=mode,
                status="failed",
                duration_sec=duration,
                log_path=str(log_path),
                reason="; ".join(failures),
                return_code=1,
                metrics=metrics,
            )

        return ModelResult(
            name=name,
            mode=mode,
            status="passed",
            duration_sec=duration,
            log_path=str(log_path),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        duration = time.monotonic() - start
        return ModelResult(
            name=name,
            mode=mode,
            status="failed",
            duration_sec=duration,
            log_path=str(log_path),
            reason=str(exc),
            return_code=1,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def write_reports(report_dir: Path, selected_models: list[str], results: list[ModelResult]) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "passed": sum(1 for r in results if r.status == "passed"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
    }

    summary_json = {
        "generated_at": now_utc_iso(),
        "selected_models": selected_models,
        "counts": counts,
        "results": [
            {
                "name": r.name,
                "mode": r.mode,
                "status": r.status,
                "duration_sec": round(r.duration_sec, 3),
                "log_path": r.log_path,
                "reason": r.reason,
                "return_code": r.return_code,
                "metrics": r.metrics or {},
            }
            for r in results
        ],
    }

    summary_json_path = report_dir / "summary.json"
    summary_txt_path = report_dir / "summary.txt"

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
        f.write("\n")

    lines = [
        "Codec E2E Summary",
        f"generated_at: {summary_json['generated_at']}",
        f"selected_models: {', '.join(selected_models)}",
        f"passed={counts['passed']} failed={counts['failed']} skipped={counts['skipped']}",
        "",
    ]
    for r in results:
        metric = r.metrics or {}
        mse = metric.get("mse")
        corr = metric.get("corr")
        metric_txt = ""
        if mse is not None and corr is not None:
            metric_txt = f" | mse={float(mse):.8f} corr={float(corr):.6f}"
        lines.append(
            f"- {r.name}: {r.status} [{r.mode}] ({r.duration_sec:.2f}s)"
            + metric_txt
            + (f" | reason: {r.reason}" if r.reason else "")
            + f" | log: {r.log_path}"
        )

    with summary_txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return summary_json_path, summary_txt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run codec E2E tests")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to runner config JSON")
    parser.add_argument("--report-dir", default=str(REPO_ROOT / "tests/e2e/reports"), help="Report output directory")
    parser.add_argument("--input-audio", default=str(DEFAULT_INPUT_AUDIO), help="Input audio path")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Input resample rate for test")
    parser.add_argument(
        "--models",
        action="append",
        help="Run only specific model names (repeatable or comma-separated)",
    )
    parser.add_argument("--list-models", action="store_true", help="List configured models and exit")
    return parser.parse_args()


def parse_model_selection(raw_values: list[str] | None) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for raw in raw_values or []:
        for part in raw.split(","):
            name = part.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            selected.append(name)
    return selected


def main() -> int:
    args = parse_args()
    require_tools()

    config = load_config(Path(args.config))
    model_cfgs = config.get("models", [])
    model_cfg_by_name = {m["name"]: m for m in model_cfgs}
    configured_names = [m["name"] for m in model_cfgs]

    if args.list_models:
        print("\n".join(configured_names))
        return 0

    selected_names = parse_model_selection(args.models) or configured_names
    unknown = sorted(set(selected_names) - set(configured_names))
    if unknown:
        print(f"ERROR: unknown model(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    selected_cfgs = [model_cfg_by_name[name] for name in selected_names]
    report_dir = Path(args.report_dir).resolve()
    input_audio = Path(args.input_audio).resolve()

    results: list[ModelResult] = []
    for model_cfg in selected_cfgs:
        result = run_model(report_dir, model_cfg, input_audio, int(args.sample_rate))
        results.append(result)
        print(f"[{result.name}] result: {result.status}")
        if result.reason:
            print(f"[{result.name}] detail: {result.reason}")

    summary_json, summary_txt = write_reports(report_dir, selected_names, results)
    failed = any(r.status == "failed" for r in results)

    print("\n=== E2E Summary ===")
    print(summary_txt.read_text(encoding="utf-8"))
    print(f"JSON report: {summary_json}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
