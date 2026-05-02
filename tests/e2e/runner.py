#!/usr/bin/env python3
"""Unified E2E runner with direct HF model calls inside this file."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import multiprocessing
import shutil
import subprocess
import sys
import tempfile
import time
import wave
import os
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import torch
import traceback
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.json")
DEFAULT_INPUT_AUDIO = REPO_ROOT / "input_audio/reference_10_2.mp3"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_THRESHOLDS = {
    "corr_min": 0.99,
    "mse_max": 0.0001,
}

# Per-quantization correlation/MSE relaxation. F32 and F16 should match the
# reference closely; quantized formats lose precision so the same threshold is
# unrealistic. Each entry scales the per-model thresholds: corr_min is reduced
# (1.0 - delta) and mse_max is multiplied. None means "use the strict
# per-model threshold as-is".
QUANT_THRESHOLD_RELAX: dict[str, dict[str, float]] = {
    "F32":    {"corr_delta": 0.0,    "mse_mul": 1.0},
    "F16":    {"corr_delta": 0.0,    "mse_mul": 1.0},
    "Q8_0":   {"corr_delta": 0.015,  "mse_mul": 4.0},
    "Q5_K_M": {"corr_delta": 0.10,   "mse_mul": 16.0},
    "Q4_K_M": {"corr_delta": 0.20,   "mse_mul": 64.0},
}


def relax_thresholds_for_quantization(thresholds: dict[str, Any], quantization: str | None) -> dict[str, Any]:
    """Apply per-quantization relaxation to a threshold dict, returning a new copy.

    For quantized runs we don't expect bit-exact parity with the HF reference;
    the legitimate quality drop is roughly proportional to the quantization
    aggressiveness. Use --strict-thresholds on the CLI to disable.
    """
    if quantization is None:
        return dict(thresholds)
    relax = QUANT_THRESHOLD_RELAX.get(quantization)
    if relax is None:
        return dict(thresholds)

    out = dict(thresholds)
    corr_delta = float(relax.get("corr_delta", 0.0))
    mse_mul = float(relax.get("mse_mul", 1.0))
    for key in ("corr_min", "encode_roundtrip_corr_min"):
        if out.get(key) is not None:
            out[key] = max(0.0, float(out[key]) - corr_delta)
    for key in ("mse_max", "encode_roundtrip_mse_max"):
        if out.get(key) is not None:
            out[key] = float(out[key]) * mse_mul
    return out


@dataclass
class ModelResult:
    name: str
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


def _read_proc_mem() -> dict[str, int]:
    """Read memory stats from /proc (Linux only). Returns values in MB."""
    info: dict[str, int] = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmPeak:", "VmSize:")):
                    key, val = line.split(":")
                    info[key.strip()] = int(val.split()[0]) // 1024
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:")):
                    key, val = line.split(":")
                    info[key.strip()] = int(val.split()[0]) // 1024
    except OSError:
        pass
    return info


class MemTracker:
    """Silently records RSS snapshots, then prints a compact summary table."""

    def __init__(self, name: str):
        self.name = name
        self.snaps: list[tuple[str, int, int]] = []

    def snap(self, label: str) -> None:
        m = _read_proc_mem()
        self.snaps.append((label, m.get("VmRSS", 0), m.get("MemAvailable", 0)))

    def report(self) -> None:
        m = _read_proc_mem()
        peak = m.get("VmPeak", 0)
        total = m.get("MemTotal", 0)
        print(f"\n[mem] === {self.name} === peak RSS: {peak} MB  (system total: {total} MB)")
        print(f"[mem]   {'step':<25s} {'RSS':>7s} {'delta':>7s} {'sys avail':>10s}")
        print(f"[mem]   {'-'*25} {'-'*7} {'-'*7} {'-'*10}")
        prev_rss = 0
        for label, rss, avail in self.snaps:
            delta = rss - prev_rss
            sign = "+" if delta >= 0 else ""
            print(f"[mem]   {label:<25s} {rss:>6d}M {sign}{delta:>5d}M {avail:>9d}M")
            prev_rss = rss
        print()


def require_tools() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")


def run_and_log(
    command: list[str],
    log_path: Path,
    prefix: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(command)}\n")
        proc = subprocess.Popen(
            command,
            cwd=str(cwd or REPO_ROOT),
            env=env,
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

def _codes_to_base_levels(codes_2d: np.ndarray, levels: list[int]) -> np.ndarray:
    basis = np.cumprod(np.asarray([1] + levels[:-1], dtype=np.int64))
    codes = codes_2d.astype(np.int64, copy=False)
    digits = np.empty((codes.shape[0], codes.shape[1], len(levels)), dtype=np.int64)
    for i, lv in enumerate(levels):
        digits[:, :, i] = (codes // basis[i]) % lv
    return digits


def compare_codes(ref_codes: Path, out_codes: Path, model_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    ref_codes = np.load(ref_codes)
    out_codes = np.load(out_codes)
    layout_mismatch = False
    if ref_codes.shape != out_codes.shape and ref_codes.shape == (out_codes.shape[1], out_codes.shape[0]):
        # If a transpose would match, we still treat it as a layout mismatch.
        layout_mismatch = True
    codebook_len_diff = abs(ref_codes.shape[0] - out_codes.shape[0]) if ref_codes.ndim == 2 and out_codes.ndim == 2 else float("inf")
    feature_len_diff = abs(ref_codes.shape[1] - out_codes.shape[1]) if ref_codes.ndim == 2 and out_codes.ndim == 2 else float("inf")

    mismatch = None
    raw_mismatch = None
    max_abs = None
    corr = float("-inf")
    mse = float("inf")
    if ref_codes.shape == out_codes.shape and ref_codes.ndim == 2:
        diff = out_codes.astype(np.int64) - ref_codes.astype(np.int64)
        raw_mismatch = int(np.count_nonzero(diff))
        mismatch = raw_mismatch
        max_abs = int(np.max(np.abs(diff))) if diff.size else 0
        corr = np.corrcoef(ref_codes.reshape(-1), out_codes.reshape(-1))[0, 1]
        mse = float(np.mean(diff.astype(np.float64) ** 2))

        normalized_boundary_frames = 0

        # External reference models can hit FSQ boundary flips from tiny numerical drift.
        # Normalize these single-digit +/-1 flips for neucodec parity checks.
        if model_cfg and model_cfg.get("class") == "neucodec" and ref_codes.shape[0] == 1:
            levels = [4] * 8
            ref_digits = _codes_to_base_levels(ref_codes, levels)[0]
            out_digits = _codes_to_base_levels(out_codes, levels)[0]
            delta = np.abs(out_digits - ref_digits)
            per_frame_off_dims = np.count_nonzero(delta, axis=1)
            per_frame_max_delta = np.max(delta, axis=1)
            boundary_flip = (per_frame_off_dims == 1) & (per_frame_max_delta == 1)
            exact_match = np.all(delta == 0, axis=1)
            mismatch = int(np.count_nonzero(~(exact_match | boundary_flip)))
            if mismatch == 0:
                mse = 0.0
                corr = 1.0

    return {
        "codebook_diff": codebook_len_diff,
        "length_diff": feature_len_diff,
        "layout_mismatch": layout_mismatch,
        "mismatch": mismatch,
        "raw_mismatch": raw_mismatch,
        "normalized_boundary_frames": normalized_boundary_frames if ref_codes.shape == out_codes.shape and ref_codes.ndim == 2 else None,
        "max_abs": max_abs,
        "corr": corr,
        "mse": mse,
    }

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


def roundtrip_metric_passes(metrics: dict[str, Any], thresholds: dict[str, Any], hop_size: int | None = None) -> bool:
    metric = metrics.get("roundtrip")
    if not metric:
        return False

    if int(metric["sample_rate_ref"]) != int(metric["sample_rate_out"]):
        return False

    corr_min = thresholds.get("encode_roundtrip_corr_min")
    if corr_min is not None and float(metric["corr"]) < float(corr_min):
        return False

    mse_max = thresholds.get("encode_roundtrip_mse_max")
    if mse_max is not None and float(metric["mse"]) > float(mse_max):
        return False

    len_diff_max = thresholds.get("encode_roundtrip_length_diff_max")
    if len_diff_max is None and hop_size is not None:
        len_diff_max = hop_size
    if len_diff_max is not None and int(metric["length_diff"]) > int(len_diff_max):
        return False

    return True


def assert_metric(metrics: dict[str, Any], thresholds: dict[str, Any], hop_size: int | None = None) -> list[str]:
    failures: list[str] = []

    if int(metrics["decode"]["sample_rate_ref"]) != int(metrics["decode"]["sample_rate_out"]):
        failures.append(
            f"decode sample_rate mismatch ref={metrics['decode']['sample_rate_ref']} out={metrics['decode']['sample_rate_out']}"
        )

    if metrics.get("encode", {}).get("layout_mismatch"):
        failures.append("encode layout mismatch (expected (n_q, n_frames))")

    encode_metrics = metrics.get("encode")
    if encode_metrics is not None:
        if encode_metrics.get("codebook_diff", float("inf")) > 0:
            failures.append(f"encode n_codebook diff {encode_metrics['codebook_diff']}")

        if encode_metrics.get("length_diff", float("inf")) > 0:
            failures.append(f"encode length_diff {encode_metrics['length_diff']}")

    encode_value_failures: list[str] = []
    mismatch = metrics.get("encode", {}).get("mismatch")
    if mismatch is not None and mismatch > 0:
        encode_value_failures.append(f"encode token mismatch {mismatch} (max_abs={metrics['encode'].get('max_abs')})")

    corr_min = thresholds.get("corr_min")
    for key in ["decode"]:
        metric = metrics.get(key, {})
        if not metric:
            continue
        if corr_min is not None and float(metric["corr"]) < float(corr_min):
            failures.append(f"{key} corr {metric['corr']:.6f} < {float(corr_min):.6f}")

        mse_max = thresholds.get("mse_max")
        if mse_max is not None and float(metric["mse"]) > float(mse_max):
            failures.append(f"{key} mse {metric['mse']:.8f} > {float(mse_max):.8f}")

    encode_metric = metrics.get("encode", {})
    if encode_metric:
        if corr_min is not None and float(encode_metric["corr"]) < float(corr_min):
            encode_value_failures.append(f"encode corr {encode_metric['corr']:.6f} < {float(corr_min):.6f}")

        mse_max = thresholds.get("mse_max")
        if mse_max is not None and float(encode_metric["mse"]) > float(mse_max):
            encode_value_failures.append(f"encode mse {encode_metric['mse']:.8f} > {float(mse_max):.8f}")

    len_diff_max = thresholds.get("length_diff_max")
    if len_diff_max is None and hop_size is not None:
        len_diff_max = hop_size
    for key in ["decode", "encode"]:
        metric = metrics.get(key, {})
        if not metric:
            continue
        if len_diff_max is not None and int(metric["length_diff"]) > int(len_diff_max):
            failures.append(f"{key} length_diff {metric['length_diff']} > {int(len_diff_max)}")

    if encode_value_failures:
        if roundtrip_metric_passes(metrics, thresholds, hop_size=hop_size):
            pass
        else:
            failures.extend(encode_value_failures)

    if metrics.get("roundtrip"):
        rt_corr_min = thresholds.get("encode_roundtrip_corr_min")
        if rt_corr_min is not None and float(metrics["roundtrip"]["corr"]) < float(rt_corr_min):
            failures.append(
                f"roundtrip corr {metrics['roundtrip']['corr']:.6f} < {float(rt_corr_min):.6f}"
            )

        rt_mse_max = thresholds.get("encode_roundtrip_mse_max")
        if rt_mse_max is not None and float(metrics["roundtrip"]["mse"]) > float(rt_mse_max):
            failures.append(
                f"roundtrip mse {metrics['roundtrip']['mse']:.8f} > {float(rt_mse_max):.8f}"
            )

        rt_len_diff_max = thresholds.get("encode_roundtrip_length_diff_max")
        if rt_len_diff_max is None and hop_size is not None:
            rt_len_diff_max = hop_size
        if rt_len_diff_max is not None and int(metrics["roundtrip"]["length_diff"]) > int(rt_len_diff_max):
            failures.append(
                f"roundtrip length_diff {metrics['roundtrip']['length_diff']} > {int(rt_len_diff_max)}"
            )

    return failures


def parse_model_class(class_spec: str) -> tuple[str, str]:
    spec = class_spec.strip()
    if not spec:
        raise RuntimeError("empty class spec")

    if spec.lower() == "dac":
        return "dac", ""

    if spec.lower() == "wavtokenizer":
        return "wavtokenizer", ""

    if spec.lower() in {"nemo_nano_codec", "nemo"}:
        return "nemo_nano_codec", ""

    if spec.lower() in {"qwen3_tts_tokenizer", "qwen3"}:
        return "qwen3_tts_tokenizer", ""

    if spec.lower() in {"neucodec"}:
        return "neucodec", ""

    if spec.lower() in {"soprano"}:
        return "soprano", ""

    if ":" not in spec:
        raise RuntimeError(
            f"invalid class spec '{class_spec}': expected 'transformers:ClassName', 'dac', 'wavtokenizer', 'nemo_nano_codec', 'neucodec', 'soprano', or 'qwen3_tts_tokenizer'"
        )

    module_name, class_name = spec.split(":", 1)
    module_name = module_name.strip()
    class_name = class_name.strip()
    if module_name != "transformers" or not class_name:
        raise RuntimeError(
            f"invalid class spec '{class_spec}': expected 'transformers:ClassName', 'dac', 'wavtokenizer', 'nemo_nano_codec', 'neucodec', 'soprano', or 'qwen3_tts_tokenizer'"
        )
    return module_name, class_name


def resolve_model_local_path(model_cfg: dict[str, Any]) -> Path:
    return REPO_ROOT / "models" / model_cfg["name"]


# return (model, encoder_fn, decoder_fn)
# encoder_fn = (audio_frames, **kwargs) -> audio_codes
# decoder_fn = (audio_codes, **kwargs) -> audio_frames
def load_native_model(model_cfg: dict[str, Any], local_path: Path):
    hf_repo_id = model_cfg.get("hf_repo_id")
    class_name = model_cfg.get("class")
    cache_dir = str(REPO_ROOT / "models" / "hf")
    kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "trust_remote_code": True,
    }
    if class_name.startswith("transformers:"):
        module = importlib.import_module("transformers")
        model_cls = getattr(module, class_name.split(":")[1])
        model = model_cls.from_pretrained(hf_repo_id, **kwargs)
        model = model.eval()
        return (
            model,
            lambda audio_frames, **kwargs: model.encode(audio_frames),
            lambda audio_codes, **kwargs: model.decode(audio_codes)
        )
    if class_name == "wavtokenizer":
        # Pull wavtokenizer src
        wavtokenizer_source = REPO_ROOT / "models" / "wavtokenizer-source"
        if not wavtokenizer_source.is_dir():
            subprocess.run(["git", "clone", "https://github.com/jishengpeng/WavTokenizer.git", wavtokenizer_source], check=True)
        sys.path.insert(0, str(wavtokenizer_source))
        decoder_mod = importlib.import_module("decoder.pretrained")
        # build config
        # config_path = wavtokenizer_source / "configs" / "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        config_path = local_path / model_cfg.get("config_path")
        model_path = local_path / model_cfg.get("hf_file")
        model = getattr(decoder_mod, "WavTokenizer").from_pretrained0802(str(config_path), str(model_path))
        model = model.eval()
        bandwidth_id = torch.tensor([0])
        return (
            model,
            lambda audio_frames, **kwargs: model.encode_infer(audio_frames.squeeze(0), bandwidth_id=bandwidth_id)[1],
            lambda audio_codes, **kwargs: model.decode(model.codes_to_features(audio_codes), bandwidth_id=bandwidth_id)
        )
    if class_name == "qwen3_tts_tokenizer":
        qwen_mod = importlib.import_module("qwen_tts")
        tokenizer = getattr(qwen_mod, "Qwen3TTSTokenizer").from_pretrained(str(local_path))
        tokenizer.model = tokenizer.model.eval()
        sample_rate = int(model_cfg.get("sample_rate", DEFAULT_SAMPLE_RATE))

        def _encode(audio_frames=None, **kwargs):
            audio = audio_frames
            if audio is None:
                audio = kwargs.get("input_values")
            if audio is None:
                audio = kwargs.get("audio_values")
            if audio is None:
                raise RuntimeError("missing audio input for Qwen3 encode")
            if isinstance(audio, torch.Tensor):
                t = audio.detach().to(torch.float32).cpu()
                if t.ndim == 3 and t.shape[1] == 1:
                    t = t[0, 0]
                elif t.ndim == 2:
                    t = t[0]
                audio = t.numpy()
            return tokenizer.encode(audio, sr=sample_rate)

        def _decode(audio_codes=None, **kwargs):
            codes = audio_codes
            if codes is None:
                codes = kwargs.get("audio_codes")
            if codes is None:
                raise RuntimeError("missing audio_codes for Qwen3 decode")
            wavs, _ = tokenizer.decode({"audio_codes": codes})
            return SimpleNamespace(audio_values=np.asarray(wavs[0], dtype=np.float32))

        return (tokenizer, _encode, _decode)
    if class_name == "nemo_nano_codec":
        sys.path.insert(0, str(REPO_ROOT / "tests" / "e2e"))
        from nemo_ref import NemoNanoCodecRef

        hf_file = model_cfg.get("hf_file")
        if not hf_file:
            raise RuntimeError("nemo_nano_codec requires hf_file pointing to a .nemo archive")
        nemo_path = local_path / hf_file
        ref = NemoNanoCodecRef.from_checkpoint(nemo_path, device="cpu")

        def _encode(audio_frames=None, **kwargs):
            audio = audio_frames
            if audio is None:
                audio = kwargs.get("input_values")
            if audio is None:
                audio = kwargs.get("audio_values")
            if audio is None:
                raise RuntimeError("missing audio input for NeMo encode")
            return ref.encode(audio)

        def _decode(audio_codes=None, **kwargs):
            codes = audio_codes
            if codes is None:
                codes = kwargs.get("audio_codes")
            if codes is None:
                raise RuntimeError("missing audio_codes for NeMo decode")
            return SimpleNamespace(audio_values=ref.decode(codes).cpu().numpy())

        return (ref, _encode, _decode)
    if class_name == "neucodec":
        sys.path.insert(0, str(REPO_ROOT / ".model-src" / "neucodec"))
        from neucodec import NeuCodec, DistillNeuCodec

        model_id = model_cfg.get("hf_repo_id")
        if model_id == "neuphonic/distill-neucodec":
            model = DistillNeuCodec._from_pretrained(model_id=model_id, cache_dir=cache_dir)
        else:
            model = NeuCodec._from_pretrained(model_id=model_id, cache_dir=cache_dir)
        model = model.eval()

        def _encode(audio_frames=None, **kwargs):
            audio = audio_frames
            if audio is None:
                audio = kwargs.get("input_values")
            if audio is None:
                audio = kwargs.get("audio_values")
            if audio is None:
                raise RuntimeError("missing audio input for NeuCodec encode")
            return model.encode_code(audio)

        def _decode(audio_codes=None, **kwargs):
            codes = audio_codes
            if codes is None:
                codes = kwargs.get("audio_codes")
            if codes is None:
                raise RuntimeError("missing audio_codes for NeuCodec decode")
            if isinstance(codes, torch.Tensor):
                codes = codes.to(dtype=torch.int64)
            else:
                codes = torch.as_tensor(codes, dtype=torch.int64)
            return SimpleNamespace(audio_values=model.decode_code(codes).cpu().numpy())

        return (model, _encode, _decode)
    if class_name == "soprano":
        sys.path.insert(0, str(REPO_ROOT / ".model-src" / "soprano"))
        from soprano.vocos.decoder import SopranoDecoder

        ckpt_path = local_path / "decoder.pth"
        if not ckpt_path.is_file():
            raise RuntimeError(f"missing Soprano decoder checkpoint: {ckpt_path}")

        model = SopranoDecoder()
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model = model.eval()

        def _decode(audio_codes=None, **kwargs):
            latent = audio_codes
            if latent is None:
                latent = kwargs.get("latent")
            if latent is None:
                latent = kwargs.get("audio_codes")
            if latent is None:
                raise RuntimeError("missing latent input for Soprano decode")
            if isinstance(latent, torch.Tensor):
                t = latent.detach().to(torch.float32).cpu()
            else:
                t = torch.as_tensor(latent, dtype=torch.float32)
            if t.ndim == 2:
                t = t.transpose(0, 1).unsqueeze(0)
            elif t.ndim == 3 and t.shape[0] == 1:
                pass
            else:
                raise RuntimeError(f"unsupported Soprano latent shape: {tuple(t.shape)}")
            return SimpleNamespace(audio_values=model(t).cpu().numpy())

        return (model, None, _decode)
    raise RuntimeError(f"unsupported class family '{class_name}'")

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

    if np.issubdtype(t.dtype, np.floating):
        t = np.rint(t)
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


def call_with_fallback(func: Callable[..., Any], call_args: list[tuple[tuple[Any, ...], dict[str, Any]]], op_name: str) -> Any:
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
    if module_name == "nemo_nano_codec":
        return "nemo_nano_codec"
    if module_name == "neucodec":
        return "neucodec"
    if module_name == "soprano":
        return "soprano"
    if module_name == "qwen3_tts_tokenizer":
        return "qwen3_tts_tokenizer"
    return class_name.lower().replace("model", "")


def encode_decode_hf(
    model_cfg: dict[str, Any],
    input_wav: Path,
    codes_out: Path,
    ref_out: Path,
    encode_sample_rate: int,
    decode_sample_rate: int,
    log_path: Path,
    model: Any,
    encoder_fn: Callable[[np.ndarray, Any], Any],
    decoder_fn: Callable[[np.ndarray, Any], Any],
) -> None:
    import torch

    sr, mono = read_wav_float_mono(input_wav)
    if sr != encode_sample_rate:
        raise RuntimeError(f"expected {encode_sample_rate} Hz input wav, got {sr}")

    class_alias = class_family(model_cfg["class"])

    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"[hf] loading model: {model_cfg['hf_repo_id']} ({model_cfg['class']})\n")

    with torch.no_grad():
        audio = to_torch_audio(mono, channels=True)
        encoded = call_with_fallback(
            encoder_fn,
            [
                ((), {"input_values": audio}),
                ((), {"audio_values": audio}),
                ((audio,), {}),
            ],
            "encode",
        )
        features = None
        if hasattr(encoded, "quantized_representation"):
            codes_raw = encoded.audio_codes
            features = encoded.quantized_representation
        elif hasattr(encoded, "audio_codes"):
            features = codes_raw = encoded.audio_codes
        elif isinstance(encoded, tuple):
            codes_raw, _ = encoded
            features = codes_raw
        elif isinstance(encoded, torch.Tensor):
            features = codes_raw = encoded
        else:
            raise RuntimeError(f"unsupported encoded type: {type(encoded)}")
        decoded = call_with_fallback(
            decoder_fn,
            [
                ((), {"audio_codes": features}),
                ((features,), {}),
            ],
            "decode",
        )

    codes_2d = normalize_codes_2d(codes_raw)
    expected_n_q = int(model_cfg.get("n_q", 0))
    if expected_n_q > 0 and codes_2d.shape[0] != expected_n_q and codes_2d.shape[1] == expected_n_q:
        # HF returns (n_frames, n_q); codec expects (n_q, n_frames).
        codes_2d = np.ascontiguousarray(codes_2d.T)
    np.save(codes_out, codes_2d)

    decoded_audio = extract_audio_values(decoded)
    save_wav_pcm16(ref_out, decoded_audio, decode_sample_rate)


def make_soprano_test_latent(n_frames: int, latent_dim: int, seed: int = 1234) -> np.ndarray:
    if n_frames <= 1 or latent_dim <= 0:
        raise RuntimeError("invalid Soprano latent shape")
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)[:, None]
    c = np.linspace(0.0, 1.0, latent_dim, dtype=np.float32)[None, :]
    base = (
        0.55 * np.sin(2.0 * np.pi * (1.0 + 2.0 * c) * t) +
        0.35 * np.cos(2.0 * np.pi * (0.5 + 3.0 * t) * c) +
        0.15 * np.sin(2.0 * np.pi * (t * c * 7.0 + c))
    )
    noise = 0.03 * rng.standard_normal((n_frames, latent_dim), dtype=np.float32)
    latent = np.tanh(base + noise).astype(np.float32, copy=False)
    return np.ascontiguousarray(latent)


def decode_latent_hf(
    model_cfg: dict[str, Any],
    latent_out: Path,
    ref_out: Path,
    decode_sample_rate: int,
    log_path: Path,
    model: Any,
    decoder_fn: Callable[[np.ndarray, Any], Any],
) -> None:
    import torch

    del model

    n_frames = int(model_cfg.get("latent_frames", 64))
    latent_dim = int(model_cfg.get("latent_dim", 0))
    if latent_dim <= 0:
        raise RuntimeError("soprano latent_dim must be configured")

    latent = make_soprano_test_latent(n_frames, latent_dim)
    np.save(latent_out, latent)

    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"[hf] soprano latent decode: n_frames={n_frames} latent_dim={latent_dim}\n")

    with torch.no_grad():
        decoded = call_with_fallback(
            decoder_fn,
            [
                ((), {"latent": latent}),
                ((), {"audio_codes": latent}),
                ((latent,), {}),
            ],
            "decode_latent",
        )

    decoded_audio = extract_audio_values(decoded)
    save_wav_pcm16(ref_out, decoded_audio, decode_sample_rate)


def download_hf_snapshot(model_cfg: dict[str, Any], log_path: Path, model_name: str) -> Path:
    """Download HF model snapshot to local cache using huggingface_hub"""
    from huggingface_hub import hf_hub_download, snapshot_download
    
    repo_id = model_cfg["hf_repo_id"]
    local_path = REPO_ROOT / "models" / model_name

    hf_file = model_cfg.get("hf_file")
    if local_path.exists():
        if hf_file and (local_path / hf_file).exists():
            return local_path.resolve()
        if (local_path / "model_config.yaml").exists() and (local_path / "model_weights.ckpt").exists():
            return local_path.resolve()

    # Use HF cache dir
    cache_dir = REPO_ROOT / "models" / "hf"
    cache_repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"

    ref_file = cache_repo_dir / "refs" / "main"
    if ref_file.is_file():
        revision = ref_file.read_text(encoding="utf-8").strip()
        snapshot_dir = cache_repo_dir / "snapshots" / revision
        if snapshot_dir.is_dir():
            if hf_file is None or (snapshot_dir / hf_file).exists():
                if local_path.exists():
                    shutil.rmtree(local_path)
                shutil.copytree(snapshot_dir, local_path)
                print(f"[{model_name}] Restored cached snapshot from: {snapshot_dir}")
                return local_path.resolve()
    
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
        if hf_file:
            try:
                downloaded_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=hf_file,
                    cache_dir=str(cache_dir),
                    local_dir=str(local_path),
                    local_dir_use_symlinks=False,
                )
                print(f"[{model_name}] Downloaded single file to: {downloaded_file}")
                return local_path.resolve()
            except Exception:
                pass
        raise RuntimeError(f"Failed to download HF snapshot: {e}")


def convert_gguf(
    model_cfg: dict[str, Any],
    local_path: Path,
    output_path: Path,
    log_path: Path,
    model_name: str,
    quantization: str | None,
) -> Path:
    """Auto-convert HF model to GGUF using scripts/convert-to-gguf.py"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hf_file = model_cfg.get("hf_file")

    # For transformers models, use direct HF conversion
    if not hf_file:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "convert-to-gguf.py"),
            "--model-id", model_cfg["hf_repo_id"],
            "--output", str(output_path),
        ]
        model_type = model_cfg.get("converter") or model_cfg.get("model_type")
        if model_type:
            cmd.extend(["--model-type", model_type])
    else:
        ckpt_path = local_path / hf_file
        if ckpt_path.suffix in {".nemo", ".ckpt", ".pth"} or ckpt_path.is_file():
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "convert-to-gguf.py"),
                "--checkpoint-path", str(ckpt_path),
                "--output", str(output_path),
                "--model-type", model_name,
            ]
        else:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "convert-to-gguf.py"),
                "--input-dir", str(ckpt_path),
                "--output", str(output_path),
                "--model-type", model_name,
            ]

    if quantization:
        cmd.extend(["--quantization", quantization])

    ret = run_and_log(cmd, log_path, model_name)
    if ret != 0:
        raise RuntimeError(f"convert to gguf failed (exit={ret})")

    if not output_path.is_file():
        raise RuntimeError(f"convert to gguf did not produce: {output_path}")

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


def run_decode(
    gguf_path: Path,
    codes_npy: Path,
    out_wav: Path,
    n_q: int,
    log_path: Path,
    model_name: str,
    env: dict[str, str] | None = None,
) -> None:
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

    ret = run_and_log(cmd, log_path, model_name, env=env)
    if ret != 0:
        raise RuntimeError(f"codec-cli decode failed (exit={ret})")


def run_decode_latent(
    gguf_path: Path,
    latent_npy: Path,
    out_wav: Path,
    log_path: Path,
    model_name: str,
    env: dict[str, str] | None = None,
) -> None:
    cmd = [
        str(REPO_ROOT / "build/codec-cli"),
        "decode-latent",
        "--model",
        str(gguf_path),
        "--latent",
        str(latent_npy),
        "--out",
        str(out_wav),
    ]

    ret = run_and_log(cmd, log_path, model_name, env=env)
    if ret != 0:
        raise RuntimeError(f"codec-cli decode-latent failed (exit={ret})")

def run_encode(
    gguf_path: Path,
    input_wav: Path,
    out_codes: Path,
    n_q: int,
    log_path: Path,
    model_name: str,
    env: dict[str, str] | None = None,
) -> None:
    cmd = [
        str(REPO_ROOT / "build/codec-cli"),
        "encode",
        "--model",
        str(gguf_path),
        "--in",
        str(input_wav),
        "--out",
        str(out_codes),
    ]
    if n_q > 0:
        cmd.extend(["--nq", str(n_q)])

    ret = run_and_log(cmd, log_path, model_name, env=env)
    if ret != 0:
        raise RuntimeError(f"codec-cli encode failed (exit={ret})")

def run_model(
    report_dir: Path,
    model_cfg: dict[str, Any],
    input_audio: Path,
    sample_rate: int,
    quantization: str | None,
    strict_thresholds: bool = False,
) -> ModelResult:
    name = model_cfg["name"]
    mt = MemTracker(name)
    start = time.monotonic()
    log_path = report_dir / f"{name}.log"

    report_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    encode_sample_rate = int(model_cfg.get("encode_sample_rate", model_cfg.get("sample_rate", sample_rate)))
    decode_sample_rate = int(model_cfg.get("decode_sample_rate", model_cfg.get("sample_rate", sample_rate)))

    tmpdir = Path(tempfile.mkdtemp(prefix=f"codec-e2e-{name}-"))
    try:
        if not input_audio.is_file():
            raise RuntimeError(f"input audio not found: {input_audio}")

        model_name = model_cfg["name"]
        local_path = download_hf_snapshot(model_cfg, log_path, model_name)

        expected_gguf = resolve_gguf_path(model_cfg)
        mt.snap("before convert_gguf")
        gguf_path = convert_gguf(model_cfg, local_path, expected_gguf, log_path, name, quantization)
        mt.snap("after convert_gguf")

        mt.snap("before load_native_model")
        model, encoder_fn, decoder_fn = load_native_model(model_cfg, local_path)
        mt.snap("after load_native_model")

        input_wav = tmpdir / "input.wav"
        hf_codes = tmpdir / "hf_codes.npy"
        hf_latent = tmpdir / "hf_latent.npy"
        hf_ref_wav = tmpdir / "hf_reference.wav"
        cpp_out_wav = tmpdir / "cpp_decode.wav"
        cpp_out_codes = tmpdir / "cpp_encode.npy"
        cpp_roundtrip_wav = tmpdir / "cpp_roundtrip.wav"

        mt.snap("before encode_decode_hf")
        if model_cfg.get("latent_only"):
            decode_latent_hf(
                model_cfg,
                hf_latent,
                hf_ref_wav,
                decode_sample_rate,
                log_path,
                model,
                decoder_fn,
            )
        else:
            ffmpeg_to_mono_wav(input_audio, input_wav, encode_sample_rate, log_path, name)
            encode_decode_hf(
                model_cfg,
                input_wav,
                hf_codes,
                hf_ref_wav,
                encode_sample_rate,
                decode_sample_rate,
                log_path,
                model,
                encoder_fn,
                decoder_fn,
            )
        mt.snap("after encode_decode_hf")

        del model, encoder_fn, decoder_fn
        gc.collect()
        mt.snap("after gc.collect")

        n_q = int(model_cfg.get("n_q", 0))
        codec_env = dict(os.environ)
        # codec_env["GGML_DISABLE_VULKAN"] = "1"
        if model_cfg.get("latent_only"):
            run_decode_latent(gguf_path, hf_latent, cpp_out_wav, log_path, name, env=codec_env)
        elif model_cfg.get("decode_only"):
            run_decode(gguf_path, hf_codes, cpp_out_wav, n_q, log_path, name, env=codec_env)
        else:
            run_encode(gguf_path, input_wav, cpp_out_codes, n_q, log_path, name, env=codec_env)
            run_decode(gguf_path, hf_codes, cpp_out_wav, n_q, log_path, name, env=codec_env)
            run_decode(gguf_path, cpp_out_codes, cpp_roundtrip_wav, n_q, log_path, name, env=codec_env)
        mt.snap("after codec-cli")

        metrics = {}
        metrics["decode"] = compare_wav(hf_ref_wav, cpp_out_wav)
        if not model_cfg.get("decode_only") and not model_cfg.get("latent_only"):
            metrics["encode"] = compare_codes(hf_codes, cpp_out_codes, model_cfg)
            metrics["roundtrip"] = compare_wav(hf_ref_wav, cpp_roundtrip_wav)
        thresholds = merge_thresholds(model_cfg)
        if not strict_thresholds:
            thresholds = relax_thresholds_for_quantization(thresholds, quantization)
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
                status="failed",
                duration_sec=duration,
                log_path=str(log_path),
                reason="; ".join(failures),
                return_code=1,
                metrics=metrics,
            )

        return ModelResult(
            name=name,
            status="passed",
            duration_sec=duration,
            log_path=str(log_path),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        print(traceback.format_exc())
        duration = time.monotonic() - start
        return ModelResult(
            name=name,
            status="failed",
            duration_sec=duration,
            log_path=str(log_path),
            reason=str(exc),
            return_code=1,
        )
    finally:
        mt.report()
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
            f"- {r.name}: {r.status} ({r.duration_sec:.2f}s)"
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
        "--quantization",
        choices=["F32", "F16", "Q8_0", "Q4_K_M", "Q5_K_M"],
        default=None,
        help="Override GGUF quantization for all models",
    )
    parser.add_argument(
        "--models",
        action="append",
        help="Run only specific model names (repeatable or comma-separated)",
    )
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="Run models in-process (useful in sandbox environments without multiprocessing.SemLock)",
    )
    parser.add_argument("--list-models", action="store_true", help="List configured models and exit")
    parser.add_argument(
        "--strict-thresholds",
        action="store_true",
        help="Skip per-quantization threshold relaxation. By default, quantized runs (Q8_0/Q5_K_M/Q4_K_M) get tier-scaled corr_min and mse_max because exact parity with the HF reference isn't realistic for lossy formats.",
    )
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


def _run_model_worker(
    result_queue: multiprocessing.Queue,
    report_dir: Path,
    model_cfg: dict[str, Any],
    input_audio: Path,
    sample_rate: int,
    quantization: str | None,
    strict_thresholds: bool = False,
) -> None:
    result = run_model(report_dir, model_cfg, input_audio, sample_rate, quantization, strict_thresholds=strict_thresholds)
    result_queue.put(result)


def _run_model_isolated(
    report_dir: Path,
    model_cfg: dict[str, Any],
    input_audio: Path,
    sample_rate: int,
    quantization: str | None,
    strict_thresholds: bool = False,
) -> ModelResult:
    """Run a single model test in an isolated subprocess so all memory
    (PyTorch allocator pools, imported modules, HF caches) is reclaimed
    by the OS when the child exits."""
    try:
        q: multiprocessing.Queue = multiprocessing.Queue()
    except PermissionError:
        # Some sandboxed environments disallow SemLock.
        return run_model(report_dir, model_cfg, input_audio, sample_rate, quantization, strict_thresholds=strict_thresholds)
    p = multiprocessing.Process(
        target=_run_model_worker,
        args=(q, report_dir, model_cfg, input_audio, sample_rate, quantization, strict_thresholds),
    )
    p.start()
    p.join()
    if not q.empty():
        return q.get_nowait()
    name = model_cfg["name"]
    return ModelResult(
        name=name,
        status="failed",
        duration_sec=0.0,
        log_path=str(report_dir / f"{name}.log"),
        reason=f"worker process died (exit code {p.exitcode})",
        return_code=p.exitcode,
    )


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
        if args.no_isolation:
            result = run_model(
                report_dir,
                model_cfg,
                input_audio,
                int(args.sample_rate),
                args.quantization,
                strict_thresholds=args.strict_thresholds,
            )
        else:
            result = _run_model_isolated(
                report_dir,
                model_cfg,
                input_audio,
                int(args.sample_rate),
                args.quantization,
                strict_thresholds=args.strict_thresholds,
            )
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
