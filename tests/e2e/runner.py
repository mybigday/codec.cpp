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

def compare_codes(ref_codes: Path, out_codes: Path) -> dict[str, Any]:
    ref_codes = np.load(ref_codes)
    out_codes = np.load(out_codes)
    codebook_len_diff = abs(ref_codes.shape[0] - out_codes.shape[0])
    feature_len_diff = abs(ref_codes.shape[1] - out_codes.shape[1])
    corr = float("-inf")
    mse = float("inf")
    if codebook_len_diff == 0:
        corr = np.corrcoef(ref_codes[:, :-feature_len_diff], out_codes[:, :-feature_len_diff])[0, 1]
        mse = np.mean((ref_codes[:, :-feature_len_diff] - out_codes[:, :-feature_len_diff]) ** 2)
    return {
        "codebook_diff": codebook_len_diff,
        "length_diff": feature_len_diff,
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


def assert_metric(metrics: dict[str, Any], thresholds: dict[str, Any], hop_size: int | None = None) -> list[str]:
    failures: list[str] = []

    if int(metrics["decode"]["sample_rate_ref"]) != int(metrics["decode"]["sample_rate_out"]):
        failures.append(
            f"decode sample_rate mismatch ref={metrics['decode']['sample_rate_ref']} out={metrics['decode']['sample_rate_out']}"
        )

    if metrics.get("encode", {}).get("codebook_diff", float("inf")) > 0:
        failures.append(f"encode n_codebook diff {metrics['encode']['codebook_diff']}")

    if metrics.get("encode", {}).get("length_diff", float("inf")) > 0:
        failures.append(f"encode length_diff {metrics['encode']['length_diff']}")

    corr_min = thresholds.get("corr_min")
    for key in ["decode", "encode"]:
        metric = metrics.get(key, {})
        if not metric:
            continue
        if corr_min is not None and float(metric["corr"]) < float(corr_min):
            failures.append(f"{key} corr {metric['corr']:.6f} < {float(corr_min):.6f}")

        mse_max = thresholds.get("mse_max")
        if mse_max is not None and float(metric["mse"]) > float(mse_max):
            failures.append(f"{key} mse {metric['mse']:.8f} > {float(mse_max):.8f}")

    len_diff_max = thresholds.get("length_diff_max")
    if len_diff_max is None and hop_size is not None:
        len_diff_max = hop_size
    for key in ["decode", "encode"]:
        metric = metrics.get(key, {})
        if not metric:
            continue
        if len_diff_max is not None and int(metric["length_diff"]) > int(len_diff_max):
            failures.append(f"{key} length_diff {metric['length_diff']} > {int(len_diff_max)}")

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


def resolve_model_local_path(model_cfg: dict[str, Any]) -> Path:
    local_path = model_cfg.get("local_path")
    if not local_path:
        return REPO_ROOT / "models" / model_cfg["name"]
    path = Path(local_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


# return (model, encoder_fn, decoder_fn)
# encoder_fn = (audio_frames, **kwargs) -> audio_codes
# decoder_fn = (audio_codes, **kwargs) -> audio_frames
def load_native_model(model_cfg: dict[str, Any]):
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
        model_name = model_cfg["name"]
        log_path = REPO_ROOT / "models" / model_name / f"{model_name}.log"
        local_path = download_hf_snapshot(model_cfg, log_path, model_name)
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
    return class_name.lower().replace("model", "")


def encode_decode_hf(
    model_cfg: dict[str, Any],
    input_wav: Path,
    codes_out: Path,
    ref_out: Path,
    sample_rate: int,
    log_path: Path,
    model: Any,
    encoder_fn: Callable[[np.ndarray, Any], Any],
    decoder_fn: Callable[[np.ndarray, Any], Any],
) -> None:
    import torch

    sr, mono = read_wav_float_mono(input_wav)
    if sr != sample_rate:
        raise RuntimeError(f"expected {sample_rate} Hz input wav, got {sr}")

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


def convert_gguf(model_cfg: dict[str, Any], output_path: Path, log_path: Path, model_name: str) -> Path:
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
    else:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "convert-to-gguf.py"),
            "--input-dir", str(REPO_ROOT / "models" / model_name / hf_file),
            "--output", str(output_path),
            "--model-type", model_name,
        ]

    quantization = model_cfg.get("quantization")
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

def run_encode(
    gguf_path: Path,
    input_wav: Path,
    out_codes: Path,
    n_q: int,
    log_path: Path,
    model_name: str,
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

    ret = run_and_log(cmd, log_path, model_name)
    if ret != 0:
        raise RuntimeError(f"codec-cli encode failed (exit={ret})")

def run_model(report_dir: Path, model_cfg: dict[str, Any], input_audio: Path, sample_rate: int) -> ModelResult:
    name = model_cfg["name"]
    start = time.monotonic()
    log_path = report_dir / f"{name}.log"
    model, encoder_fn, decoder_fn = load_native_model(model_cfg)

    report_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    model_sample_rate = int(model_cfg.get("sample_rate", sample_rate))

    tmpdir = Path(tempfile.mkdtemp(prefix=f"codec-e2e-{name}-"))
    try:
        if not input_audio.is_file():
            raise RuntimeError(f"input audio not found: {input_audio}")

        expected_gguf = resolve_gguf_path(model_cfg)
        gguf_path = convert_gguf(model_cfg, expected_gguf, log_path, name)

        input_wav = tmpdir / "input.wav"
        hf_codes = tmpdir / "hf_codes.npy"
        hf_ref_wav = tmpdir / "hf_reference.wav"
        cpp_out_wav = tmpdir / "cpp_decode.wav"
        cpp_out_codes = tmpdir / "cpp_encode.npy"

        ffmpeg_to_mono_wav(input_audio, input_wav, model_sample_rate, log_path, name)
        encode_decode_hf(model_cfg, input_wav, hf_codes, hf_ref_wav, model_sample_rate, log_path, model, encoder_fn, decoder_fn)

        n_q = int(model_cfg.get("n_q", 0))
        if model_cfg.get("decode_only"):
            run_decode(gguf_path, hf_codes, cpp_out_wav, n_q, log_path, name)
        else:
            run_encode(gguf_path, input_wav, cpp_out_codes, n_q, log_path, name)
            run_decode(gguf_path, hf_codes, cpp_out_wav, n_q, log_path, name)

        metrics = {}
        metrics["decode"] = compare_wav(hf_ref_wav, cpp_out_wav)
        if not  model_cfg.get("decode_only"):
            metrics["encode"] = compare_codes(hf_codes, cpp_out_codes)
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
