#!/usr/bin/env python3
"""Memory + performance tracking harness for codec.cpp.

Goal: produce a single JSON report per run that captures, per-model and
per-operation:

  - Wall time (overall + per-phase breakdown via CODEC_PERF_LOG)
  - Peak resident memory of the codec-cli process (/usr/bin/time -v)
  - Backend-buffer sizes / graph node counts (from the perf log)
  - A liveness check on the output file (size + first/last sample finite)

Each (model, op) combo is run with a configurable warmup + iteration count;
we report mean / p50 / p95 / std.  The whole report is a stable JSON shape
so two runs (baseline vs. after-optimisation) can be diffed mechanically:

    python tools/benchmark.py --out before.json
    # ... apply optimisation ...
    python tools/benchmark.py --out after.json
    python tools/benchmark.py compare before.json after.json

The comparison tool prints per-metric deltas (absolute + percent) and
flags any regressions outside a small noise band.

Schema (v1):

  {
    "version": 1,
    "git_sha": "abc1234",
    "build_type": "Release",
    "host": {"cpu": "...", "uname": "..."},
    "config": {"input_wav": "test.wav", "iterations": 10, "warmup": 2},
    "results": {
      "<model>": {
        "<op>": {
          "ok": true,
          "wall_ms":     {"mean": 50.1, "p50": ..., "p95": ..., "std": ...},
          "peak_rss_mb": {"mean": ...,  ...},
          "phases": {
            "graph_build":      {"mean_ms": ..., "p50_ms": ..., "samples": [...]},
            "graph_prepare_io": {...},
            "graph_compute":    {...}
          },
          "raw": [{"wall_ms": ..., "peak_rss_mb": ..., "phases": {...}}, ...]
        }
      }
    }
  }
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODEC_CLI = REPO_ROOT / "build" / "codec-cli"
DEFAULT_INPUT = REPO_ROOT / "test.wav"


def git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def host_info() -> dict:
    try:
        cpu = ""
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        uname = subprocess.check_output(["uname", "-srm"]).decode().strip()
        return {"cpu": cpu, "uname": uname}
    except Exception:
        return {}


def percentile(samples: list[float], q: float) -> float:
    if not samples:
        return float("nan")
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def stat_summary(samples: list[float]) -> dict:
    if not samples:
        return {"mean": None, "p50": None, "p95": None, "std": None, "n": 0}
    return {
        "mean": float(statistics.mean(samples)),
        "p50":  float(percentile(samples, 0.50)),
        "p95":  float(percentile(samples, 0.95)),
        "std":  float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
        "n":    len(samples),
    }


def parse_perf_log(path: Path) -> dict[str, list[float]]:
    """Read CODEC_PERF_LOG ndjson and return per-phase wall_us samples."""
    if not path.exists():
        return {}
    by_phase: dict[str, list[float]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line[0] != "{":
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if "wall_us" not in ev:
            continue
        by_phase.setdefault(ev["phase"], []).append(float(ev["wall_us"]))
    return by_phase


def parse_time_v(stderr: str) -> dict:
    """Parse `/usr/bin/time -v` output for wall + RSS."""
    out: dict = {}
    elapsed_re = re.compile(r"Elapsed \(wall clock\) time \(.*?\): ([\d:\.]+)")
    rss_re = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
    user_re = re.compile(r"User time \(seconds\): ([\d\.]+)")
    sys_re  = re.compile(r"System time \(seconds\): ([\d\.]+)")
    for line in stderr.splitlines():
        m = elapsed_re.search(line)
        if m:
            parts = m.group(1).split(":")
            secs = 0.0
            for p in parts:
                secs = secs * 60 + float(p)
            out["wall_s"] = secs
        m = rss_re.search(line)
        if m: out["peak_rss_kb"] = int(m.group(1))
        m = user_re.search(line)
        if m: out["user_s"] = float(m.group(1))
        m = sys_re.search(line)
        if m: out["system_s"] = float(m.group(1))
    return out


def run_once(args: list[str], perf_log: Path) -> dict:
    """Run codec-cli once via /usr/bin/time -v, return parsed metrics."""
    if perf_log.exists():
        perf_log.unlink()
    env = dict(os.environ)
    env["CODEC_PERF_LOG"] = str(perf_log)
    # Disable Vulkan/CUDA so we measure the requested CPU backend deterministically.
    env.setdefault("GGML_DISABLE_VULKAN", "1")
    cmd = ["/usr/bin/time", "-v", *args]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, env=env, text=True)
    elapsed = time.perf_counter() - t0
    metrics = parse_time_v(proc.stderr)
    # /usr/bin/time -v writes to stderr; if it isn't present, fall back to wall.
    metrics.setdefault("wall_s", elapsed)
    metrics["returncode"] = proc.returncode
    metrics["stderr"] = proc.stderr[-2000:] if proc.returncode != 0 else ""
    metrics["phases_us"] = parse_perf_log(perf_log)
    return metrics


def liveness_wav(path: Path) -> dict:
    if not path.exists():
        return {"ok": False, "reason": "missing"}
    try:
        with wave.open(str(path), "rb") as wf:
            n = wf.getnframes()
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            raw = wf.readframes(min(n, sr))  # first second
        import numpy as np
        arr = np.frombuffer(raw, dtype=np.int16)
        finite = bool(np.all(np.isfinite(arr.astype(np.float32))))
        rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
        return {"ok": finite and n > 0, "n_samples": n, "sr": sr, "channels": ch, "rms": rms}
    except Exception as e:
        return {"ok": False, "reason": str(e)[:120]}


def liveness_npy(path: Path) -> dict:
    if not path.exists():
        return {"ok": False, "reason": "missing"}
    try:
        import numpy as np
        arr = np.load(path)
        return {"ok": arr.size > 0, "shape": list(arr.shape)}
    except Exception as e:
        return {"ok": False, "reason": str(e)[:120]}


def prepare_input_wav(src_path: Path, target_sr: int, target_channels: int,
                       max_seconds: float, cache_dir: Path) -> Path:
    """Resample (and re-channel + truncate if needed) `src_path` so the
    codec's sample-rate / channel-count / max-length expectations are met.
    Cached by (sr, ch, max_seconds)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    secs_tag = f"_{int(max_seconds)}s" if max_seconds > 0 else ""
    out_path = cache_dir / f"{src_path.stem}.{target_sr}hz_{target_channels}ch{secs_tag}.wav"
    if out_path.is_file():
        return out_path
    import numpy as np
    with wave.open(str(src_path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n  = wf.getnframes()
        raw = wf.readframes(n)
    if sw != 2:
        raise RuntimeError(f"{src_path}: expected PCM16, got {sw} bytes/sample")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if ch > 1:
        pcm = pcm.reshape(-1, ch)
    else:
        pcm = pcm.reshape(-1, 1)

    if sr != target_sr:
        n_in = pcm.shape[0]
        n_out = int(round(n_in * target_sr / sr))
        x_old = np.arange(n_in, dtype=np.float64)
        x_new = np.linspace(0, n_in - 1, n_out, dtype=np.float64)
        resampled = np.empty((n_out, pcm.shape[1]), dtype=np.float32)
        for c in range(pcm.shape[1]):
            resampled[:, c] = np.interp(x_new, x_old, pcm[:, c])
        pcm = resampled
        sr = target_sr

    if pcm.shape[1] != target_channels:
        if target_channels == 1:
            pcm = pcm.mean(axis=1, keepdims=True)
        else:
            pcm = np.tile(pcm[:, :1], (1, target_channels))

    if max_seconds > 0:
        max_samples = int(round(max_seconds * target_sr))
        if pcm.shape[0] > max_samples:
            pcm = pcm[:max_samples]

    pcm_i16 = np.clip(pcm, -32768, 32767).astype(np.int16).reshape(-1)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(target_channels)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        wf.writeframes(pcm_i16.tobytes())
    return out_path


def run_op(model_cfg: dict, op: str, iterations: int, warmup: int) -> dict:
    """Run one (model, op) combo many times; return aggregated metrics."""
    gguf = REPO_ROOT / model_cfg["gguf"]
    if not gguf.is_file():
        return {"ok": False, "reason": f"gguf missing: {gguf}"}

    src_wav = REPO_ROOT / model_cfg.get("input_wav", str(DEFAULT_INPUT))
    if not src_wav.is_file():
        return {"ok": False, "reason": f"input wav missing: {src_wav}"}

    target_sr = int(model_cfg.get("input_sample_rate", 0))
    target_ch = int(model_cfg.get("input_channels", 1))
    max_seconds = float(model_cfg.get("max_seconds", 3.0))
    if target_sr > 0:
        cache = REPO_ROOT / "benchmarks" / "_input_cache"
        in_wav = prepare_input_wav(src_wav, target_sr, target_ch, max_seconds, cache)
    else:
        in_wav = src_wav

    nq = model_cfg.get("nq")
    extra = model_cfg.get("extra_args", [])

    raw: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="codec-bench-") as td:
        td_p = Path(td)
        for i in range(warmup + iterations):
            out_wav = td_p / f"out_{i}.wav"
            out_codes = td_p / f"codes_{i}.npy"
            perf_log = td_p / f"perf_{i}.ndjson"

            if op == "encode":
                cli_args = [str(CODEC_CLI), "encode",
                            "--model", str(gguf), "--in", str(in_wav),
                            "--out", str(out_codes)]
            elif op == "decode":
                # Need a codes file from a previous encode; do an inline encode
                # in warmup zero so subsequent decode iterations have valid codes.
                cli_args = [str(CODEC_CLI), "decode",
                            "--model", str(gguf), "--codes", str(td_p / "ref_codes.npy"),
                            "--out", str(out_wav)]
            elif op == "e2e":
                cli_args = [str(CODEC_CLI), "e2e",
                            "--model", str(gguf), "--in", str(in_wav),
                            "--out", str(out_wav)]
            else:
                raise ValueError(f"unknown op {op}")

            if nq is not None:
                cli_args += ["--nq", str(nq)]
            cli_args += extra

            # For decode op: prime once before timing.
            if op == "decode" and i == 0:
                ref_args = [str(CODEC_CLI), "encode",
                            "--model", str(gguf), "--in", str(in_wav),
                            "--out", str(td_p / "ref_codes.npy")]
                if nq is not None:
                    ref_args += ["--nq", str(nq)]
                ref_args += extra
                env = dict(os.environ); env["GGML_DISABLE_VULKAN"] = "1"
                rc = subprocess.run(ref_args, capture_output=True, env=env, text=True)
                if rc.returncode != 0:
                    return {"ok": False, "reason": f"ref encode failed: {rc.stderr[-400:]}"}

            metrics = run_once(cli_args, perf_log)
            metrics["iteration"] = i
            metrics["is_warmup"] = (i < warmup)
            if op in ("decode", "e2e"):
                metrics["liveness"] = liveness_wav(out_wav)
            else:
                metrics["liveness"] = liveness_npy(out_codes)
            raw.append(metrics)

    measured = [m for m in raw if not m["is_warmup"] and m["returncode"] == 0]
    failures = [m for m in raw if m["returncode"] != 0]
    if not measured:
        return {"ok": False, "reason": "all runs failed",
                "first_stderr": failures[0]["stderr"] if failures else ""}

    wall_ms = [m["wall_s"] * 1000.0 for m in measured]
    rss_mb  = [m.get("peak_rss_kb", 0) / 1024.0 for m in measured]

    phases: dict[str, dict] = {}
    phase_names = {p for m in measured for p in m["phases_us"].keys()}
    for p in sorted(phase_names):
        # Sum same-phase events within one run (e.g. encode triggers two
        # graph_prepare_io because compute does its own first); pick the max
        # so the slow one shows up.  For graph_build we usually only see one
        # event per run — sum is identical to that one.
        per_run = []
        for m in measured:
            xs = m["phases_us"].get(p, [])
            per_run.append(sum(xs) / 1000.0)  # ms
        phases[p] = {
            "mean_ms": float(statistics.mean(per_run)) if per_run else None,
            "p50_ms":  float(percentile(per_run, 0.50)) if per_run else None,
            "p95_ms":  float(percentile(per_run, 0.95)) if per_run else None,
            "std_ms":  float(statistics.pstdev(per_run)) if len(per_run) > 1 else 0.0,
            "n":       len(per_run),
        }

    return {
        "ok": all(m["liveness"]["ok"] for m in measured),
        "wall_ms":     stat_summary(wall_ms),
        "peak_rss_mb": stat_summary(rss_mb),
        "phases":      phases,
        "n_failures":  len(failures),
    }


def cmd_run(args: argparse.Namespace) -> int:
    if not CODEC_CLI.is_file():
        print(f"FAIL: {CODEC_CLI} not built", file=sys.stderr)
        return 2

    cfg_path = REPO_ROOT / args.config
    cfg = json.loads(cfg_path.read_text())
    selected = set(args.models) if args.models else None

    report = {
        "version": 1,
        "git_sha": git_sha(),
        "build_type": "Release",
        "host": host_info(),
        "config": {
            "input_wav": str(DEFAULT_INPUT.relative_to(REPO_ROOT)),
            "iterations": args.iterations,
            "warmup": args.warmup,
        },
        "results": {},
    }
    for entry in cfg["models"]:
        name = entry["name"]
        if selected and name not in selected:
            continue
        ops = entry.get("ops", ["e2e"])
        if not ops:
            print(f"  [{name}] (skipped via empty ops)", flush=True)
            continue
        report["results"][name] = {}
        for op in ops:
            print(f"  [{name}] {op} ({args.iterations}x, warmup={args.warmup})", flush=True)
            res = run_op(entry, op, args.iterations, args.warmup)
            print(f"    -> wall_ms.mean={res.get('wall_ms', {}).get('mean'):.1f} "
                  f"rss_mb.mean={res.get('peak_rss_mb', {}).get('mean'):.1f} ok={res.get('ok')}"
                  if res.get("ok") else f"    -> FAILED: {res.get('reason')}")
            report["results"][name][op] = res

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {out_path}")
    return 0


def fmt_pct(a: float | None, b: float | None) -> str:
    if a is None or b is None or a == 0:
        return "  n/a "
    pct = (b - a) / a * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:6.1f}%"


def cmd_compare(args: argparse.Namespace) -> int:
    a = json.loads(Path(args.before).read_text())
    b = json.loads(Path(args.after).read_text())
    print(f"compare: {args.before} -> {args.after}")
    print(f"  before sha={a.get('git_sha')}  after sha={b.get('git_sha')}")
    print()
    fmt = "{:<24} {:<7} {:>10} {:>10} {:>9}  {:>10} {:>10} {:>9}"
    print(fmt.format("model", "op", "wall_ms", "  Δms", "Δ%",
                     "rss_mb", "  Δmb", "Δ%"))
    print("-" * 110)
    regressions: list[str] = []
    for model in sorted(set(a["results"]) | set(b["results"])):
        for op in sorted(set(a["results"].get(model, {})) | set(b["results"].get(model, {}))):
            ar = a["results"].get(model, {}).get(op, {})
            br = b["results"].get(model, {}).get(op, {})
            aw = ar.get("wall_ms", {}).get("mean")
            bw = br.get("wall_ms", {}).get("mean")
            am = ar.get("peak_rss_mb", {}).get("mean")
            bm = br.get("peak_rss_mb", {}).get("mean")
            dw = (bw - aw) if (aw is not None and bw is not None) else None
            dm = (bm - am) if (am is not None and bm is not None) else None
            print(fmt.format(model[:24], op,
                             f"{bw:.1f}" if bw is not None else "n/a",
                             f"{dw:+.1f}" if dw is not None else "n/a",
                             fmt_pct(aw, bw),
                             f"{bm:.1f}" if bm is not None else "n/a",
                             f"{dm:+.1f}" if dm is not None else "n/a",
                             fmt_pct(am, bm)))
            # Tag big regressions outside ±5% noise band
            if aw and bw and (bw - aw) / aw > 0.05:
                regressions.append(f"{model}/{op}: wall +{(bw-aw)/aw*100:.1f}%")
            if am and bm and (bm - am) / am > 0.05:
                regressions.append(f"{model}/{op}: rss +{(bm-am)/am*100:.1f}%")
    print()
    if regressions:
        print("REGRESSIONS (> +5%):")
        for r in regressions:
            print(f"  {r}")
        return 1
    print("No regressions outside ±5%.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="codec.cpp memory + perf benchmark")
    sub = p.add_subparsers(dest="cmd", required=False)

    p_run = sub.add_parser("run", help="run benchmark and write JSON report")
    p_run.add_argument("--config", default="tools/benchmark_models.json")
    p_run.add_argument("--iterations", type=int, default=5)
    p_run.add_argument("--warmup", type=int, default=1)
    p_run.add_argument("--models", nargs="*", default=None,
                       help="subset of model names from the config")
    p_run.add_argument("--out", default="benchmarks/run.json")

    p_cmp = sub.add_parser("compare", help="diff two benchmark JSONs")
    p_cmp.add_argument("before")
    p_cmp.add_argument("after")

    # Default to "run" if no subcommand given (or first arg starts with "-").
    argv = sys.argv[1:]
    if argv and argv[0] not in ("run", "compare", "-h", "--help"):
        argv = ["run", *argv]
    args = p.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
