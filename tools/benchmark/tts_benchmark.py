#!/usr/bin/env python3
"""End-to-end TTS timing: every `tts-cli synthesize` case of the e2e smokes.

Collects the cases of tests/e2e/ttscli_synthesize_smoke.py and
tests/e2e/cosyvoice_synthesize_smoke.py (same models, text, seeds and flags),
runs each through tts-cli and records the `timing:` line (load / generate /
decode / total), the audio duration and the Whisper CER.

    .venv/bin/python tools/benchmark/tts_benchmark.py run --out benchmarks/tts_cpu.json
    .venv/bin/python tools/benchmark/tts_benchmark.py run --gpu --out benchmarks/tts_vk.json
    .venv/bin/python tools/benchmark/tts_benchmark.py compare before.json after.json

`synth_ms` (speaker + generate + decode) is the headline; load is reported on
its own since a host pays it once.  Each case runs `--repeat` times (default 2)
and keeps the fastest.  `compare` normalises by output length (generate per
frame, decode / synth per audio second) and exits non-zero when a case fails or
its CER rises by more than CER_SLACK.  `--only a,b` restricts the run to the
named cases, `--no-cer` skips Whisper.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "e2e"))

import ttscli_synthesize_smoke as tts_smoke  # noqa: E402
import cosyvoice_synthesize_smoke as cv_smoke  # noqa: E402

TIMING_RE = re.compile(r"timing: load (-?\d+) ms, speaker (\d+) ms, generate (-?\d+) ms, decode (\d+) ms, "
                       r"total (\d+) ms, frames (\d+)")
CER_SLACK = 0.1  # compare: a case regresses when its CER rises by more than this


def collect_cases() -> list[dict]:
    cases: list[dict] = []

    def tts_case(name, codec, backbone, text, lang, want_stop, cer_max, extra=None, greedy=True, **_):
        cmd = ["--model", str(codec), "--text", text]
        if backbone is not None:
            cmd += ["--backbone", str(backbone)]
        if greedy and backbone is not None:
            cmd += ["--temp", "0"]
        cmd += list(extra or [])
        assets = [codec] + ([backbone] if backbone is not None else [])
        cases.append({"name": name, "args": cmd, "lang": lang, "text": text, "assets": assets})
        return True

    def cv_case(label, name, text, lang, extra, cer_max, tmp):
        codec, backbone = cv_smoke.MODELS / f"{name}.gguf", cv_smoke.MODELS / f"{name}-backbone.gguf"
        cmd = ["--model", str(codec), "--backbone", str(backbone), "--text", text, "--seed", "1"] + list(extra)
        cases.append({"name": label, "args": cmd, "lang": lang, "text": text, "assets": [codec, backbone]})
        return True

    class _NotRun:
        returncode, stdout, stderr = 1, "", ""

    def run_synth(codec, backbone, text, out_wav, extra=None, greedy=True):
        # Cases that call _run_synth directly (OuteTTS, NeuTTS): their text may be
        # phonemes, so no CER.
        name = Path(out_wav).stem.removeprefix("ttscli_")
        tts_case(name, codec, backbone, text, None, None, None, extra=extra, greedy=greedy)
        return _NotRun()

    tts_smoke._case = tts_case
    tts_smoke._run_synth = run_synth
    tts_smoke.TTS_CLI = Path(sys.executable)  # main() only checks that it exists
    with open(os.devnull, "w") as devnull:
        stdout, sys.stdout = sys.stdout, devnull
        try:
            tts_smoke.main()
            cv_smoke.case = cv_case
            cv_smoke.main()
        finally:
            sys.stdout = stdout
    return [c for c in cases if all(Path(a).exists() for a in c["assets"])]


def swap_file(args: list[str], swap: str | None, flag: str = "--backbone") -> list[str]:
    """--backbone-swap / --codec-swap DIR:TAG: use DIR/<model dir>__<stem>-<TAG>.gguf where it exists."""
    if not swap or flag not in args:
        return args
    d, tag = swap.rsplit(":", 1)
    i = args.index(flag) + 1
    bb = Path(args[i])
    alt = Path(d) / f"{bb.parent.name}__{bb.stem}-{tag}.gguf"
    return args[:i] + [str(alt)] + args[i + 1:] if alt.exists() else args


def run_case(c: dict, n_threads: int, gpu: bool, cer: bool, tmp: Path, swap: str | None = None,
             extra: list[str] | None = None, codec_swap: str | None = None) -> dict:
    out = tmp / f"{c['name']}.wav"
    args = swap_file(swap_file(c["args"], swap), codec_swap, "--model")
    cmd = [str(REPO / "build" / "tts-cli"), "synthesize", *args, *(extra or []),
           "--output", str(out), "--n-threads", str(n_threads)]
    if gpu:
        cmd.append("--gpu")
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    m = TIMING_RE.search(proc.stdout)
    if proc.returncode != 0 or m is None or not out.exists():
        return {"ok": False, "rc": proc.returncode, "tail": (proc.stdout + proc.stderr)[-600:]}
    load, spk, gen, dec, total, frames = (int(v) for v in m.groups())
    sr, dur, _ = tts_smoke._wav_stats(out)
    # synth_ms (speaker + generate + decode) is the headline: load is one-time in a host.
    r = {"ok": True, "load_ms": load, "speaker_ms": spk, "generate_ms": gen, "decode_ms": dec,
         "synth_ms": spk + gen + dec, "total_ms": total, "frames": frames, "audio_s": round(dur, 3)}
    if cer and c["lang"] is not None:
        res = tts_smoke._try_cer(out, c["text"], c["lang"])
        if res is not None:
            r["asr"], r["cer"] = res[0], round(res[1], 3)
    return r


def cmd_run(a) -> int:
    cases = collect_cases()
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c["name"] in keep]
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for c in cases:
            # The first run warms the page cache and graph caches of a fresh process alike;
            # keep the faster of --repeat runs (CER from the kept one).
            runs = [run_case(c, a.n_threads, a.gpu, not a.no_cer, Path(tmp), a.backbone_swap, shlex.split(a.extra),
                             a.codec_swap)
                    for _ in range(a.repeat)]
            good = [x for x in runs if x["ok"]]
            r = min(good, key=lambda x: x["synth_ms"]) if good else runs[-1]
            results[c["name"]] = r
            if r["ok"]:
                cer = f" CER={r['cer']:.3f}" if "cer" in r else ""
                print(f"{c['name']:28s} synth {r['synth_ms']:6d} ms (spk {r['speaker_ms']:5d} gen {r['generate_ms']:6d} "
                      f"dec {r['decode_ms']:6d})  load {r['load_ms']:5d}  frames {r['frames']:4d}  "
                      f"audio {r['audio_s']:.2f}s{cer}", flush=True)
            else:
                print(f"{c['name']:28s} FAILED rc={r['rc']}\n{r['tail']}", flush=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    doc = {"git_sha": sha, "gpu": a.gpu, "n_threads": a.n_threads, "results": results}
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return 0 if all(r["ok"] for r in results.values()) else 1


def cmd_compare(a) -> int:
    before = json.loads(Path(a.before).read_text())["results"]
    after = json.loads(Path(a.after).read_text())["results"]
    # Speedups are per unit of output (sampled runs differ in length): generate per
    # frame, decode and synth per audio second.
    print(f"{'case':28s} {'synth/s before':>14s} {'after':>8s} {'x':>6s}  {'gen/fr x':>8s} {'dec/s x':>8s}  CER")
    bad = []

    def norm(r):
        # Results from before the speaker / frames columns existed.
        r = dict(r)
        r.setdefault("synth_ms", r["generate_ms"] + r["decode_ms"])
        r.setdefault("frames", 0)
        return r

    for name, b in before.items():
        f = after.get(name)
        if f is None:
            continue  # not in this run (--only)
        if not (b.get("ok") and f.get("ok")):
            if b.get("ok"):
                bad.append(f"{name}: failed after")
            continue
        b, f = norm(b), norm(f)
        per_s = lambda r, k: r[k] / max(r["audio_s"], 1e-3)  # noqa: E731
        per_fr = lambda r, k: r[k] / (r["frames"] if r["frames"] > 0 else max(r["audio_s"], 1e-3))  # noqa: E731
        sx = lambda x, y: x / y if y > 0 else float("inf")  # noqa: E731
        cb, ca = b.get("cer"), f.get("cer")
        if cb is not None and ca is not None and ca > cb + CER_SLACK:
            bad.append(f"{name}: CER {cb} -> {ca}")
        print(f"{name:28s} {per_s(b, 'synth_ms'):14.0f} {per_s(f, 'synth_ms'):8.0f} "
              f"{sx(per_s(b, 'synth_ms'), per_s(f, 'synth_ms')):6.2f}  "
              f"{sx(per_fr(b, 'generate_ms'), per_fr(f, 'generate_ms')):8.2f} "
              f"{sx(per_s(b, 'decode_ms'), per_s(f, 'decode_ms')):8.2f}  {cb if cb is not None else '-'} -> "
              f"{ca if ca is not None else '-'}")
    for line in bad:
        print("REGRESSION " + line)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out")
    r.add_argument("--n-threads", type=int, default=8)
    r.add_argument("--gpu", action="store_true")
    r.add_argument("--only")
    r.add_argument("--no-cer", action="store_true")
    r.add_argument("--repeat", type=int, default=2)
    r.add_argument("--backbone-swap", metavar="DIR:TAG",
                   help="run each backbone's DIR/<model dir>__<stem>-<TAG>.gguf variant where present")
    r.add_argument("--codec-swap", metavar="DIR:TAG", help="same for the codec GGUF")
    r.add_argument("--extra", default="", help="extra tts-cli arguments for every case (one shell-quoted string)")
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    a = ap.parse_args()
    return cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
