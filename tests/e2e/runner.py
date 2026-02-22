#!/usr/bin/env python3
"""Unified E2E regression runner for codec models."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.json")


@dataclass
class ModelResult:
    name: str
    status: str
    duration_sec: float
    command: list[str]
    log_path: str
    reason: str = ""
    return_code: int | None = None
    artifacts: dict[str, str] | None = None


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_candidate(root: Path, rel_pattern: str, kind: str = "file", contains: list[str] | None = None) -> Path | None:
    matches = sorted(root.glob(rel_pattern))
    for candidate in matches:
        if kind == "dir":
            if not candidate.is_dir():
                continue
            if contains:
                if not all((candidate / p).exists() for p in contains):
                    continue
            return candidate
        if candidate.is_file():
            return candidate
    return None


def detect_model(root: Path, model_cfg: dict[str, Any]) -> bool:
    detect_patterns = model_cfg.get("detect", [])
    if detect_patterns:
        return any(resolve_candidate(root, p) is not None for p in detect_patterns)

    artifacts = model_cfg.get("artifacts", {})
    for artifact_cfg in artifacts.values():
        for candidate in artifact_cfg.get("candidates", []):
            if resolve_candidate(root, candidate, artifact_cfg.get("kind", "file"), artifact_cfg.get("contains")):
                return True
    return False


def resolve_artifacts(root: Path, model_cfg: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    resolved: dict[str, str] = {}
    missing_required: list[str] = []

    artifacts = model_cfg.get("artifacts", {})
    for key, artifact_cfg in artifacts.items():
        found: Path | None = None
        kind = artifact_cfg.get("kind", "file")
        contains = artifact_cfg.get("contains")
        for candidate in artifact_cfg.get("candidates", []):
            found = resolve_candidate(root, candidate, kind=kind, contains=contains)
            if found is not None:
                break

        if found is not None:
            resolved[key] = str(found)
            continue

        default_rel = artifact_cfg.get("default")
        if default_rel:
            resolved[key] = str((root / default_rel).resolve())

        if artifact_cfg.get("required", False):
            missing_required.append(key)

    return resolved, missing_required


def run_hf_snapshot_setup(root: Path, model_name: str, setup_cfg: dict[str, Any]) -> None:
    repo_id = setup_cfg["repo_id"]
    local_dir = str((root / setup_cfg["local_dir"]).resolve())
    token_env = setup_cfg.get("token_env")

    env = os.environ.copy()
    token = env.get(token_env) if token_env else None

    print(f"[{model_name}] setup: downloading snapshot {repo_id} -> {local_dir}")
    script = (
        "import os\n"
        "from huggingface_hub import snapshot_download\n"
        f"snapshot_download(repo_id={repo_id!r}, local_dir={local_dir!r}, local_dir_use_symlinks=False, token={token!r})\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True, cwd=root, env=env)


def setup_model(root: Path, model_cfg: dict[str, Any]) -> None:
    setup_cfg = model_cfg.get("setup")
    if not setup_cfg:
        return

    setup_type = setup_cfg.get("type")
    if setup_type == "hf_snapshot":
        run_hf_snapshot_setup(root, model_cfg["name"], setup_cfg)
        return

    raise ValueError(f"unsupported setup type: {setup_type}")


def format_command(parts: list[str], values: dict[str, str]) -> list[str]:
    return [part.format(**values) for part in parts]


def run_model(root: Path, report_dir: Path, model_cfg: dict[str, Any], explicit_selection: bool) -> ModelResult:
    name = model_cfg["name"]
    start = time.monotonic()
    log_path = report_dir / f"{name}.log"
    command: list[str] = []

    detected = detect_model(root, model_cfg)
    if not detected and not explicit_selection and not model_cfg.get("required", False) and not model_cfg.get("setup"):
        duration = time.monotonic() - start
        return ModelResult(
            name=name,
            status="skipped",
            duration_sec=duration,
            command=[],
            log_path=str(log_path),
            reason="not detected under models/",
        )

    resolved, missing_required = resolve_artifacts(root, model_cfg)

    if missing_required and model_cfg.get("setup"):
        try:
            setup_model(root, model_cfg)
            resolved, missing_required = resolve_artifacts(root, model_cfg)
        except Exception as exc:  # noqa: BLE001
            duration = time.monotonic() - start
            status = "failed" if model_cfg.get("required", False) else "skipped"
            return ModelResult(
                name=name,
                status=status,
                duration_sec=duration,
                command=[],
                log_path=str(log_path),
                reason=f"setup failed: {exc}",
                artifacts=resolved,
            )

    if missing_required:
        duration = time.monotonic() - start
        status = "failed" if model_cfg.get("required", False) else "skipped"
        return ModelResult(
            name=name,
            status=status,
            duration_sec=duration,
            command=[],
            log_path=str(log_path),
            reason=f"missing required artifacts: {', '.join(missing_required)}",
            artifacts=resolved,
        )

    command = format_command(model_cfg["command"], resolved)
    env = os.environ.copy()

    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] running: {' '.join(command)}")
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{name}] {line}", end="")
            log_file.write(line)
        ret = proc.wait()

    duration = time.monotonic() - start
    status = "passed" if ret == 0 else "failed"
    reason = "" if ret == 0 else f"command exited with code {ret}"
    return ModelResult(
        name=name,
        status=status,
        duration_sec=duration,
        command=command,
        log_path=str(log_path),
        reason=reason,
        return_code=ret,
        artifacts=resolved,
    )


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
                "command": r.command,
                "log_path": r.log_path,
                "reason": r.reason,
                "return_code": r.return_code,
                "artifacts": r.artifacts or {},
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
        lines.append(
            f"- {r.name}: {r.status} ({r.duration_sec:.2f}s)"
            + (f" | reason: {r.reason}" if r.reason else "")
            + f" | log: {r.log_path}"
        )

    with summary_txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return summary_json_path, summary_txt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run codec E2E regressions for discovered models")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to runner config JSON")
    parser.add_argument("--report-dir", default=str(REPO_ROOT / "tests/e2e/reports"), help="Report output directory")
    parser.add_argument("--model", action="append", dest="models", help="Run only this model name (repeatable)")
    parser.add_argument("--list-models", action="store_true", help="List configured models and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    model_cfgs = config.get("models", [])
    model_names = [m["name"] for m in model_cfgs]

    if args.list_models:
        print("\n".join(model_names))
        return 0

    selected_names = args.models if args.models else model_names
    unknown = sorted(set(selected_names) - set(model_names))
    if unknown:
        print(f"ERROR: unknown model(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    selected_cfgs = [m for m in model_cfgs if m["name"] in set(selected_names)]
    report_dir = Path(args.report_dir).resolve()

    results: list[ModelResult] = []
    for cfg in selected_cfgs:
        explicit = args.models is not None
        result = run_model(REPO_ROOT, report_dir, cfg, explicit_selection=explicit)
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
