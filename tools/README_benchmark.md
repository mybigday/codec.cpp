# Benchmark harness

Tracks wall time, peak memory, and per-phase breakdowns for every model so
optimisation work can be measured against a saved baseline.

## What gets measured

For each `(model, op)` combo (`op ∈ {encode, decode, e2e}`) the harness runs
`codec-cli` `--warmup` + `--iterations` times and aggregates:

| Metric | Source |
|---|---|
| `wall_ms` | `/usr/bin/time -v` "Elapsed (wall clock) time" — overall CLI wall time including process spawn, model load, I/O |
| `peak_rss_mb` | `/usr/bin/time -v` "Maximum resident set size" — peak RSS across the entire process lifetime (includes GGUF mmap, eval arena, scheduler buffers) |
| `phases.graph_build.{mean,p50,p95,std}_ms` | C++ scope `CODEC_PERF_SCOPE("graph_build")` in `codec_graph_cache_get_or_build` — graph construction (cached after first call per shape) |
| `phases.graph_prepare_io.…` | `codec_graph_prepare_io` — backend tensor allocation / scheduler reset |
| `phases.graph_compute.…` | `codec_graph_compute` — actual `ggml_backend_sched_graph_compute` |
| `phases.encode_total.…` / `decode_total.…` | Top-level public API entry points |
| `liveness.{ok,n_samples,rms,…}` | Output WAV (or NPY) is non-empty, finite, and has plausible energy |

Each iteration writes to its own `CODEC_PERF_LOG=…ndjson`, parsed by the
harness.  Phase events are summed within an iteration (multiple
`graph_prepare_io` calls per encode pass collapse to one number).

## Schema (`benchmarks/<run>.json`, version 1)

```jsonc
{
  "version": 1,
  "git_sha": "abc1234",
  "build_type": "Release",
  "host": {"cpu": "...", "uname": "..."},
  "config": {"input_wav": "test.wav", "iterations": 5, "warmup": 1},
  "results": {
    "snac": {
      "e2e": {
        "ok": true,
        "wall_ms":     {"mean": 6586.7, "p50": 6590, "p95": 6599, "std": 12.5, "n": 5},
        "peak_rss_mb": {"mean": 26545,  ...},
        "phases": {
          "graph_compute":  {"mean_ms": 6020, "p50_ms": ..., "p95_ms": ...},
          "graph_build":    {"mean_ms": 0.6, ...},
          "graph_prepare_io": {"mean_ms": 0.07, ...},
          "encode_total":   {...},
          "decode_total":   {...}
        },
        "n_failures": 0
      }
    },
    ...
  }
}
```

Stable shape — two runs on the same git SHA can be byte-diffed (mod
trivial timing noise) or fed into the `compare` subcommand.

## Workflow

```bash
# 1. Build (Release, no GPU backends so timings are deterministic)
cmake --build build -j

# 2. Capture baseline before optimising
python tools/benchmark.py --iterations 5 --warmup 1 --out benchmarks/before.json

# 3. Apply your optimisation, rebuild

# 4. Capture after
python tools/benchmark.py --iterations 5 --warmup 1 --out benchmarks/after.json

# 5. Diff
python tools/benchmark.py compare benchmarks/before.json benchmarks/after.json
```

`compare` prints a per-model table with absolute and percent deltas for
`wall_ms.mean` and `peak_rss_mb.mean`, and exits non-zero if any model
regresses by more than ±5 % on either metric.

### Subset / fast iteration

```bash
# Only run two specific models, fewer iterations during dev:
python tools/benchmark.py --models snac neucodec --iterations 3 --warmup 1 --out /tmp/dev.json
```

## C++ instrumentation contract

`src/runtime/perf_log.{h,cpp}` exposes:

```cpp
codec_perf_scope("phase_name");                  // RAII timer
codec_perf_event("phase_name", "free-form");     // one-off snapshot, no timing
```

When `CODEC_PERF_LOG` is unset (the default for normal users), every scope
costs one `ggml_time_us()` call at construction and a no-op at destruction
(early return on `resolve_path() == nullptr`).  When set, each scope
appends one JSON-lines record.

Add a scope to a hot path with `CODEC_PERF_SCOPE("name")` at the top of
the function — the `__LINE__`-suffixed macro avoids name clashes when
multiple scopes nest in the same function.  Pass a free-form `detail`
string with `CODEC_PERF_SCOPE_D("name", detail_str)` for caller-specific
metadata (graph kind, sequence length, etc.).

### Adding a new instrumented site

1. `#include "runtime/perf_log.h"` (already included by callers in
   `src/runtime/`).
2. `CODEC_PERF_SCOPE("your_phase")` at the top of the function.
3. The Python harness picks up the new phase automatically — it appears
   as a key in `results.<model>.<op>.phases` on the next run.

## Caveats

- `peak_rss_mb` from `/usr/bin/time -v` is the kernel-reported peak RSS,
  which includes mmap'd file pages.  GGUF weights are mmap'd, so the
  number reflects working-set memory rather than malloc'd memory.
- `wall_ms` includes process spawn (~50 ms) and weight load.  For
  optimisations that target only `graph_compute` you'll usually see a
  tighter signal in `phases.graph_compute.mean_ms`.
- The harness sets `GGML_DISABLE_VULKAN=1` so CPU vs. Vulkan timings
  don't mix.  Override `GGML_DISABLE_VULKAN` in your shell to benchmark
  the GPU backend.
- HF reference parity is not re-checked here — the existing
  `tests/e2e/runner.py` and `tests/e2e/*_smoke.py` cover that.  This
  harness only confirms output liveness (non-empty, finite, plausible
  energy) so a "fast but broken" optimisation gets caught.
