"""End-to-end parity check: codec_lm parallel_heads_delay vs HF reference.

Loads the HF MOSS-TTSD-v0.5 model, runs a forward pass on a synthetic
input to capture the backbone's last hidden state and the reference per-
codebook logits + the reference next-step audio embedding (channel-sum).
Then drives our codec_lm via the codec-lm-cli binary against the same
hidden / codes and compares.

This isolates LM-adaptor correctness from any llama.cpp backbone work:
- If logits match, our `lm.heads_{i}.weight` (tied to audio_embd_{i}) are
  correctly converted and the parallel_heads_delay step graph computes
  `head @ h` faithfully.
- If audio embeddings match, our `lm.audio_embd_{i}.weight` tensors are
  correctly converted and `compose_audio_embd` sums them faithfully.

Drift tolerance: F16 weights vs HF's BF16 → ~5e-3 max-abs on logits is
expected; correlation must be very high.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GGUF = REPO / "models/moss_ttsd_v0_5/moss_ttsd_v0_5.gguf"
CODEC_LM_CLI = REPO / "build/codec-lm-cli"

# The HF v0.5 release.  Tied to the GGUF the converter produced.
HF_LM_REPO = "fnlp/MOSS-TTSD-v0.5"

# Per-codebook similarity thresholds.  Pulled empirically — F16 vs BF16
# round-trip on the logits is very tight; the embedding sum is even
# tighter (no matmul, just lookup + add).
LOGITS_CORR_MIN     = 0.9999
LOGITS_MAX_ABS_DIFF = 5e-3
EMBD_CORR_MIN       = 0.99999
EMBD_MAX_ABS_DIFF   = 1e-3


def must(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


def run_codec_lm_step(hidden: np.ndarray, gguf: Path, tmp: Path) -> dict[int, np.ndarray]:
    """Drive codec-lm-cli step → returns {cb_idx: logits np.ndarray}."""
    h_path = tmp / "h.npy"
    np.save(h_path, hidden.astype(np.float32))
    pfx = tmp / "logits"
    cmd = [
        str(CODEC_LM_CLI), "step",
        "--model",  str(gguf),
        "--hidden", str(h_path),
        "--logits-prefix", str(pfx),
    ]
    env = {**os.environ, "GGML_DISABLE_VULKAN": "1"}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"codec-lm-cli step exit={r.returncode}")
    out: dict[int, np.ndarray] = {}
    i = 0
    while True:
        p = tmp / f"logits_{i}.npy"
        if not p.exists():
            break
        out[i] = np.load(p)
        i += 1
    if not out:
        raise RuntimeError(f"no logits_*.npy produced; stderr:\n{r.stderr}")
    return out


def run_codec_lm_compose(codes: np.ndarray, gguf: Path, tmp: Path) -> np.ndarray:
    c_path = tmp / "c.npy"
    np.save(c_path, codes.astype(np.int32))
    e_path = tmp / "e.npy"
    cmd = [
        str(CODEC_LM_CLI), "compose",
        "--model",     str(gguf),
        "--codes",     str(c_path),
        "--embd-out",  str(e_path),
    ]
    env = {**os.environ, "GGML_DISABLE_VULKAN": "1"}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"codec-lm-cli compose exit={r.returncode}")
    return np.load(e_path)


def hf_reference(prompt_text: str):
    """Run HF MOSS-TTSD-v0.5 forward to capture
       (last_hidden, list[ref_logits_i], list[ref_audio_embd_i_for_chosen_code])."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    print("[hf] loading MOSS-TTSD-v0.5 …", flush=True)
    tok   = AutoTokenizer.from_pretrained(HF_LM_REPO, trust_remote_code=True)
    model = AutoModel.from_pretrained(HF_LM_REPO, trust_remote_code=True,
                                      torch_dtype=torch.float32).eval()

    # AutoModel resolves to MossTTSDForCausalLM (per the repo's auto_map);
    # the underlying base model with the multi-channel embed sum is
    # `model.model`, whose forward returns Qwen3 `BaseModelOutputWithPast`
    # exposing last_hidden_state.  Heads / embedding_list live on the
    # outer MossTTSDForCausalLM.
    base = model.model
    embedding_list = base.embedding_list
    n_cb = model.config.channels
    speech_pad = int(model.config.speech_pad_token)

    text_ids = tok(prompt_text, return_tensors="pt").input_ids[0]
    seq_len  = int(text_ids.shape[0])

    # Build (1, seq_len, n_cb): channel 0 is text, channels 1..n_cb-1 are
    # speech-pad — same shape MOSS-TTSDForCausalLM's forward expects.
    input_ids = torch.full((1, seq_len, n_cb), speech_pad, dtype=torch.long)
    input_ids[0, :, 0] = text_ids

    print(f"[hf] forward seq_len={seq_len} n_cb={n_cb} …", flush=True)
    with torch.no_grad():
        base_out = base(input_ids=input_ids, return_dict=True)
        last_hidden = base_out.last_hidden_state[0, -1, :]   # (hidden,)

        # Heads parity — matches MossTTSDForCausalLM.forward's per-cb
        # logit projection.  All heads are tied to embedding_list[i] via
        # _tie_or_clone_weights, so reading lm_heads[i].weight is fine.
        ref_logits: list[np.ndarray] = []
        for i in range(n_cb):
            head_w = model.lm_heads[i].weight
            ref_logits.append((last_hidden @ head_w.t()).cpu().float().numpy())
        ref_hidden = last_hidden.cpu().float().numpy()

        # Compose: pick greedy code per cb, look up the embedding, sum.
        # Mirrors MossTTSDModel._prepare_multi_modal_inputs at a single
        # position with codes drawn from greedy on the reference logits.
        codes = np.array([int(np.argmax(l)) for l in ref_logits], dtype=np.int32)
        ref_embd_terms = []
        for i in range(n_cb):
            row = embedding_list[i].weight[int(codes[i])]
            ref_embd_terms.append(row.cpu().float().numpy())
        ref_embd = np.sum(ref_embd_terms, axis=0)

    return ref_hidden, ref_logits, codes, ref_embd


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    must(GGUF.is_file(),         f"missing GGUF: {GGUF} (run convert-to-gguf first)")
    must(CODEC_LM_CLI.is_file(), f"missing binary: {CODEC_LM_CLI}")

    # Use a deterministic prompt so the test is reproducible.
    prompt = "[S1] Hello, this is a quick parity test."

    ref_hidden, ref_logits, codes, ref_embd = hf_reference(prompt)
    n_cb = len(ref_logits)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # ---- step parity ----
        print("[cpp] running codec-lm-cli step …", flush=True)
        cpp_logits = run_codec_lm_step(ref_hidden, GGUF, tmp)
        must(len(cpp_logits) == n_cb,
             f"cpp produced {len(cpp_logits)} logits, expected {n_cb}")

        all_pass = True
        for i in range(n_cb):
            r = ref_logits[i]
            c = cpp_logits[i]
            must(c.shape == r.shape,
                 f"cb={i}: shape mismatch cpp={c.shape} hf={r.shape}")
            mad = float(np.max(np.abs(c - r)))
            cc  = corr(c, r)
            ok  = (mad <= LOGITS_MAX_ABS_DIFF) and (cc >= LOGITS_CORR_MIN)
            tag = "OK " if ok else "FAIL"
            print(f"  [{tag}] cb={i:2d}  vocab={r.shape[0]:6d}  "
                  f"max_abs_diff={mad:.5g}  corr={cc:.6f}")
            all_pass &= ok

        # ---- compose parity ----
        print("[cpp] running codec-lm-cli compose …", flush=True)
        cpp_embd = run_codec_lm_compose(codes, GGUF, tmp)
        must(cpp_embd.shape == ref_embd.shape,
             f"embd shape mismatch cpp={cpp_embd.shape} hf={ref_embd.shape}")
        mad = float(np.max(np.abs(cpp_embd - ref_embd)))
        cc  = corr(cpp_embd, ref_embd)
        ok  = (mad <= EMBD_MAX_ABS_DIFF) and (cc >= EMBD_CORR_MIN)
        print(f"  [{'OK ' if ok else 'FAIL'}] compose  hidden={ref_embd.shape[0]:5d}  "
              f"max_abs_diff={mad:.5g}  corr={cc:.6f}")
        all_pass &= ok

    if all_pass:
        print("\nMOSS-TTSD-v0.5 codec_lm parity test PASSED")
        return 0
    print("\nMOSS-TTSD-v0.5 codec_lm parity test FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
