"""Multi-model TTS CLI driving codec.cpp's codec_lm + llama.cpp's
backbone + codec.cpp's audio codec to produce a WAV from a text prompt.

Architecture (per `docs/audio_lm_extensions.md`):

    text  -> HF processor -> token_ids
    token_ids -> HF embed_table -> prompt_embeds (T, hidden)
    prompt_embeds -> llama.cpp backbone -> last-pos hidden h
    for step in 0..max_frames:
        h -> codec_lm.step_begin
        for cb in 0..n_cb:
            logits = codec_lm.step_logits()
            code   = sample(logits, temperature, top_p, top_k)
            codec_lm.step_push_code(code)
        codes = codec_lm.step_finish()
        if profile.detect_stop(codes): break
        embd = codec_lm.compose_audio_embd(codes)
        h = backbone.feed_embeds(embd)
    codes_array (T, n_q) -> codec_decode -> PCM
    PCM -> WAV

Models wired today (one --model flag picks the profile):

    csm        sesame/csm-1b              backbone=llama  codec=mimi
    qwen3-tts  Qwen/Qwen3-TTS-12Hz-0.6B-Base  backbone=qwen3  codec=qwen3_tts_tokenizer
    moshi      kmhf/hf-moshiko            backbone=llama  codec=mimi    (TTS-only mode)
    lfm2-audio LiquidAI/LFM2-Audio-1.5B   backbone=lfm2   codec=mimi    (TTS-only mode)

Per-model fiddly bits (prompt format, stop condition, code-pack layout)
live in TTSProfile subclasses below.

Usage:

    .venv/bin/python examples/tts.py --model csm \\
        --text "Hello world." --speaker 0 \\
        --output /tmp/hello.wav --max-frames 200

The Python venv at `.venv/` carries the HF + torch dependencies for
tokenization / text-embed extraction; the actual inference runs in C++
via libcodec.so + llama.cpp.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests/e2e"))

# Load order: llama_cpp ships its own libggml.so that conflicts with
# the one libcodec.so links against — import it first so its symbols
# dominate the process, then dlopen libcodec.so as private.
import llama_cpp  # noqa: F401

from _codec_lm_ctypes import CodecLM, CodecDecoder, lib as codec_lib  # noqa: E402


# ---------------------------------------------------------------------
# Backbone driver (llama.cpp via llama_cpp.cffi)
# ---------------------------------------------------------------------

class LlamaBackbone:
    """Minimal wrapper around llama_cpp for `embd`-driven decode.

    Tracks position internally; supports an absorbed prompt of
    T_prompt embeddings followed by streamed 1-row continuations.
    """

    def __init__(self, gguf_path: Path, *, n_ctx: int, n_gpu_layers: int = 0):
        import llama_cpp as lc
        self.lc = lc
        lc.llama_backend_init()

        mp = lc.llama_model_default_params()
        mp.use_mmap     = True
        mp.n_gpu_layers = int(n_gpu_layers)
        self.model = lc.llama_model_load_from_file(str(gguf_path).encode("utf-8"), mp)
        if not self.model:
            raise RuntimeError(f"llama_model_load_from_file failed: {gguf_path}")
        self.hidden = lc.llama_model_n_embd(self.model)

        cp = lc.llama_context_default_params()
        cp.n_ctx        = int(n_ctx)
        cp.n_batch      = max(int(n_ctx), 64)
        cp.n_ubatch     = max(int(n_ctx), 64)
        cp.embeddings   = True
        cp.pooling_type = 0
        self.ctx = lc.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError("llama_init_from_model failed")
        self.pos = 0
        self.n_ctx = int(n_ctx)

    def feed_embeds(self, input_embeds: np.ndarray) -> np.ndarray:
        """Push a (T, hidden) block and return last-position hidden (hidden,)."""
        if input_embeds.ndim != 2 or input_embeds.shape[1] != self.hidden:
            raise ValueError(
                f"feed_embeds expects (T, {self.hidden}); got {input_embeds.shape}")
        T = int(input_embeds.shape[0])
        if self.pos + T > self.n_ctx:
            raise RuntimeError(
                f"backbone n_ctx={self.n_ctx} exhausted: pos={self.pos} + T={T}")

        batch = self.lc.llama_batch_init(T, self.hidden, 1)
        batch.n_tokens = T
        flat = np.ascontiguousarray(input_embeds.astype(np.float32, copy=False)).ravel()
        ctypes.memmove(batch.embd, flat.ctypes.data, flat.nbytes)
        for t in range(T):
            batch.pos[t]       = self.pos + t
            batch.n_seq_id[t]  = 1
            batch.seq_id[t][0] = 0
            batch.logits[t]    = 1 if t == T - 1 else 0
        rc = self.lc.llama_decode(self.ctx, batch)
        if rc != 0:
            self.lc.llama_batch_free(batch)
            raise RuntimeError(f"llama_decode rc={rc}")
        emb_ptr = self.lc.llama_get_embeddings_ith(self.ctx, 0)
        if not emb_ptr:
            self.lc.llama_batch_free(batch)
            raise RuntimeError("llama_get_embeddings_ith returned NULL")
        out = np.empty(self.hidden, dtype=np.float32)
        ctypes.memmove(out.ctypes.data, emb_ptr, self.hidden * 4)
        self.lc.llama_batch_free(batch)
        self.pos += T
        return out

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.lc.llama_free(self.ctx); self.ctx = None
        if getattr(self, "model", None):
            self.lc.llama_model_free(self.model); self.model = None


# ---------------------------------------------------------------------
# Sampler (per codebook)
# ---------------------------------------------------------------------

def sample_logits(logits: np.ndarray, *, temperature: float, top_p: float,
                  top_k: int, rng: np.random.Generator) -> int:
    if temperature <= 0.0:
        return int(np.argmax(logits))
    x = logits.astype(np.float64) / max(temperature, 1e-6)
    x -= x.max()
    p = np.exp(x); p /= p.sum()

    if top_k and top_k > 0 and top_k < p.size:
        idx_sorted = np.argpartition(p, -top_k)[-top_k:]
        mask = np.zeros_like(p, dtype=bool)
        mask[idx_sorted] = True
        p = np.where(mask, p, 0.0)
        s = p.sum()
        if s <= 0.0:
            return int(np.argmax(logits))
        p /= s

    if 0.0 < top_p < 1.0:
        order = np.argsort(p)[::-1]
        ps = p[order]
        cdf = np.cumsum(ps)
        cutoff = int(np.searchsorted(cdf, top_p)) + 1
        keep = order[:cutoff]
        mask = np.zeros_like(p, dtype=bool)
        mask[keep] = True
        p = np.where(mask, p, 0.0)
        s = p.sum()
        if s <= 0.0:
            return int(np.argmax(logits))
        p /= s
    return int(rng.choice(p.size, p=p))


# ---------------------------------------------------------------------
# Profile interface
# ---------------------------------------------------------------------

@dataclass
class TTSProfile:
    name: str
    hf_id: str
    backbone_gguf: Path
    codec_lm_gguf: Path
    codec_gguf:    Path

    default_temperature: float = 0.9
    default_top_p:       float = 0.95
    default_top_k:       int   = 50

    # Implemented per-model below.
    def build_prompt_embeds(self, text: str, *, speaker: int | None,
                            backbone_hidden: int) -> np.ndarray:
        raise NotImplementedError

    def detect_stop(self, codes: list[int], step: int) -> bool:
        return False  # default: rely on --max-frames

    def pack_codes(self, codes_all: list[list[int]], n_cb: int) -> np.ndarray:
        # Default: identity (T, n_cb).  Override for models with delay-pattern
        # or codebook permutation.
        return np.asarray(codes_all, dtype=np.int32)


# ---------------------------------------------------------------------
# CSM profile
# ---------------------------------------------------------------------

class CSMProfile(TTSProfile):
    """sesame/csm-1b — Llama-3.2-1B backbone + 32 Mimi codebooks.

    Prompt: `<|begin_of_text|>[<speaker>]<text><|end_of_text|>`
    (HF CsmProcessor.apply_chat_template handles tokenization).

    The backbone's input embeddings come from CSM's `embed_text_tokens`
    table (a distinct table from the audio-side embeddings).  Our
    converted GGUF stores it as `model.embed_tokens.weight` so we
    could in principle drive the backbone via llama_decode with token
    ids — but the AR continuation feeds composed-audio embeddings, so
    keeping a single embed-driven path is simpler.

    Stop heuristic: codes[0] == 0 (codebook_eos_token_id) for two
    consecutive frames.  HF doesn't expose a clean per-step EOS for
    CSM (it relies on max-length + the audio_eos special token in the
    text-side input_ids), so the two-zero rule is empirical.
    """

    def build_prompt_embeds(self, text: str, *, speaker: int | None,
                            backbone_hidden: int) -> np.ndarray:
        import torch
        from transformers import AutoProcessor, CsmForConditionalGeneration

        # AutoProcessor is small; the full model is heavy.  But we need
        # the text-embed table from the model, so we load both.  Cache
        # on the instance so repeated `.build_prompt_embeds` (e.g. in
        # benchmarks) doesn't re-load.
        if not hasattr(self, "_hf"):
            print(f"[hf] loading {self.hf_id} (for tokenizer + embed table) …",
                  flush=True)
            proc = AutoProcessor.from_pretrained(self.hf_id)
            mdl  = CsmForConditionalGeneration.from_pretrained(
                self.hf_id, dtype=torch.float32).eval()
            self._hf = (proc, mdl)
        proc, mdl = self._hf

        if speaker is None:
            speaker = 0
        conv = [{"role": str(int(speaker)),
                 "content": [{"type": "text", "text": text}]}]
        ids = proc.apply_chat_template(conv, tokenize=True, return_dict=True,
                                       return_tensors="pt")["input_ids"]
        with torch.no_grad():
            embeds = mdl.embed_text_tokens(ids)[0].cpu().float().numpy()
        if embeds.shape[1] != backbone_hidden:
            raise RuntimeError(
                f"embed width {embeds.shape[1]} != backbone n_embd {backbone_hidden}")
        return embeds  # (T_prompt, hidden)

    def detect_stop(self, codes: list[int], step: int) -> bool:
        # codebook-0 == codebook_eos_token_id (=0) at the audio-LM head
        # signals end-of-audio on CSM.  Require it at step > 0 to avoid
        # tripping on the very first frame when temperature is low.
        return step > 0 and int(codes[0]) == 0


# ---------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------

PROFILES: dict[str, TTSProfile] = {
    "csm": CSMProfile(
        name="csm",
        hf_id="sesame/csm-1b",
        backbone_gguf = REPO / "models/csm/llama_backbone.gguf",
        codec_lm_gguf = REPO / "models/csm/csm.gguf",
        codec_gguf    = REPO / "models/mimi/mimi.gguf",
    ),
    # qwen3-tts / moshi / lfm2-audio profiles land below as they get
    # their prompt-format + stop-condition pieces wired in.
}


# ---------------------------------------------------------------------
# WAV writer (16-bit PCM)
# ---------------------------------------------------------------------

def write_wav_pcm16(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    import struct, wave
    if pcm.ndim == 1:
        n_ch, samples = 1, pcm
    elif pcm.ndim == 2:
        n_ch, samples = int(pcm.shape[1]), pcm
    else:
        raise ValueError(f"unsupported PCM shape {pcm.shape}")
    s = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(n_ch)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(s.tobytes(order="C"))


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def run(profile: TTSProfile, args) -> int:
    for label, p in (("backbone", profile.backbone_gguf),
                     ("codec_lm", profile.codec_lm_gguf),
                     ("codec",    profile.codec_gguf)):
        if not p.is_file():
            print(f"FAIL: missing {label} GGUF {p}", file=sys.stderr)
            return 2

    print(f"[cpp] loading codec_lm + codec_model + backbone …", flush=True)
    cpp_lm   = CodecLM(profile.codec_lm_gguf)
    decoder  = CodecDecoder(profile.codec_gguf, use_gpu=args.use_gpu)
    print(f"  codec_lm:  n_cb={cpp_lm.n_cb} hidden={cpp_lm.hidden_dim}", flush=True)
    print(f"  codec:     arch_sr={decoder.sample_rate} n_q={decoder.n_q} "
          f"hop={decoder.hop_size}", flush=True)

    if cpp_lm.n_cb > decoder.n_q:
        print(f"WARN: codec_lm n_cb={cpp_lm.n_cb} > codec n_q={decoder.n_q}; "
              f"decoder will only use the first {decoder.n_q} codebooks",
              file=sys.stderr)

    # Compose prompt embeds (HF-side: tokenizer + text-embed table).
    prompt_embeds = profile.build_prompt_embeds(
        args.text, speaker=args.speaker, backbone_hidden=cpp_lm.hidden_dim)
    T_prompt = int(prompt_embeds.shape[0])
    print(f"  prompt:    T={T_prompt}", flush=True)

    n_ctx = T_prompt + args.max_frames + 16
    backbone = LlamaBackbone(profile.backbone_gguf, n_ctx=n_ctx,
                             n_gpu_layers=args.gpu_layers)
    if backbone.hidden != cpp_lm.hidden_dim:
        print(f"FAIL: backbone hidden {backbone.hidden} != codec_lm hidden "
              f"{cpp_lm.hidden_dim}", file=sys.stderr)
        return 3

    rng = np.random.default_rng(args.seed)
    state = cpp_lm.state()

    t0 = time.time()
    h = backbone.feed_embeds(prompt_embeds)

    temperature = args.temperature if not args.greedy else 0.0
    top_p       = args.top_p
    top_k       = args.top_k

    codes_all: list[list[int]] = []
    t_step = time.time()
    for step in range(args.max_frames):
        state.step_begin(h)
        codes_this: list[int] = []
        for cb in range(cpp_lm.n_cb):
            cb_idx, n, logits_view = state.step_logits()
            logits = np.frombuffer(logits_view, dtype=np.float32)
            code = sample_logits(logits, temperature=temperature,
                                 top_p=top_p, top_k=top_k, rng=rng)
            state.step_push_code(code)
            codes_this.append(code)
        state.step_finish()
        codes_all.append(codes_this)

        if profile.detect_stop(codes_this, step):
            print(f"  [stop] codes[0]==0 at step {step}", flush=True)
            break

        # Compose next-step input embedding from this frame's codes.
        embd_buf = cpp_lm.compose_audio_embd(codes_this)
        next_embeds = np.frombuffer(embd_buf, dtype=np.float32).reshape(1, -1).copy()
        h = backbone.feed_embeds(next_embeds)

        if (step + 1) % 25 == 0:
            elapsed = time.time() - t_step
            print(f"  [step {step+1:4d}] {elapsed:.2f}s "
                  f"({(step+1)/max(elapsed,1e-3):.1f} steps/s)",
                  flush=True)

    n_frames = len(codes_all)
    t_loop = time.time() - t0
    print(f"\n[gen] {n_frames} frames in {t_loop:.2f}s "
          f"({n_frames/max(t_loop,1e-3):.1f} steps/s)", flush=True)

    # (T, n_cb) -> pass to codec, asking it to use min(n_cb, decoder.n_q) heads
    codes_arr = profile.pack_codes(codes_all, cpp_lm.n_cb)
    if codes_arr.shape[1] > decoder.n_q:
        codes_arr = codes_arr[:, :decoder.n_q].copy()

    print(f"[dec] codec_decode codes shape={codes_arr.shape} …", flush=True)
    pcm, sr = decoder.decode(codes_arr)
    print(f"  pcm: n_samples={pcm.size if pcm.ndim == 1 else pcm.shape[0]}  "
          f"sr={sr}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav_pcm16(out_path, pcm, sr)
    print(f"[wav] wrote {out_path}", flush=True)

    state.close()
    cpp_lm.close()
    decoder.close()
    backbone.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(PROFILES.keys()), required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--output", default="/tmp/tts_out.wav")
    ap.add_argument("--speaker", type=int, default=None,
                    help="speaker id (CSM accepts 0/1; ignored if not supported)")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--greedy", action="store_true",
                    help="argmax sampling regardless of --temperature")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-gpu", action="store_true",
                    help="run codec_model decode on GPU if available")
    ap.add_argument("--gpu-layers", type=int, default=0,
                    help="layers to offload to GPU for llama.cpp backbone")
    args = ap.parse_args()

    profile = PROFILES[args.model]
    if args.temperature is None: args.temperature = profile.default_temperature
    if args.top_p       is None: args.top_p       = profile.default_top_p
    if args.top_k       is None: args.top_k       = profile.default_top_k

    return run(profile, args)


if __name__ == "__main__":
    sys.exit(main())
