#!/usr/bin/env python3
"""E2E smoke for codec_cpp.AudioLM driven by a host-owned llama.cpp loop.

The backbone runs in llama-cpp-python (its own llama.cpp build, stock
architectures only), the way an application embeds codec.cpp next to its own
llama.cpp.  Everything model-specific comes from codec_audio_lm: the context
requirements, the prompt chunks, the backbone sampler settings, the per-step
verdicts and the audio.  Cases (skipped when their GGUFs are missing):

  * CSM-1B         — codec heads read the backbone hidden (depth decoder).
  * MOSS-TTS-Realtime — text lane from the backbone's embedding table
                     (backbone_path), audio fed back as embeddings.

Each must stop on the model's end, produce non-silent audio of a plausible
length and (when Whisper is available) transcribe with CER below the threshold.
llama-cpp-python's llama.cpp differs from the bundled one, so the audio is
not compared bit for bit with tts-cli.

Run from the repo root: .venv/bin/python tests/e2e/audio_lm_host_smoke.py
"""

import ctypes
import sys
import tempfile
from pathlib import Path

import llama_cpp as lc  # before codec_cpp: llama-cpp-python ships its own libggml
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codec_cpp  # noqa: E402
from _common import asr_cer, write_wav_pcm16  # noqa: E402


def make_sampler(sp: dict, allowed, n_vocab: int):
    """A llama.cpp sampler chain from the completion's settings."""
    chain = lc.llama_sampler_chain_init(lc.llama_sampler_chain_default_params())
    if allowed:
        ok = np.zeros(n_vocab, bool)
        for b, e in allowed:
            ok[max(0, b):min(n_vocab, e)] = True
        bias = [(t, float("-inf")) for t in np.nonzero(~ok)[0]]
        arr = (lc.llama_logit_bias * len(bias))(*[lc.llama_logit_bias(int(t), v) for t, v in bias])
        lc.llama_sampler_chain_add(chain, lc.llama_sampler_init_logit_bias(n_vocab, len(bias), arr))
    if sp["penalty_repeat"] != 1.0:
        last_n = sp["penalty_last_n"] if sp["penalty_last_n"] > 0 else 1 << 16
        lc.llama_sampler_chain_add(chain, lc.llama_sampler_init_penalties(
            last_n, sp["penalty_repeat"], sp["penalty_freq"], sp["penalty_present"]))
    if sp["temp"] <= 0:
        lc.llama_sampler_chain_add(chain, lc.llama_sampler_init_greedy())
        return chain
    if sp["top_k"] > 0:
        lc.llama_sampler_chain_add(chain, lc.llama_sampler_init_top_k(sp["top_k"]))
    if 0 < sp["top_p"] < 1:
        lc.llama_sampler_chain_add(chain, lc.llama_sampler_init_top_p(sp["top_p"], 1))
    if sp["min_p"] > 0:
        lc.llama_sampler_chain_add(chain, lc.llama_sampler_init_min_p(sp["min_p"], 1))
    lc.llama_sampler_chain_add(chain, lc.llama_sampler_init_temp(sp["temp"]))
    lc.llama_sampler_chain_add(chain, lc.llama_sampler_init_dist(sp["seed"]))
    return chain


class Backbone:
    """llama-cpp-python context sized by codec_audio_lm_backbone_requirements."""

    def __init__(self, path: Path, alm: codec_cpp.AudioLM, max_frames: int):
        lc.llama_backend_init()
        mp = lc.llama_model_default_params()
        self.model = lc.llama_model_load_from_file(str(path).encode(), mp)
        if not self.model:
            raise RuntimeError(f"cannot load {path}")
        self.vocab = codec_cpp.Vocab.from_llama_cpp(self.model)
        req = alm.backbone_requirements(self.vocab, max_frames)
        cp = lc.llama_context_default_params()
        cp.n_ctx = req["n_ctx"]
        cp.n_batch = req["n_ctx"]
        cp.n_ubatch = req["n_ubatch"] or req["n_ctx"]
        cp.n_seq_max = req["n_seq_max"]
        cp.embeddings = req["embeddings"]
        cp.pooling_type = 0
        if not req["flash_attn"]:
            cp.flash_attn = False
        self.ctx = lc.llama_init_from_model(self.model, cp)
        self.n_embd = lc.llama_model_n_embd(self.model)
        self.n_vocab = lc.llama_vocab_n_tokens(lc.llama_model_get_vocab(self.model))

    def decode(self, pos: int, tokens=None, embd=None, all_out: bool = False) -> None:
        n = len(tokens) if tokens is not None else embd.shape[0]
        b = lc.llama_batch_init(n, 0 if tokens is not None else self.n_embd, 1)
        try:
            b.n_tokens = n
            if tokens is not None:
                for i, t in enumerate(tokens):
                    b.token[i] = int(t)
            else:
                e = np.ascontiguousarray(embd, np.float32)
                ctypes.memmove(b.embd, e.ctypes.data, e.nbytes)
            for i in range(n):
                b.pos[i] = pos + i
                b.n_seq_id[i] = 1
                b.seq_id[i][0] = 0
                b.logits[i] = 1 if (all_out or i == n - 1) else 0
            if lc.llama_decode(self.ctx, b) != 0:
                raise RuntimeError("llama_decode failed")
        finally:
            lc.llama_batch_free(b)

    def hidden(self, i: int = -1) -> np.ndarray:
        p = lc.llama_get_embeddings_ith(self.ctx, i)
        return np.ctypeslib.as_array(p, shape=(self.n_embd,)).copy()

    def clear(self) -> None:
        if hasattr(lc, "llama_memory_clear"):
            lc.llama_memory_clear(lc.llama_get_memory(self.ctx), True)
        else:   # llama-cpp-python before the memory API
            lc.llama_kv_self_clear(self.ctx)

    def close(self) -> None:
        lc.llama_free(self.ctx)
        lc.llama_model_free(self.model)


def synthesize(alm: codec_cpp.AudioLM, bb: Backbone, text: str, **req):
    """The host loop of include/codec_audio_lm.h, one segment."""
    alm.tts_prompt(text, vocab=bb.vocab, **req)
    c = alm.completion()
    assert c.n_seq == 1, "this smoke drives single-sequence models"
    bb.clear()
    smpl = make_sampler(c.sampler, c.allowed, bb.n_vocab)
    pos, prompt_hidden = 0, []
    for ch in c.chunks[0]:
        if ch.type == codec_cpp.AudioLM.CHUNK_EMBD:
            if len(ch.embd):
                bb.decode(pos, embd=ch.embd)
                pos += len(ch.embd)
            continue
        toks = ch.tokens if ch.type == codec_cpp.AudioLM.CHUNK_TOKENS else \
            bb.vocab._tokenize(ch.text, ch.add_special, ch.parse_special)
        if len(toks):
            bb.decode(pos, tokens=toks, all_out=c.needs_prompt_hidden)
            if c.needs_prompt_hidden:
                prompt_hidden += [bb.hidden(i) for i in range(len(toks))]
            pos += len(toks)
    if c.needs_prompt_hidden:
        alm.prefill(np.stack(prompt_hidden))
    sample, frames = c.sample_first, 0
    while frames < c.max_frames:
        h = bb.hidden() if c.needs_embeddings else None
        tok = lc.llama_sampler_sample(smpl, bb.ctx, -1) if sample else None
        v = alm.observe(tokens=None if tok is None else [tok], hiddens=h)
        frames += v.is_audio
        if v.action == codec_cpp.AudioLM.STOP:
            break
        if v.action == codec_cpp.AudioLM.FEED_TOKEN:
            bb.decode(pos, tokens=[v.token if v.token >= 0 else tok])
        else:
            bb.decode(pos, embd=alm.next_embd(0, c.n_embd)[None, :])
        pos += 1
        sample = v.sample_backbone
    lc.llama_sampler_free(smpl)
    audio, sr = alm.decode()
    # The exported decoder input decodes to the same audio.
    again, _ = alm.decode_generation(alm.generation())
    assert np.array_equal(again, audio), "decode_generation(generation()) != decode()"
    return audio, sr, frames, alm.stop_reason


CASES = [
    dict(name="csm", codec=REPO / "models/csm/csm.gguf", backbone=REPO / "models/csm/llama_backbone.gguf",
         text="Hello, this is a test of the emergency broadcast system.", lang="en", stop="eos_code_c0",
         cer_max=0.2, req=dict(temperature=0)),   # greedy, as in the tts-cli smoke
    dict(name="moss_tts_realtime", codec=REPO / "models/moss_tts_realtime/moss_tts_realtime.gguf",
         backbone=REPO / "models/moss_tts_realtime/qwen3_backbone_bf16.gguf", backbone_path=True,
         text="你好，欢迎使用语音合成。", lang="zh", stop="eos_code_c0", cer_max=0.35,
         req=dict(temperature=0.8, top_p=0.6, top_k=30, seed=42)),
]


def main() -> int:
    failed = 0
    for case in CASES:
        name = case["name"]
        if not case["codec"].exists() or not case["backbone"].exists():
            print(f"[skip] {name}: missing GGUFs")
            continue
        alm = codec_cpp.AudioLM(case["codec"], backbone_path=case["backbone"] if case.get("backbone_path") else None)
        bb = Backbone(case["backbone"], alm, max_frames=400)
        try:
            alm.set_voice()
            audio, sr, frames, stop = synthesize(alm, bb, case["text"], **case["req"])
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1
            continue
        finally:
            bb.close()
        dur = len(audio) / sr
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if len(audio) else 0.0
        ok = stop == case["stop"] and 0.5 < dur < 20.0 and rms > 1e-3
        note = ""
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            write_wav_pcm16(Path(f.name), audio, sr)
            r = asr_cer(f.name, case["text"], case["lang"])
        if r is not None:
            note = f" asr={r[0]!r} cer={r[1]:.2f}"
            ok = ok and r[1] <= case["cer_max"]
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: frames={frames} stop={stop} dur={dur:.2f}s rms={rms:.3f}{note}")
        failed += not ok
        alm.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
