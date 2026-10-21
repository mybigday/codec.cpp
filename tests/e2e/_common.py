"""Shared helpers for the tests/e2e smoke scripts.

The C API is reached through the `codec_cpp` package (python/codec_cpp);
this module only adds what tests need on top: assertions, metrics, WAV I/O
and a llama.cpp backbone driver.

When `codec_cpp` is not installed (`pip install -e python`), the source
tree's python/ directory is put on sys.path so scripts run straight from a
checkout.

Load order: llama-cpp-python ships its own libggml.  Scripts that drive a
backbone through `llama_cpp` must import it before the first codec_cpp
call (codec_cpp loads libcodec lazily, so importing codec_cpp is fine).
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

try:
    import codec_cpp  # noqa: F401
except ImportError:
    sys.path.insert(0, str(REPO / "python"))
    import codec_cpp  # noqa: F401


# --- assertions / metrics ----------------------------------------------


def must(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over flattened arrays (1.0 for two equal constants)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run_cmd(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, echo it, and fail with its output when it exits non-zero."""
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise RuntimeError(f"command failed (exit={proc.returncode}): {cmd[0]}")
    return proc


# --- WAV I/O (PCM16) ---------------------------------------------------


def read_wav_mono_f32(path: Path) -> tuple[np.ndarray, int]:
    """PCM16 WAV -> (float32 mono in [-1, 1), sample_rate); channels are averaged."""
    with wave.open(str(path), "rb") as wf:
        sr, nch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw != 2:
        raise RuntimeError(f"{path}: expected PCM16, got sample_width={sw}")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm.reshape(-1, nch).mean(axis=1), sr


def write_wav_pcm16(path: Path, pcm: np.ndarray, sr: int) -> None:
    """float32 (n,) or (n, channels) -> PCM16 WAV."""
    pcm = np.asarray(pcm, dtype=np.float32)
    nch = 1 if pcm.ndim == 1 else int(pcm.shape[1])
    s = np.clip(np.round(pcm * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(s.tobytes())


# --- llama.cpp backbone (llama-cpp-python) -----------------------------


def asr_cer(wav, ref, lang):
    """Best-effort Whisper transcript + CER vs `ref`: (text, cer), or None
    when torch / transformers / the model are unavailable."""
    try:
        import torch  # noqa
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except Exception:
        return None
    try:
        x, sr = read_wav_mono_f32(wav)
        if sr != 16000:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        m = "openai/whisper-large-v3-turbo"
        proc = WhisperProcessor.from_pretrained(m)
        model = WhisperForConditionalGeneration.from_pretrained(m)
        inp = proc(x, sampling_rate=16000, return_tensors="pt")
        ids = model.generate(inp.input_features, language=lang, task="transcribe")
        txt = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    except Exception:
        return None

    def norm(s):
        import unicodedata
        return "".join(c for c in unicodedata.normalize("NFKC", s.lower()) if c.isalnum())

    a, b = norm(ref), norm(txt)
    d = np.zeros((len(a) + 1, len(b) + 1), int)
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return txt, d[len(a)][len(b)] / max(1, len(a))


class LlamaBackbone:
    """Embedding-in / hidden-out driver for an AR backbone GGUF.

    Mirrors how tts-cli runs the backbone: inputs are embedding rows, the
    output is the last-layer hidden state (embeddings=True, no pooling).
    """

    def __init__(self, gguf: Path, n_ctx: int, *, n_gpu_layers: int = 0):
        import llama_cpp as lc
        self.lc = lc
        lc.llama_backend_init()
        mp = lc.llama_model_default_params()
        mp.use_mmap = True
        mp.n_gpu_layers = n_gpu_layers
        self.model = lc.llama_model_load_from_file(str(gguf).encode("utf-8"), mp)
        if not self.model:
            raise RuntimeError(f"llama_model_load_from_file failed: {gguf}")
        self.hidden = lc.llama_model_n_embd(self.model)
        cp = lc.llama_context_default_params()
        cp.n_ctx = n_ctx
        cp.n_batch = max(n_ctx, 64)
        cp.n_ubatch = max(n_ctx, 64)
        cp.embeddings = True
        cp.pooling_type = 0
        self.ctx = lc.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError("llama_init_from_model failed")
        self.pos = 0

    def feed_embeds(self, embeds: np.ndarray, *, all_hidden: bool = False) -> np.ndarray:
        """Append rows (T, hidden) at the current position.

        Returns the last row's hidden state (hidden,), or every row's
        (T, hidden) when ``all_hidden``.
        """
        embeds = np.ascontiguousarray(embeds, dtype=np.float32)
        if embeds.shape[1] != self.hidden:
            raise ValueError(f"feed_embeds: width {embeds.shape[1]} != model n_embd {self.hidden}")
        return self._decode(int(embeds.shape[0]), embeds, all_hidden)

    def feed_tokens(self, tokens, *, all_hidden: bool = False) -> np.ndarray:
        """Append token ids (looked up in the backbone's token_embd); returns
        hidden states as ``feed_embeds`` does."""
        tokens = np.asarray(tokens, dtype=np.int32).reshape(-1)
        return self._decode(int(tokens.size), tokens, all_hidden)

    def _decode(self, T: int, rows: np.ndarray, all_hidden: bool) -> np.ndarray:
        lc = self.lc
        is_embd = rows.dtype == np.float32
        batch = lc.llama_batch_init(T, self.hidden if is_embd else 0, 1)
        try:
            batch.n_tokens = T
            if is_embd:
                ctypes.memmove(batch.embd, rows.ctypes.data, rows.nbytes)
            for t in range(T):
                if not is_embd:
                    batch.token[t] = int(rows[t])
                batch.pos[t] = self.pos + t
                batch.n_seq_id[t] = 1
                batch.seq_id[t][0] = 0
                batch.logits[t] = 1 if (all_hidden or t == T - 1) else 0
            rc = lc.llama_decode(self.ctx, batch)
            if rc != 0:
                raise RuntimeError(f"llama_decode rc={rc}")
            idx = range(T) if all_hidden else (T - 1,)
            out = np.zeros((len(idx), self.hidden), dtype=np.float32)
            for i, t in enumerate(idx):
                p = lc.llama_get_embeddings_ith(self.ctx, t if all_hidden else -1)
                if not p:
                    raise RuntimeError(f"llama_get_embeddings_ith({t}) returned NULL")
                ctypes.memmove(out[i].ctypes.data, p, self.hidden * 4)
        finally:
            lc.llama_batch_free(batch)
        self.pos += T
        return out if all_hidden else out[0]

    def tokenize(self, text: str, *, add_bos: bool = False, special: bool = True) -> list[int]:
        lc = self.lc
        vocab = lc.llama_model_get_vocab(self.model)
        b = text.encode("utf-8")
        buf = (lc.llama_token * (len(b) + 16))()
        n = lc.llama_tokenize(vocab, b, len(b), buf, len(buf), add_bos, special)
        if n < 0:
            raise RuntimeError(f"llama_tokenize failed ({n})")
        return list(buf[:n])

    def feed_tokens(self, tokens) -> np.ndarray:
        """Append token ids at the current position; returns the last hidden state."""
        lc = self.lc
        T = len(tokens)
        batch = lc.llama_batch_init(T, 0, 1)
        try:
            batch.n_tokens = T
            for t, tok in enumerate(tokens):
                batch.token[t] = int(tok)
                batch.pos[t] = self.pos + t
                batch.n_seq_id[t] = 1
                batch.seq_id[t][0] = 0
                batch.logits[t] = 1 if t == T - 1 else 0
            rc = lc.llama_decode(self.ctx, batch)
            if rc != 0:
                raise RuntimeError(f"llama_decode rc={rc}")
            p = lc.llama_get_embeddings_ith(self.ctx, -1)
            out = np.zeros(self.hidden, dtype=np.float32)
            ctypes.memmove(out.ctypes.data, p, self.hidden * 4)
        finally:
            lc.llama_batch_free(batch)
        self.pos += T
        return out

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.lc.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            self.lc.llama_model_free(self.model)
            self.model = None

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def llamacpp_hidden_seq(embeds: np.ndarray, gguf: Path) -> np.ndarray:
    """One-shot: run a whole (T, hidden) prompt and return every hidden row."""
    T = int(embeds.shape[0])
    with LlamaBackbone(gguf, n_ctx=max(T + 8, 64)) as bb:
        return bb.feed_embeds(embeds, all_hidden=True)


# --- model-card sampling ------------------------------------------------
# Each model's recommended llama.cpp sampling (scripts/utils/sampling.py, the
# backbone GGUF's general.sampling.*), by common_params_sampling name.  Tests
# pass them explicitly so they do not depend on the local GGUFs carrying the keys.
sys.path.insert(0, str(REPO / "scripts"))
from utils.sampling import MODEL_SAMPLING  # noqa: E402

_FLAGS = {"temp": "--temp", "top_k": "--top-k", "top_p": "--top-p", "min_p": "--min-p",
          "penalty_last_n": "--repeat-last-n", "penalty_repeat": "--repeat-penalty",
          "penalty_freq": "--frequency-penalty", "penalty_present": "--presence-penalty",
          "dry_multiplier": "--dry-multiplier", "dry_base": "--dry-base",
          "dry_allowed_length": "--dry-allowed-length", "dry_penalty_last_n": "--dry-penalty-last-n",
          "samplers": "--samplers", "seed": "--seed"}


def sampling_flags(sampling: dict) -> list[str]:
    """tts-cli flags (llama.cpp names) for a sampling dict."""
    out = []
    for k, v in sampling.items():
        out += [_FLAGS[k], str(v)]
    return out
