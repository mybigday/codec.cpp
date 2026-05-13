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
        if session.detect_stop(codes, step): break
        next_embeds = session.compose_next_embed(codes, step)
        h = backbone.feed_embeds(next_embeds)
    codes_array (T, n_q) -> codec_decode -> PCM
    PCM -> session.trim_output_pcm -> WAV

Models wired today (one --model flag picks the profile):

    csm        sesame/csm-1b              zero-shot TTS, no ref needed
    qwen3-tts  Qwen/Qwen3-TTS-12Hz-0.6B-Base    voice-clone TTS, needs ref audio

Voice-clone profiles take a JSON config via --speaker-config:

    {
      "ref_audio": "alice.wav",     # path (relative to JSON file or absolute)
      "ref_text":  "Hello, my name is Alice.",
      "language":  "Auto",          # optional: en / zh / Auto / ...
      "x_vector_only_mode": false   # Qwen3-TTS: true = no ICL, just spk_emb
    }

Usage:

    .venv/bin/python examples/tts.py --model csm \\
        --text "Hello world." --speaker 0 --output /tmp/hello.wav

    .venv/bin/python examples/tts.py --model qwen3-tts \\
        --text "How are you today?" --speaker-config alice.json \\
        --output /tmp/clone.wav

The Python venv at `.venv/` carries the HF + torch dependencies for
tokenization / text-embed extraction + speaker-encoder forward; the
actual AR inference runs in C++ via libcodec.so + llama.cpp.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests/e2e"))

# Load order: llama_cpp ships its own libggml.so that conflicts with
# the one libcodec.so links against — import it first so its symbols
# dominate the process, then dlopen libcodec.so as private.
import llama_cpp  # noqa: F401

from _codec_lm_ctypes import CodecLM, CodecDecoder  # noqa: E402


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
# Speaker config
# ---------------------------------------------------------------------

@dataclass
class SpeakerConfig:
    """Voice-clone conditioning loaded from a JSON file.

    Per-profile interpretation:

      - Qwen3-TTS, MOSS-TTSD:  ref_audio + ref_text drive the ICL prompt.
        `language` controls the codec language tag; `x_vector_only_mode`
        (Qwen3-TTS only) disables ICL and uses spk_emb only.
      - CSM:  speaker config is unused (zero-shot); `--speaker` int picks
        the speaker token instead.
    """
    ref_audio: Path | None = None
    ref_text:  str  | None = None
    language:  str         = "Auto"
    x_vector_only_mode:    bool   = False
    speakers:  list[str]   = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "SpeakerConfig":
        d = json.loads(path.read_text())
        base = path.parent
        ra = d.get("ref_audio")
        ra_path: Path | None = None
        if ra:
            p = Path(ra)
            ra_path = p if p.is_absolute() else (base / p).resolve()
            if not ra_path.is_file():
                raise FileNotFoundError(f"speaker config ref_audio: {ra_path}")
        return cls(
            ref_audio=ra_path,
            ref_text=d.get("ref_text"),
            language=d.get("language", "Auto"),
            x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
            speakers=list(d.get("speakers", [])),
        )


def load_audio_mono(path: Path, target_sr: int) -> np.ndarray:
    """Load a WAV / FLAC / MP3 ... as a float32 mono ndarray @ target_sr."""
    import soundfile as sf
    try:
        import librosa  # type: ignore
    except ImportError:
        librosa = None

    pcm, sr = sf.read(str(path), dtype="float32")
    if pcm.ndim == 2:
        pcm = pcm.mean(axis=1)
    if sr != target_sr:
        if librosa is None:
            raise RuntimeError(
                f"ref_audio sr={sr} != target {target_sr}; install librosa to resample")
        pcm = librosa.resample(pcm, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(pcm.astype(np.float32))


# ---------------------------------------------------------------------
# Profile / Session interface
# ---------------------------------------------------------------------

@dataclass
class TTSProfile:
    name: str
    hf_id: str
    backbone_gguf: Path
    codec_lm_gguf: Path
    codec_gguf:    Path

    needs_speaker_config: bool = False

    default_temperature: float = 0.9
    default_top_p:       float = 0.95
    default_top_k:       int   = 50

    def create_session(self, *, codec_lm: CodecLM, codec: CodecDecoder,
                       args, speaker_cfg: SpeakerConfig | None) -> "TTSSession":
        raise NotImplementedError


class TTSSession:
    """Per-run mutable state created by a TTSProfile.create_session().

    The driver calls these methods in this order:

        prompt = session.initial_prompt_embeds(text)
        for step in ...:
            ...sample codes...
            if session.detect_stop(codes, step): break
            next_embd = session.compose_next_embed(codes, step)
        codes_arr = session.pack_codes(codes_all)
        pcm = decoder.decode(codes_arr)
        pcm = session.trim_output_pcm(pcm, sr)
    """

    def __init__(self, profile: TTSProfile, codec_lm: CodecLM,
                 codec: CodecDecoder, args, speaker_cfg: SpeakerConfig | None):
        self.profile  = profile
        self.codec_lm = codec_lm
        self.codec    = codec
        self.args     = args
        self.speaker_cfg = speaker_cfg

    # ----- required hooks -----

    def initial_prompt_embeds(self, text: str) -> np.ndarray:
        raise NotImplementedError

    # ----- optional hooks (sensible defaults) -----

    def compose_next_embed(self, codes: list[int], step: int) -> np.ndarray:
        embd_buf = self.codec_lm.compose_audio_embd(codes)
        return np.frombuffer(embd_buf, dtype=np.float32).reshape(1, -1).copy()

    def detect_stop(self, codes: list[int], step: int) -> bool:
        return False

    def pack_codes(self, codes_all: list[list[int]]) -> np.ndarray:
        return np.asarray(codes_all, dtype=np.int32)

    def trim_output_pcm(self, pcm: np.ndarray, sr: int,
                        n_gen_frames: int) -> np.ndarray:
        return pcm


# ---------------------------------------------------------------------
# CSM (zero-shot, single-speaker, Llama-3.2-1B + Mimi)
# ---------------------------------------------------------------------

class CSMProfile(TTSProfile):
    def create_session(self, *, codec_lm, codec, args, speaker_cfg):
        return CSMSession(self, codec_lm, codec, args, speaker_cfg)


class CSMSession(TTSSession):
    """sesame/csm-1b prompt assembly + stop heuristic.

    Prompt format: `<|begin_of_text|>[<speaker>]<text><|end_of_text|>`
    (HF CsmProcessor handles tokenization).  The text-side embed table
    is CSM's `embed_text_tokens` (distinct from the audio embed table).

    Stop: codes[0] == 0 (codebook_eos_token_id) at step > 0 signals
    the model's audio-EOS — HF doesn't expose a clean per-step stop;
    this matches the training-time pattern where the audio_eos frame's
    labels are all zeros.
    """

    def initial_prompt_embeds(self, text: str) -> np.ndarray:
        import torch
        from transformers import AutoProcessor, CsmForConditionalGeneration

        if not hasattr(self, "_hf"):
            print(f"[hf] loading {self.profile.hf_id} (for tokenizer + embed table) …",
                  flush=True)
            proc = AutoProcessor.from_pretrained(self.profile.hf_id)
            mdl  = CsmForConditionalGeneration.from_pretrained(
                self.profile.hf_id, dtype=torch.float32).eval()
            self._hf = (proc, mdl)
        proc, mdl = self._hf

        speaker = int(self.args.speaker if self.args.speaker is not None else 0)
        conv = [{"role": str(speaker),
                 "content": [{"type": "text", "text": text}]}]
        ids = proc.apply_chat_template(conv, tokenize=True, return_dict=True,
                                       return_tensors="pt")["input_ids"]
        with torch.no_grad():
            embeds = mdl.embed_text_tokens(ids)[0].cpu().float().numpy()
        return embeds  # (T_prompt, hidden)

    def detect_stop(self, codes, step):
        return step > 0 and int(codes[0]) == 0


# ---------------------------------------------------------------------
# Qwen3-TTS (voice-clone, Qwen3 talker + qwen3_tts_tokenizer)
# ---------------------------------------------------------------------

class Qwen3TTSProfile(TTSProfile):
    def create_session(self, *, codec_lm, codec, args, speaker_cfg):
        if speaker_cfg is None:
            raise SystemExit(
                "qwen3-tts requires --speaker-config alice.json with "
                "ref_audio + ref_text (voice-clone TTS)")
        return Qwen3TTSSession(self, codec_lm, codec, args, speaker_cfg)


class Qwen3TTSSession(TTSSession):
    """Qwen/Qwen3-TTS-12Hz-0.6B-Base TTS (voice clone).

    We mirror the HF prompt-assembly path
    (`Qwen3TTSModel.generate_voice_clone` ->
     `Qwen3TTSForConditionalGeneration.generate`) up to the point where
    HF produces `talker_input_embeds` + `trailing_text_hidden` +
    `tts_pad_embed`, then hand `talker_input_embeds` to our llama.cpp
    backbone and drive the AR loop via codec_lm.

    The per-step compose is `compose_audio_embd(codes) +
    trailing_text_hidden[step]` while step < len(trailing_text), else
    `compose_audio_embd(codes) + tts_pad_embed` — this matches the HF
    talker's `forward()` generate-branch.

    Stop: codes[0] == codec_eos_token_id.
    """

    def __init__(self, profile, codec_lm, codec, args, speaker_cfg):
        super().__init__(profile, codec_lm, codec, args, speaker_cfg)
        self._hf_ready = False
        self._trailing: np.ndarray | None = None  # (T_trail, H)
        self._pad_emb:  np.ndarray | None = None  # (1, H)
        self._codec_eos_id: int = -1
        self._ref_n_frames: int = 0
        self._ref_codes_for_decode: np.ndarray | None = None

    # ----- HF preprocessing (slow path: runs once per session) -----

    def _load_hf(self):
        if self._hf_ready:
            return
        sys.path.insert(0, str(REPO / ".model-src/Qwen3-TTS"))
        import torch
        from qwen_tts import Qwen3TTSModel

        print(f"[hf] loading {self.profile.hf_id} (for prompt + spk_emb) …",
              flush=True)
        self._hf_tts = Qwen3TTSModel.from_pretrained(
            self.profile.hf_id, device_map="cpu", dtype=torch.float32,
            attn_implementation="eager",
        )
        self._torch = torch
        self._hf_ready = True

    def _build_voice_clone_prompt(self):
        cfg = self.speaker_cfg
        if cfg.ref_audio is None:
            raise SystemExit("speaker config missing ref_audio")
        if not cfg.x_vector_only_mode and not cfg.ref_text:
            raise SystemExit("ref_text required (set x_vector_only_mode=true to skip)")

        tts = self._hf_tts
        prompt_items = tts.create_voice_clone_prompt(
            ref_audio=str(cfg.ref_audio),
            ref_text=cfg.ref_text,
            x_vector_only_mode=cfg.x_vector_only_mode,
        )
        self._prompt_items = prompt_items
        # ref_code (encoded by HF speech_tokenizer); we keep it for decode-time concat
        if prompt_items[0].ref_code is not None:
            self._ref_codes_for_decode = prompt_items[0].ref_code.cpu().numpy().astype(np.int32)
            self._ref_n_frames = int(self._ref_codes_for_decode.shape[0])

    def initial_prompt_embeds(self, text: str) -> np.ndarray:
        self._load_hf()
        self._build_voice_clone_prompt()
        torch = self._torch
        tts = self._hf_tts
        model = tts.model       # Qwen3TTSForConditionalGeneration

        prompt_dict = tts._prompt_items_to_voice_clone_prompt(self._prompt_items)

        # Tokenize ref + target text via HF processor, matching
        # `generate_voice_clone`'s _build_assistant_text / _build_ref_text.
        target_text  = tts._build_assistant_text(text)
        input_ids    = tts._tokenize_texts([target_text])
        ref_text_str = self._prompt_items[0].ref_text
        ref_ids: list = [None]
        if ref_text_str:
            ref_ids = [tts._tokenize_texts([tts._build_ref_text(ref_text_str)])[0]]

        # Mirror Qwen3TTSForConditionalGeneration.generate up to the
        # construction of talker_input_embeds + trailing_text_hidden +
        # tts_pad_embed.  Non-streaming mode keeps the AR compose path
        # simple: trailing_text covers the audio axis position-by-
        # position; tts_pad_embed is added once trailing_text is exhausted.
        non_streaming_mode = True
        cfg_l = self.speaker_cfg.language

        # Speaker embedding (always present in voice-clone mode).
        voice_clone_spk_embeds = model.generate_speaker_prompt(prompt_dict)

        with torch.no_grad():
            input_id  = input_ids[0]              # (1, T_text)
            ref_id    = ref_ids[0]                # (1, T_ref) or None
            spk_emb   = voice_clone_spk_embeds[0]

            # language id
            tk_cfg = model.config.talker_config
            if cfg_l.lower() == "auto":
                language_id = None
            else:
                language_id = tk_cfg.codec_language_id.get(cfg_l.lower())
                if language_id is None:
                    raise SystemExit(f"unsupported language={cfg_l!r}")

            tts_special = torch.tensor([
                [model.config.tts_bos_token_id,
                 model.config.tts_eos_token_id,
                 model.config.tts_pad_token_id]],
                device=model.talker.device, dtype=input_id.dtype)
            tts_bos_embed, tts_eos_embed, tts_pad_embed = \
                model.talker.text_projection(
                    model.talker.get_text_embeddings()(tts_special)
                ).chunk(3, dim=1)

            if language_id is None:
                codec_prefill = [[tk_cfg.codec_nothink_id,
                                  tk_cfg.codec_think_bos_id,
                                  tk_cfg.codec_think_eos_id]]
            else:
                codec_prefill = [[tk_cfg.codec_think_id,
                                  tk_cfg.codec_think_bos_id,
                                  language_id,
                                  tk_cfg.codec_think_eos_id]]
            codec_input_emb_0 = model.talker.get_input_embeddings()(
                torch.tensor(codec_prefill, device=model.talker.device,
                             dtype=input_id.dtype))
            codec_input_emb_1 = model.talker.get_input_embeddings()(
                torch.tensor([[tk_cfg.codec_pad_id, tk_cfg.codec_bos_id]],
                             device=model.talker.device, dtype=input_id.dtype))
            codec_input_emb = torch.cat([
                codec_input_emb_0,
                spk_emb.view(1, 1, -1),
                codec_input_emb_1], dim=1)

            role_embed = model.talker.text_projection(
                model.talker.get_text_embeddings()(input_id[:, :3]))
            pre_embed = torch.cat([
                tts_pad_embed.expand(-1, codec_input_emb.shape[1] - 2, -1),
                tts_bos_embed], dim=1) + codec_input_emb[:, :-1]
            talker_input_embed = torch.cat([role_embed, pre_embed], dim=1)

            if (self._prompt_items[0].icl_mode
                    and self._prompt_items[0].ref_code is not None):
                ref_code_t = self._prompt_items[0].ref_code.to(model.talker.device)
                icl_emb, trailing = model.generate_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_id[:, 3:-2],
                    ref_code=ref_code_t,
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_input_embed = torch.cat([talker_input_embed, icl_emb], dim=1)
            else:
                # x_vector_only path: feed the assistant tts text inline,
                # mirroring the `else:` non_streaming branch in HF generate.
                first_text_embed = model.talker.text_projection(
                    model.talker.get_text_embeddings()(input_id[:, 3:4])
                ) + codec_input_emb[:, -1:]
                rest_text_embed  = torch.cat([
                    model.talker.text_projection(
                        model.talker.get_text_embeddings()(input_id[:, 3:-5])),
                    tts_eos_embed], dim=1) + model.talker.get_input_embeddings()(
                        torch.tensor([[tk_cfg.codec_pad_id] *
                                      (input_id[:, 3:-5].shape[1] + 1)],
                                     device=model.talker.device,
                                     dtype=input_id.dtype))
                bos_embed = tts_pad_embed + model.talker.get_input_embeddings()(
                    torch.tensor([[tk_cfg.codec_bos_id]],
                                 device=model.talker.device, dtype=input_id.dtype))
                talker_input_embed = torch.cat([
                    talker_input_embed[:, :-1],   # drop pre-placed text row
                    rest_text_embed,
                    bos_embed], dim=1)
                trailing = tts_pad_embed

            self._trailing = trailing.squeeze(0).cpu().float().numpy()
            self._pad_emb  = tts_pad_embed.squeeze(0).cpu().float().numpy()
            self._codec_eos_id = int(tk_cfg.codec_eos_token_id)

        out = talker_input_embed.squeeze(0).cpu().float().numpy()
        return out  # (T_prompt, hidden)

    def compose_next_embed(self, codes, step):
        embd_buf = self.codec_lm.compose_audio_embd(codes)
        base = np.frombuffer(embd_buf, dtype=np.float32).reshape(1, -1).copy()
        if self._trailing is not None and step < self._trailing.shape[0]:
            base = base + self._trailing[step:step+1]
        elif self._pad_emb is not None:
            base = base + self._pad_emb
        return base

    def detect_stop(self, codes, step):
        return int(codes[0]) == self._codec_eos_id

    def pack_codes(self, codes_all):
        # Drop the c0 stop frame (if any was sampled before --max-frames hit).
        codes = np.asarray(codes_all, dtype=np.int32)
        # Decode prepends ref_codes (matches HF's
        # `codes_for_decode = cat([ref_code, gen_codes])`).
        if self._ref_codes_for_decode is not None:
            codes = np.concatenate([self._ref_codes_for_decode, codes], axis=0)
        return codes

    def trim_output_pcm(self, pcm, sr, n_gen_frames):
        # HF trims the ref-portion proportionally:
        #   cut = int(ref_len / total_len * pcm.shape[0])
        # We mirror that.  pcm is decoded from [ref_codes + gen_codes].
        if self._ref_n_frames <= 0 or pcm.ndim != 1:
            return pcm
        n_total = self._ref_n_frames + n_gen_frames
        if n_total <= 0:
            return pcm
        cut = int(self._ref_n_frames / n_total * pcm.shape[0])
        return pcm[cut:] if cut < pcm.shape[0] else pcm


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
        needs_speaker_config = False,
    ),
    "qwen3-tts": Qwen3TTSProfile(
        name="qwen3-tts",
        hf_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        backbone_gguf = REPO / "models/qwen3_tts/qwen3_tts_talker.gguf",
        codec_lm_gguf = REPO / "models/qwen3_tts/qwen3_tts_06b_base.gguf",
        codec_gguf    = REPO / "models/qwen3_tts_tokenizer/qwen3_tts_tokenizer.gguf",
        needs_speaker_config = True,
        default_top_p = 1.0,
    ),
    # moss-ttsd / lfm2-audio profiles land below as they get their
    # prompt-format + stop-condition pieces wired in.
}


# ---------------------------------------------------------------------
# WAV writer (16-bit PCM)
# ---------------------------------------------------------------------

def write_wav_pcm16(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    import wave
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

def run(profile: TTSProfile, args, speaker_cfg: SpeakerConfig | None) -> int:
    for label, p in (("backbone", profile.backbone_gguf),
                     ("codec_lm", profile.codec_lm_gguf),
                     ("codec",    profile.codec_gguf)):
        if not p.is_file():
            print(f"FAIL: missing {label} GGUF {p}", file=sys.stderr)
            return 2

    print(f"[cpp] loading codec_lm + codec_model …", flush=True)
    cpp_lm = CodecLM(profile.codec_lm_gguf)
    codec  = CodecDecoder(profile.codec_gguf, use_gpu=args.use_gpu)
    print(f"  codec_lm:  n_cb={cpp_lm.n_cb} hidden={cpp_lm.hidden_dim}", flush=True)
    print(f"  codec:     sr={codec.sample_rate} n_q={codec.n_q} hop={codec.hop_size}",
          flush=True)
    if cpp_lm.n_cb > codec.n_q:
        print(f"WARN: codec_lm n_cb={cpp_lm.n_cb} > codec n_q={codec.n_q}; "
              f"decoder will use the first {codec.n_q} codebooks",
              file=sys.stderr)

    session = profile.create_session(
        codec_lm=cpp_lm, codec=codec, args=args, speaker_cfg=speaker_cfg)

    print(f"[ses] building initial prompt embeddings …", flush=True)
    prompt_embeds = session.initial_prompt_embeds(args.text)
    T_prompt = int(prompt_embeds.shape[0])
    print(f"  prompt:    T={T_prompt} hidden={prompt_embeds.shape[1]}", flush=True)
    if prompt_embeds.shape[1] != cpp_lm.hidden_dim:
        print(f"FAIL: prompt hidden {prompt_embeds.shape[1]} != codec_lm hidden "
              f"{cpp_lm.hidden_dim}", file=sys.stderr)
        return 3

    n_ctx = T_prompt + args.max_frames + 16
    backbone = LlamaBackbone(profile.backbone_gguf, n_ctx=n_ctx,
                             n_gpu_layers=args.gpu_layers)
    if backbone.hidden != cpp_lm.hidden_dim:
        print(f"FAIL: backbone hidden {backbone.hidden} != codec_lm hidden "
              f"{cpp_lm.hidden_dim}", file=sys.stderr)
        return 3

    rng   = np.random.default_rng(args.seed)
    state = cpp_lm.state()

    temperature = args.temperature if not args.greedy else 0.0
    top_p, top_k = args.top_p, args.top_k

    t0 = time.time()
    h  = backbone.feed_embeds(prompt_embeds)

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

        if session.detect_stop(codes_this, step):
            print(f"  [stop] session-detected EOS at step {step}", flush=True)
            break

        next_embeds = session.compose_next_embed(codes_this, step)
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

    codes_arr = session.pack_codes(codes_all)
    if codes_arr.shape[1] > codec.n_q:
        codes_arr = codes_arr[:, :codec.n_q].copy()

    print(f"[dec] codec_decode codes shape={codes_arr.shape} …", flush=True)
    pcm, sr = codec.decode(codes_arr)
    pcm = session.trim_output_pcm(pcm, sr, n_gen_frames=n_frames)
    print(f"  pcm: n_samples={pcm.size if pcm.ndim == 1 else pcm.shape[0]} sr={sr}",
          flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav_pcm16(out_path, pcm, sr)
    print(f"[wav] wrote {out_path}", flush=True)

    state.close()
    cpp_lm.close()
    codec.close()
    backbone.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(PROFILES.keys()), required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--output", default="/tmp/tts_out.wav")
    ap.add_argument("--speaker", type=int, default=None,
                    help="speaker id (CSM accepts 0/1; ignored if not supported)")
    ap.add_argument("--speaker-config", type=Path, default=None,
                    help="JSON config with ref_audio + ref_text (voice-clone models)")
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

    speaker_cfg: SpeakerConfig | None = None
    if args.speaker_config is not None:
        speaker_cfg = SpeakerConfig.load(args.speaker_config)
    elif profile.needs_speaker_config:
        print(f"FAIL: --model {profile.name} requires --speaker-config",
              file=sys.stderr)
        return 2

    return run(profile, args, speaker_cfg)


if __name__ == "__main__":
    sys.exit(main())
