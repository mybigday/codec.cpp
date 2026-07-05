"""Chatterbox T3 LM-adaptor dump — `parallel_heads_delay` kind.

Covers ResembleAI's Chatterbox family:

  - English-only `t3_cfg.safetensors`            (text_dict = 704)
  - Multilingual `t3_23lang.safetensors`         (text_dict = 2454)
  - Multilingual `t3_mtl23ls_v2.safetensors`     (text_dict = 2454)
  - Multilingual `t3_mtl23ls_v3.safetensors`     (text_dict = 2454)

All four share the same Llama-520M backbone (extracted separately by
`scripts/convert-backbone-to-gguf.py prep_chatterbox_t3`) + the same
adaptor shape; only the text embed table width and tokenizer differ.

Tensor mapping (`t3_*.safetensors` → codec.lm.* schema):

  speech_emb.weight              (8194, 1024)
      → lm.audio_embd_0.weight   (8194, 1024)
        Per-cb (n_cb=1) input embedding.  Used both as the
        depth-decoder's input embed (T3 has no depth decoder, so it
        only feeds the AR-step compose) and as the next-step backbone
        input.
  speech_head.weight             (8194, 1024)
      → lm.c0_head.weight        (8194, 1024)
        Speech-token head.  At AR time, only this head runs — text is
        input-only in T3 (text tokens are concatenated into the
        prompt, never autoregressively emitted).

  text_emb.weight                (text_dict, 1024)
      → lm.chatterbox.text_emb.weight       same shape
        Used by the profile to embed the prompt's text tokens before
        feeding the backbone.
  text_head.weight               (text_dict, 1024)
      → lm.chatterbox.text_head.weight      same shape
        Auxiliary text-prediction head from training.  Not used at
        TTS inference, but written for completeness / future
        reuse (e.g. text-side log-likelihood scoring).

  text_pos_emb.emb.weight        (max_text_tokens+2, 1024)   = (2050, 1024)
      → lm.chatterbox.text_pos_emb.weight   same shape
  speech_pos_emb.emb.weight      (max_speech_tokens+4, 1024) = (4100, 1024)
      → lm.chatterbox.speech_pos_emb.weight same shape
        Learned positional embeddings: T3's backbone uses RoPE on the
        QK side AND a per-token learned PE that's added to the embed
        before the backbone forward.  Profile adds the right PE row to
        each next-step speech embed during the AR loop.

  cond_enc.spkr_enc.{weight,bias}            (1024, 256), (1024,)
      → lm.chatterbox.cond.spkr_enc.{weight,bias}
        Linear projection: speaker_embed_dim → backbone hidden.
  cond_enc.emotion_adv_fc.weight             (1024, 1)
      → lm.chatterbox.cond.emotion_adv_fc.weight
        Emotion advisor scalar → backbone-hidden vector (no bias).

  cond_enc.perceiver.attn.{norm,to_q,to_k,to_v,proj_out}.{weight,bias}
  cond_enc.perceiver.pre_attention_query     (1, 32, 1024)
      → lm.chatterbox.cond.perceiver.*
        Perceiver Resampler: a single multi-head attention block
        with 32 learned queries.  Reduces the (B, T_spk, 1024)
        speaker context down to (B, 32, 1024) before concat into
        the backbone prompt.

Metadata (codec.lm.*):

  has_adaptor                = true
  kind                       = "parallel_heads_delay"
  host_arch                  = "llama"
  hidden_dim                 = 1024
  audio_embed_dim            = 1024
  n_codebook                 = 1
  codebook_sizes             = [8194]
  delay_pattern              = [0]
  parallel.tied_heads_to_embd= false

  chatterbox.text_vocab_size    = 704 or 2454
  chatterbox.start_text_token   = 255
  chatterbox.stop_text_token    = 0
  chatterbox.start_speech_token = 6561
  chatterbox.stop_speech_token  = 6562
  chatterbox.max_text_tokens    = 2048
  chatterbox.max_speech_tokens  = 4096
  chatterbox.is_multilingual    = bool
  chatterbox.has_emotion_cond   = true
  chatterbox.speaker_embed_dim  = 256
  chatterbox.cond_len           = 32
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


# ---------------------------------------------------------------------
# Constants from t3_config.py (Chatterbox upstream).  Hard-coded here
# so the converter doesn't have to import the vendored package.
# ---------------------------------------------------------------------

START_TEXT_TOKEN   = 255
STOP_TEXT_TOKEN    = 0
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN  = 6562
SPEECH_VOCAB_SIZE  = 8194
MAX_TEXT_TOKENS    = 2048
MAX_SPEECH_TOKENS  = 4096
SPEAKER_EMBED_DIM  = 256
COND_LEN           = 32         # perceiver pre_attention_query length
HIDDEN_DIM         = 1024


def dump(writer, sd: Dict[str, np.ndarray], cfg: Dict[str, Any],
         *, verbose: bool = False) -> None:
    """Write codec.lm.* metadata + lm.* tensors for a Chatterbox T3
    checkpoint into the supplied GGUFWriter."""

    text_vocab = int(cfg.get("chatterbox_text_vocab_size") or 0)
    if text_vocab == 0:
        # Derive from the actual embed shape.
        text_vocab = int(sd["text_emb.weight"].shape[0])
    is_multilingual = (text_vocab == 2454)

    # ---- metadata ------------------------------------------------------
    writer.add_bool   ("codec.lm.has_adaptor",     True)
    writer.add_string ("codec.lm.kind",            "parallel_heads_delay")
    writer.add_string ("codec.lm.host_arch",       "llama")
    writer.add_uint32 ("codec.lm.hidden_dim",      HIDDEN_DIM)
    writer.add_uint32 ("codec.lm.audio_embed_dim", HIDDEN_DIM)
    writer.add_uint32 ("codec.lm.n_codebook",      1)
    writer.add_array  ("codec.lm.codebook_sizes",  [SPEECH_VOCAB_SIZE])
    writer.add_array  ("codec.lm.delay_pattern",   [0])
    writer.add_bool   ("codec.lm.parallel.tied_heads_to_embd", False)

    writer.add_uint32 ("codec.lm.chatterbox.text_vocab_size",
                       text_vocab)
    writer.add_uint32 ("codec.lm.chatterbox.start_text_token",
                       START_TEXT_TOKEN)
    writer.add_uint32 ("codec.lm.chatterbox.stop_text_token",
                       STOP_TEXT_TOKEN)
    writer.add_uint32 ("codec.lm.chatterbox.start_speech_token",
                       START_SPEECH_TOKEN)
    writer.add_uint32 ("codec.lm.chatterbox.stop_speech_token",
                       STOP_SPEECH_TOKEN)

    # End-of-audio (uniform schema): T3's single speech codebook stops
    # when it samples the stop-speech token.  Mirror stop_speech_token
    # under eos_code_c0 (eos_min_step=0); chatterbox.* keys kept for
    # back-compat.
    writer.add_int32  ("codec.lm.eos_code_c0", STOP_SPEECH_TOKEN)
    writer.add_int32  ("codec.lm.eos_min_step", 0)
    writer.add_int32  ("codec.lm.bos_code_c0", START_SPEECH_TOKEN)
    writer.add_uint32 ("codec.lm.chatterbox.max_text_tokens",
                       MAX_TEXT_TOKENS)
    writer.add_uint32 ("codec.lm.chatterbox.max_speech_tokens",
                       MAX_SPEECH_TOKENS)
    writer.add_bool   ("codec.lm.chatterbox.is_multilingual",
                       is_multilingual)
    writer.add_bool   ("codec.lm.chatterbox.has_emotion_cond", True)
    writer.add_uint32 ("codec.lm.chatterbox.speaker_embed_dim",
                       SPEAKER_EMBED_DIM)
    writer.add_uint32 ("codec.lm.chatterbox.cond_len", COND_LEN)

    # ---- speech embed + head (the n_cb=1 codec_lm pair) --------------
    # Tensor names follow the parallel_heads_delay convention
    # (`lm.heads_{i}.weight` + `lm.audio_embd_{i}.weight`) so the
    # standard `codec_lm_check_unfused_audio_tables` validator picks
    # them up at codec_lm_create time.
    _emit(writer, "lm.audio_embd_0.weight",
          _require(sd, "speech_emb.weight"),
          shape=(SPEECH_VOCAB_SIZE, HIDDEN_DIM))
    _emit(writer, "lm.heads_0.weight",
          _require(sd, "speech_head.weight"),
          shape=(SPEECH_VOCAB_SIZE, HIDDEN_DIM))

    # ---- text-side prompt assembly tables ----------------------------
    _emit(writer, "lm.chatterbox.text_emb.weight",
          _require(sd, "text_emb.weight"),
          shape=(text_vocab, HIDDEN_DIM))
    _emit(writer, "lm.chatterbox.text_head.weight",
          _require(sd, "text_head.weight"),
          shape=(text_vocab, HIDDEN_DIM))

    # ---- learned positional embeddings -------------------------------
    _emit(writer, "lm.chatterbox.text_pos_emb.weight",
          _require(sd, "text_pos_emb.emb.weight"),
          shape=(MAX_TEXT_TOKENS + 2, HIDDEN_DIM))
    _emit(writer, "lm.chatterbox.speech_pos_emb.weight",
          _require(sd, "speech_pos_emb.emb.weight"),
          shape=(MAX_SPEECH_TOKENS + 4, HIDDEN_DIM))

    # ---- conditioning encoder (speaker + emotion + perceiver) --------
    _emit(writer, "lm.chatterbox.cond.spkr_enc.weight",
          _require(sd, "cond_enc.spkr_enc.weight"),
          shape=(HIDDEN_DIM, SPEAKER_EMBED_DIM))
    _emit(writer, "lm.chatterbox.cond.spkr_enc.bias",
          _require(sd, "cond_enc.spkr_enc.bias"),
          shape=(HIDDEN_DIM,))
    _emit(writer, "lm.chatterbox.cond.emotion_adv_fc.weight",
          _require(sd, "cond_enc.emotion_adv_fc.weight"),
          shape=(HIDDEN_DIM, 1))

    # Perceiver resampler — single self-attn block over 32 learned queries.
    _emit(writer, "lm.chatterbox.cond.perceiver.queries",
          _require(sd, "cond_enc.perceiver.pre_attention_query"),
          shape=(1, COND_LEN, HIDDEN_DIM))
    for suf in ("norm.weight", "norm.bias",
                "to_q.weight", "to_q.bias",
                "to_k.weight", "to_k.bias",
                "to_v.weight", "to_v.bias",
                "proj_out.weight", "proj_out.bias"):
        _emit(writer, f"lm.chatterbox.cond.perceiver.{suf}",
              _require(sd, f"cond_enc.perceiver.attn.{suf}"))

    # ---- tokenizer (EnTokenizer BPE) + builtin conditioning ----------
    # Both live next to the T3 checkpoint (the ResembleAI Chatterbox dir).
    # Baking them into the GGUF lets the runtime tokenize English text and
    # synthesise from a builtin voice with no external files (the
    # equivalent of ChatterboxTTS + conds.pt).
    src_dir = cfg.get("chatterbox_source_dir")
    if src_dir:
        _dump_tokenizer(writer, Path(src_dir), verbose=verbose)
        _dump_builtin_conds(writer, Path(src_dir), verbose=verbose)

    if verbose:
        print(f"[lm_adaptor:chatterbox] parallel_heads_delay: "
              f"n_codebook=1 hidden={HIDDEN_DIM} "
              f"text_vocab={text_vocab} (multilingual={is_multilingual}) "
              f"speech_vocab={SPEECH_VOCAB_SIZE}")


def _dump_tokenizer(writer, src_dir: Path, *, verbose: bool = False) -> None:
    """Bake the EnTokenizer BPE (tokenizer.json) into GGUF metadata.

    English Chatterbox uses a small HF `tokenizers` BPE: Whitespace
    pre-tokenizer, no normalizer, `continuing_subword_prefix=""`,
    `end_of_word_suffix=""`.  The runtime replays it as: replace ' ' with
    the `[SPACE]` added token, split on the Whitespace regex
    (`\\w+|[^\\w\\s]+`, extracting added-token strings first), then greedy
    rank-based BPE merges.  We store the vocab (token strings, id-ordered)
    and merge rules so the C++ side needs no HF dependency.
    """
    tok_path = src_dir / "tokenizer.json"
    if not tok_path.is_file():
        if verbose:
            print(f"[lm_adaptor:chatterbox] no tokenizer.json at {tok_path} — skipping")
        return
    with open(tok_path, "r", encoding="utf-8") as f:
        tj = json.load(f)
    model = tj.get("model", {})
    if model.get("type") != "BPE":
        raise RuntimeError(f"chatterbox tokenizer: unexpected model type {model.get('type')!r}")

    vocab: Dict[str, int] = model.get("vocab", {})
    n_vocab = len(vocab)
    # id-ordered token strings
    id_to_tok = [""] * n_vocab
    for tok, tid in vocab.items():
        if 0 <= tid < n_vocab:
            id_to_tok[tid] = tok
    # merges: list of "a b" (or ["a","b"] in newer tokenizers.json)
    merges_raw = model.get("merges", [])
    merges: list[str] = []
    for m in merges_raw:
        if isinstance(m, (list, tuple)):
            merges.append(f"{m[0]} {m[1]}")
        else:
            merges.append(str(m))
    # added tokens (special atoms extracted before BPE, e.g. [SPACE])
    added = tj.get("added_tokens", [])
    added_content = [a["content"] for a in added]
    added_ids = [int(a["id"]) for a in added]

    # The project's GGUF writer only supports numeric arrays, so token
    # strings + merge rules are stored as newline-joined blobs (tokens are
    # single-line each; '\n' never appears in this vocab).  The runtime
    # splits on '\n' to recover the id-ordered vocab and the merge list.
    writer.add_string ("codec.lm.chatterbox.tokenizer.model", "bpe")
    writer.add_uint32 ("codec.lm.chatterbox.tokenizer.n_vocab", n_vocab)
    writer.add_string ("codec.lm.chatterbox.tokenizer.tokens", "\n".join(id_to_tok))
    writer.add_string ("codec.lm.chatterbox.tokenizer.merges", "\n".join(merges))
    # Added tokens as "content\tid" pairs, newline-joined.
    writer.add_string ("codec.lm.chatterbox.tokenizer.added",
                       "\n".join(f"{c}\t{i}" for c, i in zip(added_content, added_ids)))
    unk = model.get("unk_token")
    if unk is not None:
        writer.add_string("codec.lm.chatterbox.tokenizer.unk_token", str(unk))
    if verbose:
        print(f"[lm_adaptor:chatterbox] tokenizer: {n_vocab} tokens, "
              f"{len(merges)} merges, {len(added_content)} added")


def _dump_builtin_conds(writer, src_dir: Path, *, verbose: bool = False) -> None:
    """Bake the T3 builtin speaker conditioning (conds.pt `t3` dict) so the
    runtime can synthesise with a default voice and no ref audio.

    conds.t3 carries: speaker_emb (1,256), cond_prompt_speech_tokens
    (1,150), emotion_adv (1,1,1).  These feed `chatterbox_speaker_encode_
    from_emb` (spkr_enc | perceiver(speech_emb(prompt)+pos) | emotion) to
    produce the (34,1024) cond prefix.
    """
    conds_path = src_dir / "conds.pt"
    if not conds_path.is_file():
        if verbose:
            print(f"[lm_adaptor:chatterbox] no conds.pt at {conds_path} — skipping builtin conds")
        return
    try:
        import torch
    except ImportError:
        if verbose:
            print("[lm_adaptor:chatterbox] torch unavailable — skipping builtin conds")
        return
    obj = torch.load(str(conds_path), map_location="cpu", weights_only=False)
    t3 = obj["t3"] if isinstance(obj, dict) else getattr(obj, "t3", None)
    if t3 is None:
        return
    def _get(d, k):
        return d[k] if isinstance(d, dict) else getattr(d, k, None)
    spk   = _get(t3, "speaker_emb")
    toks  = _get(t3, "cond_prompt_speech_tokens")
    emo   = _get(t3, "emotion_adv")
    if spk is None or toks is None:
        return
    spk_np  = spk.detach().cpu().float().numpy().reshape(-1)             # (256,)
    toks_np = toks.detach().cpu().to(torch.int64).numpy().reshape(-1)    # (150,)
    emo_f   = float(emo.reshape(-1)[0].item()) if emo is not None else 0.5

    writer.add_bool   ("codec.lm.chatterbox.has_builtin_conds", True)
    writer.add_array  ("codec.lm.chatterbox.builtin.speaker_emb",
                       [float(x) for x in spk_np.tolist()])
    writer.add_array  ("codec.lm.chatterbox.builtin.cond_prompt_speech_tokens",
                       [int(x) for x in toks_np.tolist()])
    writer.add_float32("codec.lm.chatterbox.builtin.emotion_adv", emo_f)
    if verbose:
        print(f"[lm_adaptor:chatterbox] builtin conds: speaker_emb={spk_np.shape} "
              f"cond_prompt_tokens={toks_np.shape} emotion={emo_f:.3f}")


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _require(sd: Dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in sd:
        raise RuntimeError(f"missing Chatterbox T3 tensor: {key}")
    return sd[key]


def _emit(writer, name: str, arr: np.ndarray,
          shape: tuple[int, ...] | None = None, st_dtype: str = "F16") -> None:
    if shape is not None and tuple(arr.shape) != tuple(shape):
        raise RuntimeError(
            f"tensor {name!r} shape {tuple(arr.shape)} != expected {shape}")
    # Normalise to F32 numpy; GGUFWriter applies the final st_dtype quant.
    arr = arr.astype(np.float32, copy=False)
    dt = "F32" if name.endswith(".bias") or "norm.weight" in name else st_dtype
    writer.add_tensor(name, arr, st_dtype=dt)
