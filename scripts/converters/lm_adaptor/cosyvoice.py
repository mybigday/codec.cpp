"""CosyVoice LM-adaptor dump (`parallel_heads_delay`, one speech codebook).

The AR backbone runs in llama.cpp (Qwen2 for CosyVoice2/3, the `cosyvoice`
arch patch for v1); everything around it lives here, in the codec GGUF:

    speech_embedding.weight  → lm.audio_embd_0.weight   (next-step input rows)
    llm_decoder.{weight,bias}→ lm.heads_0.{weight,bias}  (speech logits)
    llm_embedding.weight     → lm.cosyvoice.llm_embd     (v1/v2: sos, task_id rows)

v1 (CosyVoice-300M / BreezyVoice) also runs the text side on codec.cpp:
    text_embedding           → lm.cosyvoice.text_embd
    text_encoder.*           → lm.cosyvoice.tenc.*      (6-layer rel-pos Conformer)
    text_encoder_affine_layer→ lm.cosyvoice.tenc_aff
    spk_embed_affine_layer   → lm.cosyvoice.spk_aff     (x-vector → prompt row)

Prompt layout (Qwen2LM / TransformerLM .inference):
    v1:   [sos] [spk_aff(norm(xvec))] [text_encoder(prompt_text ++ text)] [task_id] [speech_emb(prompt tokens)]
    v2:   [sos] [backbone embed(prompt_text ++ text)] [task_id] [speech_emb(prompt tokens)]
    v3:   same as v2, sos / task_id are speech_embedding rows 6561 / 6563.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch

V3_SILENT_TOKENS = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]
V3_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"


def _np(t) -> np.ndarray:
    return t.detach().cpu().float().numpy() if torch.is_tensor(t) else np.asarray(t)


def dump_cosyvoice_lm(writer, ckpt_dir: Path, cfg: Dict[str, Any],
                      add_tensor: Callable[..., None]) -> None:
    v = int(cfg["version"])
    sd = torch.load(Path(ckpt_dir) / "llm.pt", map_location="cpu", weights_only=True)
    speech_tokens = int(cfg["codebook_size"])
    head_w = _np(sd["llm_decoder.weight"])
    head_b = _np(sd["llm_decoder.bias"]) if "llm_decoder.bias" in sd else None
    emb = _np(sd["speech_embedding.weight"])
    n_vocab, hidden = head_w.shape
    if emb.shape[0] < n_vocab:   # v1: 4096 speech rows, head has +1 (eos); eos is never embedded
        emb = np.concatenate([emb, np.zeros((n_vocab - emb.shape[0], hidden), np.float32)], 0)

    if v == 1:
        sos, task, eos = 0, 1, speech_tokens
        stop_begin, stop_end = speech_tokens, speech_tokens + 1
    elif v == 2:
        sos, task, eos = 0, 1, speech_tokens
        stop_begin, stop_end = speech_tokens, speech_tokens + 3
    else:
        sos, task, eos = speech_tokens, speech_tokens + 2, speech_tokens + 1
        stop_begin, stop_end = speech_tokens, n_vocab

    w = writer
    w.add_bool("codec.lm.has_adaptor", True)
    w.add_string("codec.lm.kind", "parallel_heads_delay")
    w.add_string("codec.lm.host_arch", "cosyvoice" if v == 1 else "qwen2")
    w.add_uint32("codec.lm.hidden_dim", hidden)
    w.add_uint32("codec.lm.audio_embed_dim", hidden)
    w.add_uint32("codec.lm.n_codebook", 1)
    w.add_array("codec.lm.codebook_sizes", [n_vocab])
    w.add_array("codec.lm.delay_pattern", [0])
    w.add_bool("codec.lm.parallel.tied_heads_to_embd", False)
    w.add_int32("codec.lm.eos_code_c0", eos)
    w.add_int32("codec.lm.eos_min_step", 0)

    p = "codec.lm.cosyvoice."
    w.add_uint32(p + "version", v)
    w.add_uint32(p + "speech_token_size", speech_tokens)
    # sos / task_id: rows of lm.cosyvoice.llm_embd (v1/v2) or of the speech table (v3).
    w.add_bool(p + "special_in_speech_table", v == 3)
    w.add_uint32(p + "sos", sos)
    w.add_uint32(p + "task_id", task)
    w.add_uint32(p + "stop_begin", stop_begin)
    w.add_uint32(p + "stop_end", stop_end)
    # sampling_ids masks this id while i < min_len; the reference masks
    # speech_token_size (= sos in v3, so v3 min_len is effectively not enforced).
    w.add_uint32(p + "ignore_eos_id", speech_tokens)
    w.add_float32(p + "min_token_text_ratio", 2.0)
    w.add_float32(p + "max_token_text_ratio", 20.0)
    ras = bool(cfg.get("ras_sampling", True))
    w.add_string(p + "sampling", "ras" if ras else "top_k")
    w.add_uint32(p + "top_k", 25)
    w.add_float32(p + "top_p", 0.8)
    w.add_uint32(p + "win_size", 10)
    w.add_float32(p + "tau_r", 0.1)
    w.add_bool(p + "text_in_backbone", v != 1)
    if v == 3:
        w.add_array(p + "silent_tokens", V3_SILENT_TOKENS)
        w.add_uint32(p + "max_silent_run", 5)
        w.add_string(p + "prompt_prefix", V3_PROMPT_PREFIX)
    if cfg.get("breezyvoice"):
        # BreezyVoice expects 注音 (bopomofo) annotations from g2pw in the text;
        # the frontend runs outside codec.cpp.
        w.add_bool(p + "bopomofo_frontend", True)

    add_tensor("lm.audio_embd_0.weight", emb)
    add_tensor("lm.heads_0.weight", head_w)
    if head_b is not None:
        add_tensor("lm.heads_0.bias", head_b, "F32")
    if "llm_embedding.weight" in sd:
        add_tensor("lm.cosyvoice.llm_embd", _np(sd["llm_embedding.weight"]), "F32")

    if v != 1:
        return

    # ---- v1 text side ----
    breezy = bool(cfg.get("breezyvoice"))
    add_tensor("lm.cosyvoice.text_embd", _np(sd["text_embedding.weight"]))
    add_tensor("lm.cosyvoice.tenc.embed.lin.w", _np(sd["text_encoder.embed.out.0.weight"]))
    add_tensor("lm.cosyvoice.tenc.embed.lin.b", _np(sd["text_encoder.embed.out.0.bias"]), "F32")
    add_tensor("lm.cosyvoice.tenc.embed.ln.w", _np(sd["text_encoder.embed.out.1.weight"]), "F32")
    add_tensor("lm.cosyvoice.tenc.embed.ln.b", _np(sd["text_encoder.embed.out.1.bias"]), "F32")
    add_tensor("lm.cosyvoice.tenc.after_norm.w", _np(sd["text_encoder.after_norm.weight"]), "F32")
    add_tensor("lm.cosyvoice.tenc.after_norm.b", _np(sd["text_encoder.after_norm.bias"]), "F32")
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("text_encoder.encoders."))
    for i in range(n_layers):
        s, d = f"text_encoder.encoders.{i}", f"lm.cosyvoice.tenc.blk.{i}"
        a, f = s + ".self_attn", s + ".feed_forward"
        pairs = [("norm_mha.w", s + ".norm_mha.weight"), ("norm_mha.b", s + ".norm_mha.bias"),
                 ("norm_ff.w", s + ".norm_ff.weight"), ("norm_ff.b", s + ".norm_ff.bias"),
                 ("attn.pos.w", a + ".linear_pos.weight"), ("attn.pbu", a + ".pos_bias_u"),
                 ("attn.pbv", a + ".pos_bias_v"),
                 ("ff.w1.w", f + ".w_1.weight"), ("ff.w1.b", f + ".w_1.bias"),
                 ("ff.w2.w", f + ".w_2.weight"), ("ff.w2.b", f + ".w_2.bias")]
        for dd, ss in (("q", "linear_q"), ("k", "linear_k"), ("v", "linear_v"), ("o", "linear_out")):
            pairs += [(f"attn.{dd}.w", f"{a}.{ss}.weight"), (f"attn.{dd}.b", f"{a}.{ss}.bias")]
        for dn, key in pairs:
            arr = _np(sd[key])
            add_tensor(f"{d}.{dn}", arr, "F32" if arr.ndim < 2 or dn.endswith(("pbu", "pbv")) else None)
    add_tensor("lm.cosyvoice.tenc_aff.w", _np(sd["text_encoder_affine_layer.weight"]))
    add_tensor("lm.cosyvoice.tenc_aff.b", _np(sd["text_encoder_affine_layer.bias"]), "F32")
    add_tensor("lm.cosyvoice.spk_aff.w", _np(sd["spk_embed_affine_layer.weight"]))
    add_tensor("lm.cosyvoice.spk_aff.b", _np(sd["spk_embed_affine_layer.bias"]), "F32")
    heads = int(_np(sd["text_encoder.encoders.0.self_attn.pos_bias_u"]).shape[0])
    w.add_uint32(p + "tenc.layers", n_layers)
    w.add_uint32(p + "tenc.heads", heads)
    w.add_uint32(p + "tenc.hidden", int(_np(sd["text_encoder.embed.out.0.weight"]).shape[0]))
    w.add_float32(p + "tenc.block_eps", 1e-5 if breezy else 1e-12)
    # TransformerLM.encode runs the encoder with decoding_chunk_size=1 and the
    # yaml's static_chunk_size=1 → a chunk-of-one (causal) attention mask.
    w.add_bool(p + "tenc.causal", True)
