"""Raw ctypes declarations mirroring include/codec.h and include/codec_lm.h.

Struct field order must match the headers exactly; the headers follow an
ABI-append discipline (new fields at the end), so keep the same here.
"""

from __future__ import annotations

import ctypes
from ctypes import (POINTER, Structure, Union, c_bool, c_char, c_char_p, c_double,
                    c_float, c_int, c_int32, c_int64, c_size_t, c_void_p)

# ─── enums (codec.h) ────────────────────────────────────────────────

CODEC_DEFAULT_SEED = -1  # (int32_t)0xFFFFFFFF

CODEC_STATUS_SUCCESS = 0
CODEC_STATUS_INVALID_ARG = 1
CODEC_STATUS_INVALID_STATE = 2
CODEC_STATUS_IO_ERROR = 3
CODEC_STATUS_NOT_SUPPORTED = 4
CODEC_STATUS_INTERNAL_ERROR = 5

CODEC_PCM_TYPE_F32 = 0
CODEC_PCM_TYPE_I16 = 1

CODEC_BATCH_MODE_CODES = 0
CODEC_BATCH_MODE_LATENT = 1

CODEC_KV_OVERRIDE_TYPE_INT = 0
CODEC_KV_OVERRIDE_TYPE_FLOAT = 1
CODEC_KV_OVERRIDE_TYPE_BOOL = 2
CODEC_KV_OVERRIDE_TYPE_STR = 3

CODEC_SPEAKER_SLOT_F32 = 0
CODEC_SPEAKER_SLOT_I32 = 1

# ─── enums (codec_lm.h) ─────────────────────────────────────────────

CODEC_LM_KIND_UNKNOWN = 0
CODEC_LM_KIND_PARALLEL_HEADS_DELAY = 1
CODEC_LM_KIND_RESIDUAL_DEPTH_AR = 2
CODEC_LM_KIND_CONTINUOUS_LATENT_CFM = 3
CODEC_LM_KIND_FLOW_LM = 4

# ─── structs (codec.h) ──────────────────────────────────────────────


class codec_model_params(Structure):
    _fields_ = [
        ("use_gpu", c_bool),
        ("n_threads", c_int32),
    ]


class _kv_override_value(Union):
    _fields_ = [
        ("val_i64", c_int64),
        ("val_f64", c_double),
        ("val_bool", c_bool),
        ("val_str", c_char * 128),
    ]


class codec_model_kv_override(Structure):
    _anonymous_ = ("value",)
    _fields_ = [
        ("tag", c_int),
        ("key", c_char * 128),
        ("value", _kv_override_value),
    ]


class codec_context_params(Structure):
    _fields_ = [
        ("seed", c_int32),
    ]


class codec_encode_params(Structure):
    _fields_ = [
        ("n_threads", c_int32),
        ("frame_size", c_int32),
        ("hop_size", c_int32),
        ("n_q", c_int32),
    ]


class codec_decode_params(Structure):
    _fields_ = [
        ("n_threads", c_int32),
        ("n_q", c_int32),
        ("speaker", c_void_p),
    ]


class codec_audio(Structure):
    _fields_ = [
        ("data", c_void_p),
        ("n_samples", c_int32),
        ("sample_rate", c_int32),
        ("n_channels", c_int32),
        ("pcm_type", c_int),
    ]


class codec_token_buffer(Structure):
    _fields_ = [
        ("data", POINTER(c_int32)),
        ("n_tokens", c_int32),
        ("n_frames", c_int32),
        ("n_q", c_int32),
        ("codebook_size", c_int32),
        ("sample_rate", c_int32),
        ("hop_size", c_int32),
    ]


class codec_pcm_buffer(Structure):
    _fields_ = [
        ("data", POINTER(c_float)),
        ("n_samples", c_int32),
        ("sample_rate", c_int32),
        ("n_channels", c_int32),
    ]


class codec_latent_buffer(Structure):
    _fields_ = [
        ("data", POINTER(c_float)),
        ("latent_dim", c_int32),
        ("n_frames", c_int32),
        ("sample_rate", c_int32),
        ("hop_size", c_int32),
    ]


class codec_batch(Structure):
    _fields_ = [
        ("n_seq", c_int32),
        ("n_seq_alloc", c_int32),
        ("n_seq_max", c_int32),
        ("seq_id", POINTER(c_int32)),
        ("n_frames", POINTER(c_int32)),
        ("n_q", POINTER(c_int32)),
        ("mode", c_int),
        ("codes", POINTER(c_int32)),
        ("codes_size", c_int32),
        ("codes_used", c_int32),
        ("latent", POINTER(c_float)),
        ("latent_dim", c_int32),
        ("latent_size", c_int32),
        ("latent_used", c_int32),
        ("codes_offset", POINTER(c_int32)),
        ("latent_offset", POINTER(c_int32)),
        ("sample_rate", c_int32),
        ("hop_size", c_int32),
    ]


class codec_gguf_kv(Structure):
    _fields_ = [
        ("key", c_char_p),
        ("value", c_char_p),
    ]


class codec_gguf_metadata(Structure):
    _fields_ = [
        ("items", POINTER(codec_gguf_kv)),
        ("n_items", c_size_t),
    ]


# ─── structs (codec_lm.h) ───────────────────────────────────────────


class codec_lm_info(Structure):
    _fields_ = [
        ("kind", c_int),
        ("hidden_dim", c_int32),
        ("audio_embed_dim", c_int32),
        ("compose_audio_embed_dim", c_int32),
        ("n_codebook", c_int32),
        ("codebook_sizes", POINTER(c_int32)),
        ("delay_pattern", POINTER(c_int32)),
        ("host_arch", c_char_p),
        ("is_continuous", c_bool),
        ("patch_size", c_int32),
        ("latent_dim", c_int32),
        ("eos_code_c0", c_int32),
        ("eos_min_step", c_int32),
    ]


class codec_lm_speaker_info(Structure):
    _fields_ = [
        ("needs_ref_pcm", c_bool),
        ("needs_ref_speech_tokens", c_bool),
        ("needs_emotion_scalar", c_bool),
        ("ref_sample_rate", c_int32),
        ("emotion_default", c_float),
        ("n_rows", c_int32),
        ("hidden_dim", c_int32),
        ("speaker_emb_dim", c_int32),
    ]


class codec_lm_chatterbox_info(Structure):
    _fields_ = [
        ("hidden_dim", c_int32),
        ("text_vocab_size", c_int32),
        ("speech_vocab_size", c_int32),
        ("start_text_token", c_int32),
        ("stop_text_token", c_int32),
        ("start_speech_token", c_int32),
        ("stop_speech_token", c_int32),
        ("cond_rows", c_int32),
        ("has_tokenizer", c_int32),
        ("has_builtin_conds", c_int32),
        ("is_multilingual", c_int32),
        ("is_turbo", c_int32),
        ("backbone_tokenizer", c_int32),
        ("silence_token", c_int32),
        ("n_trailing_silence", c_int32),
        ("temperature", c_float),
        ("top_k", c_int32),
        ("top_p", c_float),
        ("min_p", c_float),
        ("repetition_penalty", c_float),
        ("cfg_weight", c_float),
    ]


class codec_lm_cosyvoice_info(Structure):
    _fields_ = [
        ("version", c_int32),
        ("hidden_dim", c_int32),
        ("speech_token_size", c_int32),
        ("n_vocab", c_int32),
        ("special_in_speech_table", c_int32),
        ("sos", c_int32),
        ("task_id", c_int32),
        ("stop_begin", c_int32),
        ("stop_end", c_int32),
        ("ignore_eos_id", c_int32),
        ("text_in_backbone", c_int32),
        ("has_text_encoder", c_int32),
        ("ras_sampling", c_int32),
        ("top_k", c_int32),
        ("top_p", c_float),
        ("win_size", c_int32),
        ("tau_r", c_float),
        ("min_token_text_ratio", c_float),
        ("max_token_text_ratio", c_float),
        ("n_silent_tokens", c_int32),
        ("silent_tokens", POINTER(c_int32)),
        ("max_silent_run", c_int32),
        ("prompt_prefix", c_char_p),
        ("bopomofo_frontend", c_int32),
    ]


class codec_lm_flow_info(Structure):
    _fields_ = [
        ("d_model", c_int32),
        ("ldim", c_int32),
        ("n_txt_bins", c_int32),
        ("insert_bos_before_voice", c_int32),
        ("frames_after_eos", c_int32),
        ("temperature", c_float),
        ("eos_threshold", c_float),
        ("lsd_decode_steps", c_int32),
        ("has_tokenizer", c_int32),
    ]


# ─── prototypes ─────────────────────────────────────────────────────

_i32p = POINTER(c_int32)
_f32p = POINTER(c_float)
_status = c_int

# codec_metadata_free is left out: the only metadata a caller can get is
# model-owned (codec_model_metadata) and freed with the model.
PROTOTYPES: dict[str, tuple[object, list[object]]] = {
    # codec.h — params / lifetime
    "codec_model_default_params": (codec_model_params, []),
    "codec_context_default_params": (codec_context_params, []),
    "codec_encode_default_params": (codec_encode_params, []),
    "codec_decode_default_params": (codec_decode_params, []),
    "codec_model_load_from_file": (c_void_p, [c_char_p, codec_model_params]),
    "codec_model_load_from_file_ext": (
        c_void_p, [c_char_p, codec_model_params, POINTER(codec_model_kv_override)]),
    "codec_model_free": (None, [c_void_p]),
    "codec_init_from_model": (c_void_p, [c_void_p, codec_context_params]),
    "codec_free": (None, [c_void_p]),
    # codec.h — encode / decode
    "codec_encode": (
        _status, [c_void_p, POINTER(codec_audio), POINTER(codec_token_buffer), codec_encode_params]),
    "codec_encode_latent": (
        _status, [c_void_p, POINTER(codec_audio), POINTER(codec_token_buffer),
                  POINTER(codec_latent_buffer), codec_encode_params]),
    "codec_decode": (
        _status, [c_void_p, POINTER(codec_token_buffer), POINTER(codec_pcm_buffer), codec_decode_params]),
    "codec_decode_quantized_representation": (
        _status, [c_void_p, _f32p, c_int32, c_int32, POINTER(codec_pcm_buffer), codec_decode_params]),
    "codec_batch_init_codes": (codec_batch, [c_int32, c_int32, c_int32]),
    "codec_batch_init_latent": (codec_batch, [c_int32, c_int32, c_int32, c_int32]),
    "codec_batch_free": (None, [codec_batch]),
    "codec_batch_add_seq_codes": (c_int32, [POINTER(codec_batch), c_int32, c_int32, c_int32, _i32p]),
    "codec_batch_add_seq_latent": (c_int32, [POINTER(codec_batch), c_int32, c_int32, _f32p, c_int32]),
    "codec_decode_batch": (
        _status, [c_void_p, POINTER(codec_batch), POINTER(codec_pcm_buffer), codec_decode_params]),
    "codec_token_buffer_free": (None, [POINTER(codec_token_buffer)]),
    "codec_pcm_buffer_free": (None, [POINTER(codec_pcm_buffer)]),
    "codec_latent_buffer_free": (None, [POINTER(codec_latent_buffer)]),
    "codec_get_last_error": (c_char_p, [c_void_p]),
    # codec.h — model info
    "codec_model_arch": (c_int, [c_void_p]),
    "codec_model_name": (c_char_p, [c_void_p]),
    "codec_model_n_tensors": (c_int32, [c_void_p]),
    "codec_model_sample_rate": (c_int32, [c_void_p]),
    "codec_model_has_encoder": (c_bool, [c_void_p]),
    "codec_model_has_decoder": (c_bool, [c_void_p]),
    "codec_model_n_q": (c_int32, [c_void_p]),
    "codec_model_codebook_size": (c_int32, [c_void_p]),
    "codec_model_hop_size": (c_int32, [c_void_p]),
    "codec_model_n_fft": (c_int32, [c_void_p]),
    "codec_model_win_length": (c_int32, [c_void_p]),
    "codec_model_n_mels": (c_int32, [c_void_p]),
    "codec_model_latent_dim": (c_int32, [c_void_p]),
    "codec_model_metadata": (POINTER(codec_gguf_metadata), [c_void_p]),
    "codec_arch_name": (c_char_p, [c_int]),
    # codec.h — speakers
    "codec_model_n_speakers": (c_int32, [c_void_p]),
    "codec_model_speaker_name": (c_char_p, [c_void_p, c_int32]),
    "codec_get_speaker": (c_void_p, [c_void_p, c_char_p]),
    "codec_model_can_speaker_from_audio": (c_bool, [c_void_p]),
    "codec_speaker_from_audio": (_status, [c_void_p, POINTER(codec_audio), POINTER(c_void_p)]),
    "codec_speaker_save": (_status, [c_void_p, c_void_p, c_char_p, c_char_p]),
    "codec_speaker_load": (_status, [c_void_p, c_char_p, c_char_p, POINTER(c_void_p)]),
    "codec_speaker_free": (None, [c_void_p]),
    "codec_speaker_file_open": (_status, [c_void_p, c_char_p, POINTER(c_void_p)]),
    "codec_speaker_file_n_speakers": (c_int32, [c_void_p]),
    "codec_speaker_file_speaker_name": (c_char_p, [c_void_p, c_int32]),
    "codec_speaker_file_get_speaker": (c_void_p, [c_void_p, c_char_p]),
    "codec_speaker_file_free": (None, [c_void_p]),
    "codec_speaker_name": (c_char_p, [c_void_p]),
    "codec_speaker_ref_text": (c_char_p, [c_void_p]),
    "codec_speaker_set_ref_text": (_status, [c_void_p, c_char_p]),
    "codec_speaker_get_slot": (
        c_bool, [c_void_p, c_char_p, POINTER(c_int), POINTER(c_int64), POINTER(c_void_p)]),
    # codec_lm.h — lifetime / info
    "codec_lm_kind_name": (c_char_p, [c_int]),
    "codec_lm_create": (c_void_p, [c_void_p]),
    "codec_lm_free": (None, [c_void_p]),
    "codec_lm_get_info": (POINTER(codec_lm_info), [c_void_p]),
    "codec_lm_get_last_error": (c_char_p, [c_void_p]),
    "codec_lm_state_get_last_error": (c_char_p, [c_void_p]),
    "codec_lm_get_create_error": (c_char_p, []),
    "codec_lm_state_new": (c_void_p, [c_void_p]),
    "codec_lm_state_free": (None, [c_void_p]),
    "codec_lm_state_reset": (None, [c_void_p]),
    "codec_lm_state_set_text_context": (_status, [c_void_p, c_int32]),
    # codec_lm.h — embeddings
    "codec_lm_audio_embd": (_f32p, [c_void_p, c_int32, c_int32]),
    "codec_lm_project_text": (c_bool, [c_void_p, c_int32, _f32p, c_int32]),
    "codec_lm_text_proj_dim": (c_int32, [c_void_p]),
    "codec_lm_codec_embd_row": (c_bool, [c_void_p, c_int32, _f32p, c_int32]),
    "codec_lm_compose_audio_embd": (_status, [c_void_p, _i32p, _f32p]),
    "codec_lm_compose_next_embd": (_status, [c_void_p, _i32p, c_int32, _f32p]),
    # codec_lm.h — codebook step API
    "codec_lm_step_begin": (_status, [c_void_p, _f32p]),
    "codec_lm_step_pending": (c_bool, [c_void_p]),
    "codec_lm_step_logits": (_f32p, [c_void_p, _i32p, _i32p]),
    "codec_lm_step_push_code": (_status, [c_void_p, c_int32]),
    "codec_lm_step_finish": (_status, [c_void_p, _i32p]),
    "codec_lm_step_is_eos": (_status, [c_void_p, _i32p, c_int32, _i32p]),
    # codec_lm.h — continuous latent
    "codec_lm_step_generate": (_status, [c_void_p, _f32p, c_float, c_int32, _f32p, _f32p, _i32p]),
    "codec_lm_step_feedback_embd": (_status, [c_void_p, _f32p]),
    "codec_lm_text_prefill": (_status, [c_void_p, _f32p, c_int32, c_int32]),
    "codec_lm_set_continuous_min_len": (_status, [c_void_p, c_int32]),
    "codec_lm_set_teacher_patch": (_status, [c_void_p, _f32p, c_int32]),
    # codec_lm.h — speaker encoder
    "codec_lm_speaker_get_info": (POINTER(codec_lm_speaker_info), [c_void_p]),
    "codec_lm_speaker_encode": (
        _status, [c_void_p, POINTER(codec_audio), _i32p, c_int32, _f32p, _f32p, c_int32]),
    "codec_lm_speaker_encode_from_embedding": (
        _status, [c_void_p, _f32p, c_int32, _i32p, c_int32, _f32p, _f32p, c_int32]),
    # codec_lm.h — Chatterbox T3
    "codec_lm_chatterbox_get_info": (POINTER(codec_lm_chatterbox_info), [c_void_p]),
    "codec_lm_chatterbox_tokenize": (_status, [c_void_p, c_char_p, _i32p, c_int32, _i32p]),
    "codec_lm_chatterbox_normalize_text": (_status, [c_void_p, c_char_p, ctypes.c_char_p, c_int32]),
    "codec_lm_chatterbox_build_prompt": (
        _status, [c_void_p, _i32p, c_int32, c_float, _f32p, c_int32, _i32p, c_int32, _f32p,
                  _f32p, c_int32, c_int32, _f32p, c_int32, _i32p, _i32p]),
    "codec_lm_chatterbox_compose_speech_embd": (_status, [c_void_p, c_int32, c_int32, _f32p, c_int32]),
    # codec_lm.h — CosyVoice
    "codec_lm_cosyvoice_get_info": (POINTER(codec_lm_cosyvoice_info), [c_void_p]),
    "codec_lm_cosyvoice_special_embd": (_status, [c_void_p, c_int32, _f32p, c_int32]),
    "codec_lm_cosyvoice_speech_embd": (_status, [c_void_p, _i32p, c_int32, _f32p, c_int32]),
    "codec_lm_cosyvoice_text_rows": (
        _status, [c_void_p, _i32p, c_int32, _f32p, c_int32, _f32p, c_int32, _i32p]),
    # codec_lm.h — FlowLM
    "codec_lm_flow_get_info": (POINTER(codec_lm_flow_info), [c_void_p]),
    "codec_lm_flow_tokenize": (_status, [c_void_p, c_char_p, _i32p, c_int32, _i32p]),
    "codec_lm_flow_speaker_rows": (_status, [c_void_p, _f32p, c_int32, _f32p, c_int32]),
    "codec_lm_flow_prefill": (_status, [c_void_p, _i32p, c_int32, _f32p, c_int32]),
    "codec_lm_flow_step": (_status, [c_void_p, _f32p, _f32p, _f32p, _i32p]),
    "codec_lm_flow_denorm_latent": (_status, [c_void_p, _f32p, _f32p]),
}


def declare(lib: ctypes.CDLL) -> None:
    """Attach restype/argtypes for every public symbol.

    Raises AttributeError when the library lacks a symbol from the headers,
    i.e. the Python bindings and the built library are out of sync.
    """
    for name, (restype, argtypes) in PROTOTYPES.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
