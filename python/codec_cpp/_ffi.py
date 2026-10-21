"""Raw ctypes declarations mirroring include/codec.h, include/codec_speaker.h,
include/codec_lm.h and include/codec_audio_lm.h; the subset of the bundled
llama.cpp's llama.h that codec_cpp.TTS drives at the end.

Struct field order must match the headers exactly; the headers follow an
ABI-append discipline (new fields at the end), so keep the same here.
"""

from __future__ import annotations

import ctypes
from ctypes import (POINTER, Structure, Union, c_bool, c_char, c_char_p, c_double,
                    c_float, c_int, c_int32, c_int64, c_size_t, c_uint32, c_void_p)

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

CODEC_LOG_LEVEL_NONE = 0
CODEC_LOG_LEVEL_INFO = 1
CODEC_LOG_LEVEL_WARN = 2
CODEC_LOG_LEVEL_ERROR = 3
CODEC_LOG_LEVEL_DEBUG = 4
CODEC_LOG_LEVEL_CONT = 5

# typedef void (*codec_log_callback)(enum codec_log_level, const char *, void *)
codec_log_callback = ctypes.CFUNCTYPE(None, c_int, c_char_p, c_void_p)

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


# ─── codec_audio_lm.h ───────────────────────────────────────────────

CODEC_AUDIO_LM_INPUT_TEXT = 1 << 0
CODEC_AUDIO_LM_INPUT_AUDIO = 1 << 1
CODEC_AUDIO_LM_OUTPUT_TEXT = 1 << 2
CODEC_AUDIO_LM_OUTPUT_AUDIO = 1 << 3

CODEC_AUDIO_LM_CHUNK_TEXT = 0
CODEC_AUDIO_LM_CHUNK_TOKENS = 1
CODEC_AUDIO_LM_CHUNK_EMBD = 2

CODEC_AUDIO_LM_FLOW_TOKENS = 0
CODEC_AUDIO_LM_FLOW_CODEC_HEAD = 1
CODEC_AUDIO_LM_FLOW_HIDDEN = 2
CODEC_AUDIO_LM_FLOW_SELF_CONTAINED = 3

CODEC_AUDIO_LM_FEED_TOKEN = 0
CODEC_AUDIO_LM_FEED_EMBD = 1
CODEC_AUDIO_LM_STOP = 2
CODEC_AUDIO_LM_SAMPLE_HEAD = 3


class codec_audio_lm_params(Structure):
    _fields_ = [
        ("codec_path", c_char_p),
        ("backbone_path", c_char_p),
        ("kv_overrides", POINTER(codec_model_kv_override)),
        ("n_kv_overrides", c_int32),
        ("n_threads", c_int32),
        ("use_gpu", c_bool),
    ]


class codec_audio_lm_caps(Structure):
    _fields_ = [
        ("modality", ctypes.c_uint32),
        ("needs_backbone", c_bool),
        ("needs_backbone_path", c_bool),
        ("text_needs_phonemes", c_bool),
        ("takes_ref_audio", c_bool),
        ("takes_ref_text", c_bool),
        ("takes_instruct", c_bool),
        ("needs_voice", c_bool),
        ("sample_rate", c_int32),
        ("segment_bytes", c_int32),
    ]


# int32_t (*tokenize)(void *, const char *, int32_t, int32_t *, int32_t, bool, bool)
codec_audio_lm_tokenize_fn = ctypes.CFUNCTYPE(c_int32, c_void_p, POINTER(c_char), c_int32, POINTER(c_int32),
                                              c_int32, c_bool, c_bool)
# int32_t (*token_to_piece)(void *, int32_t, char *, int32_t)
codec_audio_lm_piece_fn = ctypes.CFUNCTYPE(c_int32, c_void_p, c_int32, POINTER(c_char), c_int32)
# bool (*is_eog)(void *, int32_t)
codec_audio_lm_is_eog_fn = ctypes.CFUNCTYPE(c_bool, c_void_p, c_int32)


class codec_audio_lm_vocab(Structure):
    _fields_ = [
        ("user_data", c_void_p),
        ("tokenize", codec_audio_lm_tokenize_fn),
        ("token_to_piece", codec_audio_lm_piece_fn),
        ("n_tokens", c_int32),
        ("eos", c_int32),
        ("is_eog", codec_audio_lm_is_eog_fn),
    ]


class codec_audio_lm_backbone_req(Structure):
    _fields_ = [
        ("n_ctx", c_int32),
        ("n_ubatch", c_int32),
        ("n_seq_max", c_int32),
        ("embeddings", c_bool),
        ("flash_attn", c_bool),
    ]


CODEC_AUDIO_LM_DEFAULT_SEED = 0xFFFFFFFF


class codec_audio_lm_tts_request(Structure):
    _fields_ = [
        ("text", c_char_p),
        ("voice", c_void_p),
        ("instruct", c_char_p),
        ("min_frames", c_int32),
        ("max_frames", c_int32),
        ("lm_guidance", c_float),
        ("flow_guidance", c_float),
        ("flow_steps", c_int32),
    ]


class codec_audio_lm_chunk(Structure):
    _fields_ = [
        ("type", c_int),
        ("text", c_char_p),
        ("add_special", c_bool),
        ("parse_special", c_bool),
        ("tokens", POINTER(c_int32)),
        ("embd", POINTER(c_float)),
        ("n", c_int32),
    ]


class codec_audio_lm_token_range(Structure):
    _fields_ = [("begin", c_int32), ("end", c_int32)]


class codec_audio_lm_head_group(Structure):
    _fields_ = [
        ("penalties", c_bool),
        ("history", POINTER(c_int32)),
        ("n_history", c_int32),
        ("penalty_exempt", POINTER(c_int32)),
        ("n_penalty_exempt", c_int32),
    ]


class codec_audio_lm_completion(Structure):
    _fields_ = [
        ("flow", c_int),
        ("n_seq", c_int32),
        ("n_embd", c_int32),
        ("chunks", POINTER(codec_audio_lm_chunk)),
        ("n_chunks", c_int32),
        ("grammar", c_char_p),
        ("penalty_exempt", POINTER(c_int32)),
        ("n_penalty_exempt", c_int32),
        ("allowed", POINTER(codec_audio_lm_token_range)),
        ("n_allowed", c_int32),
        ("needs_embeddings", c_bool),
        ("needs_prompt_hidden", c_bool),
        ("sample_first", c_bool),
        ("streamable", c_bool),
        ("max_frames", c_int32),
        ("head_groups", POINTER(codec_audio_lm_head_group)),
        ("n_head_groups", c_int32),
    ]


class codec_audio_lm_verdict(Structure):
    _fields_ = [
        ("action", c_int),
        ("sample_backbone", c_bool),
        ("is_audio", c_bool),
        ("token", c_int32),
        ("head_group", c_int32),
        ("n_vocab", c_int32),
    ]


CODEC_AUDIO_LM_DATA_CODES = 0
CODEC_AUDIO_LM_DATA_LATENTS = 1


class codec_audio_lm_generation(Structure):
    _fields_ = [
        ("kind", c_int),
        ("n_frames", c_int32),
        ("width", c_int32),
        ("codes", POINTER(c_int32)),
        ("latents", POINTER(c_float)),
    ]


# ─── prototypes ─────────────────────────────────────────────────────

_i32p = POINTER(c_int32)
_f32p = POINTER(c_float)
_status = c_int

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
    # codec_speaker.h
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
    # codec_lm.h — speaker encoder
    "codec_lm_speaker_get_info": (POINTER(codec_lm_speaker_info), [c_void_p]),
    "codec_lm_speaker_encode": (
        _status, [c_void_p, POINTER(codec_audio), _i32p, c_int32, _f32p, _f32p, c_int32]),
    "codec_lm_speaker_encode_from_embedding": (
        _status, [c_void_p, _f32p, c_int32, _i32p, c_int32, _f32p, _f32p, c_int32]),
    # codec.h — logging (codec_log is variadic: call it as codec_log(level, b"%s", msg))
    "codec_log_set": (None, [codec_log_callback, c_void_p]),
    # codec_audio_lm.h
    "codec_audio_lm_default_params": (codec_audio_lm_params, []),
    "codec_audio_lm_init": (c_void_p, [POINTER(codec_audio_lm_params), c_char_p, c_size_t]),
    "codec_audio_lm_free": (None, [c_void_p]),
    "codec_audio_lm_last_error": (c_char_p, [c_void_p]),
    "codec_audio_lm_meta_str": (c_char_p, [c_void_p, c_char_p]),
    "codec_audio_lm_get_caps": (None, [c_void_p, POINTER(codec_audio_lm_caps)]),
    "codec_audio_lm_list_voices": (c_int32, [c_void_p, c_char_p]),
    "codec_audio_lm_voice": (c_bool, [c_void_p, c_int32, POINTER(c_char_p), POINTER(c_char_p), POINTER(c_bool)]),
    "codec_audio_lm_backbone_requirements": (
        c_bool, [c_void_p, POINTER(codec_audio_lm_vocab), c_int32, POINTER(codec_audio_lm_backbone_req)]),
    "codec_audio_lm_get_voice": (c_void_p, [c_void_p, c_char_p]),
    "codec_audio_lm_create_voice": (c_void_p, [c_void_p, _f32p, c_int32, c_int32, c_char_p]),
    "codec_audio_lm_load_voice": (c_void_p, [c_void_p, c_char_p, c_char_p]),
    "codec_audio_lm_save_voice": (c_bool, [c_void_p, c_void_p, c_char_p, c_char_p]),
    "codec_audio_lm_default_tts_request": (codec_audio_lm_tts_request, []),
    "codec_audio_lm_tts_prompt": (
        c_bool, [c_void_p, POINTER(codec_audio_lm_tts_request), POINTER(codec_audio_lm_vocab)]),
    "codec_audio_lm_tts_split": (c_int32, [c_void_p, c_char_p, c_int32]),
    "codec_audio_lm_tts_segment": (c_char_p, [c_void_p, c_int32]),
    "codec_audio_lm_tts_join": (c_int32, [c_void_p, POINTER(codec_pcm_buffer), c_int32, c_int32]),
    "codec_audio_lm_get_completion": (c_bool, [c_void_p, POINTER(codec_audio_lm_completion)]),
    "codec_audio_lm_prefill": (c_bool, [c_void_p, _f32p, c_int32]),
    "codec_audio_lm_observe": (c_bool, [c_void_p, _i32p, _f32p, POINTER(codec_audio_lm_verdict)]),
    "codec_audio_lm_next_embd": (_f32p, [c_void_p, c_int32]),
    "codec_audio_lm_head_logits": (_f32p, [c_void_p]),
    "codec_audio_lm_set_seed": (c_bool, [c_void_p, c_uint32]),
    "codec_audio_lm_head_token": (c_bool, [c_void_p, c_int32, POINTER(codec_audio_lm_verdict)]),
    "codec_audio_lm_stop_reason": (c_char_p, [c_void_p]),
    "codec_audio_lm_generate_self_contained": (c_bool, [c_void_p]),
    "codec_audio_lm_decode": (c_bool, [c_void_p, POINTER(codec_pcm_buffer)]),
    "codec_audio_lm_get_generation": (c_bool, [c_void_p, POINTER(codec_audio_lm_generation)]),
    "codec_audio_lm_decode_generation": (c_bool, [c_void_p, POINTER(codec_audio_lm_generation), c_void_p,
                                                  POINTER(codec_pcm_buffer)]),
    "codec_audio_lm_stream_poll": (c_bool, [c_void_p, c_int32, POINTER(codec_pcm_buffer)]),
    "codec_audio_lm_stream_finish": (c_bool, [c_void_p, POINTER(codec_pcm_buffer)]),
    "codec_audio_lm_decodable_frames": (c_int32, [c_void_p]),
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


# ─── llama.h (the bundled llama.cpp, libttsbackbone) ─────────────────
# The subset codec_cpp.TTS drives, mirroring the pinned
# common/third-party/llama.cpp/include/llama.h (with codec.cpp's patches);
# update it with the submodule.

llama_token = c_int32


class llama_model_params(Structure):
    _fields_ = [
        ("devices", c_void_p),
        ("tensor_buft_overrides", c_void_p),
        ("n_gpu_layers", c_int32),
        ("split_mode", c_int),
        ("main_gpu", c_int32),
        ("tensor_split", c_void_p),
        ("progress_callback", c_void_p),
        ("progress_callback_user_data", c_void_p),
        ("kv_overrides", c_void_p),
        ("vocab_only", c_bool),
        ("use_mmap", c_bool),
        ("use_direct_io", c_bool),
        ("use_mlock", c_bool),
        ("check_tensors", c_bool),
        ("use_extra_bufts", c_bool),
        ("no_host", c_bool),
        ("no_alloc", c_bool),
    ]


LLAMA_POOLING_TYPE_NONE = 0
LLAMA_FLASH_ATTN_TYPE_DISABLED = 0


class llama_context_params(Structure):
    _fields_ = [
        ("n_ctx", c_uint32),
        ("n_batch", c_uint32),
        ("n_ubatch", c_uint32),
        ("n_seq_max", c_uint32),
        ("n_rs_seq", c_uint32),
        ("n_outputs_max", c_uint32),
        ("n_threads", c_int32),
        ("n_threads_batch", c_int32),
        ("ctx_type", c_int),
        ("rope_scaling_type", c_int),
        ("pooling_type", c_int),
        ("attention_type", c_int),
        ("flash_attn_type", c_int),
        ("rope_freq_base", c_float),
        ("rope_freq_scale", c_float),
        ("yarn_ext_factor", c_float),
        ("yarn_attn_factor", c_float),
        ("yarn_beta_fast", c_float),
        ("yarn_beta_slow", c_float),
        ("yarn_orig_ctx", c_uint32),
        ("defrag_thold", c_float),
        ("cb_eval", c_void_p),
        ("cb_eval_user_data", c_void_p),
        ("type_k", c_int),
        ("type_v", c_int),
        ("abort_callback", c_void_p),
        ("abort_callback_data", c_void_p),
        ("embeddings", c_bool),
        ("offload_kqv", c_bool),
        ("no_perf", c_bool),
        ("op_offload", c_bool),
        ("swa_full", c_bool),
        ("kv_unified", c_bool),
        ("samplers", c_void_p),
        ("n_samplers", c_size_t),
        ("ctx_other", c_void_p),
    ]


class llama_batch(Structure):
    _fields_ = [
        ("n_tokens", c_int32),
        ("token", POINTER(llama_token)),
        ("embd", POINTER(c_float)),
        ("pos", POINTER(c_int32)),
        ("n_seq_id", POINTER(c_int32)),
        ("seq_id", POINTER(POINTER(c_int32))),
        ("logits", POINTER(ctypes.c_int8)),
    ]


class llama_sampler_chain_params(Structure):
    _fields_ = [("no_perf", c_bool)]


class llama_token_data(Structure):
    _fields_ = [("id", llama_token), ("logit", c_float), ("p", c_float)]


class llama_token_data_array(Structure):
    _fields_ = [("data", POINTER(llama_token_data)), ("size", c_size_t), ("selected", c_int64), ("sorted", c_bool)]


class llama_logit_bias(Structure):
    _fields_ = [("token", llama_token), ("bias", c_float)]


_P = c_void_p
LLAMA_PROTOTYPES: dict[str, tuple[object, list[object]]] = {
    "llama_backend_init": (None, []),
    "llama_log_set": (None, [codec_log_callback, c_void_p]),
    "llama_model_default_params": (llama_model_params, []),
    "llama_model_load_from_file": (_P, [c_char_p, llama_model_params]),
    "llama_model_free": (None, [_P]),
    "llama_model_get_vocab": (_P, [_P]),
    "llama_model_n_embd": (c_int32, [_P]),
    "llama_model_n_ctx_train": (c_int32, [_P]),
    "llama_model_meta_val_str": (c_int32, [_P, c_char_p, c_char_p, c_size_t]),
    "llama_context_default_params": (llama_context_params, []),
    "llama_init_from_model": (_P, [_P, llama_context_params]),
    "llama_free": (None, [_P]),
    "llama_n_ctx": (c_uint32, [_P]),
    "llama_get_memory": (_P, [_P]),
    "llama_memory_clear": (None, [_P, c_bool]),
    "llama_batch_init": (llama_batch, [c_int32, c_int32, c_int32]),
    "llama_batch_free": (None, [llama_batch]),
    "llama_decode": (c_int32, [_P, llama_batch]),
    "llama_get_embeddings_ith": (POINTER(c_float), [_P, c_int32]),
    "llama_tokenize": (c_int32, [_P, c_char_p, c_int32, POINTER(llama_token), c_int32, c_bool, c_bool]),
    "llama_token_to_piece": (c_int32, [_P, llama_token, c_char_p, c_int32, c_int32, c_bool]),
    "llama_vocab_n_tokens": (c_int32, [_P]),
    "llama_vocab_type": (c_int, [_P]),
    "llama_vocab_eos": (llama_token, [_P]),
    "llama_vocab_is_eog": (c_bool, [_P, llama_token]),
    "llama_sampler_chain_default_params": (llama_sampler_chain_params, []),
    "llama_sampler_chain_init": (_P, [llama_sampler_chain_params]),
    "llama_sampler_chain_add": (None, [_P, _P]),
    "llama_sampler_chain_get": (_P, [_P, c_int32]),
    "llama_sampler_init_logit_bias": (_P, [c_int32, c_int32, POINTER(llama_logit_bias)]),
    "llama_sampler_init_grammar": (_P, [_P, c_char_p, c_char_p]),
    "llama_sampler_init_penalties_exempt": (_P, [c_int32, c_float, c_float, c_float, POINTER(llama_token),
                                                 c_size_t]),
    "llama_sampler_init_dry": (_P, [_P, c_int32, c_float, c_float, c_int32, c_int32, POINTER(c_char_p),
                                    c_size_t]),
    "llama_sampler_init_top_k": (_P, [c_int32]),
    "llama_sampler_init_top_p": (_P, [c_float, c_size_t]),
    "llama_sampler_init_min_p": (_P, [c_float, c_size_t]),
    "llama_sampler_init_typical": (_P, [c_float, c_size_t]),
    "llama_sampler_init_temp_ext": (_P, [c_float, c_float, c_float]),
    "llama_sampler_init_dist": (_P, [c_uint32]),
    "llama_sampler_sample": (llama_token, [_P, _P, c_int32]),
    "llama_sampler_accept": (None, [_P, llama_token]),
    "llama_sampler_apply": (None, [_P, POINTER(llama_token_data_array)]),
    "llama_sampler_free": (None, [_P]),
}


def declare_llama(lib: ctypes.CDLL) -> None:
    """declare() for the bundled llama.cpp."""
    for name, (restype, argtypes) in LLAMA_PROTOTYPES.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
