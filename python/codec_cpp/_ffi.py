"""Raw ctypes declarations mirroring include/codec.h, include/codec_lm.h and
include/codec_audio_lm.h; codec_tts.h (libcodec_tts) at the end.

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


# ─── enums + structs (codec_audio_lm.h) ─────────────────────────────

CODEC_AUDIO_LM_INPUT_TEXT = 1 << 0
CODEC_AUDIO_LM_INPUT_AUDIO = 1 << 1
CODEC_AUDIO_LM_OUTPUT_TEXT = 1 << 2
CODEC_AUDIO_LM_OUTPUT_AUDIO = 1 << 3

CODEC_AUDIO_LM_OBSERVE_PASSTHROUGH = 0
CODEC_AUDIO_LM_OBSERVE_CONSUMED = 1
CODEC_AUDIO_LM_OBSERVE_CONSUMED_EMBED = 2
CODEC_AUDIO_LM_OBSERVE_STOP = 3

CODEC_AUDIO_LM_KIND_UNKNOWN = 0
CODEC_AUDIO_LM_KIND_RESIDUAL_DEPTH_AR = 1
CODEC_AUDIO_LM_KIND_PARALLEL_HEADS_DELAY = 2
CODEC_AUDIO_LM_KIND_CONTINUOUS_CFM = 3
CODEC_AUDIO_LM_KIND_TOKEN_SINGLE_CB = 4

CODEC_AUDIO_LM_OUTETTS_V1 = 0
CODEC_AUDIO_LM_OUTETTS_V2 = 1
CODEC_AUDIO_LM_OUTETTS_V3 = 2


class codec_audio_lm_params(Structure):
    _fields_ = [
        ("codec_path", c_char_p),
        ("use_gpu", c_bool),
        ("n_threads", c_int32),
        ("kv_overrides", POINTER(codec_model_kv_override)),
        ("n_kv_overrides", c_int32),
    ]


class codec_audio_lm_input(Structure):
    _fields_ = [
        ("ref_pcm", POINTER(c_float)),
        ("ref_n_samples", c_int32),
        ("ref_sample_rate", c_int32),
        ("speaker_emb", POINTER(c_float)),
        ("speaker_emb_dim", c_int32),
        ("emotion", POINTER(c_float)),
        ("ref_text", c_char_p),
        ("text", c_char_p),
        ("extra_keys", POINTER(c_char_p)),
        ("extra_values", POINTER(c_char_p)),
        ("n_extra", c_int32),
    ]


class codec_audio_lm_prompt(Structure):
    _fields_ = [
        ("tokens", POINTER(c_int32)),
        ("n_tokens", c_int32),
        ("embeds_prefix", POINTER(c_float)),
        ("embeds_prefix_rows", c_int32),
        ("embeds_prefix_hidden", c_int32),
        ("embeds_uncond", POINTER(c_float)),
        ("default_temperature", c_float),
        ("default_top_p", c_float),
        ("default_min_p", c_float),
        ("default_repetition_penalty", c_float),
        ("default_cfg_weight", c_float),
        ("start_token", c_int32),
        ("stop_token", c_int32),
    ]


class codec_audio_lm_prompt_info(Structure):
    _fields_ = [
        ("model_kind", c_int),
        ("host_arch", c_char_p),
        ("prompt_prefix", c_char_p),
        ("prompt_suffix", c_char_p),
        ("add_bos", c_bool),
        ("parse_special", c_bool),
        ("cb0_from_backbone", c_bool),
        ("audio_codebook_offset", c_int32),
        ("is_continuous", c_bool),
        ("n_codebook", c_int32),
        ("hidden_dim", c_int32),
        ("eos_code_c0", c_int32),
        ("eos_min_step", c_int32),
        ("default_temperature", c_float),
        ("default_top_p", c_float),
        ("default_top_k", c_int32),
        ("streaming_interleave", c_bool),
        ("text_externally_added", c_bool),
        ("prefill_text_len", c_int32),
        ("text_pad_id", c_int32),
        ("audio_pad_code", c_int32),
        ("bos_code_c0", c_int32),
        ("default_repetition_penalty", c_float),
        ("repetition_window", c_int32),
        ("sequential_text_audio", c_bool),
        ("audio_start_id", c_int32),
        ("text_end_id", c_int32),
        ("max_text_tokens", c_int32),
        ("cb0_speech_range_start", c_int32),
        ("cb0_speech_range_end", c_int32),
        ("audio_tok_offset", c_int32),
        ("audio_tok_count", c_int32),
        ("audio_tok_eos", c_int32),
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
    # codec_audio_lm.h — lifecycle / capabilities
    "codec_audio_lm_default_params": (codec_audio_lm_params, []),
    "codec_audio_lm_init": (c_void_p, [POINTER(codec_audio_lm_params), c_char_p, c_size_t]),
    "codec_audio_lm_free": (None, [c_void_p]),
    "codec_audio_lm_reset": (None, [c_void_p]),
    "codec_audio_lm_alloc_bare": (c_void_p, []),
    "codec_audio_lm_parse_kv_override": (c_bool, [c_char_p, POINTER(codec_model_kv_override)]),
    "codec_audio_lm_modality": (ctypes.c_uint32, [c_void_p]),
    "codec_audio_lm_has_speaker_enc": (c_bool, [c_void_p]),
    "codec_audio_lm_n_codebook": (c_int32, [c_void_p]),
    "codec_audio_lm_hidden_dim": (c_int32, [c_void_p]),
    "codec_audio_lm_is_continuous": (c_bool, [c_void_p]),
    "codec_audio_lm_last_error": (c_char_p, [c_void_p]),
    "codec_audio_lm_meta_str": (c_char_p, [c_void_p, c_char_p]),
    "codec_audio_lm_get_lm": (c_void_p, [c_void_p]),
    # codec_audio_lm.h — decoder speaker
    "codec_audio_lm_codec_has_speakers": (c_bool, [c_void_p]),
    "codec_audio_lm_codec_can_speaker_from_audio": (c_bool, [c_void_p]),
    "codec_audio_lm_set_decode_speaker": (c_bool, [c_void_p, c_char_p]),
    "codec_audio_lm_load_decode_speaker": (c_bool, [c_void_p, c_char_p, c_char_p]),
    "codec_audio_lm_decode_speaker_from_audio": (c_bool, [c_void_p, _f32p, c_int32, c_int32, c_char_p]),
    "codec_audio_lm_save_decode_speaker": (c_bool, [c_void_p, c_char_p, c_char_p]),
    "codec_audio_lm_decode_speaker": (c_void_p, [c_void_p]),
    "codec_audio_lm_list_codec_speakers": (c_int32, [c_void_p]),
    "codec_audio_lm_list_speaker_file": (c_int32, [c_void_p, c_char_p]),
    "codec_audio_lm_listed_speaker": (
        c_bool, [c_void_p, c_int32, POINTER(c_char_p), POINTER(c_char_p), POINTER(c_bool)]),
    # codec_audio_lm.h — prompt
    "codec_audio_lm_default_input": (codec_audio_lm_input, []),
    "codec_audio_lm_build_prompt": (
        c_bool, [c_void_p, POINTER(codec_audio_lm_input), POINTER(codec_audio_lm_prompt)]),
    "codec_audio_lm_get_prompt_info": (c_bool, [c_void_p, POINTER(codec_audio_lm_prompt_info)]),
    "codec_audio_lm_outetts_build_grammar": (c_int32, [c_int, c_char_p, c_char_p, c_int32]),
    "codec_audio_lm_tts_auto_grammar": (c_int32, [c_void_p, c_char_p, c_char_p, c_int32]),
    # codec_audio_lm.h — token ranges / accumulator
    "codec_audio_lm_set_audio_token_range": (None, [c_void_p, c_int32, c_int32, c_int32]),
    "codec_audio_lm_set_audio_token_ranges": (None, [c_void_p, _i32p, _i32p, c_int32, c_int32, c_int32]),
    "codec_audio_lm_get_audio_token_range": (None, [c_void_p, _i32p, _i32p, _i32p]),
    "codec_audio_lm_codes_n_frames": (c_int32, [c_void_p]),
    "codec_audio_lm_codes": (_i32p, [c_void_p, _i32p, _i32p]),
    "codec_audio_lm_get_lm_eos": (None, [c_void_p, _i32p, _i32p]),
    "codec_audio_lm_set_uses_embed_override": (None, [c_void_p, c_bool, c_int32]),
    "codec_audio_lm_get_uses_embed_override": (c_bool, [c_void_p]),
    # codec_audio_lm.h — per-step hooks
    "codec_audio_lm_observe_token": (c_int, [c_void_p, c_int32, _f32p, c_int32]),
    "codec_audio_lm_get_next_embed": (_f32p, [c_void_p, _i32p]),
    "codec_audio_lm_observe_hidden": (c_int, [c_void_p, _f32p, c_int32, _f32p]),
    "codec_audio_lm_set_continuous_params": (None, [c_void_p, c_float, c_int32, c_int32]),
    "codec_audio_lm_text_prefill": (c_bool, [c_void_p, _f32p, c_int32, c_int32]),
    "codec_audio_lm_observe_codes": (c_int, [c_void_p, _i32p, c_int32, _f32p, c_int32]),
    "codec_audio_lm_push_codes": (c_bool, [c_void_p, _i32p, c_int32, c_int32]),
    "codec_audio_lm_step_set_text_context": (c_bool, [c_void_p, c_int32]),
    "codec_audio_lm_step_begin": (c_bool, [c_void_p, _f32p, c_int32]),
    "codec_audio_lm_step_logits": (_f32p, [c_void_p, _i32p, _i32p]),
    "codec_audio_lm_step_push_code": (c_bool, [c_void_p, c_int32]),
    "codec_audio_lm_step_finish": (c_bool, [c_void_p, _i32p, c_int32]),
    # codec_audio_lm.h — embeddings
    "codec_audio_lm_prompt_needs_composed_embd": (c_bool, [c_void_p]),
    "codec_audio_lm_compose_prompt_embd": (c_bool, [c_void_p, c_int32, _f32p, c_int32]),
    "codec_audio_lm_compose_audio_codes_embd": (c_bool, [c_void_p, _i32p, c_int32, _f32p, c_int32]),
    "codec_audio_lm_talker_has_projection": (c_bool, [c_void_p]),
    "codec_audio_lm_build_talker_prefix": (
        c_bool, [c_void_p, _i32p, c_int32, _i32p, c_int32, _f32p, c_int32, _f32p, c_int32, _i32p, _i32p]),
    "codec_audio_lm_talker_trailing_text_embd": (c_bool, [c_void_p, _i32p, c_int32, c_int32, _f32p, c_int32]),
    # codec_audio_lm.h — end of sequence
    "codec_audio_lm_decode_audio": (c_bool, [c_void_p, POINTER(codec_pcm_buffer)]),
    "codec_audio_lm_decode_latents": (c_bool, [c_void_p, _f32p, c_int32, c_int32, POINTER(codec_pcm_buffer)]),
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


# ─── codec_tts.h (libcodec_tts) ─────────────────────────────────────


class codec_tts_params(Structure):
    _fields_ = [
        ("codec_path", c_char_p),
        ("backbone_path", c_char_p),
        ("kv_overrides", POINTER(codec_model_kv_override)),
        ("n_kv_overrides", c_int32),
        ("n_threads", c_int32),
        ("use_gpu", c_bool),
        ("max_frames", c_int32),
        ("cfg_weight", c_float),
    ]


class codec_tts_request(Structure):
    _fields_ = [
        ("text", c_char_p),
        ("voice", c_char_p),
        ("speaker_file", c_char_p),
        ("ref_audio_path", c_char_p),
        ("ref_text", c_char_p),
        ("save_speaker", c_char_p),
        ("instruct", c_char_p),
        ("preset", c_char_p),
        ("grammar", c_char_p),
        ("seed", ctypes.c_uint32),
        ("temperature", c_float),
        ("top_p", c_float),
        ("top_k", c_int32),
        ("min_p", c_float),
        ("repetition_penalty", c_float),
        ("cfg", c_float),
        ("timesteps", c_int32),
        ("min_len", c_int32),
    ]


class codec_tts_result(Structure):
    _fields_ = [
        ("n_frames", c_int32),
        ("stop_reason", c_char * 32),
        ("total_ms", c_double),
        ("decode_ms", c_double),
    ]


TTS_PROTOTYPES: dict[str, tuple[object, list[object]]] = {
    "codec_tts_default_params": (codec_tts_params, []),
    "codec_tts_default_request": (codec_tts_request, []),
    "codec_tts_load": (c_void_p, [POINTER(codec_tts_params), c_char_p, c_size_t]),
    "codec_tts_free": (None, [c_void_p]),
    "codec_tts_run": (c_bool, [c_void_p, POINTER(codec_tts_request), POINTER(codec_pcm_buffer),
                               POINTER(codec_tts_result)]),
    "codec_tts_last_error": (c_char_p, [c_void_p]),
    "codec_tts_voices": (c_int32, [c_void_p, c_char_p]),
    "codec_tts_voice": (c_char_p, [c_void_p, c_int32]),
    "codec_tts_default_voice": (c_char_p, [c_void_p]),
    "codec_tts_takes_instruct": (c_bool, [c_void_p]),
}


def declare_tts(lib: ctypes.CDLL) -> None:
    """declare() for libcodec_tts."""
    for name, (restype, argtypes) in TTS_PROTOTYPES.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
