"""Recommended sampling of each TTS model, written into its BACKBONE GGUF.

Sampling belongs to the backbone (the AR model whose output distribution it
shapes), so its recommended settings ship as llama.cpp's standard
`general.sampling.*` keys (plus the freq / presence / DRY keys of
patches/sampling-meta-llamacpp.patch).  Hosts read them as llama.cpp does
(common_init_sampler_from_model); a setting the user gives wins.  The values
reproduce each model's reference generate(); docs/models/<model>.md lists them.

Names are llama.cpp common_params_sampling's;
"samplers" is the stage order (general.sampling.sequence).
"""

from __future__ import annotations

from pathlib import Path

_HF = "penalties;temperature;top_k;top_p"   # HF warper order (temperature first)

MODEL_SAMPLING: dict[str, dict] = {
    "csm":               dict(temp=0.9, top_k=50, top_p=1.0, min_p=0.0, samplers=_HF),   # generation_config.json
    "qwen3_tts":         dict(temp=0.9, top_k=50, top_p=1.0, min_p=0.0, penalty_repeat=1.05, penalty_last_n=-1,
                              samplers=_HF),
    "moss_ttsd":         dict(temp=0.9, top_k=50, top_p=0.95, min_p=0.0, samplers=_HF),
    "moss_tts_realtime": dict(temp=0.8, top_k=30, top_p=0.6, min_p=0.0, penalty_repeat=1.1, penalty_last_n=50,
                              samplers=_HF),
    "lfm2_audio":        dict(temp=0.0, top_k=0, top_p=1.0, min_p=0.0),
    # tts.py::generate.  mtl_tts.py uses repetition_penalty 2.0, but only with
    # its AlignmentStreamAnalyzer (attention-driven EOS forcing), which this
    # runtime does not have; without it 2.0 runs to max length on half the
    # test sentences, so both variants keep 1.2.
    "chatterbox":        dict(temp=0.8, top_k=0, top_p=1.0, min_p=0.05, penalty_repeat=1.2, penalty_last_n=-1,
                              samplers="penalties;temperature;min_p;top_p"),
    "chatterbox_turbo":  dict(temp=0.8, top_k=1000, top_p=0.95, min_p=0.0, penalty_repeat=1.2, penalty_last_n=-1,
                              samplers="temperature;top_k;top_p;penalties"),
    "kitten_tts2":       dict(temp=0.8, top_k=50, top_p=0.8, min_p=0.0, penalty_repeat=1.1, penalty_last_n=50,
                              dry_multiplier=1.3, dry_base=1.5, dry_allowed_length=10, dry_penalty_last_n=-1,
                              samplers="dry;penalties;temperature;top_k;top_p;min_p"),
    "soprano":           dict(temp=0.0, top_k=0, top_p=0.95, min_p=0.0, penalty_repeat=1.2, penalty_last_n=-1),
    "neutts":            dict(temp=1.0, top_k=50, top_p=1.0, min_p=0.0),
    "outetts_1_0":       dict(temp=1.0, top_k=50, top_p=1.0, min_p=0.0, penalty_repeat=1.1, penalty_last_n=64),
    "outetts_legacy":    dict(temp=0.1, top_k=50, top_p=1.0, min_p=0.0, penalty_repeat=1.1, penalty_last_n=-1),
    # RAS sampling (nucleus top-p 0.8 / top-k 25, redrawn when the pick repeats
    # in the last 10 tokens) approximated by a repetition penalty over that window.
    "cosyvoice":         dict(temp=1.0, top_k=25, top_p=0.8, min_p=0.0, penalty_repeat=1.1, penalty_last_n=10,
                              samplers="penalties;top_k;top_p"),
    "breezyvoice":       dict(temp=1.0, top_k=25, top_p=1.0, min_p=0.0),
}

# common_params_sampling name -> general.sampling.<key>, GGUF type
_KEYS = {
    "samplers":           ("sequence", "str"),
    "temp":               ("temp", "f32"),
    "top_k":              ("top_k", "i32"),
    "top_p":              ("top_p", "f32"),
    "min_p":              ("min_p", "f32"),
    "penalty_last_n":     ("penalty_last_n", "i32"),
    "penalty_repeat":     ("penalty_repeat", "f32"),
    "penalty_freq":       ("penalty_freq", "f32"),
    "penalty_present":    ("penalty_present", "f32"),
    "dry_multiplier":     ("dry_multiplier", "f32"),
    "dry_base":           ("dry_base", "f32"),
    "dry_allowed_length": ("dry_allowed_length", "i32"),
    "dry_penalty_last_n": ("dry_penalty_last_n", "i32"),
}


def sampling_kv(model: str) -> dict[str, tuple[str, object]]:
    """{general.sampling.<key>: (type, value)} for a MODEL_SAMPLING entry."""
    out = {}
    for name, value in MODEL_SAMPLING[model].items():
        key, ty = _KEYS[name]
        out[f"general.sampling.{key}"] = (ty, value)
    return out


def add_sampling_to_gguf(src: Path, dst: Path, model: str) -> None:
    """Copy `src` to `dst` with the model's general.sampling.* keys (replacing any)."""
    import gguf
    from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata

    types = {"str": gguf.GGUFValueType.STRING, "f32": gguf.GGUFValueType.FLOAT32,
             "i32": gguf.GGUFValueType.INT32}
    reader = gguf.GGUFReader(str(src), "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    writer = gguf.GGUFWriter(str(dst), arch=arch.contents(), endianess=reader.endianess)
    align = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if align is not None:
        writer.data_alignment = align.contents()
    new = {k: MetadataDetails(types[ty], int(v) if ty == "i32" else (float(v) if ty == "f32" else str(v)))
           for k, (ty, v) in sampling_kv(model).items()}
    stale = [f.name for f in reader.fields.values() if f.name.startswith("general.sampling.") and f.name not in new]
    copy_with_new_metadata(reader, writer, new, stale)
