#!/usr/bin/env python3
"""Bake TTS reference voices into a codec GGUF as builtin speakers.

The output is a copy of the input codec GGUF (tensors and metadata
unchanged) plus `spk.<name>.<slot>` tensors and `<arch>.spk.*` metadata
(docs/codec_speaker.md), or with --speaker-file a speaker file holding only
the voices. A codec GGUF with baked voices is published for one TTS model:
OuteTTS prompts are stored as that backbone's token ids.

  outetts  ref_text = the speaker's text section, audio_prompt = I32 token ids
           of the speaker's audio section (everything the reference puts
           after "<|audio_start|>\\n").  Speaker sources: outetts JSON files
           (interface 1: words[].codes; interface 3: words[].c1/c2/features),
           or the 0.3 package's speakers.parquet (interface 2).
  neutts   ref_codes = I32 NeuCodec tokens (upstream samples/<voice>.pt),
           ref_text = espeak phonemes of samples/<voice>.txt, as neutts does.

  bake-speakers.py outetts IN.gguf OUT.gguf --interface 3 \\
      --tokenizer models/hf/outetts_tok/OuteTTS-1.0-0.6B --model-id OuteAI/OuteTTS-1.0-0.6B \\
      --speakers outetts/version/v3/default_speakers/json --default en_female_1_neutral
  bake-speakers.py neutts IN.gguf OUT.gguf --samples neutts/samples \\
      --voice jo=en-us --voice greta=de --default jo
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf  # noqa: E402

from converters.speaker import encoder_hash, speaker_name  # noqa: E402

FORMAT_VERSION = 1


# ── OuteTTS ──────────────────────────────────────────────────────────────────

def _load_outetts_sources(paths: list[Path]) -> list[tuple[str, dict]]:
    out = []
    for p in paths:
        if p.is_dir():
            out += _load_outetts_sources(sorted(p.glob("*.json")) + sorted(p.glob("*.parquet")))
        elif p.suffix == ".parquet":
            import pandas as pd
            for _, row in pd.read_parquet(p).iterrows():
                words = row["words"]
                if isinstance(words, str):
                    words = ast.literal_eval(words)
                words = [{**w, "codes": [int(c) for c in w["codes"]]} for w in words]
                out.append((speaker_name(row["speaker"]), {"text": row["text"], "words": words}))
        else:
            out.append((speaker_name(p.stem), json.loads(p.read_text())))
    return out


def _render_v1(spk: dict) -> tuple[str, str]:
    """OuteTTS 0.2 (interface 1, outetts/version/v1/prompt_processor.py)."""
    words = [w["word"] for w in spk["words"]]
    audio = "\n".join(
        f"{w['word']}<|t_{w['duration']:.2f}|><|code_start|>" + "".join(f"<|{c}|>" for c in w["codes"]) + "<|code_end|>"
        for w in spk["words"])
    return " ".join(words), audio


def _render_v2(spk: dict) -> tuple[str, str]:
    """OuteTTS 0.3 (interface 2): punctuation tokens come from the speaker text."""
    from outetts.version.v2.prompt_processor import PromptProcessor
    pp = PromptProcessor(None)
    processed = pp.text_processor.process_text(spk["text"])
    audio = pp.create_audio_prompt(processed, spk) + pp.special_tokens.space + "\n"
    text = pp.text_processor.get_text(processed).replace(pp.special_tokens.space, " ")
    return text, audio


def _render_v3(spk: dict) -> tuple[str, str]:
    """OuteTTS 1.0 (interface 3): get_completion_prompt minus the input text."""
    from outetts.version.v3.prompt_processor import PromptProcessor
    pp = PromptProcessor(None)
    text = pp.text_normalizations(spk["text"])
    # merge_speaker_text appends the separator it inserts to the last word.
    _, sep = pp.merge_speaker_text("x", text)
    words = [dict(w) for w in spk["words"]]
    words[-1]["word"] += sep
    return text, pp.create_codes(words) + "\n"


def cmd_outetts(args) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer))
    render = {1: _render_v1, 2: _render_v2, 3: _render_v3}[args.interface]
    voices = {}
    for name, spk in _load_outetts_sources(args.speakers):
        if args.interface == 3 and "c1" not in spk["words"][0]:
            raise SystemExit(f"{name}: not an interface-3 speaker (no c1/c2 codes)")
        if args.interface != 3 and "codes" not in spk["words"][0]:
            raise SystemExit(f"{name}: not an interface-1/2 speaker (no codes)")
        text, audio = render(spk)
        ids = tok.encode(audio, add_special_tokens=False)
        if tok.decode(ids) != audio:
            raise SystemExit(f"{name}: audio prompt does not round-trip through the tokenizer")
        voices[name] = ({"audio_prompt": np.asarray(ids, dtype=np.int32)}, text)
        print(f"  {name}: {len(ids)} prompt tokens, ref_text {text[:60]!r}")
    enc = encoder_hash([], f"outetts.spk.v1;interface={args.interface};model={args.model_id}")
    return {"voices": voices, "encoder_hash": enc}


# ── NeuTTS ───────────────────────────────────────────────────────────────────

def _phonemize(text: str, lang: str) -> str:
    # neutts/phonemizers.py BasePhonemizer (+ FrenchPhonemizer.clean), then
    # NeuTTS._to_phones whitespace normalization.
    from phonemizer.backend import EspeakBackend
    g2p = EspeakBackend(language=lang, preserve_punctuation=True, with_stress=True,
                        words_mismatch="ignore", language_switch="remove-flags")
    phones = g2p.phonemize([text])[0]
    if lang == "fr-fr":
        phones = phones.replace("-", "")
    return " ".join(phones.split())


def cmd_neutts(args, reader) -> dict:
    import torch
    voices = {}
    for spec in args.voice:
        name, _, lang = spec.partition("=")
        if not lang:
            raise SystemExit(f"--voice {spec}: expected NAME=ESPEAK_LANG")
        codes = torch.load(args.samples / f"{name}.pt", map_location="cpu", weights_only=True)
        codes = np.asarray(codes, dtype=np.int64).reshape(-1)
        text = (args.samples / f"{name}.txt").read_text().strip()
        phones = _phonemize(text, lang)
        voices[speaker_name(name)] = ({"ref_codes": codes.astype(np.int32)}, phones)
        print(f"  {name} ({lang}): {codes.size} codes, {phones[:60]!r}")
    arch = reader.fields["general.architecture"].contents()
    field = reader.fields.get(f"{arch}.spk.encoder_hash")
    return {"voices": voices, "encoder_hash": field.contents() if field is not None else None}


# ── GGUF output ──────────────────────────────────────────────────────────────

def write(reader, out: Path, arch: str, voices: dict, default: str, enc_hash: str | None, speaker_file: bool):
    if default not in voices:
        raise SystemExit(f"--default {default!r} is not one of {sorted(voices)}")
    w = gguf.GGUFWriter(str(out), arch=arch, endianess=reader.endianess)
    spk_prefix = f"{arch}.spk."
    if speaker_file:
        w.add_string("general.type", "speaker")
        w.add_uint32(f"{arch}.spk.format_version", FORMAT_VERSION)
    else:
        alignment = reader.fields.get(gguf.Keys.General.ALIGNMENT)
        if alignment is not None:
            w.data_alignment = int(alignment.contents())
        for field in reader.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
                continue
            if field.name.startswith(spk_prefix):  # replaced below
                continue
            val_type = field.types[0]
            sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
        for t in reader.tensors:
            if t.name.startswith("spk."):
                continue
            raw = t.data.dtype == np.uint8
            w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type if raw else None)
    w.add_string(f"{arch}.spk.default", default)
    if enc_hash:
        w.add_string(f"{arch}.spk.encoder_hash", enc_hash)
    for name, (slots, ref_text) in voices.items():
        for slot, arr in slots.items():
            w.add_tensor(f"spk.{name}.{slot}", np.ascontiguousarray(arr, dtype=np.int32))
        w.add_string(f"{arch}.spk.{name}.ref_text", ref_text)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out} ({len(voices)} speakers, default {default}, {out.stat().st_size / 2**20:.1f} MiB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("outetts", "neutts"):
        p = sub.add_parser(name)
        p.add_argument("input", type=Path, help="codec GGUF")
        p.add_argument("output", type=Path)
        p.add_argument("--default", required=True, help="default speaker name")
        p.add_argument("--speaker-file", action="store_true", help="write a speaker file instead of a codec copy")
        if name == "outetts":
            p.add_argument("--interface", type=int, choices=(1, 2, 3), required=True,
                           help="1 = OuteTTS 0.2, 2 = 0.3, 3 = 1.0")
            p.add_argument("--tokenizer", type=Path, required=True, help="the backbone's HF tokenizer dir")
            p.add_argument("--model-id", required=True, help="backbone identity, hashed into encoder_hash")
            p.add_argument("--speakers", type=Path, nargs="+", required=True)
        else:
            p.add_argument("--samples", type=Path, required=True, help="neutts samples/ dir (.pt + .txt)")
            p.add_argument("--voice", action="append", required=True, metavar="NAME=ESPEAK_LANG")
    args = ap.parse_args()

    reader = gguf.GGUFReader(str(args.input))
    arch = reader.fields["general.architecture"].contents()
    res = cmd_outetts(args) if args.cmd == "outetts" else cmd_neutts(args, reader)
    write(reader, args.output, arch, res["voices"], speaker_name(args.default), res["encoder_hash"], args.speaker_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
