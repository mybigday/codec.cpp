"""KittenTTS 2 codec-side GGUF (KittenML/kitten-tts-2).

KittenTTS 2 is a Qwen3 speech LM (converted separately with
convert-backbone-to-gguf.py) whose S3 tokens are decoded by Chatterbox-Turbo's
MeanFlow S3Gen. This GGUF is therefore a `chatterbox_s3g` model carrying:

  * the S3Gen decoder + its reference-audio encoders (S3 tokenizer, CAMPPlus),
  * the LM-side speaker encoder: pyannote XVectorSincNet + the LM's `spk_proj`
    head (Linear 512→2048 + LayerNorm), flagged by
    `chatterbox_s3g.spk.lm_encoder = "xvector_sincnet"`,
  * the 47 builtin voices as speakers (`spk.<voice>.{prompt_token, prompt_feat,
    embedding, lm_cond, ref_tokens}` + `chatterbox_s3g.spk.<voice>.ref_text`),
  * the LM prompt / sampling contract under `codec.lm.kitten.*`.

The S3Gen weights are `s3gen_meanflow.safetensors` from ResembleAI/chatterbox-turbo
(what the `kittenml` package downloads): looked up next to the KittenTTS
checkpoint, else fetched from the Hub.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .chatterbox import ChatterboxS3GConverter, S3G_SPEAKER_PARAMS, _load_safetensors
from .speaker import speaker_name, write_speaker, write_speaker_meta
from .xvector_sincnet import write_xvector
from utils.gguf_writer import GGUFWriter

S3GEN_REPO = "ResembleAI/chatterbox-turbo"
S3GEN_FILE = "s3gen_meanflow.safetensors"

KITTEN_SPEAKER_PARAMS = S3G_SPEAKER_PARAMS + (
    ";lm_cond=xvector_sincnet(16k,full clip)->l2norm->spk_proj(linear,layernorm);"
    "ref_tokens=s3tokenizer_v2(16k,full clip)"
)


class KittenTTS2Converter(ChatterboxS3GConverter):
    model_name = "KittenTTS-2"

    @property
    def model_type(self) -> str:
        return "kitten_tts2"

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.kitten_dir = checkpoint_dir
        self.kitten_cfg = json.loads((checkpoint_dir / "config.json").read_text())
        if self.kitten_cfg.get("type") != "KITTEN2":
            raise ValueError(f"{checkpoint_dir}/config.json is not a KittenTTS 2 repository")

        s3gen = checkpoint_dir / S3GEN_FILE
        if not s3gen.is_file():
            from huggingface_hub import hf_hub_download
            s3gen = Path(hf_hub_download(S3GEN_REPO, S3GEN_FILE))
        self.state_dict = _load_safetensors(s3gen)
        self.conds = None
        self.ve_state = None
        self.config = {
            "sample_rate": 24000, "hop_size": 960, "n_q": 1, "codebook_size": 6561,
            "token_rate_hz": 25, "meanflow": True, "has_encoder": False, "has_decoder": True,
        }

        lm_dir = checkpoint_dir / self.kitten_cfg.get("lm_dir", "lm")
        with safe_open(str(lm_dir / "model.safetensors"), framework="pt") as f:
            self.spk_proj = {k: f.get_tensor(k).float().numpy() for k in f.keys() if k.startswith("spk_proj.")}
        if len(self.spk_proj) != 4:
            raise ValueError(f"expected spk_proj.{{0,1}}.{{weight,bias}} in {lm_dir}, got {sorted(self.spk_proj)}")
        self.xvector_state = _load_safetensors(checkpoint_dir / self.kitten_cfg.get(
            "speaker_embedding", "speaker/model.safetensors"))

        voices_path = checkpoint_dir / "cpp" / "default" / "voices.json"
        self.voices = json.loads(voices_path.read_text())
        self.log(f"Loaded KittenTTS 2 from {checkpoint_dir}: S3Gen {s3gen.name}, {len(self.voices)} voices")

    def load_from_huggingface(self, model_id: str) -> None:
        from huggingface_hub import snapshot_download
        self.load_from_checkpoint(Path(snapshot_download(model_id, allow_patterns=[
            "config.json", "lm/*.json", "lm/merges.txt", "lm/model.safetensors",
            "speaker/*", "cpp/default/voices.json"])))

    # ── S3G converter hooks ────────────────────────────────────────────
    def _extra_speaker_encoder_tensors(self):
        return ([("xvector." + k, v) for k, v in self.xvector_state.items()] +
                [(k, v) for k, v in self.spk_proj.items()])

    def _speaker_params(self) -> str:
        return KITTEN_SPEAKER_PARAMS

    def _write_builtin_speakers(self, writer: GGUFWriter, enc_hash: str) -> bool:
        default = speaker_name(self.kitten_cfg.get("default_voice", "Bruno"))
        for voice, v in self.voices.items():
            write_speaker(writer, self.architecture, speaker_name(voice), {
                "prompt_token": np.asarray(v["prompt_token"][0], np.int32),
                "prompt_feat": np.asarray(v["prompt_feat"][0], np.float32),   # [F, 80] -> ggml [80, F]
                "embedding": np.asarray(v["embedding"][0], np.float32),
                "lm_cond": np.asarray(v["speaker"], np.float32),
                "ref_tokens": np.asarray(v["reference_tokens"], np.int32),
            }, ref_text=v["transcript"].strip())
        write_speaker_meta(writer, self.architecture, default=default, enc_hash=enc_hash)
        return True

    def _write_extra_encoders(self, writer: GGUFWriter) -> None:
        f32 = lambda n, a: self._add_tensor(writer, n, a, st_dtype="F32")  # noqa: E731
        write_xvector(f32, self.xvector_state, "kt.xv")
        f32("kt.spk_proj.w", self.spk_proj["spk_proj.0.weight"])
        f32("kt.spk_proj.b", self.spk_proj["spk_proj.0.bias"])
        f32("kt.spk_proj.ln.w", self.spk_proj["spk_proj.1.weight"])
        f32("kt.spk_proj.ln.b", self.spk_proj["spk_proj.1.bias"])
        writer.add_string("chatterbox_s3g.spk.lm_encoder", "xvector_sincnet")
        writer.add_uint32("chatterbox_s3g.spk.lm_cond_dim", int(self.spk_proj["spk_proj.0.weight"].shape[0]))
        self._write_lm_contract(writer)

    def _write_lm_contract(self, writer: GGUFWriter) -> None:
        """Prompt layout the host runner needs (kittenml defaults); the sampling
        (decode presets, repetition / token-run penalties) is the backbone GGUF's."""
        cfg = self.kitten_cfg
        p = "codec.lm.kitten."
        for key, val in cfg["token_map"].items():
            writer.add_uint32(p + "token." + key, int(val))
        gen = cfg.get("generation", {})
        writer.add_uint32(p + "silence_token", 4299)
        writer.add_uint32(p + "n_trailing_silence", 3)
        writer.add_bool(p + "use_reference_prompt", bool(gen.get("use_reference_prompt", True)))
        writer.add_string(p + "emotion_control", gen.get("emotion_control", "{emo: 1}"))
        writer.add_uint32(p + "chunk_chars", int(gen.get("chunk_chars", 380)))
        writer.add_uint32(p + "chunk_min_chars", int(gen.get("chunk_min_chars", 130)))
        writer.add_float32(p + "chunk_gap_s", 0.16)
        writer.add_uint32(p + "max_new_tokens", 1000)
