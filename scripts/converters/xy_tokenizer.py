"""OpenMOSS-Team/XY_Tokenizer_TTSD_V0_hf converter.

XY-Tokenizer is a 16 kHz-in / 24 kHz-out neural codec built around a parallel
Whisper-style semantic + acoustic pair, residual vector quantisation, and a
ConvNeXt-Vocos vocoder.  Pipeline (encode):

    mel(80, hop=160, n_fft=400, 16 kHz) ──► OmniAudioEncoder × 2  (semantic + acoustic)
                                          │  └ conv1+conv2 (stride 2) → +sinusoid PE
                                          │  └ 12 OmniWhisperTransformerLayer (LayerNorm + GELU MLP + bias-Q/K/V SDPA)
                                          │  └ final LayerNorm
                                          ▼
              semantic_encoder_adapter   (4-layer Transformer)
              acoustic_encoder           (raw output, shape (B, 768, T))
                                          │
                                          ▼
                                concat ── pre_rvq_adapter (Linear 1536→768 + 4 transformer layers)
                                          │
                                          ▼
                            ResidualDownConv (avg_pool=4: gate × up → linear → +residual + LN)
                                          │
                                          ▼
                       Linear 3072→512 (WNConv) ─► RVQ8 (1024×512 codebooks, EMA codebook
                                                           stored as `codebook` + `embed_avg`/
                                                           `cluster_size`).
                                          │
                                          ▼
                       Linear 512→3072 (WNConv) ─► post_rvq_adapter (3072→768 + 4 layers + 768→3072)
                                          │
                                          ▼
                          UpConv (deconv stride=4, 3072→768)
                                          │
                                          ▼
                       OmniAudioDecoder (12 layers + deconv1+deconv2 → 80 mel bins, stride 2)
                                          │
                                          ▼
                       enhanced Vocos (embed conv 80→512, 30 ConvNeXt blocks,
                                       iSTFT head: Linear 512→962, n_fft=960, hop=240, 24 kHz)

The converter mirrors this structure 1:1 in the GGUF tensor namespace:

    xy.sem_enc.{conv1,conv2,layer_norm,pos_emb}
    xy.sem_enc.l{i}.{norm1,norm2,attn.q,attn.k,attn.v,attn.out,mlp.fc1,mlp.fc2}
    xy.acoust_enc.* (same shape)
    xy.sem_enc_adapter.l{i}.* + xy.sem_enc_adapter.{layer_norm,pos_emb}
    xy.pre_rvq_adapter.{proj,layer_norm,pos_emb} + xy.pre_rvq_adapter.l{i}.*
    xy.downsample.{gate,up,down,layer_norm}
    xy.q.{in_proj,out_proj} + xy.q.{i}.{codebook,codebook_norm}
    xy.upsample.up_conv
    xy.post_rvq_adapter.{proj,out_proj,layer_norm,pos_emb} + xy.post_rvq_adapter.l{i}.*
    xy.acoust_dec.{deconv1,deconv2,layer_norm,pos_emb} + xy.acoust_dec.l{i}.*
    xy.vocos.embed (conv 80→512 k=7)
    xy.vocos.norm  (initial LN)
    xy.vocos.b{i}.{dwconv,norm,pwconv1,pwconv2,gamma}   (ConvNeXt block × 30)
    xy.vocos.final_layer_norm
    xy.vocos.head.out  (Linear 512→962)

Weight-norm is baked into `quantizer.{input,output}_proj` at convert time.

The RVQ codebook is also baked as a per-row L2-normalised companion tensor
(`xy.q.{i}.codebook_norm`) so the runtime can compute cosine NN with a single
matmul.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .base import BaseConverter
from utils.gguf_writer import GGUFWriter


def _to_numpy(t):
    if torch is not None and isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def _apply_weight_norm(weight_v: np.ndarray, weight_g: np.ndarray, dim: int = 0) -> np.ndarray:
    if weight_v.ndim < 2:
        raise ValueError(f"weight_norm expects ndim >= 2, got {weight_v.ndim}")
    axes = tuple(i for i in range(weight_v.ndim) if i != dim)
    norm = np.linalg.norm(weight_v, axis=axes, keepdims=True)
    if weight_g.shape != norm.shape:
        weight_g = weight_g.reshape(norm.shape)
    return weight_v * (weight_g / (norm + 1e-12))


def _load_state_dict(path: Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    if path.is_dir():
        st_files = sorted(path.glob("model-*.safetensors")) + sorted(path.glob("*.safetensors"))
        st_files = sorted(set(st_files))
        if st_files:
            from safetensors.numpy import load_file as st_load
            sd: Dict[str, np.ndarray] = {}
            for f in st_files:
                sd.update(st_load(str(f)))
            return sd
        bin_path = path / "pytorch_model.bin"
        if bin_path.is_file() and torch is not None:
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
            return OrderedDict((k, _to_numpy(v)) for k, v in state.items())
        raise FileNotFoundError(f"no XY-Tokenizer checkpoint files in {path}")
    if path.suffix == ".safetensors":
        from safetensors.numpy import load_file as st_load
        return dict(st_load(str(path)))
    if torch is None:
        raise RuntimeError("torch required to load .bin checkpoints")
    state = torch.load(path, map_location="cpu", weights_only=True)
    return OrderedDict((k, _to_numpy(v)) for k, v in state.items())


class XYTokenizerConverter(BaseConverter):
    """Converter for the XY-Tokenizer codec (used by MOSS-TTS-family TTS LMs).

    Optionally bundles an LM-side adaptor (`lm.*` tensors + `codec.lm.*`
    metadata) into the same GGUF when `lm_source` is supplied — this is
    how MOSS-TTSD-v0.5/v0.7/v1.0/MOSS-TTS get converted.  The LM source
    is auto-detected from its `config.json` and dispatched to the right
    handler in `scripts/converters/lm_adaptor/`."""

    def __init__(
        self,
        quantization: str = "F16",
        quantize_codebook: bool = False,
        verbose: bool = False,
        lm_source=None,
    ):
        super().__init__(
            quantization=quantization,
            quantize_codebook=quantize_codebook,
            verbose=verbose,
        )
        self.lm_source = lm_source

    @property
    def model_type(self) -> str:
        return "xy_tokenizer"

    @property
    def architecture(self) -> str:
        return "xy_tokenizer"

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        path = Path(checkpoint_dir)
        cfg_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"missing config.json next to {path}")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        self.state_dict = _load_state_dict(path)
        self.config = cfg

    def load_from_huggingface(self, model_id: str) -> None:
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.safetensors", "*.json", "*.py", "*.bin"],
        )
        self.load_from_checkpoint(Path(local))

    def convert_and_save(self, output_path: Path) -> None:
        writer = GGUFWriter(output_path, self.architecture)
        self._reset_quant_stats()
        self.write_into(writer)
        if self.lm_source is not None:
            from .lm_adaptor import dump_lm_into
            dump_lm_into(writer, self.lm_source, verbose=self.verbose)
        self._warn_if_no_quantized()
        writer.write()
        if self.lm_source is None:
            self.log(f"Wrote XY-Tokenizer GGUF to {output_path}")
        else:
            self.log(f"Wrote XY-Tokenizer codec + LM adaptor GGUF to {output_path} "
                     f"(lm_source={self.lm_source})")

    def write_into(self, writer: GGUFWriter) -> None:
        """Write XY-Tokenizer codec metadata + tensors into a caller-supplied
        GGUFWriter.  Lets a higher-level invocation bundle the codec section
        alongside `lm.*` tensors in a single output GGUF (passed via
        `lm_source`).  Caller owns the writer's lifecycle (creation + final
        `write()`)."""
        if self.state_dict is None or self.config is None:
            raise RuntimeError("No model loaded.")
        sd = self.state_dict
        cfg = self.config
        params = cfg["params"]

        # ---- top-level metadata ------------------------------------------
        in_sr = int(cfg["input_sample_rate"])
        out_sr = int(cfg["output_sample_rate"])
        enc_down = int(cfg["encoder_downsample_rate"])  # 16k → 12.5 codes/s = 1280 spc
        dec_up   = int(cfg["decoder_upsample_rate"])    # 24k / 12.5 = 1920 spc
        code_dim = int(cfg["code_dim"])

        qz = params["quantizer_kwargs"]
        n_q = int(qz["num_quantizers"])
        cb_size = int(qz["codebook_size"])
        cb_dim = int(qz["codebook_dim"])
        rvq_dim = int(qz["rvq_dim"])

        sem_enc = params["semantic_encoder_kwargs"]
        acoust_enc = params["acoustic_encoder_kwargs"]
        sem_adapt = params["semantic_encoder_adapter_kwargs"]
        pre_rvq = params["pre_rvq_adapter_kwargs"]
        downsample = params["downsample_kwargs"]
        upsample = params["upsample_kwargs"]
        post_rvq = params["post_rvq_adapter_kwargs"]
        acoust_dec = params["acoustic_decoder_kwargs"]
        vocos = params["vocos_kwargs"]

        writer.add_name("XY-Tokenizer")
        writer.add_uint32("codec.sample_rate", out_sr)
        writer.add_uint32("codec.encode_sample_rate", in_sr)
        writer.add_uint32("codec.hop_size", dec_up)
        writer.add_uint32("codec.n_q", n_q)
        writer.add_uint32("codec.codebook_size", cb_size)
        writer.add_uint32("codec.codebook_dim", cb_dim)
        writer.add_uint32("codec.latent_dim", code_dim)
        writer.add_bool("codec.has_encoder", True)
        writer.add_bool("codec.has_decoder", True)

        writer.add_uint32("xy.encoder_downsample_rate", enc_down)
        writer.add_uint32("xy.decoder_upsample_rate", dec_up)
        writer.add_uint32("xy.rvq_dim", rvq_dim)

        # Mel-fbank parameters (encode-side feature extractor, runs CPU-side).
        fe = params["feature_extractor_kwargs"]
        writer.add_uint32("xy.mel.n_mels", int(fe["feature_size"]))
        writer.add_uint32("xy.mel.n_fft", int(fe["n_fft"]))
        writer.add_uint32("xy.mel.hop_length", int(fe["hop_length"]))
        writer.add_uint32("xy.mel.sample_rate", int(fe["sampling_rate"]))
        writer.add_uint32("xy.mel.chunk_length_seconds", int(fe.get("chunk_length", 30)))

        # ---- Whisper-style transformer modules ---------------------------
        for key, prefix_in, prefix_out in [
            ("semantic_encoder",         "semantic_encoder",         "xy.sem_enc"),
            ("acoustic_encoder",         "acoustic_encoder",         "xy.acoust_enc"),
            ("semantic_encoder_adapter", "semantic_encoder_adapter", "xy.sem_enc_adapter"),
            ("pre_rvq_adapter",          "pre_rvq_adapter",          "xy.pre_rvq_adapter"),
            ("post_rvq_adapter",         "post_rvq_adapter",         "xy.post_rvq_adapter"),
            ("acoustic_decoder",         "acoustic_decoder",         "xy.acoust_dec"),
        ]:
            self._save_transformer_module(writer, sd, prefix_in, prefix_out)

        # ---- ConvNeXt Vocos vocoder + iSTFT head -------------------------
        self._save_vocos(writer, sd, vocos)

        # ---- ResidualDownConv (avg_pooler=4) -----------------------------
        # Three Linears + 1 LayerNorm, but `gate_proj` and `up_proj` are stored
        # as Conv1d (out, in, k=avg_pooler).
        self._add_tensor(writer, "xy.downsample.gate.w", sd["downsample.gate_proj.weight"])
        self._add_tensor(writer, "xy.downsample.up.w",   sd["downsample.up_proj.weight"])
        self._add_tensor(writer, "xy.downsample.down.w", sd["downsample.down_proj.weight"])
        self._add_tensor(writer, "xy.downsample.layer_norm.w", sd["downsample.layer_norm.weight"], "F32")
        self._add_tensor(writer, "xy.downsample.layer_norm.b", sd["downsample.layer_norm.bias"], "F32")
        writer.add_uint32("xy.downsample.avg_pooler", int(downsample["avg_pooler"]))

        # ---- UpConv (a single ConvTranspose1d, stride=4) -----------------
        self._add_tensor(writer, "xy.upsample.up_conv.w", sd["upsample.up_conv.weight"])
        writer.add_uint32("xy.upsample.stride", int(upsample["stride"]))

        # ---- Quantiser ---------------------------------------------------
        # input_proj / output_proj are 1×1 WNConv1d's stored as
        #   weight_g: (out, 1, 1)
        #   weight_v: (out, in, 1)
        # We bake the weight-norm here so the runtime sees a plain conv1d.
        self._save_wn_conv1d(writer, sd, "quantizer.input_proj",  "xy.q.in_proj")
        self._save_wn_conv1d(writer, sd, "quantizer.output_proj", "xy.q.out_proj")

        # Each VectorQuantize keeps both `codebook` (the working code book) and
        # `embed_avg` / `cluster_size` (EMA running totals).  At inference we
        # only need `codebook`.  XY uses Euclidean-distance NN (not cosine), so
        # we additionally bake `codebook_sq_norm[i] = ||codebook[i]||^2` to
        # turn the per-step cost into a single matmul + bias argmax:
        #   argmin_i ||z - cb[i]||^2  ==  argmax_i (2 z · cb[i] - sq_norm[i]).
        for qi in range(n_q):
            cb = _to_numpy(sd[f"quantizer.quantizers.{qi}.codebook"]).astype(np.float32)
            sq = (cb * cb).sum(axis=1).astype(np.float32)  # (codebook_size,)
            self._add_tensor(writer, f"xy.q.{qi}.codebook",         cb, "F32")
            self._add_tensor(writer, f"xy.q.{qi}.codebook_sq_norm", sq, "F32")

    # ------------------------------------------------------------------
    # Helpers — Whisper-style transformer modules
    # ------------------------------------------------------------------
    def _save_transformer_module(
        self,
        writer: GGUFWriter,
        sd: Dict[str, np.ndarray],
        prefix_in: str,
        prefix_out: str,
    ) -> None:
        """Write one of:
        - OmniAudioEncoder       (conv1+conv2+layers+layer_norm+positional_embedding)
        - OmniAudioDecoder       (deconv1+deconv2+layers+layer_norm+positional_embedding)
        - Transformer (adapter)  (proj/out_proj+layers+layer_norm+positional_embedding)

        and mirrors the per-layer Whisper encoder layer (q/k/v/out + 2 LN + 2 MLP)."""

        # Common pieces.
        if (prefix_in + ".positional_embedding") in sd:
            self._add_tensor(writer, prefix_out + ".pos_emb",
                             sd[prefix_in + ".positional_embedding"], "F32")
        if (prefix_in + ".layer_norm.weight") in sd:
            self._add_tensor(writer, prefix_out + ".layer_norm.w",
                             sd[prefix_in + ".layer_norm.weight"], "F32")
            self._add_tensor(writer, prefix_out + ".layer_norm.b",
                             sd[prefix_in + ".layer_norm.bias"], "F32")

        # Conv pre/post stacks (encoder/decoder only).
        for k_in, k_out in [
            ("conv1",   "conv1"),
            ("conv2",   "conv2"),
            ("deconv1", "deconv1"),
            ("deconv2", "deconv2"),
        ]:
            wname = prefix_in + "." + k_in + ".weight"
            bname = prefix_in + "." + k_in + ".bias"
            if wname in sd:
                self._add_tensor(writer, prefix_out + "." + k_out + ".w", sd[wname])
                self._add_tensor(writer, prefix_out + "." + k_out + ".b", sd[bname], "F32")

        # Adapter `proj` (input projection) and `out_proj` (output projection).
        for k_in, k_out in [("proj", "proj"), ("out_proj", "out_proj")]:
            wname = prefix_in + "." + k_in + ".weight"
            bname = prefix_in + "." + k_in + ".bias"
            if wname in sd:
                self._add_tensor(writer, prefix_out + "." + k_out + ".w", sd[wname])
                if bname in sd:
                    self._add_tensor(writer, prefix_out + "." + k_out + ".b", sd[bname], "F32")

        # Per-layer.  Detect the number of layers by walking the state-dict.
        n_layers = 0
        while (prefix_in + f".layers.{n_layers}.self_attn.q_proj.weight") in sd:
            n_layers += 1
        for li in range(n_layers):
            lp_in = f"{prefix_in}.layers.{li}"
            lp_out = f"{prefix_out}.l{li}"
            # Pre-attention LayerNorm.
            self._add_tensor(writer, lp_out + ".norm1.w",
                             sd[lp_in + ".self_attn_layer_norm.weight"], "F32")
            self._add_tensor(writer, lp_out + ".norm1.b",
                             sd[lp_in + ".self_attn_layer_norm.bias"], "F32")
            # SDPA Q/K/V/out.  Note: q_proj, v_proj, out_proj have biases; k_proj
            # has no bias (Whisper convention).
            self._add_tensor(writer, lp_out + ".attn.q.w", sd[lp_in + ".self_attn.q_proj.weight"])
            self._add_tensor(writer, lp_out + ".attn.q.b", sd[lp_in + ".self_attn.q_proj.bias"], "F32")
            self._add_tensor(writer, lp_out + ".attn.k.w", sd[lp_in + ".self_attn.k_proj.weight"])
            self._add_tensor(writer, lp_out + ".attn.v.w", sd[lp_in + ".self_attn.v_proj.weight"])
            self._add_tensor(writer, lp_out + ".attn.v.b", sd[lp_in + ".self_attn.v_proj.bias"], "F32")
            self._add_tensor(writer, lp_out + ".attn.out.w", sd[lp_in + ".self_attn.out_proj.weight"])
            self._add_tensor(writer, lp_out + ".attn.out.b", sd[lp_in + ".self_attn.out_proj.bias"], "F32")
            # Final LayerNorm + GELU MLP.
            self._add_tensor(writer, lp_out + ".norm2.w",
                             sd[lp_in + ".final_layer_norm.weight"], "F32")
            self._add_tensor(writer, lp_out + ".norm2.b",
                             sd[lp_in + ".final_layer_norm.bias"], "F32")
            self._add_tensor(writer, lp_out + ".mlp.fc1.w", sd[lp_in + ".mlp.fc1.weight"])
            self._add_tensor(writer, lp_out + ".mlp.fc1.b", sd[lp_in + ".mlp.fc1.bias"], "F32")
            self._add_tensor(writer, lp_out + ".mlp.fc2.w", sd[lp_in + ".mlp.fc2.weight"])
            self._add_tensor(writer, lp_out + ".mlp.fc2.b", sd[lp_in + ".mlp.fc2.bias"], "F32")

        writer.add_uint32(prefix_out + ".n_layers", n_layers)

    # ------------------------------------------------------------------
    # Helpers — Vocos backbone + iSTFT head
    # ------------------------------------------------------------------
    def _save_vocos(self, writer: GGUFWriter, sd: Dict[str, np.ndarray], vocos_cfg: dict) -> None:
        # Initial embed conv 80→512 k=7 + LN + final LN.
        self._add_tensor(writer, "xy.vocos.embed.w", sd["enhanced_vocos.backbone.embed.weight"])
        self._add_tensor(writer, "xy.vocos.embed.b", sd["enhanced_vocos.backbone.embed.bias"], "F32")
        self._add_tensor(writer, "xy.vocos.norm.w",  sd["enhanced_vocos.backbone.norm.weight"], "F32")
        self._add_tensor(writer, "xy.vocos.norm.b",  sd["enhanced_vocos.backbone.norm.bias"], "F32")
        self._add_tensor(writer, "xy.vocos.final_layer_norm.w",
                         sd["enhanced_vocos.backbone.final_layer_norm.weight"], "F32")
        self._add_tensor(writer, "xy.vocos.final_layer_norm.b",
                         sd["enhanced_vocos.backbone.final_layer_norm.bias"], "F32")

        # Per-block ConvNeXt.  Detect count by walking.
        n_blocks = 0
        while f"enhanced_vocos.backbone.convnext.{n_blocks}.dwconv.weight" in sd:
            n_blocks += 1
        for bi in range(n_blocks):
            sp = f"enhanced_vocos.backbone.convnext.{bi}"
            op = f"xy.vocos.b{bi}"
            self._add_tensor(writer, op + ".dwconv.w", sd[sp + ".dwconv.weight"])
            self._add_tensor(writer, op + ".dwconv.b", sd[sp + ".dwconv.bias"], "F32")
            self._add_tensor(writer, op + ".norm.w",   sd[sp + ".norm.weight"],   "F32")
            self._add_tensor(writer, op + ".norm.b",   sd[sp + ".norm.bias"],     "F32")
            self._add_tensor(writer, op + ".pwconv1.w", sd[sp + ".pwconv1.weight"])
            self._add_tensor(writer, op + ".pwconv1.b", sd[sp + ".pwconv1.bias"], "F32")
            self._add_tensor(writer, op + ".pwconv2.w", sd[sp + ".pwconv2.weight"])
            self._add_tensor(writer, op + ".pwconv2.b", sd[sp + ".pwconv2.bias"], "F32")
            self._add_tensor(writer, op + ".gamma",     sd[sp + ".gamma"],         "F32")
        writer.add_uint32("xy.vocos.n_blocks", n_blocks)

        # Head: out Linear 512→962 (= 1 + n_fft/2 * 2: real + imag interleaved
        # in 2*(n_fft/2 + 1) channels — the runtime splits and runs iSTFT).
        self._add_tensor(writer, "xy.vocos.head.out.w", sd["enhanced_vocos.head.out.weight"])
        self._add_tensor(writer, "xy.vocos.head.out.b", sd["enhanced_vocos.head.out.bias"], "F32")
        # The shipped iSTFT window (Hann, length=n_fft).  Runtime uses it as-is.
        self._add_tensor(writer, "xy.vocos.head.istft_window",
                         sd["enhanced_vocos.head.istft.window"], "F32")
        writer.add_uint32("xy.vocos.head.n_fft",     int(vocos_cfg["n_fft"]))
        writer.add_uint32("xy.vocos.head.hop_size",  int(vocos_cfg["hop_size"]))

    # ------------------------------------------------------------------
    # Helpers — WN conv 1×1 (used inside the quantiser's I/O projections)
    # ------------------------------------------------------------------
    def _save_wn_conv1d(
        self,
        writer: GGUFWriter,
        sd: Dict[str, np.ndarray],
        prefix_in: str,
        prefix_out: str,
    ) -> None:
        wv = _to_numpy(sd[prefix_in + ".weight_v"]).astype(np.float32)
        wg = _to_numpy(sd[prefix_in + ".weight_g"]).astype(np.float32)
        w = _apply_weight_norm(wv, wg, dim=0)  # baked
        self._add_tensor(writer, prefix_out + ".w", w)
        if (prefix_in + ".bias") in sd:
            self._add_tensor(writer, prefix_out + ".b", sd[prefix_in + ".bias"], "F32")
