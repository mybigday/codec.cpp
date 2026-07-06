"""Pocket-TTS converter (kyutai/pocket-tts, MIT).

Pocket-TTS = FlowLM (100M continuous-latent AR transformer + flow head + EOS
head + SentencePiece-4000 text LUT) driving a CUSTOM small Mimi (24 kHz,
12.5 Hz frames, SEANet ratios [6,5,4], inner_dim 32 / outer 512, 2-layer
LayerScale transformer, DummyQuantizer output_proj 32->512).

This converter writes ONE gguf holding both the codec (Mimi-variant) and the
LM (FlowLM), mirroring bluemagpie's audio-codec + lm split:

  codec.*        Mimi-variant hyperparams (decode/encode)
  <arch>.*       tensors for the Mimi-variant encoder/decoder
  codec.lm.*     FlowLM hyperparams + LSD/EOS semantics
  lm.*           FlowLM transformer + flow head + EOS head + text LUT
  codec.lm.tokenizer.* SentencePiece model (base64 KV) + metadata

Phase 1 delivers the foundation + the Mimi-variant decode/encode graphs.  The
FlowLM codec_lm kind + synthesize land in Phase 2 (the lm.* tensors + metadata
are baked here so Phase 2 only needs runtime wiring).

The single english model.safetensors is the full TTSModel state dict (keys
`mimi.*` and `flow_lm.*`), with weight-norm ALREADY folded to plain `.weight`
(the upstream `POCKET_TTS_SAVE_WEIGHTS` path saved the eval model), so no
weight_g/weight_v baking is needed here — unlike bluemagpie.

Key layout facts (verified against the checkpoint):
- Mimi SEANet decoder `decoder.model.{0,2,3,5,6,8,9,11}` (n_residual_layers=1):
  0 conv(512->512,k7); 2 convtr(512->256,k12,s6); 3 resnet; 5 convtr(256->128,
  k10,s5); 6 resnet; 8 convtr(128->64,k8,s4); 9 resnet; 11 conv(64->1,k3).
  ELU sits at odd indices 1/4/7/10 (activation, no weights).
- `upsample.convtr.convtr.weight` is DEPTHWISE (in=512, out/groups=1, k=32,
  stride 16). ggml_conv_transpose_1d isn't depthwise, so we expand it to a
  dense block-diagonal (k, out=512, in=512) — same trick as mimi's `up.cv.w`.
- `downsample.conv.conv.weight` (out=32, in=512, k=32, stride 16, groups=1,
  bias=False, pad_mode=replicate).
- `quantizer.output_proj.weight` (512, 32, 1): Conv1d k1, 32->512, no bias.
- Transformer blocks (enc+dec) have a FUSED `self_attn.in_proj.weight`
  [3*d, d]; we split into q/k/v at convert time.  LayerScale `layer_scale_*`
  vectors, `norm1/norm2` (LayerNorm w/bias), `linear1/linear2` (FFN, no bias),
  `out_proj` (no bias).  RoPE max_period=10000, num_heads=8, head_dim=64,
  context=250.  NB: pocket_tts apply_rope is INTERLEAVED (real=x[...,0],
  imag=x[...,1] on adjacent pairs) => the runtime uses ggml NORMAL rope, NOT
  NEOX.  The same interleaved convention applies to the FlowLM AR transformer.
"""

from __future__ import annotations

import base64
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


# english.yaml pins (weights + tokenizer revisions).
_HF_MODEL_REPO = "kyutai/pocket-tts"
_HF_MODEL_REV = "39592ff23c9ef80098bb74895d104c26275fe2c9"
_HF_MODEL_FILE = "languages/english/model.safetensors"
_HF_NOVC_REPO = "kyutai/pocket-tts-without-voice-cloning"
_HF_NOVC_REV = "d29db7978e464fb90cb3359ee0c69a273b9142cc"
_HF_NOVC_FILE = "languages/english/model.safetensors"
_HF_TOK_FILE = "languages/english/tokenizer.model"


def _to_numpy(t):
    if torch is not None and isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return np.asarray(t)


class PocketTTSConverter(BaseConverter):
    @property
    def model_type(self) -> str:
        return "pocket_tts"

    @property
    def architecture(self) -> str:
        return "pocket_mimi"

    # ------------------------------------------------------------------ #
    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        path = Path(checkpoint_dir)
        if path.is_dir():
            st_path = path / "model.safetensors"
            tok_path = path / "tokenizer.model"
        else:
            st_path = path
            tok_path = path.parent / "tokenizer.model"
        if not st_path.is_file():
            raise FileNotFoundError(f"missing model.safetensors: {st_path}")

        import safetensors
        sd: "OrderedDict[str, np.ndarray]" = OrderedDict()
        with safetensors.safe_open(str(st_path), framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = _to_numpy(f.get_tensor(k))
        self.state_dict = sd

        self.tokenizer_bytes = None
        if tok_path.is_file():
            self.tokenizer_bytes = tok_path.read_bytes()

        # Config is fixed for english; mirror config/english.yaml.
        self.config = {
            "sample_rate": 24000,
            "frame_rate": 12.5,
            "inner_dim": 32,
            "outer_dim": 512,
            "channels": 1,
            "seanet_dim": 512,
            "n_filters": 64,
            "decoder_ratios": [6, 5, 4],       # PyTorch order (decode = upsample)
            "encoder_ratios": [4, 5, 6],       # encoder reverses ratios internally
            "quantizer_dim": 32,
            "quantizer_out": 512,
            # Mimi transformer (enc + dec share hyperparams)
            "tf_d_model": 512,
            "tf_heads": 8,
            "tf_layers": 2,
            "tf_head_dim": 512 // 8,
            "tf_ffn": 2048,
            "tf_context": 250,
            "tf_max_period": 10000.0,
            "layer_scale": 0.01,
            # FlowLM
            "flow_lm_dtype": "float32",
            "insert_bos_before_voice": True,
            "flow_depth": 6,
            "flow_dim": 512,
            "lm_d_model": 1024,
            "lm_heads": 16,
            "lm_layers": 6,
            "lm_head_dim": 1024 // 16,
            "lm_ffn": 1024 * 4,
            "lm_max_period": 10000.0,
            "lut_dim": 1024,
            "lut_n_bins": 4000,
            "ldim": 32,                        # latent dim (= quantizer.dimension)
            # LSD / EOS decode defaults (default_parameters.py)
            "lsd_decode_steps": 1,
            "temperature": 0.7,
            "eos_threshold": -4.0,
            "noise_clamp": 0.0,                # None -> 0 sentinel (unused = no clamp)
            "frames_after_eos": -1,            # DEFAULT_FRAMES_AFTER_EOS = None -> -1
        }

    def load_from_huggingface(self, model_id: str) -> None:
        from huggingface_hub import hf_hub_download
        try:
            st = hf_hub_download(repo_id=_HF_MODEL_REPO, filename=_HF_MODEL_FILE, revision=_HF_MODEL_REV)
        except Exception:
            st = hf_hub_download(repo_id=_HF_NOVC_REPO, filename=_HF_NOVC_FILE, revision=_HF_NOVC_REV)
        try:
            tok = hf_hub_download(repo_id=_HF_NOVC_REPO, filename=_HF_TOK_FILE, revision=_HF_NOVC_REV)
        except Exception:
            tok = None
        d = Path(st).parent
        if tok is not None:
            import shutil
            shutil.copy(tok, d / "tokenizer.model")
        self.load_from_checkpoint(d / "model.safetensors")

    # ------------------------------------------------------------------ #
    def convert_and_save(self, output_path: Path) -> None:
        if self.state_dict is None or self.config is None:
            raise RuntimeError("No model loaded.")
        cfg = self.config
        sd = self.state_dict
        writer = GGUFWriter(output_path, self.architecture)
        self._reset_quant_stats()

        def t(name: str) -> np.ndarray:
            arr = sd.get(name)
            if arr is None:
                raise KeyError(f"missing tensor: {name}")
            return np.asarray(arr)

        # ---- codec metadata --------------------------------------------
        hop = int(cfg["sample_rate"] / cfg["frame_rate"])  # 24000/12.5 = 1920
        writer.add_name("Pocket-TTS (Mimi-variant)")
        writer.add_uint32("codec.sample_rate", cfg["sample_rate"])
        writer.add_uint32("codec.encode_sample_rate", cfg["sample_rate"])
        writer.add_uint32("codec.hop_size", hop)
        writer.add_uint32("codec.decode_hop_size", hop)
        writer.add_uint32("codec.latent_dim", cfg["ldim"])   # 32-dim continuous latent input
        writer.add_uint32("codec.n_q", 0)                    # continuous latent — no codebooks
        writer.add_bool("codec.has_encoder", True)
        writer.add_bool("codec.has_decoder", True)
        writer.add_bool("codec.continuous_latent", True)
        writer.add_float32("codec.frame_rate", float(cfg["frame_rate"]))

        # Mimi-variant hyperparams.
        writer.add_array("pocket_mimi.decoder_ratios", list(cfg["decoder_ratios"]))
        writer.add_array("pocket_mimi.encoder_ratios", list(cfg["encoder_ratios"]))
        writer.add_uint32("pocket_mimi.seanet_dim", cfg["seanet_dim"])
        writer.add_uint32("pocket_mimi.n_filters", cfg["n_filters"])
        writer.add_uint32("pocket_mimi.inner_dim", cfg["inner_dim"])
        writer.add_uint32("pocket_mimi.outer_dim", cfg["outer_dim"])
        writer.add_uint32("pocket_mimi.quantizer_dim", cfg["quantizer_dim"])
        writer.add_uint32("pocket_mimi.tf_layers", cfg["tf_layers"])
        writer.add_uint32("pocket_mimi.tf_heads", cfg["tf_heads"])
        writer.add_uint32("pocket_mimi.tf_head_dim", cfg["tf_head_dim"])
        writer.add_uint32("pocket_mimi.tf_ffn", cfg["tf_ffn"])
        writer.add_uint32("pocket_mimi.tf_context", cfg["tf_context"])
        writer.add_float32("pocket_mimi.tf_max_period", float(cfg["tf_max_period"]))

        # ---- codec tensors ---------------------------------------------
        self._emit_mimi(writer, cfg, t)

        # ---- LM tensors + metadata (Phase 2 wiring reads these) --------
        self._emit_flow_lm(writer, cfg, t)

        self._warn_if_no_quantized()
        writer.write()
        self.log(f"wrote {output_path}")

    # ------------------------------------------------------------------ #
    def _emit_mimi(self, writer: GGUFWriter, cfg: Dict[str, Any], t) -> None:
        # All Mimi conv/linear weights ship F16 (small, quality-sensitive; conv
        # kernels aren't Q8-eligible anyway).  Match bluemagpie's audio-codec
        # convention: convs/proj F16, norms/scales/bias F32.
        def add_w(name: str, arr: np.ndarray) -> None:
            self._add_tensor(writer, name, np.asarray(arr).astype(np.float32), "F16")

        def add_f32(name: str, arr: np.ndarray) -> None:
            self._add_tensor(writer, name, np.asarray(arr).astype(np.float32), "F32")

        def add_conv(src: str, out: str) -> None:
            add_w(out + ".w", t(src + ".weight"))
            if (src + ".bias") in self.state_dict:
                add_f32(out + ".b", t(src + ".bias"))

        # ----- decoder SEANet conv stack --------------------------------
        # decoder.model.{0,2,3,5,6,8,9,11}
        add_conv("mimi.decoder.model.0.conv", "pocket_mimi.dec.l0")           # conv 512->512 k7
        add_conv("mimi.decoder.model.2.convtr", "pocket_mimi.dec.l2")         # convtr 512->256 k12 s6
        add_conv("mimi.decoder.model.3.block.1.conv", "pocket_mimi.dec.r0.c1")  # resnet k3
        add_conv("mimi.decoder.model.3.block.3.conv", "pocket_mimi.dec.r0.c2")  # resnet k1
        add_conv("mimi.decoder.model.5.convtr", "pocket_mimi.dec.l5")         # convtr 256->128 k10 s5
        add_conv("mimi.decoder.model.6.block.1.conv", "pocket_mimi.dec.r1.c1")
        add_conv("mimi.decoder.model.6.block.3.conv", "pocket_mimi.dec.r1.c2")
        add_conv("mimi.decoder.model.8.convtr", "pocket_mimi.dec.l8")         # convtr 128->64 k8 s4
        add_conv("mimi.decoder.model.9.block.1.conv", "pocket_mimi.dec.r2.c1")
        add_conv("mimi.decoder.model.9.block.3.conv", "pocket_mimi.dec.r2.c2")
        add_conv("mimi.decoder.model.11.conv", "pocket_mimi.dec.l11")         # conv 64->1 k3

        # ----- quantizer output_proj (Conv1d k1, 32->512, no bias) ------
        add_w("pocket_mimi.quant.out_proj.w", t("mimi.quantizer.output_proj.weight"))

        # ----- upsample (depthwise convtr, in=512, groups=512, k=32, s16) -
        # Expand depthwise (in, 1, k) -> dense block-diagonal ggml layout
        # (k, out=in, in) so codec_convtr1d works (no depthwise convtr in ggml).
        up = t("mimi.upsample.convtr.convtr.weight")   # torch (in=512, out/groups=1, k=32)
        C, _, K = up.shape
        dense = np.zeros((C, C, K), dtype=np.float32)   # (out, in, k) torch layout
        for c in range(C):
            dense[c, c, :] = up[c, 0, :]
        add_w("pocket_mimi.upsample.w", dense)

        # ----- downsample (Conv1d, out=32, in=512, k=32, s16, no bias) --
        add_w("pocket_mimi.downsample.w", t("mimi.downsample.conv.conv.weight"))

        # ----- encoder SEANet conv stack --------------------------------
        # encoder.model.{0,1,3,4,6,7,9,11}
        add_conv("mimi.encoder.model.0.conv", "pocket_mimi.enc.l0")           # conv 1->64 k7
        add_conv("mimi.encoder.model.1.block.1.conv", "pocket_mimi.enc.r0.c1")
        add_conv("mimi.encoder.model.1.block.3.conv", "pocket_mimi.enc.r0.c2")
        add_conv("mimi.encoder.model.3.conv", "pocket_mimi.enc.l3")           # down 64->128 k8 s4
        add_conv("mimi.encoder.model.4.block.1.conv", "pocket_mimi.enc.r1.c1")
        add_conv("mimi.encoder.model.4.block.3.conv", "pocket_mimi.enc.r1.c2")
        add_conv("mimi.encoder.model.6.conv", "pocket_mimi.enc.l6")           # down 128->256 k10 s5
        add_conv("mimi.encoder.model.7.block.1.conv", "pocket_mimi.enc.r2.c1")
        add_conv("mimi.encoder.model.7.block.3.conv", "pocket_mimi.enc.r2.c2")
        add_conv("mimi.encoder.model.9.conv", "pocket_mimi.enc.l9")           # down 256->512 k12 s6
        add_conv("mimi.encoder.model.11.conv", "pocket_mimi.enc.l11")         # conv 512->512 k3

        # ----- transformer blocks (enc + dec) ---------------------------
        d_model = cfg["tf_d_model"]
        self._emit_mimi_transformer(writer, "mimi.encoder_transformer.transformer",
                                    "pocket_mimi.etr", cfg["tf_layers"], d_model, t)
        self._emit_mimi_transformer(writer, "mimi.decoder_transformer.transformer",
                                    "pocket_mimi.dtr", cfg["tf_layers"], d_model, t)

    def _emit_mimi_transformer(self, writer, src, out, n_layers, d_model, t) -> None:
        def add_w(name, arr):
            self._add_tensor(writer, name, np.asarray(arr).astype(np.float32), "F16")

        def add_f32(name, arr):
            self._add_tensor(writer, name, np.asarray(arr).astype(np.float32), "F32")

        for i in range(n_layers):
            s = f"{src}.layers.{i}"
            o = f"{out}.l{i}"
            # Fused in_proj (3*d, d) -> split q/k/v (each (d, d)).
            inp = t(f"{s}.self_attn.in_proj.weight")   # (3d, d)
            q, k, v = np.split(inp, 3, axis=0)
            add_w(o + ".attn.q_proj.w", q)
            add_w(o + ".attn.k_proj.w", k)
            add_w(o + ".attn.v_proj.w", v)
            add_w(o + ".attn.o_proj.w", t(f"{s}.self_attn.out_proj.weight"))
            add_w(o + ".mlp.fc1.w", t(f"{s}.linear1.weight"))
            add_w(o + ".mlp.fc2.w", t(f"{s}.linear2.weight"))
            add_f32(o + ".inln.w", t(f"{s}.norm1.weight"))
            add_f32(o + ".inln.b", t(f"{s}.norm1.bias"))
            add_f32(o + ".paln.w", t(f"{s}.norm2.weight"))
            add_f32(o + ".paln.b", t(f"{s}.norm2.bias"))
            add_f32(o + ".sa_ls.scale", t(f"{s}.layer_scale_1.scale"))
            add_f32(o + ".mlp_ls.scale", t(f"{s}.layer_scale_2.scale"))

    # ------------------------------------------------------------------ #
    def _emit_flow_lm(self, writer: GGUFWriter, cfg: Dict[str, Any], t) -> None:
        # LM matmul weights: Q8_0 if requested + row divisible by 32, else F16.
        _lm_use_q8 = (self.quantization == "Q8_0")

        def add_lw(name: str, arr: np.ndarray) -> None:
            arr = np.asarray(arr).astype(np.float32)
            if _lm_use_q8 and arr.ndim == 2 and (int(arr.shape[-1]) % 32) == 0:
                self._add_tensor(writer, name, arr, "Q8_0")
            else:
                self._add_tensor(writer, name, arr, "F16")

        def add_f32(name: str, arr: np.ndarray) -> None:
            self._add_tensor(writer, name, np.asarray(arr).astype(np.float32), "F32")

        def add_lin(src: str, out: str, bias: bool) -> None:
            add_lw(out + ".w", t(src + ".weight"))
            if bias:
                add_f32(out + ".b", t(src + ".bias"))

        # ----- text LUT conditioner (Embedding (n_bins+1, dim)) ---------
        add_f32("lm.text.embed.w", t("flow_lm.conditioner.embed.weight"))  # (4001, 1024)
        add_lw("lm.input_linear.w", t("flow_lm.input_linear.weight"))      # (1024, 32)
        add_f32("lm.bos_emb", t("flow_lm.bos_emb").reshape(-1))            # (32,)
        add_f32("lm.emb_mean", t("flow_lm.emb_mean").reshape(-1))
        add_f32("lm.emb_std", t("flow_lm.emb_std").reshape(-1))
        if "flow_lm.bos_before_voice" in self.state_dict:
            add_f32("lm.bos_before_voice", t("flow_lm.bos_before_voice").reshape(-1))  # (1024,)
        if "flow_lm.speaker_proj_weight" in self.state_dict:
            add_lw("lm.speaker_proj.w", t("flow_lm.speaker_proj_weight"))  # (1024, 32)

        # ----- AR transformer (StreamingTransformer, 6 layers) ----------
        for i in range(cfg["lm_layers"]):
            s = f"flow_lm.transformer.layers.{i}"
            o = f"lm.tf.l{i}"
            inp = t(f"{s}.self_attn.in_proj.weight")   # (3d, d)
            q, k, v = np.split(inp, 3, axis=0)
            add_lw(o + ".attn.q_proj.w", q)
            add_lw(o + ".attn.k_proj.w", k)
            add_lw(o + ".attn.v_proj.w", v)
            add_lw(o + ".attn.o_proj.w", t(f"{s}.self_attn.out_proj.weight"))
            add_lw(o + ".mlp.fc1.w", t(f"{s}.linear1.weight"))
            add_lw(o + ".mlp.fc2.w", t(f"{s}.linear2.weight"))
            add_f32(o + ".inln.w", t(f"{s}.norm1.weight"))
            add_f32(o + ".inln.b", t(f"{s}.norm1.bias"))
            add_f32(o + ".paln.w", t(f"{s}.norm2.weight"))
            add_f32(o + ".paln.b", t(f"{s}.norm2.bias"))

        # ----- out_norm + EOS head --------------------------------------
        add_f32("lm.out_norm.w", t("flow_lm.out_norm.weight"))
        add_f32("lm.out_norm.b", t("flow_lm.out_norm.bias"))
        add_lw("lm.out_eos.w", t("flow_lm.out_eos.weight"))   # (1, 1024)
        add_f32("lm.out_eos.b", t("flow_lm.out_eos.bias"))

        # ----- flow head (SimpleMLPAdaLN, depth 6, dim 512) -------------
        fn = "flow_lm.flow_net"
        add_lin(f"{fn}.input_proj", "lm.flow.input_proj", bias=True)
        add_lin(f"{fn}.cond_embed", "lm.flow.cond_embed", bias=True)
        # 2 TimestepEmbedders (s, t). freqs is a buffer (baked for reference).
        for ti in range(2):
            te = f"{fn}.time_embed.{ti}"
            add_f32(f"lm.flow.time_embed.{ti}.freqs", t(f"{te}.freqs").reshape(-1))
            add_lin(f"{te}.mlp.0", f"lm.flow.time_embed.{ti}.l1", bias=True)
            add_lin(f"{te}.mlp.2", f"lm.flow.time_embed.{ti}.l2", bias=True)
            add_f32(f"lm.flow.time_embed.{ti}.rms.alpha", t(f"{te}.mlp.3.alpha").reshape(-1))
        # res blocks
        for bi in range(cfg["flow_depth"]):
            rb = f"{fn}.res_blocks.{bi}"
            o = f"lm.flow.res.{bi}"
            add_f32(o + ".in_ln.w", t(f"{rb}.in_ln.weight"))
            add_f32(o + ".in_ln.b", t(f"{rb}.in_ln.bias"))
            add_lin(f"{rb}.mlp.0", o + ".mlp.l1", bias=True)
            add_lin(f"{rb}.mlp.2", o + ".mlp.l2", bias=True)
            add_lin(f"{rb}.adaLN_modulation.1", o + ".adaln", bias=True)
        # final layer
        add_lin(f"{fn}.final_layer.linear", "lm.flow.final.linear", bias=True)
        add_lin(f"{fn}.final_layer.adaLN_modulation.1", "lm.flow.final.adaln", bias=True)

        # ----- LM metadata ----------------------------------------------
        writer.add_bool("codec.lm.has_adaptor", True)
        writer.add_string("codec.lm.kind", "flow_lm")
        writer.add_string("codec.lm.host_arch", "flow_lm")
        writer.add_uint32("codec.lm.d_model", cfg["lm_d_model"])
        writer.add_uint32("codec.lm.n_heads", cfg["lm_heads"])
        writer.add_uint32("codec.lm.n_layers", cfg["lm_layers"])
        writer.add_uint32("codec.lm.head_dim", cfg["lm_head_dim"])
        writer.add_uint32("codec.lm.ffn_dim", cfg["lm_ffn"])
        writer.add_float32("codec.lm.max_period", float(cfg["lm_max_period"]))
        writer.add_uint32("codec.lm.ldim", cfg["ldim"])
        writer.add_uint32("codec.lm.flow_depth", cfg["flow_depth"])
        writer.add_uint32("codec.lm.flow_dim", cfg["flow_dim"])
        writer.add_uint32("codec.lm.lut_n_bins", cfg["lut_n_bins"])
        writer.add_uint32("codec.lm.lut_dim", cfg["lut_dim"])
        writer.add_bool("codec.lm.insert_bos_before_voice", bool(cfg["insert_bos_before_voice"]))
        # LSD / EOS defaults + semantics.
        writer.add_uint32("codec.lm.lsd_decode_steps", cfg["lsd_decode_steps"])
        writer.add_float32("codec.lm.temperature", float(cfg["temperature"]))
        writer.add_float32("codec.lm.eos_threshold", float(cfg["eos_threshold"]))
        writer.add_float32("codec.lm.noise_clamp", float(cfg["noise_clamp"]))
        writer.add_int32("codec.lm.frames_after_eos", int(cfg["frames_after_eos"]))

        # ----- SentencePiece tokenizer (binary protobuf, base64 KV) -----
        if getattr(self, "tokenizer_bytes", None):
            b64 = base64.b64encode(self.tokenizer_bytes).decode("ascii")
            writer.add_string("codec.lm.tokenizer.model", "sentencepiece")
            writer.add_string("codec.lm.tokenizer.spm_b64", b64)
            writer.add_uint32("codec.lm.tokenizer.n_bins", cfg["lut_n_bins"])
        else:
            self.log("WARNING: tokenizer.model not found — skipping tokenizer KV")
