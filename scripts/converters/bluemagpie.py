"""BlueMagpie-TTS converter — AudioVAE decoder (slice 1).

BlueMagpie = VoxCPM2 with its TSLM swapped to Barbet.  This converter currently
covers only the **AudioVAE V2 continuous decoder** — the bottom of the stack,
which turns the continuous latent sequence (the CFM diffusion output) into a
48 kHz waveform.  The LM / LocEnc / LocDiT / RALM stacks land in later slices.

AudioVAE decoder (from voxcpm `modules/audiovae/audio_vae_v2.py`,
`CausalDecoder`, depthwise=True, sr_bin_boundaries set):

    model.0   WNCausalConv1d(64, 64, k=7, groups=64)         depthwise in
    model.1   WNCausalConv1d(64, 2048, k=1)                  pointwise in
    model.2-7 CausalDecoderBlock(rates [8,6,5,2,2,2])        6 upsample blocks
              .block.0  Snake1d(in)
              .block.1  WNCausalTransposeConv1d(in, out, k=2*stride)
              .block.2/3/4  CausalResidualUnit(out, dilation 1/3/9)
                  .block.0 Snake → .block.1 WNCausalConv1d(k=7, dw) →
                  .block.2 Snake → .block.3 WNCausalConv1d(k=1)
    model.8   Snake1d(32)
    model.9   WNCausalConv1d(32, 1, k=7)
    model.10  Tanh

    sr_cond_model.{2..7}  SampleRateConditionLayer(scale_bias) applied to the
              block input BEFORE each CausalDecoderBlock:
                  x = x * scale_embed[sr_idx] + bias_embed[sr_idx]
              For 48 kHz output, sr_idx = bucketize(48000, [20000,30000,40000])
              = 3, so we bake row 3 of scale/bias as per-channel (C,) vectors.

Notes:
- weight_norm uses the legacy `weight_g`/`weight_v` keys (not the
  `parametrizations.*` form); folded to a plain weight at convert time.
- Snake alphas are kept signed: AudioVAE/SNAC use `(alpha + 1e-9).reciprocal()`
  and trained alphas can be negative — the runtime must use a sign-preserving
  snake (NOT `codec_op_snake`, which clamps alpha positive).
- `use_noise_block=False` → deterministic decode, no NoiseBlock tensors.
- Causal convs (left-pad) and causal transpose convs (right-trim) are handled in
  the graph; weights are stored in native PyTorch layout (matches snac.cpp).
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
        return t.detach().cpu().float().numpy()
    return np.asarray(t)


def _apply_weight_norm(weight_v: np.ndarray, weight_g: np.ndarray, dim: int = 0) -> np.ndarray:
    """Reverse legacy `torch.nn.utils.weight_norm`. AudioVAE Conv1d and
    ConvTranspose1d both default to dim=0 (weight_g shape (C0, 1, 1))."""
    if weight_v.ndim < 2:
        raise ValueError(f"weight_norm expects ndim >= 2, got {weight_v.ndim}")
    axes = tuple(i for i in range(weight_v.ndim) if i != dim)
    norm = np.linalg.norm(weight_v, axis=axes, keepdims=True)
    if weight_g.shape != norm.shape:
        weight_g = weight_g.reshape(norm.shape)
    return (weight_v * (weight_g / (norm + 1e-12))).astype(np.float32)


# bucketize(out_sample_rate=48000, [20000, 30000, 40000]) -> 3
_SR_BUCKET_48K = 3


class BlueMagpieConverter(BaseConverter):
    @property
    def model_type(self) -> str:
        return "bluemagpie"

    @property
    def architecture(self) -> str:
        return "bluemagpie_audiovae"

    # ------------------------------------------------------------------ #
    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        if torch is None:
            raise RuntimeError("torch is required for BlueMagpie conversion")
        path = Path(checkpoint_dir)
        if path.is_dir():
            cfg_path = path / "config.json"
            vae_path = path / "audiovae.pth"
            lm_path = path / "pytorch_model.bin"
        else:
            vae_path = path
            cfg_path = path.parent / "config.json"
            lm_path = path.parent / "pytorch_model.bin"
        if not vae_path.is_file():
            raise FileNotFoundError(f"missing audiovae.pth: {vae_path}")
        with open(cfg_path, "r") as f:
            full_cfg = json.load(f)
        vcfg = full_cfg["audio_vae_config"]

        vae_state = torch.load(vae_path, map_location="cpu", weights_only=True)
        vae_state = vae_state.get("state_dict", vae_state)
        self.state_dict = OrderedDict((k, _to_numpy(v)) for k, v in vae_state.items())

        # LM-side modules (everything except the Barbet base_lm, which lands in
        # llama.cpp).  Optional: decode-only conversions can skip it.
        self.lm_state = None
        if lm_path.is_file():
            lm_state = torch.load(lm_path, map_location="cpu", weights_only=True)
            self.lm_state = OrderedDict(
                (k, _to_numpy(v)) for k, v in lm_state.items() if not k.startswith("base_lm.")
            )

        vox = full_cfg["vox_lm_config"]
        self.config = {
            "encoder_rates": list(vcfg["encoder_rates"]),
            "encoder_dim":   int(vcfg["encoder_dim"]),
            "decoder_rates": list(vcfg["decoder_rates"]),
            "latent_dim":    int(vcfg["latent_dim"]),
            "decoder_dim":   int(vcfg["decoder_dim"]),
            "sample_rate":   int(vcfg["sample_rate"]),       # 16000 (encode)
            "out_sample_rate": int(vcfg["out_sample_rate"]),  # 48000 (decode)
            "depthwise":     bool(vcfg["depthwise"]),
            "use_noise_block": bool(vcfg.get("use_noise_block", False)),
            "cond_type":     str(vcfg.get("cond_type", "scale_bias")),
            "sr_bin_boundaries": list(vcfg.get("sr_bin_boundaries") or []),
            # LM-side dims
            "patch_size":    int(full_cfg["patch_size"]),
            "feat_dim":      int(full_cfg["feat_dim"]),
            "h_vox":         int(vox["hidden_size"]),
            "h_enc":         int(full_cfg["encoder_config"]["hidden_dim"]),
            "h_dit":         int(full_cfg["dit_config"]["hidden_dim"]),
            "h_barbet":      int(full_cfg["barbet_config"]["hidden_size"]),
            "n_locenc":      int(full_cfg["encoder_config"]["num_layers"]),
            "n_locdit":      int(full_cfg["dit_config"]["num_layers"]),
            "n_ralm":        int(full_cfg["residual_lm_num_layers"]),
            "n_heads":       int(vox["num_attention_heads"]),
            "n_kv":          int(vox["num_key_value_heads"]),
            "kv_channels":   int(vox["kv_channels"]),
            "rms_eps":       float(vox["rms_norm_eps"]),
            "rope_theta":    float(vox["rope_theta"]),
            "fsq_latent":    int(full_cfg["scalar_quantization_latent_dim"]),
            "fsq_scale":     int(full_cfg["scalar_quantization_scale"]),
            "speaker_dim":   int(full_cfg["speaker_embed_dim"]),
            "cfm":           dict(full_cfg["dit_config"]["cfm_config"]),
            "rope_short_factor": list(vox["rope_scaling"]["short_factor"]),
            "rope_orig_max": int(vox["rope_scaling"]["original_max_position_embeddings"]),
            "max_position_embeddings": int(vox["max_position_embeddings"]),
        }

    def load_from_huggingface(self, model_id: str) -> None:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(repo_id=model_id, filename="config.json")
        hf_hub_download(repo_id=model_id, filename="audiovae.pth")
        try:
            hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        except Exception:
            pass
        self.load_from_checkpoint(Path(cfg_path).parent)

    # ------------------------------------------------------------------ #
    def convert_and_save(self, output_path: Path) -> None:
        if self.state_dict is None or self.config is None:
            raise RuntimeError("No model loaded.")
        cfg = self.config
        if not cfg["depthwise"]:
            raise NotImplementedError("only depthwise=True AudioVAE is wired up")
        if cfg["use_noise_block"]:
            raise NotImplementedError("use_noise_block=True not supported (deterministic decode only)")
        if cfg["cond_type"] != "scale_bias":
            raise NotImplementedError(f"cond_type={cfg['cond_type']!r} not supported")

        sd = self.state_dict
        writer = GGUFWriter(output_path, self.architecture)
        self._reset_quant_stats()

        rates = cfg["decoder_rates"]
        hop = int(np.prod(cfg["encoder_rates"]))   # 16k-domain hop (encoder)
        decode_hop = int(np.prod(rates))           # 48k-domain samples per latent frame

        writer.add_name("BlueMagpie-AudioVAE")
        writer.add_uint32("codec.sample_rate", cfg["out_sample_rate"])
        writer.add_uint32("codec.encode_sample_rate", cfg["sample_rate"])
        writer.add_uint32("codec.hop_size", hop)
        writer.add_uint32("codec.decode_hop_size", decode_hop)
        writer.add_uint32("codec.latent_dim", cfg["latent_dim"])
        writer.add_uint32("codec.n_q", 0)          # continuous latent — no codebooks
        writer.add_bool("codec.has_encoder", True)
        writer.add_bool("codec.has_decoder", True)
        writer.add_bool("codec.continuous_latent", True)

        writer.add_array("bluemagpie.decoder_rates", list(rates))
        writer.add_array("bluemagpie.encoder_rates", list(cfg["encoder_rates"]))
        writer.add_uint32("bluemagpie.decoder_dim", cfg["decoder_dim"])
        writer.add_uint32("bluemagpie.encoder_dim", cfg["encoder_dim"])
        writer.add_bool("bluemagpie.depthwise", True)

        # -------------------------------------------------------------- #
        def _t(name: str) -> np.ndarray:
            arr = sd.get(name)
            if arr is None:
                raise KeyError(f"missing tensor: {name}")
            return np.asarray(arr)

        def add_wn_conv(prefix: str, out_name: str) -> None:
            w = _apply_weight_norm(_t(prefix + ".weight_v"), _t(prefix + ".weight_g"), dim=0)
            self._add_tensor(writer, out_name + ".w", w)
            if prefix + ".bias" in sd:
                self._add_tensor(writer, out_name + ".b", _t(prefix + ".bias").astype(np.float32), "F32")

        def add_alpha(prefix: str, out_name: str) -> None:
            alpha = _t(prefix + ".alpha").reshape(-1).astype(np.float32)
            self._add_tensor(writer, out_name + ".alpha", alpha, "F32")

        def add_residual_unit(prefix: str, out_name: str) -> None:
            # CausalResidualUnit: Snake → conv(k7 dw) → Snake → conv(k1)
            add_alpha(prefix + ".block.0", out_name + ".act1")
            add_wn_conv(prefix + ".block.1", out_name + ".conv1")
            add_alpha(prefix + ".block.2", out_name + ".act2")
            add_wn_conv(prefix + ".block.3", out_name + ".conv2")

        def add_sr_cond(idx: int, out_name: str) -> None:
            # scale_bias: bake row sr_idx of scale/bias as per-channel vectors.
            scale = _t(f"decoder.sr_cond_model.{idx}.scale_embed.weight")[_SR_BUCKET_48K].astype(np.float32)
            bias = _t(f"decoder.sr_cond_model.{idx}.bias_embed.weight")[_SR_BUCKET_48K].astype(np.float32)
            self._add_tensor(writer, out_name + ".scale", scale, "F32")
            self._add_tensor(writer, out_name + ".bias", bias, "F32")

        # Input convs: model.0 depthwise, model.1 pointwise.
        add_wn_conv("decoder.model.0", "bluemagpie.dec.conv_in_dw")
        add_wn_conv("decoder.model.1", "bluemagpie.dec.conv_in_pw")

        # 6 decoder blocks at model.{2..7}, sr_cond at sr_cond_model.{2..7}.
        for bi, _stride in enumerate(rates):
            model_idx = bi + 2
            base = f"decoder.model.{model_idx}.block"
            o = f"bluemagpie.dec.b{bi}"
            add_sr_cond(model_idx, o + ".cond")
            add_alpha(base + ".0", o + ".act")
            add_wn_conv(base + ".1", o + ".convtr")
            for ri in range(3):                     # residual units at block.2/3/4
                add_residual_unit(f"{base}.{ri + 2}", f"{o}.r{ri}")

        # Output: Snake → conv(k7) → Tanh.
        n_blocks = len(rates)
        add_alpha(f"decoder.model.{n_blocks + 2}", "bluemagpie.dec.act_final")
        add_wn_conv(f"decoder.model.{n_blocks + 3}", "bluemagpie.dec.conv_out")

        # -------------------------------------------------------------- #
        # Encoder (audio → latent mu, for ref/prompt voice-clone modes).
        # CausalEncoder: conv(1→128) → 4 CausalEncoderBlocks (rates
        # [2,5,8,8], each = 3 CausalResidualUnits + Snake + strided
        # downsample conv) → fc_mu(→64).
        # -------------------------------------------------------------- #
        add_wn_conv("encoder.block.0", "bluemagpie.enc.conv0")
        for bi, _stride in enumerate(cfg["encoder_rates"], start=1):
            base = f"encoder.block.{bi}.block"
            o = f"bluemagpie.enc.b{bi}"
            for ri in range(3):                          # residual units at block.0/1/2
                add_residual_unit(f"{base}.{ri}", f"{o}.r{ri}")
            add_alpha(f"{base}.3", o + ".act")           # Snake
            add_wn_conv(f"{base}.4", o + ".down")        # strided downsample
        add_wn_conv("encoder.fc_mu", "bluemagpie.enc.fc_mu")

        # -------------------------------------------------------------- #
        # LM-side modules (LocEnc / LocDiT / RALM / FSQ / projections /
        # stop / speaker / tslm_adapter) + baked RoPE table.
        # -------------------------------------------------------------- #
        if self.lm_state is not None:
            self._emit_lm(writer, cfg)

        self._warn_if_no_quantized()
        writer.write()
        self.log(f"wrote {output_path}")

    # ------------------------------------------------------------------ #
    def _emit_lm(self, writer: GGUFWriter, cfg: Dict[str, Any]) -> None:
        sd = self.lm_state

        def lm_t(name: str) -> np.ndarray:
            arr = sd.get(name)
            if arr is None:
                raise KeyError(f"missing LM tensor: {name}")
            return np.asarray(arr)

        def add_lin(prefix: str, out: str, bias: bool = False) -> None:
            self._add_tensor(writer, out + ".w", lm_t(prefix + ".weight"))  # (out,in) → ggml (in,out)
            if bias:
                self._add_tensor(writer, out + ".b", lm_t(prefix + ".bias").astype(np.float32), "F32")

        def add_norm(prefix: str, out: str) -> None:
            self._add_tensor(writer, out + ".w", lm_t(prefix + ".weight").astype(np.float32), "F32")

        def add_minicpm_stack(src: str, out: str, n_layers: int) -> None:
            for i in range(n_layers):
                s, o = f"{src}.layers.{i}", f"{out}.layers.{i}"
                add_lin(f"{s}.self_attn.q_proj", o + ".attn_q")
                add_lin(f"{s}.self_attn.k_proj", o + ".attn_k")
                add_lin(f"{s}.self_attn.v_proj", o + ".attn_v")
                add_lin(f"{s}.self_attn.o_proj", o + ".attn_o")
                add_lin(f"{s}.mlp.gate_proj", o + ".gate")
                add_lin(f"{s}.mlp.up_proj", o + ".up")
                add_lin(f"{s}.mlp.down_proj", o + ".down")
                add_norm(f"{s}.input_layernorm", o + ".ln1")
                add_norm(f"{s}.post_attention_layernorm", o + ".ln2")

        # LocEnc (feat_encoder)
        add_lin("feat_encoder.in_proj", "lm.locenc.in_proj", bias=True)
        self._add_tensor(writer, "lm.locenc.special_token",
                         lm_t("feat_encoder.special_token").reshape(-1).astype(np.float32), "F32")
        add_minicpm_stack("feat_encoder.encoder", "lm.locenc", cfg["n_locenc"])
        add_norm("feat_encoder.encoder.norm", "lm.locenc.norm")

        # LocDiT (feat_decoder.estimator)
        est = "feat_decoder.estimator"
        add_lin(f"{est}.in_proj", "lm.locdit.in_proj", bias=True)
        add_lin(f"{est}.cond_proj", "lm.locdit.cond_proj", bias=True)
        add_lin(f"{est}.out_proj", "lm.locdit.out_proj", bias=True)
        add_lin(f"{est}.time_mlp.linear_1", "lm.locdit.time_mlp.l1", bias=True)
        add_lin(f"{est}.time_mlp.linear_2", "lm.locdit.time_mlp.l2", bias=True)
        add_lin(f"{est}.delta_time_mlp.linear_1", "lm.locdit.dtime_mlp.l1", bias=True)
        add_lin(f"{est}.delta_time_mlp.linear_2", "lm.locdit.dtime_mlp.l2", bias=True)
        add_minicpm_stack(f"{est}.decoder", "lm.locdit", cfg["n_locdit"])
        add_norm(f"{est}.decoder.norm", "lm.locdit.norm")

        # RALM (residual_lm) — no rope
        add_minicpm_stack("residual_lm", "lm.ralm", cfg["n_ralm"])
        add_norm("residual_lm.norm", "lm.ralm.norm")

        # FSQ + projections + stop + speaker + tslm_adapter
        add_lin("fsq_layer.in_proj", "lm.fsq.in_proj", bias=True)
        add_lin("fsq_layer.out_proj", "lm.fsq.out_proj", bias=True)
        add_lin("enc_to_lm_proj", "lm.proj.enc_to_lm", bias=True)
        add_lin("lm_to_dit_proj", "lm.proj.lm_to_dit", bias=True)
        add_lin("res_to_dit_proj", "lm.proj.res_to_dit", bias=True)
        add_lin("fusion_concat_proj", "lm.proj.fusion_concat", bias=True)
        add_lin("enc_to_tslm_proj", "lm.proj.enc_to_tslm", bias=True)
        add_lin("stop_proj", "lm.stop.proj", bias=True)
        self._add_tensor(writer, "lm.stop.head.w", lm_t("stop_head.weight"))
        add_norm("speaker_projector.norm", "lm.speaker.norm")
        add_lin("speaker_projector.proj", "lm.speaker.proj", bias=True)
        add_norm("tslm_adapter.norm", "lm.tslm_adapter.norm")
        add_lin("tslm_adapter.proj", "lm.tslm_adapter.proj", bias=True)
        add_norm("tslm_adapter.blocks.0.norm", "lm.tslm_adapter.blk0.ln")
        add_lin("tslm_adapter.blocks.0.gate_proj", "lm.tslm_adapter.blk0.gate")
        add_lin("tslm_adapter.blocks.0.up_proj", "lm.tslm_adapter.blk0.up")
        add_lin("tslm_adapter.blocks.0.down_proj", "lm.tslm_adapter.blk0.down")

        # Baked LongRoPE cos/sin table (short_factor branch, scaling_factor=1).
        # LocEnc/LocDiT only; RALM is no_rope.  Positions 0..n_pos-1.
        head_dim = cfg["kv_channels"]
        n_pos = 16  # LocEnc seq=5, LocDiT seq=10 — 16 is ample headroom
        short = np.asarray(cfg["rope_short_factor"], dtype=np.float64)  # len head_dim/2
        inv_freq = 1.0 / (cfg["rope_theta"] ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
        scale = cfg["max_position_embeddings"] / cfg["rope_orig_max"]
        scaling = float(np.sqrt(1 + np.log(scale) / np.log(cfg["rope_orig_max"]))) if scale > 1 else 1.0
        t = np.arange(n_pos, dtype=np.float64)
        freqs = np.outer(t, 1.0 / short) * inv_freq[None, :]            # [n_pos, head_dim/2]
        emb = np.concatenate([freqs, freqs], axis=-1)                   # [n_pos, head_dim]
        self._add_tensor(writer, "lm.rope.cos", (np.cos(emb) * scaling).astype(np.float32), "F32")
        self._add_tensor(writer, "lm.rope.sin", (np.sin(emb) * scaling).astype(np.float32), "F32")

        # LM metadata
        writer.add_bool("codec.lm.has_adaptor", True)
        writer.add_string("codec.lm.kind", "continuous_latent_cfm")
        writer.add_string("codec.lm.host_arch", "barbet")
        writer.add_uint32("codec.lm.hidden_dim", cfg["h_barbet"])       # Barbet hidden fed in
        writer.add_uint32("codec.lm.h_vox", cfg["h_vox"])
        writer.add_uint32("codec.lm.h_enc", cfg["h_enc"])
        writer.add_uint32("codec.lm.h_dit", cfg["h_dit"])
        writer.add_uint32("codec.lm.patch_size", cfg["patch_size"])
        writer.add_uint32("codec.lm.latent_dim", cfg["feat_dim"])
        writer.add_uint32("codec.lm.n_locenc", cfg["n_locenc"])
        writer.add_uint32("codec.lm.n_locdit", cfg["n_locdit"])
        writer.add_uint32("codec.lm.n_ralm", cfg["n_ralm"])
        writer.add_uint32("codec.lm.n_heads", cfg["n_heads"])
        writer.add_uint32("codec.lm.n_kv", cfg["n_kv"])
        writer.add_uint32("codec.lm.head_dim", cfg["kv_channels"])
        writer.add_uint32("codec.lm.fsq_latent", cfg["fsq_latent"])
        writer.add_uint32("codec.lm.fsq_scale", cfg["fsq_scale"])
        writer.add_uint32("codec.lm.speaker_dim", cfg["speaker_dim"])
        writer.add_float32("codec.lm.rms_eps", cfg["rms_eps"])
        writer.add_float32("codec.lm.cfm_sigma_min", float(cfg["cfm"]["sigma_min"]))
        writer.add_string("codec.lm.cfm_solver", str(cfg["cfm"]["solver"]))
        # speaker section: the SpeakerProjector weights ship in the GGUF
        # (lm.speaker.*), but the ECAPA centroid front-end isn't wired into the
        # codec_lm speaker API yet, so don't advertise a speaker encoder.
