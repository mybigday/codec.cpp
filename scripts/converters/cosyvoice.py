"""CosyVoice family converter (CosyVoice-300M / -SFT / -Instruct, BreezyVoice,
CosyVoice2-0.5B, Fun-CosyVoice3-0.5B).

One `cosyvoice` codec architecture covers all releases; the variant is carried
in GGUF metadata (`cosyvoice.*`):

    v1 (300M, BreezyVoice): ConformerEncoder → InterpolateRegulator → U-Net
        CFM (channels [256, 256], GroupNorm) → HiFT 22050 Hz (SineGen).
    v2 (CosyVoice2):        UpsampleConformerEncoder → causal U-Net CFM
        (channels [256], LayerNorm) → HiFT 24 kHz (SineGen2).
    v3 (CosyVoice3):        PreLookahead + repeat ×2 → DiT CFM → causal HiFT.

Speaker-from-audio encoders are bundled too: the S3 speech tokenizer
(v1 50 Hz VQ-4096, v2/v3 25 Hz FSQ-6561), CAMPPlus (from `campplus.onnx`) and
the prompt-mel filterbank. BreezyVoice ships the CosyVoice-300M flow / HiFT /
encoders unchanged (only `llm.pt` differs); its runtime differences are the
length regulator (joint nearest interpolation, as in the BreezyVoice fork) and
plain top-k sampling (no RAS).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .base import BaseConverter
from .campplus import campplus_tensors_from_onnx
from .chatterbox import _materialize_state_dict, _materialize_weight_norm
from .speaker import encoder_hash, speaker_name, write_speaker, write_speaker_meta
from utils.gguf_writer import GGUFWriter

ARCH = "cosyvoice"

# CosyVoice-300M-SFT / BreezyVoice spk2info voices → GGUF speaker names.
SFT_SPEAKER_NAMES = {
    "中文女": "zh_female", "中文男": "zh_male", "日语男": "ja_male", "粤语女": "yue_female",
    "英文女": "en_female", "英文男": "en_male", "韩语女": "ko_female",
}

Tensors = List[Tuple[str, np.ndarray]]


def _np(t: Any) -> np.ndarray:
    if torch.is_tensor(t):
        t = t.detach().cpu().float().numpy() if t.is_floating_point() else t.detach().cpu().numpy()
    return np.asarray(t)


def _load_pt(path: Path) -> Dict[str, np.ndarray]:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return {k: _np(v) for k, v in sd.items()}


class _Taker:
    """Pop tensors from a state dict; leftovers are reported as unmapped."""

    def __init__(self, state: Dict[str, np.ndarray], what: str):
        self.state = dict(state)
        self.what = what
        self.out: Tensors = []

    def has(self, key: str) -> bool:
        return key in self.state

    def take(self, dst: str, key: str) -> None:
        if key not in self.state:
            raise KeyError(f"missing {self.what} tensor: {key}")
        self.out.append((dst, self.state.pop(key)))

    def take_opt(self, dst: str, key: str) -> None:
        if key in self.state:
            self.out.append((dst, self.state.pop(key)))

    def finish(self, ignore: Tuple[str, ...] = ()) -> Tensors:
        left = sorted(k for k in self.state if not k.startswith(ignore))
        if left:
            raise RuntimeError(f"unmapped {self.what} tensors: {left[:12]} ({len(left)} total)")
        return self.out


def _count(state: Dict[str, Any], pattern: str) -> int:
    rx = re.compile(pattern)
    idx = {int(m.group(1)) for k in state for m in [rx.match(k)] if m}
    return max(idx) + 1 if idx else 0


# ---------------------------------------------------------------- flow ----

def _conformer_layer(t: _Taker, src: str, dst: str) -> None:
    a, f = src + ".self_attn", src + ".feed_forward"
    for d, s in (("norm_mha", src + ".norm_mha"), ("norm_ff", src + ".norm_ff")):
        t.take(f"{dst}.{d}.w", s + ".weight")
        t.take(f"{dst}.{d}.b", s + ".bias")
    for d, s in (("q", "linear_q"), ("k", "linear_k"), ("v", "linear_v"), ("o", "linear_out")):
        t.take(f"{dst}.attn.{d}.w", f"{a}.{s}.weight")
        t.take(f"{dst}.attn.{d}.b", f"{a}.{s}.bias")
    t.take(f"{dst}.attn.pos.w", a + ".linear_pos.weight")
    t.take(f"{dst}.attn.pbu", a + ".pos_bias_u")
    t.take(f"{dst}.attn.pbv", a + ".pos_bias_v")
    t.take(f"{dst}.ff.w1.w", f + ".w_1.weight")
    t.take(f"{dst}.ff.w1.b", f + ".w_1.bias")
    t.take(f"{dst}.ff.w2.w", f + ".w_2.weight")
    t.take(f"{dst}.ff.w2.b", f + ".w_2.bias")


def _linear_embed(t: _Taker, src: str, dst: str) -> None:
    t.take(dst + ".lin.w", src + ".out.0.weight")
    t.take(dst + ".lin.b", src + ".out.0.bias")
    t.take(dst + ".ln.w", src + ".out.1.weight")
    t.take(dst + ".ln.b", src + ".out.1.bias")


def _cfm_resnet(t: _Taker, src: str, dst: str, norm_idx: int) -> None:
    for b in (1, 2):
        t.take(f"{dst}.b{b}.cv.w", f"{src}.block{b}.block.0.weight")
        t.take(f"{dst}.b{b}.cv.b", f"{src}.block{b}.block.0.bias")
        t.take(f"{dst}.b{b}.ln.w", f"{src}.block{b}.block.{norm_idx}.weight")
        t.take(f"{dst}.b{b}.ln.b", f"{src}.block{b}.block.{norm_idx}.bias")
    t.take(dst + ".mlp.w", src + ".mlp.1.weight")
    t.take(dst + ".mlp.b", src + ".mlp.1.bias")
    t.take(dst + ".res.w", src + ".res_conv.weight")
    t.take(dst + ".res.b", src + ".res_conv.bias")


def _cfm_transformer(t: _Taker, src: str, dst: str) -> None:
    a = src + ".attn1"
    for n in ("norm1", "norm3"):
        t.take(f"{dst}.{n}.w", f"{src}.{n}.weight")
        t.take(f"{dst}.{n}.b", f"{src}.{n}.bias")
    for d, s in (("q", "to_q"), ("k", "to_k"), ("v", "to_v")):
        t.take(f"{dst}.attn.{d}.w", f"{a}.{s}.weight")
    t.take(dst + ".attn.o.w", a + ".to_out.0.weight")
    t.take(dst + ".attn.o.b", a + ".to_out.0.bias")
    t.take(dst + ".ff.w1.w", src + ".ff.net.0.proj.weight")
    t.take(dst + ".ff.w1.b", src + ".ff.net.0.proj.bias")
    t.take(dst + ".ff.w2.w", src + ".ff.net.2.weight")
    t.take(dst + ".ff.w2.b", src + ".ff.net.2.bias")


def _cfm_unet(t: _Taker, cfg: Dict[str, Any], causal: bool) -> None:
    e = "decoder.estimator"
    t.take("cv.cfm.t.l1.w", e + ".time_mlp.linear_1.weight")
    t.take("cv.cfm.t.l1.b", e + ".time_mlp.linear_1.bias")
    t.take("cv.cfm.t.l2.w", e + ".time_mlp.linear_2.weight")
    t.take("cv.cfm.t.l2.b", e + ".time_mlp.linear_2.bias")
    norm_idx = 2 if causal else 1   # CausalBlock1D: conv, transpose, LN; Block1D: conv, GN
    n_levels = _count(t.state, rf"{e}\.down_blocks\.(\d+)\.")
    n_mid = _count(t.state, rf"{e}\.mid_blocks\.(\d+)\.")
    n_tr = _count(t.state, rf"{e}\.mid_blocks\.0\.1\.(\d+)\.")
    for group, short, n in (("down_blocks", "dn", n_levels), ("mid_blocks", "md", n_mid),
                            ("up_blocks", "up", n_levels)):
        for i in range(n):
            s, d = f"{e}.{group}.{i}", f"cv.cfm.{short}.{i}"
            _cfm_resnet(t, s + ".0", d + ".r", norm_idx)
            for j in range(n_tr):
                _cfm_transformer(t, f"{s}.1.{j}", f"{d}.t.{j}")
            if group != "mid_blocks":
                # Downsample1D / Upsample1D wrap their conv as `.conv`, plain convs do not.
                key = f"{s}.2.conv" if f"{s}.2.conv.weight" in t.state else f"{s}.2"
                t.take(d + ".x.w", key + ".weight")
                t.take(d + ".x.b", key + ".bias")
    t.take("cv.cfm.final.cv.w", e + ".final_block.block.0.weight")
    t.take("cv.cfm.final.cv.b", e + ".final_block.block.0.bias")
    t.take("cv.cfm.final.ln.w", f"{e}.final_block.block.{norm_idx}.weight")
    t.take("cv.cfm.final.ln.b", f"{e}.final_block.block.{norm_idx}.bias")
    t.take("cv.cfm.proj.w", e + ".final_proj.weight")
    t.take("cv.cfm.proj.b", e + ".final_proj.bias")
    cfg.update({"cfm_n_levels": n_levels, "cfm_n_mid": n_mid, "cfm_n_transformers": n_tr,
                "cfm_causal": causal})


def _dit(t: _Taker, cfg: Dict[str, Any]) -> None:
    e = "decoder.estimator"
    t.take("cv.dit.t.l1.w", e + ".time_embed.time_mlp.0.weight")
    t.take("cv.dit.t.l1.b", e + ".time_embed.time_mlp.0.bias")
    t.take("cv.dit.t.l2.w", e + ".time_embed.time_mlp.2.weight")
    t.take("cv.dit.t.l2.b", e + ".time_embed.time_mlp.2.bias")
    t.take("cv.dit.in.proj.w", e + ".input_embed.proj.weight")
    t.take("cv.dit.in.proj.b", e + ".input_embed.proj.bias")
    for c in (1, 2):
        t.take(f"cv.dit.in.cp{c}.w", f"{e}.input_embed.conv_pos_embed.conv{c}.0.weight")
        t.take(f"cv.dit.in.cp{c}.b", f"{e}.input_embed.conv_pos_embed.conv{c}.0.bias")
    inv_freq = t.state.pop(e + ".rotary_embed.inv_freq")
    depth = _count(t.state, rf"{e}\.transformer_blocks\.(\d+)\.")
    for i in range(depth):
        s, d = f"{e}.transformer_blocks.{i}", f"cv.dit.blk.{i}"
        t.take(d + ".ada.w", s + ".attn_norm.linear.weight")
        t.take(d + ".ada.b", s + ".attn_norm.linear.bias")
        for dd, ss in (("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("o", "to_out.0")):
            t.take(f"{d}.attn.{dd}.w", f"{s}.attn.{ss}.weight")
            t.take(f"{d}.attn.{dd}.b", f"{s}.attn.{ss}.bias")
        t.take(d + ".ff1.w", s + ".ff.ff.0.0.weight")
        t.take(d + ".ff1.b", s + ".ff.ff.0.0.bias")
        t.take(d + ".ff2.w", s + ".ff.ff.2.weight")
        t.take(d + ".ff2.b", s + ".ff.ff.2.bias")
    t.take("cv.dit.out_ada.w", e + ".norm_out.linear.weight")
    t.take("cv.dit.out_ada.b", e + ".norm_out.linear.bias")
    t.take("cv.dit.proj_out.w", e + ".proj_out.weight")
    t.take("cv.dit.proj_out.b", e + ".proj_out.bias")
    dim = int(dict(t.out)["cv.dit.in.proj.w"].shape[0])
    rope_dim = int(inv_freq.shape[0]) * 2
    ref = 1.0 / (10000.0 ** (np.arange(0, rope_dim, 2, dtype=np.float64) / rope_dim))
    if not np.allclose(inv_freq.astype(np.float64), ref, rtol=1e-5):
        raise ValueError("DiT rotary inv_freq is not theta=10000; runtime assumes it")
    cfg.update({"dit_depth": depth, "dit_dim": dim, "dit_rope_dim": rope_dim})


def _flow_tensors(state: Dict[str, np.ndarray], version: int, cfg: Dict[str, Any]) -> Tensors:
    t = _Taker(state, "flow")
    t.take("cv.flow.input_emb.w", "input_embedding.weight")
    t.take("cv.flow.spk_aff.w", "spk_embed_affine_layer.weight")
    t.take("cv.flow.spk_aff.b", "spk_embed_affine_layer.bias")
    if version == 3:
        t.take("cv.flow.pre.cv1.w", "pre_lookahead_layer.conv1.weight")
        t.take("cv.flow.pre.cv1.b", "pre_lookahead_layer.conv1.bias")
        t.take("cv.flow.pre.cv2.w", "pre_lookahead_layer.conv2.weight")
        t.take("cv.flow.pre.cv2.b", "pre_lookahead_layer.conv2.bias")
        _dit(t, cfg)
        return t.finish()

    t.take("cv.flow.proj.w", "encoder_proj.weight")
    t.take("cv.flow.proj.b", "encoder_proj.bias")
    _linear_embed(t, "encoder.embed", "cv.flow.enc.embed")
    t.take("cv.flow.enc.after_norm.w", "encoder.after_norm.weight")
    t.take("cv.flow.enc.after_norm.b", "encoder.after_norm.bias")
    n_layers = _count(t.state, r"encoder\.encoders\.(\d+)\.")
    for i in range(n_layers):
        _conformer_layer(t, f"encoder.encoders.{i}", f"cv.flow.enc.blk.{i}")
    cfg["enc_layers"] = n_layers
    cfg["enc_hidden"] = int(t.out[0][1].shape[1])
    cfg["enc_heads"] = int(state["encoder.encoders.0.self_attn.pos_bias_u"].shape[0])
    if version == 2:
        t.take("cv.flow.enc.pre.cv1.w", "encoder.pre_lookahead_layer.conv1.weight")
        t.take("cv.flow.enc.pre.cv1.b", "encoder.pre_lookahead_layer.conv1.bias")
        t.take("cv.flow.enc.pre.cv2.w", "encoder.pre_lookahead_layer.conv2.weight")
        t.take("cv.flow.enc.pre.cv2.b", "encoder.pre_lookahead_layer.conv2.bias")
        t.take("cv.flow.enc.up.w", "encoder.up_layer.conv.weight")
        t.take("cv.flow.enc.up.b", "encoder.up_layer.conv.bias")
        _linear_embed(t, "encoder.up_embed", "cv.flow.enc.up_embed")
        n_up = _count(t.state, r"encoder\.up_encoders\.(\d+)\.")
        for i in range(n_up):
            _conformer_layer(t, f"encoder.up_encoders.{i}", f"cv.flow.enc.up_blk.{i}")
        cfg["enc_up_layers"] = n_up
        _cfm_unet(t, cfg, causal=True)
    else:
        n_lr = _count(t.state, r"length_regulator\.model\.(\d+)\.")
        # Sequential(Conv, GroupNorm, Mish) × k, then the 1×1 output conv.
        for i, base in enumerate(range(0, n_lr - 1, 3)):
            t.take(f"cv.flow.lr.{i}.cv.w", f"length_regulator.model.{base}.weight")
            t.take(f"cv.flow.lr.{i}.cv.b", f"length_regulator.model.{base}.bias")
            t.take(f"cv.flow.lr.{i}.gn.w", f"length_regulator.model.{base + 1}.weight")
            t.take(f"cv.flow.lr.{i}.gn.b", f"length_regulator.model.{base + 1}.bias")
        t.take("cv.flow.lr.out.w", f"length_regulator.model.{n_lr - 1}.weight")
        t.take("cv.flow.lr.out.b", f"length_regulator.model.{n_lr - 1}.bias")
        cfg["lr_blocks"] = (n_lr - 1) // 3
        _cfm_unet(t, cfg, causal=False)
    return t.finish()


# ---------------------------------------------------------------- hift ----

def _materialize_legacy_weight_norm(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Bake legacy `weight_g` / `weight_v` weight-norm pairs (v1 checkpoints)."""
    out = {k: v for k, v in state.items() if not k.endswith((".weight_g", ".weight_v"))}
    for k in state:
        if k.endswith(".weight_g"):
            base = k[: -len(".weight_g")]
            out[base + ".weight"] = _materialize_weight_norm(state[k], state[base + ".weight_v"])
    return out


def _hift_tensors(state: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> Tensors:
    t = _Taker(_materialize_state_dict(_materialize_legacy_weight_norm(state)), "hift")
    n_f0 = _count(t.state, r"f0_predictor\.condnet\.(\d+)\.")
    for li, si in enumerate(range(0, n_f0, 2)):
        t.take(f"cv.hift.f0.cn.{li}.w", f"f0_predictor.condnet.{si}.weight")
        t.take(f"cv.hift.f0.cn.{li}.b", f"f0_predictor.condnet.{si}.bias")
    t.take("cv.hift.f0.cls.w", "f0_predictor.classifier.weight")
    t.take("cv.hift.f0.cls.b", "f0_predictor.classifier.bias")
    t.take("cv.hift.src.lin.w", "m_source.l_linear.weight")
    t.take("cv.hift.src.lin.b", "m_source.l_linear.bias")
    for n in ("conv_pre", "conv_post"):
        t.take(f"cv.hift.{n}.w", n + ".weight")
        t.take(f"cv.hift.{n}.b", n + ".bias")
    n_ups = _count(t.state, r"ups\.(\d+)\.")
    n_rb = _count(t.state, r"resblocks\.(\d+)\.") // n_ups
    n_dil = _count(t.state, r"resblocks\.0\.convs1\.(\d+)\.")

    def resblock(src: str, dst: str) -> None:
        for k in range(n_dil):
            t.take(f"{dst}.cv1.{k}.w", f"{src}.convs1.{k}.weight")
            t.take(f"{dst}.cv1.{k}.b", f"{src}.convs1.{k}.bias")
            t.take(f"{dst}.cv2.{k}.w", f"{src}.convs2.{k}.weight")
            t.take(f"{dst}.cv2.{k}.b", f"{src}.convs2.{k}.bias")
            t.take(f"{dst}.a1.{k}", f"{src}.activations1.{k}.alpha")
            t.take(f"{dst}.a2.{k}", f"{src}.activations2.{k}.alpha")

    for i in range(n_ups):
        t.take(f"cv.hift.up.{i}.w", f"ups.{i}.weight")
        t.take(f"cv.hift.up.{i}.b", f"ups.{i}.bias")
        t.take(f"cv.hift.src_dn.{i}.w", f"source_downs.{i}.weight")
        t.take(f"cv.hift.src_dn.{i}.b", f"source_downs.{i}.bias")
        resblock(f"source_resblocks.{i}", f"cv.hift.src_rb.{i}")
        for j in range(n_rb):
            resblock(f"resblocks.{i * n_rb + j}", f"cv.hift.rb.{i * n_rb + j}")
    out = t.finish()
    cfg["hift_n_ups"] = n_ups
    return out


# ----------------------------------------------------- speech tokenizer ----

def _tokenizer_tensors(onnx_path: Path, version: int) -> Tensors:
    import s3tokenizer
    import whisper

    model = s3tokenizer.load_model(str(onnx_path))
    sd = {k: _np(v) for k, v in model.state_dict().items()}
    t = _Taker(sd, "speech tokenizer")
    out: Tensors = [("s3t.mel_filters", whisper.audio.mel_filters("cpu", 128).numpy().astype(np.float32))]
    for n in ("conv1", "conv2"):
        t.take(f"s3t.enc.{n}.w", f"encoder.{n}.weight")
        t.take(f"s3t.enc.{n}.b", f"encoder.{n}.bias")
    n_layers = _count(sd, r"encoder\.blocks\.(\d+)\.")
    for li in range(n_layers):
        p, d = f"encoder.blocks.{li}", f"s3t.enc.blk.{li}"
        for n in ("attn_ln", "mlp_ln"):
            t.take(f"{d}.{n}.w", f"{p}.{n}.weight")
            t.take(f"{d}.{n}.b", f"{p}.{n}.bias")
        t.take(d + ".attn.q.w", p + ".attn.query.weight")
        t.take(d + ".attn.q.b", p + ".attn.query.bias")
        t.take(d + ".attn.k.w", p + ".attn.key.weight")
        t.take(d + ".attn.v.w", p + ".attn.value.weight")
        t.take(d + ".attn.v.b", p + ".attn.value.bias")
        t.take(d + ".attn.o.w", p + ".attn.out.weight")
        t.take(d + ".attn.o.b", p + ".attn.out.bias")
        t.take_opt(d + ".attn.fsmn.w", p + ".attn.fsmn_block.weight")
        t.take(d + ".mlp.fc1.w", p + ".mlp.0.weight")
        t.take(d + ".mlp.fc1.b", p + ".mlp.0.bias")
        t.take(d + ".mlp.fc2.w", p + ".mlp.2.weight")
        t.take(d + ".mlp.fc2.b", p + ".mlp.2.bias")
    if version == 1:
        # Whisper encoder + Euclidean VQ: argmax(2 e·x - |e|^2) over the
        # L2-normalised hidden state picks the nearest codeword.
        t.take("s3t.enc.pos", "encoder.positional_embedding")
        embed = t.state.pop("quantizer._codebook.embed").astype(np.float32)
        out.append(("s3t.q.embed", embed))
        out.append(("s3t.q.embed_sq", (embed.astype(np.float64) ** 2).sum(1).astype(np.float32)))
    else:
        t.take("s3t.q.proj.w", "quantizer._codebook.project_down.weight")
        t.take("s3t.q.proj.b", "quantizer._codebook.project_down.bias")
    return out + t.finish()


# ------------------------------------------------------------ config ----

def _yaml_list(text: str, key: str) -> List[int] | None:
    m = re.search(rf"^\s*{key}:\s*\[([^\]]*)\]", text, re.M)
    return [int(x) for x in m.group(1).split(",")] if m else None


def _yaml_scalar(text: str, key: str) -> str | None:
    m = re.search(rf"^\s*{key}:\s*([^\s#]+)", text, re.M)
    return m.group(1) if m else None


def detect_cosyvoice_dir(path: Path) -> bool:
    return any((path / n).is_file() for n in ("cosyvoice.yaml", "cosyvoice2.yaml", "cosyvoice3.yaml"))


class CosyVoiceConverter(BaseConverter):
    model_name = "CosyVoice"

    def __init__(self, quantization: str = "F16", quantize_codebook: bool = False, verbose: bool = False,
                 lm_source=None):
        super().__init__(quantization=quantization, quantize_codebook=quantize_codebook, verbose=verbose)
        self.lm_source = lm_source
        self.ckpt_dir: Path | None = None

    @property
    def model_type(self) -> str:
        return ARCH

    @property
    def architecture(self) -> str:
        return ARCH

    def load_from_huggingface(self, model_id: str) -> None:
        from huggingface_hub import snapshot_download
        self.load_from_checkpoint(Path(snapshot_download(
            model_id, ignore_patterns=["CosyVoice-ttsfrd/*", "*.batch.onnx", "flow.decoder.estimator.*",
                                       "flow.encoder.*", "llm.rl.pt", "asset/*"])))

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        d = Path(checkpoint_dir)
        for v, n in ((3, "cosyvoice3.yaml"), (2, "cosyvoice2.yaml"), (1, "cosyvoice.yaml")):
            if (d / n).is_file():
                version, yaml_text = v, (d / n).read_text(encoding="utf-8")
                break
        else:
            raise FileNotFoundError(f"{d} has no cosyvoice*.yaml")
        readme = (d / "README.md").read_text(encoding="utf-8", errors="ignore") if (d / "README.md").is_file() else ""
        breezy = version == 1 and "BreezyVoice" in readme

        cfg: Dict[str, Any] = {"version": version, "breezyvoice": breezy}
        cfg["sample_rate"] = int(_yaml_scalar(yaml_text, "sample_rate") or (22050 if version == 1 else 24000))
        cfg["token_rate"] = 50 if version == 1 else 25
        cfg["codebook_size"] = 4096 if version == 1 else 6561
        cfg["upsample_rates"] = _yaml_list(yaml_text, "upsample_rates")
        cfg["upsample_kernels"] = _yaml_list(yaml_text, "upsample_kernel_sizes")
        cfg["resblock_kernels"] = _yaml_list(yaml_text, "resblock_kernel_sizes")
        cfg["source_resblock_kernels"] = _yaml_list(yaml_text, "source_resblock_kernel_sizes")
        # Prompt-mel parameters: the `feat_extractor` block (matcha mel_spectrogram).
        m = re.search(r"^feat_extractor:.*\n((?:[ \t]+.*\n)+)", yaml_text, re.M)
        feat = m.group(1) if m else ""
        fmax = _yaml_scalar(feat, "fmax")
        cfg["mel_fmax"] = float(fmax) if fmax not in (None, "null") else cfg["sample_rate"] / 2
        cfg["mel_n_fft"] = int(_yaml_scalar(feat, "n_fft") or 1024)
        cfg["mel_hop"] = int(_yaml_scalar(feat, "hop_size") or 256)
        self.config = cfg

        self.flow = _load_pt(d / "flow.pt")
        self.hift = _load_pt(d / "hift.pt")
        tok_name = {1: "speech_tokenizer_v1.onnx", 2: "speech_tokenizer_v2.onnx", 3: "speech_tokenizer_v3.onnx"}[version]
        self.tokenizer_path = d / tok_name if (d / tok_name).is_file() else None
        self.campplus_path = d / "campplus.onnx" if (d / "campplus.onnx").is_file() else None
        self.spk2info = None
        if (d / "spk2info.pt").is_file():
            self.spk2info = torch.load(d / "spk2info.pt", map_location="cpu", weights_only=False)
        self.ckpt_dir = d
        self.state_dict = self.flow
        self.log(f"Loaded CosyVoice v{version}{' (BreezyVoice)' if breezy else ''} from {d}")

    # -------------------------------------------------------------------
    def _speaker_params(self) -> str:
        c = self.config
        return (f"cosyvoice.spk.v1;version={c['version']};resample=torchaudio_sinc_hann(width=6,rolloff=0.99);"
                f"max_seconds=30;prompt_feat=matcha_mel(sr={c['sample_rate']},n_fft={c['mel_n_fft']},"
                f"hop={c['mel_hop']},n_mels=80,fmin=0,fmax={c['mel_fmax']});embedding=campplus(fbank16k,n_mels=80);"
                f"prompt_token=s3tokenizer_v{c['version']}(16k)")

    def convert_and_save(self, output_path: Path) -> None:
        c = self.config
        v = c["version"]
        writer = GGUFWriter(output_path, ARCH)
        self._reset_quant_stats()

        flow = _flow_tensors(self.flow, v, c)
        hift = _hift_tensors(self.hift, c)

        writer.add_name({1: "BreezyVoice" if c["breezyvoice"] else "CosyVoice-300M", 2: "CosyVoice2",
                         3: "CosyVoice3"}[v])
        writer.add_uint32("codec.sample_rate", c["sample_rate"])
        writer.add_uint32("codec.hop_size", c["sample_rate"] // c["token_rate"])
        writer.add_uint32("codec.n_q", 1)
        writer.add_uint32("codec.codebook_size", c["codebook_size"])
        writer.add_uint32("codec.token_rate_hz", c["token_rate"])
        writer.add_bool("codec.has_encoder", False)
        writer.add_bool("codec.has_decoder", True)

        p = f"{ARCH}."
        writer.add_uint32(p + "version", v)
        writer.add_uint32(p + "flow.n_timesteps", 10)
        writer.add_float32(p + "flow.cfg_rate", 0.7)
        writer.add_float32(p + "flow.speed", 1.0)   # tunable: --override-kv cosyvoice.flow.speed=float:1.2
        if v == 1:
            writer.add_string(p + "flow.length_regulator", "joint_nearest" if c["breezyvoice"] else "split_linear")
            writer.add_uint32(p + "flow.lr_blocks", c["lr_blocks"])
        else:
            writer.add_uint32(p + "flow.token_mel_ratio", 2)
        if v < 3:
            writer.add_uint32(p + "flow.enc.hidden", c["enc_hidden"])
            writer.add_uint32(p + "flow.enc.heads", c["enc_heads"])
            writer.add_uint32(p + "flow.enc.layers", c["enc_layers"])
            writer.add_uint32(p + "flow.enc.up_layers", c.get("enc_up_layers", 0))
            # BreezyVoice's fork builds its encoder layers with LayerNorm eps 1e-5;
            # upstream WeNet layers use 1e-12.
            writer.add_float32(p + "flow.enc.block_eps", 1e-5 if c["breezyvoice"] else 1e-12)
            writer.add_uint32(p + "cfm.n_levels", c["cfm_n_levels"])
            writer.add_uint32(p + "cfm.n_mid", c["cfm_n_mid"])
            writer.add_uint32(p + "cfm.n_transformers", c["cfm_n_transformers"])
            writer.add_uint32(p + "cfm.n_heads", 8)
            writer.add_uint32(p + "cfm.head_dim", 64)
            writer.add_bool(p + "cfm.causal", c["cfm_causal"])
        else:
            writer.add_uint32(p + "dit.depth", c["dit_depth"])
            writer.add_uint32(p + "dit.dim", c["dit_dim"])
            writer.add_uint32(p + "dit.heads", 16)
            writer.add_uint32(p + "dit.rope_dim", c["dit_rope_dim"])
        writer.add_array(p + "hift.upsample_rates", c["upsample_rates"])
        writer.add_array(p + "hift.upsample_kernels", c["upsample_kernels"])
        writer.add_array(p + "hift.resblock_kernels", c["resblock_kernels"])
        writer.add_array(p + "hift.source_resblock_kernels", c["source_resblock_kernels"])
        writer.add_uint32(p + "hift.sinegen", 1 if c["sample_rate"] == 22050 else 2)
        writer.add_bool(p + "hift.causal", v == 3)

        for name, arr in flow + hift:
            self._add_tensor(writer, name, arr.astype(np.float32) if np.issubdtype(arr.dtype, np.floating) else arr)

        # Reference-audio speaker encoders.
        enc: Tensors = []
        if self.tokenizer_path is not None and self.campplus_path is not None:
            enc = _tokenizer_tensors(self.tokenizer_path, v) + \
                [(f"cv.cp.{n}", a) for n, a in campplus_tensors_from_onnx(self.campplus_path)]
            from librosa.filters import mel as librosa_mel
            mel_fb = librosa_mel(sr=c["sample_rate"], n_fft=c["mel_n_fft"], n_mels=80, fmin=0,
                                 fmax=c["mel_fmax"]).astype(np.float32)
            enc.append(("cv.spk.mel_fb", mel_fb))
            for name, arr in enc:
                # The tokenizer's VQ argmax and FSQ rounding are precision-critical.
                self._add_tensor(writer, name, arr, st_dtype="F32" if name.startswith(("cv.cp.", "cv.spk.", "s3t.q.", "s3t.mel")) or arr.ndim < 2 else None)
            writer.add_bool(p + "spk.has_encoder", True)
            writer.add_uint32(p + "spk.mel_n_fft", c["mel_n_fft"])
            writer.add_uint32(p + "spk.mel_hop", c["mel_hop"])
            writer.add_uint32(p + "spk.tokenizer_layers", _count(dict(enc), r"s3t\.enc\.blk\.(\d+)\."))
        enc_hash = encoder_hash(enc, self._speaker_params())

        # SFT speakers (spk2info.pt): the x-vector, plus prompt tokens / mel
        # when the checkpoint ships them (BreezyVoice).  No transcript, so the
        # LLM never sees the prompt tokens (cross-lingual / SFT modes).
        if self.spk2info:
            names = []
            for raw, info in self.spk2info.items():
                name = SFT_SPEAKER_NAMES.get(raw)
                if name is None:
                    try:
                        name = speaker_name(raw)
                    except ValueError:
                        name = f"spk{len(names)}"
                slots = {"embedding": _np(info["embedding"]).reshape(-1).astype(np.float32)}
                if "speech_token" in info and "speech_feat" in info:
                    tok = _np(info["speech_token"]).reshape(-1).astype(np.int32)
                    feat = _np(info["speech_feat"]).reshape(-1, 80).astype(np.float32)   # [F, 80] → ggml [80, F]
                    if v != 1:
                        tl = min(feat.shape[0] // 2, tok.shape[0])
                        tok, feat = tok[:tl], feat[: 2 * tl]
                    slots["prompt_token"] = tok
                    slots["prompt_feat"] = feat
                write_speaker(writer, ARCH, name, slots, ref_text=None)
                names.append(name)
            write_speaker_meta(writer, ARCH, default=names[0], enc_hash=enc_hash)
        else:
            writer.add_string(f"{ARCH}.spk.encoder_hash", enc_hash)

        if (self.ckpt_dir / "llm.pt").is_file():
            from .lm_adaptor.cosyvoice import dump_cosyvoice_lm
            dump_cosyvoice_lm(writer, self.ckpt_dir, c,
                              add_tensor=lambda n, a, st=None: self._add_tensor(writer, n, np.asarray(a), st))

        writer.write()
        self._warn_if_no_quantized()
