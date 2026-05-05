"""HKUSTAudio/xcodec2 converter (BigCodec encoder + Wav2Vec2-Bert semantic + Vocos decoder)."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .base import BaseConverter
from utils.gguf_writer import GGUFWriter


# ResidualFSQ(dim=2048, levels=[4]*8, num_quantizers=1) — same as NeuCodec.
FSQ_LEVELS: List[int] = [4, 4, 4, 4, 4, 4, 4, 4]

# Wav2Vec2-Bert / SeamlessM4T feature extractor defaults (the only mode this
# checkpoint was trained with).
W2V_BERT_N_FFT = 512
W2V_BERT_WIN = 400
W2V_BERT_HOP = 160
W2V_BERT_N_MELS = 80
W2V_BERT_PREEMPHASIS = 0.97
W2V_BERT_MEL_FLOOR = 1.192092955078125e-07
W2V_BERT_STRIDE = 2
# SeamlessM4T uses a custom Kaldi-style mel filterbank.
W2V_BERT_MEL_LOW = 20.0
W2V_BERT_MEL_HIGH = 8000.0


def _apply_weight_norm(weight_v: np.ndarray, weight_g: np.ndarray, dim: int = 0) -> np.ndarray:
    if weight_v.ndim < 2:
        raise ValueError(f"weight_norm expects ndim >= 2, got {weight_v.ndim}")
    if dim < 0:
        dim = weight_v.ndim + dim
    axes = tuple(i for i in range(weight_v.ndim) if i != dim)
    norm = np.linalg.norm(weight_v, axis=axes, keepdims=True)
    if weight_g.shape != norm.shape:
        weight_g = weight_g.reshape(norm.shape)
    return weight_v * (weight_g / (norm + 1e-12))


def _seamless_mel_filterbank(n_mels: int = W2V_BERT_N_MELS,
                              n_fft: int = W2V_BERT_N_FFT,
                              sample_rate: int = 16000,
                              fmin: float = W2V_BERT_MEL_LOW,
                              fmax: float = W2V_BERT_MEL_HIGH) -> np.ndarray:
    """Reconstruct the SeamlessM4T mel filterbank.

    Uses transformers.audio_utils.mel_filter_bank with `mel_scale="kaldi"`,
    `triangularize_in_mel_space=True` and a "slaney" norm — the exact same
    arguments that SeamlessM4TFeatureExtractor sets at construction.
    """
    from transformers.audio_utils import mel_filter_bank  # type: ignore

    return mel_filter_bank(
        num_frequency_bins=n_fft // 2 + 1,
        num_mel_filters=n_mels,
        min_frequency=fmin,
        max_frequency=fmax,
        sampling_rate=sample_rate,
        norm=None,
        mel_scale="kaldi",
        triangularize_in_mel_space=True,
    ).astype(np.float32)  # shape (n_freq=n_fft/2+1, n_mels)


def _povey_window(n: int) -> np.ndarray:
    # SeamlessM4T uses `window_function(400, "povey", periodic=False)`
    # = np.hanning(400) ** 0.85.  The numpy hanning is symmetric (vanishes at
    # the endpoints), and the resulting Povey window matches Kaldi.
    return np.power(np.hanning(n), 0.85).astype(np.float32)


def _to_numpy(tensor):
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def _fsq_implicit_codebook(levels: List[int]) -> np.ndarray:
    levels_arr = np.asarray(levels, dtype=np.int64)
    basis = np.cumprod(np.asarray([1] + levels[:-1], dtype=np.int64))
    codebook_size = int(np.prod(levels_arr))
    indices = np.arange(codebook_size, dtype=np.int64)
    level_indices = (indices[:, None] // basis[None, :]) % levels_arr[None, :]
    half_width = levels_arr // 2
    codes = (level_indices - half_width) / half_width
    return codes.T.astype(np.float32)  # [codebook_dim, codebook_size]


def _load_state_dict_from_path(path: Path) -> Dict[str, np.ndarray]:
    if path.is_dir():
        st_path = path / "model.safetensors"
        bin_path = path / "pytorch_model.bin"
    else:
        st_path = path if path.suffix == ".safetensors" else path.with_suffix(".safetensors")
        bin_path = path

    if st_path.is_file():
        try:
            from safetensors.numpy import load_file as st_load
        except Exception:
            st_load = None
        if st_load is not None:
            return OrderedDict(st_load(str(st_path)))
        # fall through to torch path
    if torch is None:
        raise RuntimeError("torch is required for xcodec2 .bin checkpoint conversion")
    if not bin_path.is_file():
        raise FileNotFoundError(f"missing xcodec2 checkpoint at {path}")
    state = torch.load(bin_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return OrderedDict((k, _to_numpy(v)) for k, v in state.items())


class XCodec2Converter(BaseConverter):
    @property
    def model_type(self) -> str:
        return "xcodec2"

    @property
    def architecture(self) -> str:
        return "xcodec2"

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        self.state_dict = _load_state_dict_from_path(Path(checkpoint_dir))
        # The HF wrapper saves keys like:
        #   semantic_model.* / SemanticEncoder_module.*  (encoder, skipped)
        #   CodecEnc.*                                  (encoder, skipped)
        #   fc_prior.*                                  (encoder, skipped)
        #   generator.*                                 (decoder, kept)
        #   fc_post_a.*                                 (decoder, kept)
        self.config = {
            "sample_rate": 16000,
            "encode_sample_rate": 16000,
            "hop_size": 320,
            "n_fft": 1280,                 # decoder.head: n_fft = hop * 4
            "n_q": 1,
            "codebook_dim": len(FSQ_LEVELS),
            "codebook_size": int(np.prod(np.asarray(FSQ_LEVELS, dtype=np.int64))),
            "latent_dim": 1024,
            "hidden_dim": 1024,
            "vq_dim": 2048,
            "num_layers": 12,
            "num_heads": 16,
            "head_dim": 64,                # pos_meb_dim == head_dim == 64 (RoPE on full head_dim)
            "rope_theta": 10000.0,
            # encoder side
            "has_encoder": True,
            "has_decoder": True,
            "ngf": 48,
            "up_ratios": [2, 2, 4, 4, 5],     # 320× total downsample
            "enc_dilations": [1, 3, 9],
            "enc_kernel_residual": 7,         # ResidualUnit dilated conv kernel
            # w2v-bert-2.0 semantic encoder slice (16 layers, take hidden_states[16])
            "w2v_layers": 16,
            "w2v_total_layers": 24,
            "w2v_hidden": 1024,
            "w2v_heads": 16,
            "w2v_head_dim": 64,
            "w2v_intermediate": 4096,
            "w2v_left_max_pos": 64,
            "w2v_right_max_pos": 8,
            "w2v_dw_kernel": 31,
            "w2v_layer_norm_eps": 1e-5,
            "w2v_input_dim": 160,             # 80 mels × stride=2
        }

    def load_from_huggingface(self, model_id: str) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise RuntimeError("huggingface_hub is required for xcodec2 HF conversion") from exc

        try:
            model_file = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        except Exception:
            model_file = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        self.load_from_checkpoint(Path(model_file))

    def convert_and_save(self, output_path: Path) -> None:
        if self.state_dict is None or self.config is None:
            raise RuntimeError("No model loaded. Call load_from_checkpoint/load_from_huggingface first.")

        sd = self.state_dict
        cfg = self.config

        writer = GGUFWriter(output_path, self.architecture)
        self._reset_quant_stats()

        writer.add_name("XCodec2")
        writer.add_uint32("codec.sample_rate", int(cfg["sample_rate"]))
        writer.add_uint32("codec.encode_sample_rate", int(cfg["encode_sample_rate"]))
        writer.add_uint32("codec.hop_size", int(cfg["hop_size"]))
        writer.add_uint32("codec.n_fft", int(cfg["n_fft"]))
        writer.add_uint32("codec.n_q", int(cfg["n_q"]))
        writer.add_uint32("codec.codebook_size", int(cfg["codebook_size"]))
        writer.add_uint32("codec.codebook_dim", int(cfg["codebook_dim"]))
        writer.add_uint32("codec.latent_dim", int(cfg["latent_dim"]))
        writer.add_bool("codec.has_encoder", bool(cfg["has_encoder"]))
        writer.add_bool("codec.has_decoder", bool(cfg["has_decoder"]))

        writer.add_uint32("xcodec2.hidden_dim", int(cfg["hidden_dim"]))
        writer.add_uint32("xcodec2.vq_dim", int(cfg["vq_dim"]))
        writer.add_uint32("xcodec2.num_layers", int(cfg["num_layers"]))
        writer.add_uint32("xcodec2.num_heads", int(cfg["num_heads"]))
        writer.add_uint32("xcodec2.head_dim", int(cfg["head_dim"]))
        writer.add_float32("xcodec2.rope_theta", float(cfg["rope_theta"]))

        codebook = _fsq_implicit_codebook(FSQ_LEVELS).T  # [codebook_size, codebook_dim] for GGUF write
        self._add_tensor(writer, "xcodec2.decode.codebook", codebook, "F16")

        def _t(name: str) -> np.ndarray:
            arr = sd.get(name)
            if arr is None:
                raise KeyError(f"missing tensor: {name}")
            return np.asarray(arr)

        def add_linear(prefix: str, out_name: str) -> None:
            self._add_tensor(writer, out_name + ".w", _t(prefix + ".weight"))
            self._add_tensor(writer, out_name + ".b", _t(prefix + ".bias"), "F32")

        def add_conv(prefix: str, out_name: str) -> None:
            self._add_tensor(writer, out_name + ".w", _t(prefix + ".weight"))
            self._add_tensor(writer, out_name + ".b", _t(prefix + ".bias"), "F32")

        def add_norm(prefix: str, out_name: str) -> None:
            self._add_tensor(writer, out_name + ".w", _t(prefix + ".weight"), "F32")
            self._add_tensor(writer, out_name + ".b", _t(prefix + ".bias"), "F32")

        # Quantizer projection out (codebook_dim -> vq_dim) lives at the top-level
        # quantizer module in vector_quantize_pytorch's ResidualFSQ.
        add_linear("generator.quantizer.project_out", "xcodec2.decode.quant.project_out")
        add_linear("fc_post_a", "xcodec2.decode.fc_post_a")

        # Vocos backbone embed (Conv1d with kernel=7, padding=3)
        add_conv("generator.backbone.embed", "xcodec2.decode.embed")

        # Prior + post resnet stacks (each: 2 GroupNorm-Conv1d-GroupNorm-Conv1d ResnetBlock)
        for i in range(2):
            base = f"generator.backbone.prior_net.{i}"
            add_norm(base + ".norm1", f"xcodec2.decode.prior.{i}.norm1")
            add_conv(base + ".conv1", f"xcodec2.decode.prior.{i}.conv1")
            add_norm(base + ".norm2", f"xcodec2.decode.prior.{i}.norm2")
            add_conv(base + ".conv2", f"xcodec2.decode.prior.{i}.conv2")

        for i in range(2):
            base = f"generator.backbone.post_net.{i}"
            add_norm(base + ".norm1", f"xcodec2.decode.post.{i}.norm1")
            add_conv(base + ".conv1", f"xcodec2.decode.post.{i}.conv1")
            add_norm(base + ".norm2", f"xcodec2.decode.post.{i}.norm2")
            add_conv(base + ".conv2", f"xcodec2.decode.post.{i}.conv2")

        # bs_roformer transformer stack: RMSNorm pre-attn + pre-ffn,
        # combined c_attn (3*dim), c_proj, MLP fc1/fc2 (no bias).
        for i in range(int(cfg["num_layers"])):
            base = f"generator.backbone.transformers.{i}"
            self._add_tensor(writer, f"xcodec2.decode.transformer.{i}.att_norm.w",
                             _t(base + ".att_norm.weight"), "F32")
            self._add_tensor(writer, f"xcodec2.decode.transformer.{i}.ffn_norm.w",
                             _t(base + ".ffn_norm.weight"), "F32")
            self._add_tensor(writer, f"xcodec2.decode.transformer.{i}.att.c_attn.w",
                             _t(base + ".att.c_attn.weight"))
            self._add_tensor(writer, f"xcodec2.decode.transformer.{i}.att.c_proj.w",
                             _t(base + ".att.c_proj.weight"))
            self._add_tensor(writer, f"xcodec2.decode.transformer.{i}.mlp.fc1.w",
                             _t(base + ".mlp.fc1.weight"))
            self._add_tensor(writer, f"xcodec2.decode.transformer.{i}.mlp.fc2.w",
                             _t(base + ".mlp.fc2.weight"))

        add_norm("generator.backbone.final_layer_norm", "xcodec2.decode.final_ln")

        # Head: Linear(hidden, n_fft+2) + ISTFT(window) buffer.
        add_linear("generator.head.out", "xcodec2.decode.head.out")
        if "generator.head.istft.window" in sd:
            self._add_tensor(writer, "xcodec2.decode.istft.window",
                             _t("generator.head.istft.window"), "F32")

        # ===================================================================
        # Encoder side: BigCodec acoustic + Wav2Vec2-Bert semantic + fc_prior
        # + FSQ.project_in. Shapes follow upstream HKUSTAudio/xcodec2.
        # ===================================================================

        writer.add_uint32("xcodec2.enc.ngf", int(cfg["ngf"]))
        writer.add_array("xcodec2.enc.up_ratios", list(cfg["up_ratios"]))
        writer.add_array("xcodec2.enc.dilations", list(cfg["enc_dilations"]))

        writer.add_uint32("xcodec2.w2v.layers", int(cfg["w2v_layers"]))
        writer.add_uint32("xcodec2.w2v.hidden", int(cfg["w2v_hidden"]))
        writer.add_uint32("xcodec2.w2v.heads", int(cfg["w2v_heads"]))
        writer.add_uint32("xcodec2.w2v.head_dim", int(cfg["w2v_head_dim"]))
        writer.add_uint32("xcodec2.w2v.intermediate", int(cfg["w2v_intermediate"]))
        writer.add_uint32("xcodec2.w2v.left_max_pos", int(cfg["w2v_left_max_pos"]))
        writer.add_uint32("xcodec2.w2v.right_max_pos", int(cfg["w2v_right_max_pos"]))
        writer.add_uint32("xcodec2.w2v.dw_kernel", int(cfg["w2v_dw_kernel"]))
        writer.add_float32("xcodec2.w2v.layer_norm_eps", float(cfg["w2v_layer_norm_eps"]))
        writer.add_uint32("xcodec2.w2v.input_dim", int(cfg["w2v_input_dim"]))

        # Mel-fbank metadata for the host-side feature extractor.
        writer.add_uint32("codec.mel.sample_rate", int(cfg["sample_rate"]))
        writer.add_uint32("codec.mel.n_mels", int(W2V_BERT_N_MELS))
        writer.add_uint32("codec.mel.n_fft", int(W2V_BERT_N_FFT))
        writer.add_uint32("codec.mel.win_length", int(W2V_BERT_WIN))
        writer.add_uint32("codec.mel.hop_length", int(W2V_BERT_HOP))
        writer.add_float32("codec.mel.preemphasis", float(W2V_BERT_PREEMPHASIS))
        writer.add_float32("codec.mel.mel_floor", float(W2V_BERT_MEL_FLOOR))
        writer.add_uint32("codec.mel.stride", int(W2V_BERT_STRIDE))
        writer.add_bool("codec.mel.remove_dc_offset", True)
        writer.add_bool("codec.mel.normalize_per_mel_bins", True)

        # Mel filters (n_mels=80, n_freq=257) and periodic-Hann window (400)
        # — recompute deterministically so we don't need transformers at runtime.
        mel_filters = _seamless_mel_filterbank()  # (257, 80)
        self._add_tensor(writer, "xcodec2.enc.mel.filters", mel_filters, "F32")
        self._add_tensor(writer, "xcodec2.enc.mel.window", _povey_window(W2V_BERT_WIN), "F32")

        # ----- helpers for weight_norm + snake-beta baking ----------------
        def add_weight_norm_conv(prefix: str, out_name: str, dim: int = 0) -> None:
            wv = _t(prefix + ".weight_v")
            wg = _t(prefix + ".weight_g")
            w = _apply_weight_norm(wv, wg, dim=dim)
            self._add_tensor(writer, out_name + ".w", w)
            if prefix + ".bias" in sd:
                b = _t(prefix + ".bias")
                self._add_tensor(writer, out_name + ".b", b, "F32")

        def add_snake_beta(prefix: str, out_name: str) -> None:
            # SnakeBeta with alpha_logscale=True. The shipped checkpoint stores
            # `act.beta`, but the upstream class actually expects `act.bias`
            # (key was renamed). HF's `load_state_dict(strict=False)` silently
            # drops the unmatched `act.beta` entry and leaves `bias` at its
            # default (zeros), so the *effective* runtime SnakeBeta is
            #   y = x + (1/(exp(0) + 1e-9)) * sin(exp(alpha)*x)^2
            #     ≈ x + sin(exp(alpha)*x)^2.
            # Bake `inv_beta = 1/(1 + 1e-9)` to match the HF reference exactly
            # (don't use the trained beta — HF doesn't load it).
            alpha = np.exp(_t(prefix + ".alpha"))
            inv_beta = np.full_like(alpha, 1.0 / (1.0 + 1e-9))
            self._add_tensor(writer, out_name + ".alpha", alpha.astype(np.float32), "F32")
            self._add_tensor(writer, out_name + ".inv_beta", inv_beta.astype(np.float32), "F32")

        # Activation1d up/down FIR filters are bit-identical across all 36
        # instances in BigCodec — store a single shared kernel.
        shared_filter = _t("CodecEnc.conv_blocks.1.block.0.block.0.upsample.filter").reshape(-1).astype(np.float32)
        # sanity-check that all stored kernels really are the same (and that
        # up == down, which they are because the kernel is symmetric).
        for k, v in sd.items():
            if "upsample.filter" in k or "downsample.lowpass.filter" in k:
                if not np.allclose(np.asarray(v).reshape(-1), shared_filter, atol=1e-7):
                    raise RuntimeError(f"alias-free FIR mismatch at {k}")
        self._add_tensor(writer, "xcodec2.enc.alias.filter", shared_filter, "F32")

        def add_act1d(prefix: str, out_name: str) -> None:
            add_snake_beta(prefix + ".act", out_name)

        # ----- Acoustic encoder (BigCodec) --------------------------------
        # initial conv
        add_weight_norm_conv("CodecEnc.conv_blocks.0", "xcodec2.enc.codec.conv0")

        up_ratios: List[int] = list(cfg["up_ratios"])
        dilations: List[int] = list(cfg["enc_dilations"])
        d_model = int(cfg["ngf"])
        for bi, stride in enumerate(up_ratios, start=1):
            d_model *= 2
            base = f"CodecEnc.conv_blocks.{bi}.block"
            for ri in range(len(dilations)):
                # ResidualUnit: SnakeBeta + Activation1d + WNConv k=7 dilated +
                #               SnakeBeta + Activation1d + WNConv k=1
                rbase = f"{base}.{ri}.block"
                act1_pref = f"{rbase}.0"
                act2_pref = f"{rbase}.2"
                add_act1d(act1_pref, f"xcodec2.enc.codec.b{bi}.r{ri}.act1")
                add_weight_norm_conv(f"{rbase}.1", f"xcodec2.enc.codec.b{bi}.r{ri}.conv1")
                add_act1d(act2_pref, f"xcodec2.enc.codec.b{bi}.r{ri}.act2")
                add_weight_norm_conv(f"{rbase}.3", f"xcodec2.enc.codec.b{bi}.r{ri}.conv2")
            # final activation + downsample conv per block
            add_act1d(f"{base}.3", f"xcodec2.enc.codec.b{bi}.act")
            add_weight_norm_conv(f"{base}.4", f"xcodec2.enc.codec.b{bi}.down")

        # final SnakeBeta + 1024-projection conv
        add_act1d("CodecEnc.conv_final_block.0", "xcodec2.enc.codec.final.act")
        add_weight_norm_conv("CodecEnc.conv_final_block.1", "xcodec2.enc.codec.final.conv")

        # ----- Wav2Vec2-Bert feature_projection + 16 conformer layers -----
        # Layer norm operates on the pre-projection 160-d feature.
        self._add_tensor(writer, "xcodec2.w2v.feat_ln.w",
                         _t("semantic_model.feature_projection.layer_norm.weight"), "F32")
        self._add_tensor(writer, "xcodec2.w2v.feat_ln.b",
                         _t("semantic_model.feature_projection.layer_norm.bias"), "F32")
        self._add_tensor(writer, "xcodec2.w2v.feat_proj.w",
                         _t("semantic_model.feature_projection.projection.weight"))
        self._add_tensor(writer, "xcodec2.w2v.feat_proj.b",
                         _t("semantic_model.feature_projection.projection.bias"), "F32")

        for li in range(int(cfg["w2v_layers"])):
            base = f"semantic_model.encoder.layers.{li}"
            o = f"xcodec2.w2v.l{li}"
            # ffn1
            self._add_tensor(writer, f"{o}.ffn1_ln.w", _t(f"{base}.ffn1_layer_norm.weight"), "F32")
            self._add_tensor(writer, f"{o}.ffn1_ln.b", _t(f"{base}.ffn1_layer_norm.bias"), "F32")
            self._add_tensor(writer, f"{o}.ffn1.fc1.w", _t(f"{base}.ffn1.intermediate_dense.weight"))
            self._add_tensor(writer, f"{o}.ffn1.fc1.b", _t(f"{base}.ffn1.intermediate_dense.bias"), "F32")
            self._add_tensor(writer, f"{o}.ffn1.fc2.w", _t(f"{base}.ffn1.output_dense.weight"))
            self._add_tensor(writer, f"{o}.ffn1.fc2.b", _t(f"{base}.ffn1.output_dense.bias"), "F32")
            # self-attn (relative-key Shaw)
            self._add_tensor(writer, f"{o}.attn_ln.w", _t(f"{base}.self_attn_layer_norm.weight"), "F32")
            self._add_tensor(writer, f"{o}.attn_ln.b", _t(f"{base}.self_attn_layer_norm.bias"), "F32")
            self._add_tensor(writer, f"{o}.attn.q.w", _t(f"{base}.self_attn.linear_q.weight"))
            self._add_tensor(writer, f"{o}.attn.q.b", _t(f"{base}.self_attn.linear_q.bias"), "F32")
            self._add_tensor(writer, f"{o}.attn.k.w", _t(f"{base}.self_attn.linear_k.weight"))
            self._add_tensor(writer, f"{o}.attn.k.b", _t(f"{base}.self_attn.linear_k.bias"), "F32")
            self._add_tensor(writer, f"{o}.attn.v.w", _t(f"{base}.self_attn.linear_v.weight"))
            self._add_tensor(writer, f"{o}.attn.v.b", _t(f"{base}.self_attn.linear_v.bias"), "F32")
            self._add_tensor(writer, f"{o}.attn.o.w", _t(f"{base}.self_attn.linear_out.weight"))
            self._add_tensor(writer, f"{o}.attn.o.b", _t(f"{base}.self_attn.linear_out.bias"), "F32")
            # distance embedding (n_buckets, head_dim) — F32 to keep table small.
            self._add_tensor(writer, f"{o}.attn.dist.w", _t(f"{base}.self_attn.distance_embedding.weight"), "F32")
            # conv module
            self._add_tensor(writer, f"{o}.conv.ln.w", _t(f"{base}.conv_module.layer_norm.weight"), "F32")
            self._add_tensor(writer, f"{o}.conv.ln.b", _t(f"{base}.conv_module.layer_norm.bias"), "F32")
            self._add_tensor(writer, f"{o}.conv.pw1.w", _t(f"{base}.conv_module.pointwise_conv1.weight"))
            self._add_tensor(writer, f"{o}.conv.dw.w", _t(f"{base}.conv_module.depthwise_conv.weight"))
            self._add_tensor(writer, f"{o}.conv.dw_ln.w", _t(f"{base}.conv_module.depthwise_layer_norm.weight"), "F32")
            self._add_tensor(writer, f"{o}.conv.dw_ln.b", _t(f"{base}.conv_module.depthwise_layer_norm.bias"), "F32")
            self._add_tensor(writer, f"{o}.conv.pw2.w", _t(f"{base}.conv_module.pointwise_conv2.weight"))
            # ffn2 + final ln
            self._add_tensor(writer, f"{o}.ffn2_ln.w", _t(f"{base}.ffn2_layer_norm.weight"), "F32")
            self._add_tensor(writer, f"{o}.ffn2_ln.b", _t(f"{base}.ffn2_layer_norm.bias"), "F32")
            self._add_tensor(writer, f"{o}.ffn2.fc1.w", _t(f"{base}.ffn2.intermediate_dense.weight"))
            self._add_tensor(writer, f"{o}.ffn2.fc1.b", _t(f"{base}.ffn2.intermediate_dense.bias"), "F32")
            self._add_tensor(writer, f"{o}.ffn2.fc2.w", _t(f"{base}.ffn2.output_dense.weight"))
            self._add_tensor(writer, f"{o}.ffn2.fc2.b", _t(f"{base}.ffn2.output_dense.bias"), "F32")
            self._add_tensor(writer, f"{o}.final_ln.w", _t(f"{base}.final_layer_norm.weight"), "F32")
            self._add_tensor(writer, f"{o}.final_ln.b", _t(f"{base}.final_layer_norm.bias"), "F32")

        # ----- SemanticEncoder (3 conv stack with ReLU) ------------------
        # initial_conv (no bias) → ReLU → conv (with bias) → ReLU → conv (with bias) → +residual → final_conv (no bias)
        self._add_tensor(writer, "xcodec2.sem.initial.w", _t("SemanticEncoder_module.initial_conv.weight"))
        self._add_tensor(writer, "xcodec2.sem.r1.w", _t("SemanticEncoder_module.residual_blocks.1.weight"))
        self._add_tensor(writer, "xcodec2.sem.r1.b", _t("SemanticEncoder_module.residual_blocks.1.bias"), "F32")
        self._add_tensor(writer, "xcodec2.sem.r3.w", _t("SemanticEncoder_module.residual_blocks.3.weight"))
        self._add_tensor(writer, "xcodec2.sem.r3.b", _t("SemanticEncoder_module.residual_blocks.3.bias"), "F32")
        self._add_tensor(writer, "xcodec2.sem.final.w", _t("SemanticEncoder_module.final_conv.weight"))

        # ----- fc_prior + FSQ.project_in --------------------------------
        add_linear("fc_prior", "xcodec2.enc.fc_prior")
        add_linear("generator.quantizer.project_in", "xcodec2.enc.quant.project_in")

        self._warn_if_no_quantized()
        writer.write()
        self.log(f"Wrote XCodec2 GGUF to {output_path}")
