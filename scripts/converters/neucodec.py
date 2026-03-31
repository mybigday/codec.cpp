"""Neuphonic NeuCodec encoder/decoder converter."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .base import BaseConverter
from utils.gguf_writer import GGUFWriter


FSQ_LEVELS: List[int] = [4, 4, 4, 4, 4, 4, 4, 4]

# SeamlessM4T feature extractor defaults (Wav2Vec2-BERT)
W2V_BERT_FBANK_N_FFT = 512
W2V_BERT_FBANK_WIN_LENGTH = 400
W2V_BERT_FBANK_HOP_LENGTH = 160
W2V_BERT_FBANK_PREEMPHASIS = 0.97
W2V_BERT_FBANK_MEL_FLOOR = 1.192092955078125e-07

def _enc_name(name: str) -> str:
    if not name.startswith("neucodec.encode."):
        return name
    h = 1469598103934665603
    for b in name.encode("utf-8"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"nce.{h:016x}"


def _to_numpy(tensor):
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def _apply_weight_norm(weight_v: np.ndarray, weight_g: np.ndarray, dim: int = 0) -> np.ndarray:
    if weight_v.ndim < 2:
        raise ValueError(f"weight_norm expects ndim >= 2, got {weight_v.ndim}")
    if dim < 0:
        dim = weight_v.ndim + dim
    if dim < 0 or dim >= weight_v.ndim:
        raise ValueError(f"invalid weight_norm dim {dim} for shape {weight_v.shape}")

    axes = tuple(i for i in range(weight_v.ndim) if i != dim)
    norm = np.linalg.norm(weight_v, axis=axes, keepdims=True)
    if weight_g.shape != norm.shape:
        weight_g = weight_g.reshape(norm.shape)
    return weight_v * (weight_g / (norm + 1e-12))


def _weight_norm_from_parametrization(weight_g: np.ndarray, weight_v: np.ndarray, dim: int = 0) -> np.ndarray:
    # torch.nn.utils.parametrizations.weight_norm stores g/v as original0/original1
    return _apply_weight_norm(weight_v, weight_g, dim=dim)


def _is_distill_state_dict(state_dict: Dict[str, np.ndarray]) -> bool:
    return any(k.startswith("codec_encoder.") for k in state_dict.keys())


def _load_hubert_state(local_path: str | None = None):
    try:
        from transformers import HubertModel
    except Exception as exc:
        raise RuntimeError("transformers is required to load Hubert weights for distill-neucodec") from exc

    try:
        model = HubertModel.from_pretrained("ntu-spml/distilhubert", local_files_only=True)
        return model.state_dict(), model.config
    except Exception:
        pass

    if local_path is None:
        snapshots = sorted(
            Path("models/hf/models--ntu-spml--distilhubert/snapshots").glob("*"),
            key=lambda p: p.name,
        )
        if snapshots:
            local_path = str(snapshots[-1])
    if local_path is not None:
        model = HubertModel.from_pretrained(local_path, local_files_only=True)
    else:
        model = HubertModel.from_pretrained("ntu-spml/distilhubert")
    return model.state_dict(), model.config


def _fsq_implicit_codebook(levels: List[int]) -> np.ndarray:
    levels_arr = np.asarray(levels, dtype=np.int64)
    basis = np.cumprod(np.asarray([1] + levels[:-1], dtype=np.int64))
    codebook_size = int(np.prod(levels_arr))
    indices = np.arange(codebook_size, dtype=np.int64)
    level_indices = (indices[:, None] // basis[None, :]) % levels_arr[None, :]
    half_width = levels_arr // 2
    codes = (level_indices - half_width) / half_width
    # [codebook_dim, codebook_size]
    return codes.T.astype(np.float32)


class NeuCodecConverter(BaseConverter):
    @property
    def model_type(self) -> str:
        return "neucodec"

    @property
    def architecture(self) -> str:
        if self.config is not None and self.config.get("encoder_type") == "distill":
            return "distill_neucodec"
        return "neucodec"

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        if torch is None:
            raise RuntimeError("torch is required for NeuCodec checkpoint conversion")

        checkpoint_dir = Path(checkpoint_dir)
        if checkpoint_dir.is_dir():
            model_file = checkpoint_dir / "pytorch_model.bin"
        else:
            model_file = checkpoint_dir

        if not model_file.is_file():
            raise FileNotFoundError(f"missing NeuCodec checkpoint: {model_file}")

        state = torch.load(model_file, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported checkpoint format at {model_file}")

        self.state_dict = OrderedDict((k, _to_numpy(v)) for k, v in state.items())

        is_distill = _is_distill_state_dict(self.state_dict)
        encoder_type = "distill" if is_distill else "base"

        self.config = {
            "sample_rate": 24000,
            "hop_size": 480,
            "n_fft": 1920,
            "n_q": 1,
            "codebook_dim": len(FSQ_LEVELS),
            "codebook_size": int(np.prod(np.asarray(FSQ_LEVELS, dtype=np.int64))),
            "latent_dim": 1024,
            "hidden_dim": 1024,
            "vq_dim": 2048,
            "num_layers": 12,
            "num_heads": 16,
            "head_dim": 64,
            "rope_theta": 10000.0,
            "has_encoder": True,
            "has_decoder": True,
            "encode_sample_rate": 16000,
            "encoder_type": encoder_type,
        }

    def load_from_huggingface(self, model_id: str) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise RuntimeError("huggingface_hub is required for NeuCodec HF conversion") from exc

        model_file = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        self.load_from_checkpoint(Path(model_file))

    def convert_and_save(self, output_path: Path) -> None:
        if self.state_dict is None or self.config is None:
            raise RuntimeError("No model loaded. Call load_from_checkpoint/load_from_huggingface first.")

        writer = GGUFWriter(output_path, self.architecture)
        self._reset_quant_stats()

        writer.add_name("NeuCodec")
        writer.add_uint32("codec.sample_rate", int(self.config["sample_rate"]))
        writer.add_uint32("codec.encode_sample_rate", int(self.config["encode_sample_rate"]))
        writer.add_uint32("codec.hop_size", int(self.config["hop_size"]))
        writer.add_uint32("codec.n_fft", int(self.config["n_fft"]))
        writer.add_uint32("codec.n_q", int(self.config["n_q"]))
        writer.add_uint32("codec.codebook_size", int(self.config["codebook_size"]))
        writer.add_uint32("codec.codebook_dim", int(self.config["codebook_dim"]))
        writer.add_uint32("codec.latent_dim", int(self.config["latent_dim"]))
        writer.add_bool("codec.has_encoder", bool(self.config["has_encoder"]))
        writer.add_bool("codec.has_decoder", bool(self.config["has_decoder"]))

        writer.add_uint32("neucodec.hidden_dim", int(self.config["hidden_dim"]))
        writer.add_uint32("neucodec.vq_dim", int(self.config["vq_dim"]))
        writer.add_uint32("neucodec.num_layers", int(self.config["num_layers"]))
        writer.add_uint32("neucodec.num_heads", int(self.config["num_heads"]))
        writer.add_uint32("neucodec.head_dim", int(self.config["head_dim"]))
        writer.add_float32("neucodec.rope_theta", float(self.config["rope_theta"]))
        writer.add_string("neucodec.encoder_type", str(self.config["encoder_type"]))

        # GGUF reverses shapes on write, so store [codebook_size, codebook_dim]
        # to get ggml tensor layout [codebook_dim, codebook_size].
        codebook = _fsq_implicit_codebook(FSQ_LEVELS).T
        self._add_tensor(writer, "neucodec.decode.codebook", codebook, "F16")

        def conv_weight(prefix: str) -> np.ndarray:
            wv = self.state_dict.get(prefix + ".weight_v")
            wg = self.state_dict.get(prefix + ".weight_g")
            if wv is not None and wg is not None:
                return _apply_weight_norm(np.asarray(wv), np.asarray(wg))
            w = self.state_dict.get(prefix + ".weight")
            if w is None:
                raise KeyError(f"missing conv weight for {prefix}")
            return np.asarray(w)

        def add_linear(prefix: str, out_name: str) -> None:
            w = np.asarray(self.state_dict[prefix + ".weight"])
            self._add_tensor(writer, _enc_name(out_name + ".w"), w)
            if prefix + ".bias" in self.state_dict:
                b = np.asarray(self.state_dict[prefix + ".bias"])
                self._add_tensor(writer, _enc_name(out_name + ".b"), b, "F32")

        def add_conv(prefix: str, out_name: str) -> None:
            w = conv_weight(prefix)
            self._add_tensor(writer, out_name + ".w", w)
            b = np.asarray(self.state_dict[prefix + ".bias"])
            self._add_tensor(writer, out_name + ".b", b, "F32")

        def add_norm(prefix: str, out_name: str) -> None:
            w = np.asarray(self.state_dict[prefix + ".weight"])
            b = np.asarray(self.state_dict[prefix + ".bias"])
            self._add_tensor(writer, out_name + ".w", w, "F32")
            self._add_tensor(writer, out_name + ".b", b, "F32")

        def add_weight_norm(prefix: str, out_name: str) -> None:
            wv = self.state_dict.get(prefix + ".weight_v")
            wg = self.state_dict.get(prefix + ".weight_g")
            if wv is None or wg is None:
                raise KeyError(f"missing weight_norm params for {prefix}")
            w = _apply_weight_norm(np.asarray(wv), np.asarray(wg))
            self._add_tensor(writer, _enc_name(out_name + ".w"), w)
            if prefix + ".bias" in self.state_dict:
                b = np.asarray(self.state_dict[prefix + ".bias"])
                self._add_tensor(writer, _enc_name(out_name + ".b"), b, "F32")

        def add_weight_norm_param(prefix: str, out_name: str) -> None:
            g = self.state_dict.get(prefix + ".parametrizations.weight.original0")
            v = self.state_dict.get(prefix + ".parametrizations.weight.original1")
            if g is None or v is None:
                raise KeyError(f"missing parametrized weight_norm params for {prefix}")
            w = _weight_norm_from_parametrization(np.asarray(g), np.asarray(v))
            self._add_tensor(writer, _enc_name(out_name + ".w"), w)
            if prefix + ".bias" in self.state_dict:
                b = np.asarray(self.state_dict[prefix + ".bias"])
                self._add_tensor(writer, _enc_name(out_name + ".b"), b, "F32")

        def add_enc_tensor(name: str, arr: np.ndarray, qtype: str | None = None) -> None:
            self._add_tensor(writer, _enc_name(name), arr, qtype)

        # quantizer + post
        add_linear("generator.quantizer.project_out", "neucodec.decode.quant.project_out")
        add_linear("fc_post_a", "neucodec.decode.fc_post_a")

        # backbone embed
        add_conv("generator.backbone.embed", "neucodec.decode.embed")

        # prior / post resnet blocks
        for i in range(2):
            base = f"generator.backbone.prior_net.{i}"
            add_norm(base + ".norm1", f"neucodec.decode.prior.{i}.norm1")
            add_conv(base + ".conv1", f"neucodec.decode.prior.{i}.conv1")
            add_norm(base + ".norm2", f"neucodec.decode.prior.{i}.norm2")
            add_conv(base + ".conv2", f"neucodec.decode.prior.{i}.conv2")

        for i in range(2):
            base = f"generator.backbone.post_net.{i}"
            add_norm(base + ".norm1", f"neucodec.decode.post.{i}.norm1")
            add_conv(base + ".conv1", f"neucodec.decode.post.{i}.conv1")
            add_norm(base + ".norm2", f"neucodec.decode.post.{i}.norm2")
            add_conv(base + ".conv2", f"neucodec.decode.post.{i}.conv2")

        # transformers
        for i in range(self.config["num_layers"]):
            base = f"generator.backbone.transformers.{i}"
            w_att = np.asarray(self.state_dict[base + ".att_norm.weight"])
            w_ffn = np.asarray(self.state_dict[base + ".ffn_norm.weight"])
            self._add_tensor(writer, f"neucodec.decode.transformer.{i}.att_norm.w", w_att, "F32")
            self._add_tensor(writer, f"neucodec.decode.transformer.{i}.ffn_norm.w", w_ffn, "F32")

            self._add_tensor(writer, f"neucodec.decode.transformer.{i}.att.c_attn.w",
                             np.asarray(self.state_dict[base + ".att.c_attn.weight"]))
            self._add_tensor(writer, f"neucodec.decode.transformer.{i}.att.c_proj.w",
                             np.asarray(self.state_dict[base + ".att.c_proj.weight"]))
            self._add_tensor(writer, f"neucodec.decode.transformer.{i}.mlp.fc1.w",
                             np.asarray(self.state_dict[base + ".mlp.fc1.weight"]))
            self._add_tensor(writer, f"neucodec.decode.transformer.{i}.mlp.fc2.w",
                             np.asarray(self.state_dict[base + ".mlp.fc2.weight"]))

        # final layer norm
        add_norm("generator.backbone.final_layer_norm", "neucodec.decode.final_ln")

        # head + window
        add_linear("generator.head.out", "neucodec.decode.head.out")
        window = np.asarray(self.state_dict["generator.head.istft.window"])
        self._add_tensor(writer, "neucodec.decode.istft.window", window, "F32")

        # -----------------------
        # Encoder weights + metadata
        # -----------------------
        # Shared encoder weights
        add_linear("generator.quantizer.project_in", "neucodec.encode.quant.project_in")
        add_linear("fc_prior", "neucodec.encode.fc_prior")
        if "fc_sq_prior.weight" in self.state_dict:
            add_linear("fc_sq_prior", "neucodec.encode.fc_sq_prior")

        # Semantic encoder (Conv1d + ReLU stack)
        if "SemanticEncoder_module.initial_conv.weight" in self.state_dict:
            add_enc_tensor(
                "neucodec.encode.semantic_encoder.initial_conv.w",
                np.asarray(self.state_dict["SemanticEncoder_module.initial_conv.weight"]),
            )
            add_enc_tensor(
                "neucodec.encode.semantic_encoder.residual.1.w",
                np.asarray(self.state_dict["SemanticEncoder_module.residual_blocks.1.weight"]),
            )
            add_enc_tensor(
                "neucodec.encode.semantic_encoder.residual.1.b",
                np.asarray(self.state_dict["SemanticEncoder_module.residual_blocks.1.bias"]),
                "F32",
            )
            add_enc_tensor(
                "neucodec.encode.semantic_encoder.residual.3.w",
                np.asarray(self.state_dict["SemanticEncoder_module.residual_blocks.3.weight"]),
            )
            add_enc_tensor(
                "neucodec.encode.semantic_encoder.residual.3.b",
                np.asarray(self.state_dict["SemanticEncoder_module.residual_blocks.3.bias"]),
                "F32",
            )
            add_enc_tensor(
                "neucodec.encode.semantic_encoder.final_conv.w",
                np.asarray(self.state_dict["SemanticEncoder_module.final_conv.weight"]),
            )

        # Acoustic encoder (base) - weight_norm convs + snake beta
        if self.config["encoder_type"] == "base":
            # Initial conv
            add_weight_norm("CodecEnc.conv_blocks.0", "neucodec.encode.acoustic.conv0")

            # Encoder blocks (5)
            for bi in range(1, 6):
                base = f"CodecEnc.conv_blocks.{bi}.block"
                # Residual units (3)
                for ri in range(3):
                    rbase = f"{base}.{ri}.block"
                    # Activation 1
                    add_enc_tensor(f"neucodec.encode.acoustic.b{bi}.r{ri}.act1.alpha",
                                     np.asarray(self.state_dict[f"{rbase}.0.act.alpha"]))
                    add_enc_tensor(f"neucodec.encode.acoustic.b{bi}.r{ri}.act1.beta",
                                     np.asarray(self.state_dict[f"{rbase}.0.act.beta"]))
                    add_enc_tensor(f"neucodec.encode.acoustic.b{bi}.r{ri}.act1.up.filter",
                                     np.asarray(self.state_dict[f"{rbase}.0.upsample.filter"]))
                    add_enc_tensor(f"neucodec.encode.acoustic.b{bi}.r{ri}.act1.down.filter",
                                     np.asarray(self.state_dict[f"{rbase}.0.downsample.lowpass.filter"]))
                    add_weight_norm(f"{rbase}.1", f"neucodec.encode.acoustic.b{bi}.r{ri}.conv1")

                    # Activation 2
                    add_enc_tensor(f"neucodec.encode.acoustic.b{bi}.r{ri}.act2.alpha",
                                     np.asarray(self.state_dict[f"{rbase}.2.act.alpha"]))
                    add_enc_tensor(f"neucodec.encode.acoustic.b{bi}.r{ri}.act2.beta",
                                     np.asarray(self.state_dict[f"{rbase}.2.act.beta"]))
                    add_enc_tensor(f"neucodec.encode.acoustic.b{bi}.r{ri}.act2.up.filter",
                                     np.asarray(self.state_dict[f"{rbase}.2.upsample.filter"]))
                    add_enc_tensor(f"neucodec.encode.acoustic.b{bi}.r{ri}.act2.down.filter",
                                     np.asarray(self.state_dict[f"{rbase}.2.downsample.lowpass.filter"]))
                    add_weight_norm(f"{rbase}.3", f"neucodec.encode.acoustic.b{bi}.r{ri}.conv2")

                # Final activation + downsample conv
                add_enc_tensor(f"neucodec.encode.acoustic.b{bi}.act.alpha",
                                 np.asarray(self.state_dict[f"{base}.3.act.alpha"]))
                add_enc_tensor(f"neucodec.encode.acoustic.b{bi}.act.beta",
                                 np.asarray(self.state_dict[f"{base}.3.act.beta"]))
                add_enc_tensor(f"neucodec.encode.acoustic.b{bi}.act.up.filter",
                                 np.asarray(self.state_dict[f"{base}.3.upsample.filter"]))
                add_enc_tensor(f"neucodec.encode.acoustic.b{bi}.act.down.filter",
                                 np.asarray(self.state_dict[f"{base}.3.downsample.lowpass.filter"]))
                add_weight_norm(f"{base}.4", f"neucodec.encode.acoustic.b{bi}.down")

            # Final block
            add_enc_tensor("neucodec.encode.acoustic.final.act.alpha",
                             np.asarray(self.state_dict["CodecEnc.conv_final_block.0.act.alpha"]))
            add_enc_tensor("neucodec.encode.acoustic.final.act.beta",
                             np.asarray(self.state_dict["CodecEnc.conv_final_block.0.act.beta"]))
            add_enc_tensor("neucodec.encode.acoustic.final.act.up.filter",
                             np.asarray(self.state_dict["CodecEnc.conv_final_block.0.upsample.filter"]))
            add_enc_tensor("neucodec.encode.acoustic.final.act.down.filter",
                             np.asarray(self.state_dict["CodecEnc.conv_final_block.0.downsample.lowpass.filter"]))
            add_weight_norm("CodecEnc.conv_final_block.1", "neucodec.encode.acoustic.final.conv")

        # Acoustic encoder (distill) - preserve all distill encoder weights
        if self.config["encoder_type"] == "distill":
            # collect weight_norm-parametrized layers
            handled_distill_keys = set()
            for k in list(self.state_dict.keys()):
                if k.endswith(".parametrizations.weight.original0"):
                    base = k[: -len(".parametrizations.weight.original0")]
                    g = np.asarray(self.state_dict[base + ".parametrizations.weight.original0"])
                    v = np.asarray(self.state_dict[base + ".parametrizations.weight.original1"])
                    w = _weight_norm_from_parametrization(g, v)
                    add_enc_tensor("neucodec.encode.distill." + base + ".weight", w)
                    handled_distill_keys.add(base + ".weight")
                    if base + ".bias" in self.state_dict:
                        add_enc_tensor(
                            "neucodec.encode.distill." + base + ".bias",
                            np.asarray(self.state_dict[base + ".bias"]),
                            "F32",
                        )
                        handled_distill_keys.add(base + ".bias")

            # add remaining distill weights (skip parametrizations handled above)
            skip_suffixes = (
                ".parametrizations.weight.original0",
                ".parametrizations.weight.original1",
            )
            for k, v in self.state_dict.items():
                if not k.startswith("codec_encoder.") and not k.startswith("SemanticEncoder_module.") \
                   and not k.startswith("fc_prior.") and not k.startswith("fc_sq_prior.") \
                   and not k.startswith("generator.quantizer.project_in."):
                    continue
                if k in handled_distill_keys:
                    continue
                if k.endswith(skip_suffixes):
                    continue
                if k.endswith(".weight") or k.endswith(".bias") or k.endswith(".alpha") or k.endswith(".beta") or k.endswith(".gamma"):
                    name = "neucodec.encode.distill." + k
                    add_enc_tensor(name, np.asarray(v), "F32" if k.endswith(".bias") or k.endswith(".gamma") else None)

            # Hubert semantic model weights for distill-neucodec
            hubert_state, hubert_cfg = _load_hubert_state()
            writer.add_uint32("neucodec.hubert.hidden_size", int(hubert_cfg.hidden_size))
            writer.add_uint32("neucodec.hubert.num_heads", int(hubert_cfg.num_attention_heads))
            writer.add_uint32("neucodec.hubert.intermediate_size", int(hubert_cfg.intermediate_size))
            writer.add_uint32("neucodec.hubert.num_layers", int(hubert_cfg.num_hidden_layers))
            writer.add_uint32("neucodec.hubert.num_conv_pos_embeddings", int(hubert_cfg.num_conv_pos_embeddings))
            writer.add_uint32("neucodec.hubert.num_conv_pos_embedding_groups", int(hubert_cfg.num_conv_pos_embedding_groups))
            writer.add_float32("neucodec.hubert.layer_norm_eps", float(hubert_cfg.layer_norm_eps))
            writer.add_array("neucodec.hubert.conv_dim", list(hubert_cfg.conv_dim))
            writer.add_array("neucodec.hubert.conv_kernel", list(hubert_cfg.conv_kernel))
            writer.add_array("neucodec.hubert.conv_stride", list(hubert_cfg.conv_stride))

            # feature extractor convs
            for i in range(hubert_cfg.num_feat_extract_layers):
                w = np.asarray(hubert_state[f"feature_extractor.conv_layers.{i}.conv.weight"])
                add_enc_tensor(f"neucodec.encode.hubert.feat.conv.{i}.w", w)
            # group norm on first layer
            gn_w = np.asarray(hubert_state["feature_extractor.conv_layers.0.layer_norm.weight"])
            gn_b = np.asarray(hubert_state["feature_extractor.conv_layers.0.layer_norm.bias"])
            add_enc_tensor("neucodec.encode.hubert.feat.conv.0.gn.w", gn_w, "F32")
            add_enc_tensor("neucodec.encode.hubert.feat.conv.0.gn.b", gn_b, "F32")

            # feature projection
            add_enc_tensor("neucodec.encode.hubert.feature_projection.w",
                             np.asarray(hubert_state["feature_projection.projection.weight"]))
            add_enc_tensor("neucodec.encode.hubert.feature_projection.b",
                             np.asarray(hubert_state["feature_projection.projection.bias"]), "F32")

            # positional conv embedding (weight norm)
            pos_w = _weight_norm_from_parametrization(
                np.asarray(hubert_state["encoder.pos_conv_embed.conv.parametrizations.weight.original0"]),
                np.asarray(hubert_state["encoder.pos_conv_embed.conv.parametrizations.weight.original1"]),
                dim=2,
            )
            add_enc_tensor("neucodec.encode.hubert.encoder.pos_conv.w", pos_w)
            add_enc_tensor("neucodec.encode.hubert.encoder.pos_conv.b",
                             np.asarray(hubert_state["encoder.pos_conv_embed.conv.bias"]), "F32")

            # encoder layer norm
            add_enc_tensor("neucodec.encode.hubert.encoder.layer_norm.w",
                             np.asarray(hubert_state["encoder.layer_norm.weight"]), "F32")
            add_enc_tensor("neucodec.encode.hubert.encoder.layer_norm.b",
                             np.asarray(hubert_state["encoder.layer_norm.bias"]), "F32")

            # encoder layers
            for i in range(hubert_cfg.num_hidden_layers):
                base = f"encoder.layers.{i}"
                add_enc_tensor(f"neucodec.encode.hubert.encoder.layers.{i}.att.q.w",
                                 np.asarray(hubert_state[f"{base}.attention.q_proj.weight"]))
                add_enc_tensor(f"neucodec.encode.hubert.encoder.layers.{i}.att.q.b",
                                 np.asarray(hubert_state[f"{base}.attention.q_proj.bias"]), "F32")
                add_enc_tensor(f"neucodec.encode.hubert.encoder.layers.{i}.att.k.w",
                                 np.asarray(hubert_state[f"{base}.attention.k_proj.weight"]))
                add_enc_tensor(f"neucodec.encode.hubert.encoder.layers.{i}.att.k.b",
                                 np.asarray(hubert_state[f"{base}.attention.k_proj.bias"]), "F32")
                add_enc_tensor(f"neucodec.encode.hubert.encoder.layers.{i}.att.v.w",
                                 np.asarray(hubert_state[f"{base}.attention.v_proj.weight"]))
                add_enc_tensor(f"neucodec.encode.hubert.encoder.layers.{i}.att.v.b",
                                 np.asarray(hubert_state[f"{base}.attention.v_proj.bias"]), "F32")
                add_enc_tensor(f"neucodec.encode.hubert.encoder.layers.{i}.att.o.w",
                                 np.asarray(hubert_state[f"{base}.attention.out_proj.weight"]))
                add_enc_tensor(f"neucodec.encode.hubert.encoder.layers.{i}.att.o.b",
                                 np.asarray(hubert_state[f"{base}.attention.out_proj.bias"]), "F32")

                add_enc_tensor(f"neucodec.encode.hubert.encoder.layers.{i}.ln.w",
                                 np.asarray(hubert_state[f"{base}.layer_norm.weight"]), "F32")
                add_enc_tensor(f"neucodec.encode.hubert.encoder.layers.{i}.ln.b",
                                 np.asarray(hubert_state[f"{base}.layer_norm.bias"]), "F32")

                add_enc_tensor(f"neucodec.encode.hubert.encoder.layers.{i}.ffn.fc1.w",
                                 np.asarray(hubert_state[f"{base}.feed_forward.intermediate_dense.weight"]))
                add_enc_tensor(f"neucodec.encode.hubert.encoder.layers.{i}.ffn.fc1.b",
                                 np.asarray(hubert_state[f"{base}.feed_forward.intermediate_dense.bias"]), "F32")
                add_enc_tensor(f"neucodec.encode.hubert.encoder.layers.{i}.ffn.fc2.w",
                                 np.asarray(hubert_state[f"{base}.feed_forward.output_dense.weight"]))
                add_enc_tensor(f"neucodec.encode.hubert.encoder.layers.{i}.ffn.fc2.b",
                                 np.asarray(hubert_state[f"{base}.feed_forward.output_dense.bias"]), "F32")

                add_enc_tensor(f"neucodec.encode.hubert.encoder.layers.{i}.ffn_ln.w",
                                 np.asarray(hubert_state[f"{base}.final_layer_norm.weight"]), "F32")
                add_enc_tensor(f"neucodec.encode.hubert.encoder.layers.{i}.ffn_ln.b",
                                 np.asarray(hubert_state[f"{base}.final_layer_norm.bias"]), "F32")

        # Feature extractor metadata for semantic model (w2v-bert default)
        try:
            from transformers import AutoFeatureExtractor
            fe = AutoFeatureExtractor.from_pretrained(
                "facebook/w2v-bert-2.0",
                local_files_only=True,
            )
            mel_filters = np.asarray(fe.mel_filters, dtype=np.float32)
            window = np.asarray(fe.window, dtype=np.float32)
            writer.add_uint32("codec.mel.sample_rate", int(fe.sampling_rate))
            writer.add_uint32("codec.mel.n_mels", int(fe.num_mel_bins))
            writer.add_uint32("codec.mel.n_fft", int(W2V_BERT_FBANK_N_FFT))
            writer.add_uint32("codec.mel.win_length", int(W2V_BERT_FBANK_WIN_LENGTH))
            writer.add_uint32("codec.mel.hop_length", int(W2V_BERT_FBANK_HOP_LENGTH))
            writer.add_float32("codec.mel.preemphasis", float(W2V_BERT_FBANK_PREEMPHASIS))
            writer.add_float32("codec.mel.mel_floor", float(W2V_BERT_FBANK_MEL_FLOOR))
            writer.add_bool("codec.mel.remove_dc_offset", True)
            writer.add_bool("codec.mel.normalize_per_mel_bins", True)
            writer.add_uint32("codec.mel.stride", 2)
            add_enc_tensor("neucodec.encode.mel.filters", mel_filters, "F32")
            add_enc_tensor("neucodec.encode.mel.window", window, "F32")
        except Exception:
            # allow conversion without feature extractor if transformers is unavailable
            pass

        self._warn_if_no_quantized()
        writer.write()
        self.log(f"Wrote NeuCodec GGUF to {output_path}")


class DistillNeuCodecConverter(NeuCodecConverter):
    @property
    def model_type(self) -> str:
        return "distill_neucodec"
