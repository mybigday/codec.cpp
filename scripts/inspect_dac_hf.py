#!/usr/bin/env python3
import argparse
import math
from collections import OrderedDict
from math import gcd
from pathlib import Path
import subprocess
import tempfile
import wave

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from transformers import DacConfig, DacModel


def ffmpeg_available() -> bool:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.returncode == 0
    except FileNotFoundError:
        return False


def ffmpeg_convert_to_wav_mono(src: Path, dst: Path, sample_rate: int) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed: {proc.stderr.strip()}")


def to_float_mono(wav_np: np.ndarray) -> np.ndarray:
    if wav_np.ndim == 1:
        mono = wav_np
    else:
        mono = wav_np.mean(axis=1)

    if np.issubdtype(mono.dtype, np.integer):
        info = np.iinfo(mono.dtype)
        mono = mono.astype(np.float32)
        if info.min < 0:
            scale = max(abs(info.min), info.max)
            mono = mono / float(scale)
        else:
            midpoint = (info.max + 1) / 2.0
            mono = (mono - midpoint) / midpoint
    else:
        mono = mono.astype(np.float32, copy=False)
        peak = float(np.max(np.abs(mono))) if mono.size > 0 else 0.0
        if peak > 1.0:
            mono = mono / peak

    return np.clip(mono, -1.0, 1.0).astype(np.float32, copy=False)


def resample_to_target_sr(wav_np: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return wav_np
    factor = gcd(src_sr, target_sr)
    up = target_sr // factor
    down = src_sr // factor
    return resample_poly(wav_np, up, down).astype(np.float32, copy=False)


def load_input_audio(path: str, target_sr: int) -> torch.Tensor:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"input audio not found: {src}")

    with tempfile.TemporaryDirectory() as tmpdir:
        load_path = src
        if src.suffix.lower() != ".wav":
            if not ffmpeg_available():
                raise RuntimeError("input is not WAV and ffmpeg is unavailable for conversion")
            load_path = Path(tmpdir) / "input_24k_mono.wav"
            ffmpeg_convert_to_wav_mono(src, load_path, target_sr)

        sr, wav_np = wavfile.read(str(load_path))
        wav_mono = to_float_mono(wav_np)
        wav_mono = resample_to_target_sr(wav_mono, int(sr), target_sr)

    return torch.from_numpy(wav_mono).view(1, 1, -1)


def make_test_audio(sample_rate: int, seconds: float, freq: float) -> torch.Tensor:
    n = max(1, int(round(sample_rate * seconds)))
    t = torch.arange(n, dtype=torch.float32) / float(sample_rate)
    x = 0.2 * torch.sin(2.0 * math.pi * freq * t)
    return x.view(1, 1, -1)


def extract_state_dict(ckpt: object) -> OrderedDict:
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return OrderedDict(ckpt["state_dict"])
    if isinstance(ckpt, dict):
        return OrderedDict(ckpt)
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")


def remap_state_dict_keys(state_dict: OrderedDict, model: DacModel) -> OrderedDict:
    model_keys = set(model.state_dict().keys())
    if all(k in model_keys for k in state_dict.keys()):
        return state_dict

    prefixes = ("module.", "model.", "generator.")
    for prefix in prefixes:
        mapped = OrderedDict()
        for key, value in state_dict.items():
            new_key = key[len(prefix) :] if key.startswith(prefix) else key
            mapped[new_key] = value
        if all(k in model_keys for k in mapped.keys()):
            return mapped

    return state_dict


def convert_weight_norm(state_dict: OrderedDict, prefix: str) -> torch.Tensor:
    weight_v = state_dict[f"{prefix}.weight_v"]
    weight_g = state_dict[f"{prefix}.weight_g"]
    return torch._weight_norm(weight_v, weight_g, 0)


def convert_original_dac_state_dict(state_dict: OrderedDict, config: DacConfig) -> OrderedDict:
    converted = OrderedDict()

    def copy_weight_norm(src_prefix: str, dst_prefix: str) -> None:
        converted[f"{dst_prefix}.weight"] = convert_weight_norm(state_dict, src_prefix)
        converted[f"{dst_prefix}.bias"] = state_dict[f"{src_prefix}.bias"]

    def copy_residual_unit(src_prefix: str, dst_prefix: str) -> None:
        converted[f"{dst_prefix}.snake1.alpha"] = state_dict[f"{src_prefix}.block.0.alpha"]
        copy_weight_norm(f"{src_prefix}.block.1", f"{dst_prefix}.conv1")
        converted[f"{dst_prefix}.snake2.alpha"] = state_dict[f"{src_prefix}.block.2.alpha"]
        copy_weight_norm(f"{src_prefix}.block.3", f"{dst_prefix}.conv2")

    copy_weight_norm("encoder.block.0", "encoder.conv1")
    for i in range(len(config.downsampling_ratios)):
        src_block = f"encoder.block.{i + 1}.block"
        dst_block = f"encoder.block.{i}"
        copy_residual_unit(f"{src_block}.0", f"{dst_block}.res_unit1")
        copy_residual_unit(f"{src_block}.1", f"{dst_block}.res_unit2")
        copy_residual_unit(f"{src_block}.2", f"{dst_block}.res_unit3")
        converted[f"{dst_block}.snake1.alpha"] = state_dict[f"{src_block}.3.alpha"]
        copy_weight_norm(f"{src_block}.4", f"{dst_block}.conv1")
    converted["encoder.snake1.alpha"] = state_dict["encoder.block.5.alpha"]
    copy_weight_norm("encoder.block.6", "encoder.conv2")

    for i in range(config.n_codebooks):
        src = f"quantizer.quantizers.{i}"
        dst = f"quantizer.quantizers.{i}"
        copy_weight_norm(f"{src}.in_proj", f"{dst}.in_proj")
        copy_weight_norm(f"{src}.out_proj", f"{dst}.out_proj")
        converted[f"{dst}.codebook.weight"] = state_dict[f"{src}.codebook.weight"]

    copy_weight_norm("decoder.model.0", "decoder.conv1")
    for i in range(len(config.upsampling_ratios)):
        src_block = f"decoder.model.{i + 1}.block"
        dst_block = f"decoder.block.{i}"
        converted[f"{dst_block}.snake1.alpha"] = state_dict[f"{src_block}.0.alpha"]
        copy_weight_norm(f"{src_block}.1", f"{dst_block}.conv_t1")
        copy_residual_unit(f"{src_block}.2", f"{dst_block}.res_unit1")
        copy_residual_unit(f"{src_block}.3", f"{dst_block}.res_unit2")
        copy_residual_unit(f"{src_block}.4", f"{dst_block}.res_unit3")
    converted["decoder.snake1.alpha"] = state_dict["decoder.model.5.alpha"]
    copy_weight_norm("decoder.model.6", "decoder.conv2")

    return converted


def maybe_convert_original_dac(state_dict: OrderedDict, config: DacConfig) -> OrderedDict:
    if "encoder.block.0.weight_v" in state_dict and "decoder.model.0.weight_v" in state_dict:
        return convert_original_dac_state_dict(state_dict, config)
    return state_dict


def save_latent_bin(path: str, quantized_representation: torch.Tensor) -> None:
    # Input shape: (B, latent_dim, n_frames). Save B=1 payload as float32 [latent_dim, n_frames].
    q = quantized_representation.detach().to(torch.float32).cpu()
    if q.ndim != 3 or q.shape[0] != 1:
        raise ValueError(f"expected quantized_representation shape (1, D, T), got {tuple(q.shape)}")
    flat = q[0].contiguous().view(-1)
    with open(path, "wb") as f:
        f.write(flat.numpy().tobytes())


def save_wav_pcm16(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    # Input shape: (B, T). Save first item as mono 16-bit PCM WAV.
    y = waveform.detach().to(torch.float32).cpu()
    if y.ndim != 2 or y.shape[0] < 1:
        raise ValueError(f"expected waveform shape (B, T), got {tuple(y.shape)}")
    mono = torch.clamp(y[0], -1.0, 1.0)
    pcm_i16 = torch.round(mono * 32767.0).to(torch.int16).tolist()
    payload = bytearray()
    for s in pcm_i16:
        payload.extend(int(s).to_bytes(2, byteorder="little", signed=True))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(payload))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a HF DAC model with one encode/decode pass")
    parser.add_argument("--model-id", default="ibm-research/DAC.speech.v1.0", help="HF model id")
    parser.add_argument("--local-dir", default=None, help="Optional local model directory (config source)")
    parser.add_argument("--weights-pth", required=True, help="Path to DAC .pth checkpoint")
    parser.add_argument("--input-audio", default=None, help="Optional input audio path (wav/mp3); resampled to model sample rate mono")
    parser.add_argument("--duration", type=float, default=0.5, help="Synthetic input duration in seconds")
    parser.add_argument("--freq", type=float, default=440.0, help="Synthetic sine frequency in Hz")
    parser.add_argument("--n-quantizers", type=int, default=None, help="Optional n_quantizers for encode")
    parser.add_argument("--offline", action="store_true", help="Force local_files_only=True")
    parser.add_argument("--save-latent", default=None, help="Output path for quantized_representation float32 binary")
    parser.add_argument("--save-reference", default=None, help="Output path for Python decode WAV")
    args = parser.parse_args()

    source = args.local_dir if args.local_dir else args.model_id
    local_files_only = args.offline or bool(args.local_dir)

    print(f"loading DAC config from: {source}")
    print(f"weights checkpoint: {args.weights_pth}")
    print(f"local_files_only: {local_files_only}")

    try:
        config = DacConfig.from_pretrained(source, local_files_only=local_files_only)
        model = DacModel(config)
    except Exception as exc:
        print("ERROR: failed to load config/create model")
        print(repr(exc))
        print("hint: pass --local-dir <path> and/or --offline when model config is already downloaded")
        return 2

    try:
        checkpoint = torch.load(args.weights_pth, map_location="cpu")
        state_dict = extract_state_dict(checkpoint)
        state_dict = remap_state_dict_keys(state_dict, model)
        state_dict = maybe_convert_original_dac(state_dict, config)
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print("ERROR: checkpoint mismatch with model config")
            print(f"missing_keys: {incompatible.missing_keys[:20]}")
            print(f"unexpected_keys: {incompatible.unexpected_keys[:20]}")
            return 3
    except Exception as exc:
        print("ERROR: failed to load .pth checkpoint")
        print(repr(exc))
        return 2

    model.eval()
    cfg = model.config

    sample_rate = int(cfg.sampling_rate)
    hop_size = int(getattr(cfg, "hop_length", 0))
    n_q = int(cfg.n_codebooks)
    codebook_size = int(cfg.codebook_size)
    latent_dim = int(cfg.hidden_size)

    print("config:")
    print(f"  sample_rate:   {sample_rate}")
    print(f"  hop_size:      {hop_size}")
    print(f"  codebook_size: {codebook_size}")
    print(f"  n_q:           {n_q}")
    print(f"  latent_dim:    {latent_dim}")
    print(f"  codebook_dim:  {int(cfg.codebook_dim)}")
    print(f"  downsampling:  {list(cfg.downsampling_ratios)}")

    if args.input_audio:
        try:
            audio = load_input_audio(args.input_audio, sample_rate)
        except Exception as exc:
            print("ERROR: failed to load input audio")
            print(repr(exc))
            return 2
        print(f"input audio: {args.input_audio} -> model input shape {tuple(audio.shape)} at {sample_rate} Hz mono")
    else:
        audio = make_test_audio(sample_rate, args.duration, args.freq)
        print(f"input audio: synthetic sine wave (duration={args.duration:.3f}s, freq={args.freq:.2f}Hz)")

    with torch.no_grad():
        enc = model.encode(audio, n_quantizers=args.n_quantizers)
        dec_from_codes = model.decode(audio_codes=enc.audio_codes)
        dec_from_quant = model.decode(quantized_representation=enc.quantized_representation)

    print("interfaces:")
    print(f"  encode input              : {tuple(audio.shape)} (B, 1, T)")
    print(f"  encode audio_codes        : {tuple(enc.audio_codes.shape)} (B, n_q, n_frames)")
    print(f"  encode quantized_repr     : {tuple(enc.quantized_representation.shape)} (B, latent_dim, n_frames)")
    print(f"  decode(audio_codes) out   : {tuple(dec_from_codes.audio_values.shape)} (B, T')")
    print(f"  decode(quantized_repr) out: {tuple(dec_from_quant.audio_values.shape)} (B, T')")

    if args.save_latent:
        save_latent_bin(args.save_latent, enc.quantized_representation)
        q = enc.quantized_representation
        print(f"saved latent: {args.save_latent} (float32, shape={(int(q.shape[1]), int(q.shape[2]))})")

    if args.save_reference:
        save_wav_pcm16(args.save_reference, dec_from_quant.audio_values, sample_rate)
        print(f"saved reference wav: {args.save_reference} (sr={sample_rate}, n_samples={int(dec_from_quant.audio_values.shape[-1])})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
