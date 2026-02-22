#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
import subprocess
import tempfile
import wave

import numpy as np
import torch
from transformers import MimiConfig, MimiModel


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
        "-f",
        "wav",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed: {proc.stderr.strip()}")


def load_wav_pcm16_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        channels = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        nframes = int(wf.getnframes())
        if sampwidth != 2:
            raise ValueError(f"expected 16-bit PCM WAV, got sample width {sampwidth} bytes")
        data = wf.readframes(nframes)

    pcm = np.frombuffer(data, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16, copy=False)
    wav = np.clip(pcm.astype(np.float32) / 32768.0, -1.0, 1.0)
    return sr, wav.astype(np.float32, copy=False)


def load_input_audio(path: str, target_sr: int) -> torch.Tensor:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"input audio not found: {src}")
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required to normalize input to 24k mono wav")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_wav = Path(tmpdir) / "input_24k_mono.wav"
        ffmpeg_convert_to_wav_mono(src, tmp_wav, target_sr)
        sr, mono = load_wav_pcm16_mono(tmp_wav)
        if sr != target_sr:
            raise RuntimeError(f"unexpected converted sample rate: got {sr}, expected {target_sr}")

    return torch.from_numpy(mono).view(1, 1, -1)


def make_test_audio(sample_rate: int, seconds: float, freq: float) -> torch.Tensor:
    n = max(1, int(round(sample_rate * seconds)))
    t = torch.arange(n, dtype=torch.float32) / float(sample_rate)
    x = 0.2 * torch.sin(2.0 * math.pi * freq * t)
    return x.view(1, 1, -1)


def save_codes(path: str, audio_codes: torch.Tensor) -> tuple[int, int]:
    codes = audio_codes.detach().to(torch.int32).cpu()
    if codes.ndim != 3 or codes.shape[0] != 1:
        raise ValueError(f"expected audio_codes shape (1, num_quantizers, codes_length), got {tuple(codes.shape)}")
    q = int(codes.shape[1])
    t = int(codes.shape[2])
    codes_2d = codes[0].contiguous()
    dst = Path(path)
    if dst.suffix.lower() == ".npy":
        np.save(str(dst), codes_2d.numpy().astype(np.int32, copy=False))
    else:
        flat = codes_2d.reshape(-1).numpy().astype(np.int32, copy=False)
        with open(dst, "wb") as f:
            f.write(flat.tobytes())
    return q, t


def save_wav_pcm16(path: str, waveform: torch.Tensor, sample_rate: int) -> int:
    y = waveform.detach().to(torch.float32).cpu()
    if y.ndim == 3:
        if y.shape[1] != 1:
            raise ValueError(f"expected mono waveform in decode output, got shape {tuple(y.shape)}")
        mono = y[0, 0]
    elif y.ndim == 2:
        mono = y[0]
    else:
        raise ValueError(f"expected waveform shape (B, T) or (B, 1, T), got {tuple(y.shape)}")

    mono = torch.clamp(mono, -1.0, 1.0)
    pcm_i16 = torch.round(mono * 32767.0).to(torch.int16).numpy().astype("<i2", copy=False)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())
    return int(mono.numel())


def get_hop_and_frame_rate(cfg: MimiConfig) -> tuple[int | None, float | None]:
    sample_rate = int(getattr(cfg, "sampling_rate", 24000))
    hop_size = getattr(cfg, "hop_size", None)
    if hop_size is None:
        hop_size = getattr(cfg, "hop_length", None)
    frame_rate = getattr(cfg, "frame_rate", None)

    if frame_rate is None and hop_size is not None:
        frame_rate = float(sample_rate) / float(hop_size)
    if hop_size is None and frame_rate is not None and float(frame_rate) > 0:
        hop_size = int(round(float(sample_rate) / float(frame_rate)))

    hop = int(hop_size) if hop_size is not None else None
    fr = float(frame_rate) if frame_rate is not None else None
    return hop, fr


def mimi_encode(model: MimiModel, audio: torch.Tensor):
    for kwargs in ({"input_values": audio}, {"audio_values": audio}):
        try:
            return model.encode(**kwargs)
        except TypeError:
            continue
    return model.encode(audio)


def mimi_decode(model: MimiModel, audio_codes: torch.Tensor):
    try:
        return model.decode(audio_codes=audio_codes)
    except TypeError:
        return model.decode(audio_codes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a HF Mimi model with one encode/decode pass")
    parser.add_argument("--model-id", default="kyutai/mimi", help="HF model id")
    parser.add_argument("--input-audio", default=None, help="Input audio path (wav/mp3); normalized to 24k mono with ffmpeg")
    parser.add_argument("--duration", type=float, default=0.5, help="Synthetic input duration when --input-audio is not provided")
    parser.add_argument("--freq", type=float, default=440.0, help="Synthetic sine frequency in Hz")
    parser.add_argument("--offline", action="store_true", help="Force local_files_only=True")
    parser.add_argument("--save-codes", default=None, help="Output path for audio codes (.npy for shaped array, otherwise int32 flat binary)")
    parser.add_argument("--save-reference", default=None, help="Output path for decoded reference WAV (PCM16 mono)")
    args = parser.parse_args()

    print(f"loading Mimi config/model from: {args.model_id}")
    print(f"local_files_only: {args.offline}")

    try:
        config = MimiConfig.from_pretrained(args.model_id, local_files_only=args.offline)
        model = MimiModel.from_pretrained(args.model_id, local_files_only=args.offline)
    except Exception as exc:
        print("ERROR: failed to load Mimi model/config")
        print(repr(exc))
        return 2

    model.eval()
    cfg = config

    sample_rate = int(getattr(cfg, "sampling_rate", 24000))
    hop_size, frame_rate = get_hop_and_frame_rate(cfg)
    num_quantizers_cfg = getattr(cfg, "num_quantizers", None)
    codebook_size = getattr(cfg, "codebook_size", None)

    print("config:")
    print(f"  sample_rate:    {sample_rate}")
    print(f"  hop_size:       {hop_size}")
    print(f"  frame_rate:     {frame_rate}")
    print(f"  num_quantizers: {num_quantizers_cfg}")
    print(f"  codebook_size:  {codebook_size}")

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
        enc = mimi_encode(model, audio)
        dec = mimi_decode(model, enc.audio_codes)

    print("interfaces:")
    print(f"  encode input         : {tuple(audio.shape)} (B, 1, T)")
    print(f"  encode audio_codes   : {tuple(enc.audio_codes.shape)} (B, num_quantizers, codes_length)")
    print(f"  decode(audio_codes)  : {tuple(dec.audio_values.shape)}")

    if args.save_codes:
        q, t = save_codes(args.save_codes, enc.audio_codes)
        print(f"saved codes: {args.save_codes} (int32, shape=({q}, {t}))")

    if args.save_reference:
        n_samples = save_wav_pcm16(args.save_reference, dec.audio_values, sample_rate)
        print(f"saved reference wav: {args.save_reference} (sr={sample_rate}, n_samples={n_samples})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
