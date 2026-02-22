import os, sys
import subprocess
from math import gcd
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

# WavTokenizer-source is vendored in this workspace.
# decoder/pretrained.py imports "decoder.*" so we must add the repo ROOT (not decoder/) to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "WavTokenizer-source"))

from decoder.pretrained import WavTokenizer

TARGET_SR = 24000


def _to_float_mono(wav_np: np.ndarray) -> np.ndarray:
    if wav_np.ndim == 1:
        mono = wav_np
    else:
        # scipy.io.wavfile returns shape (T, C) for multi-channel audio.
        mono = wav_np.mean(axis=1)

    if np.issubdtype(mono.dtype, np.integer):
        info = np.iinfo(mono.dtype)
        mono = mono.astype(np.float32)
        if info.min < 0:
            scale = max(abs(info.min), info.max)
            mono = mono / float(scale)
        else:
            # Unsigned PCM (e.g. uint8) is midpoint-centered.
            midpoint = (info.max + 1) / 2.0
            mono = (mono - midpoint) / midpoint
    else:
        mono = mono.astype(np.float32, copy=False)
        peak = float(np.max(np.abs(mono))) if mono.size > 0 else 0.0
        if peak > 1.0:
            mono = mono / peak

    return np.clip(mono, -1.0, 1.0).astype(np.float32, copy=False)


def _resample_to_24k(wav_np: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return wav_np
    factor = gcd(sr, TARGET_SR)
    up = TARGET_SR // factor
    down = sr // factor
    return resample_poly(wav_np, up, down).astype(np.float32, copy=False)


def _save_wav(path: Path, wav_1ch: torch.Tensor, sr: int):
    wav_np = wav_1ch.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    wav_np = np.clip(wav_np, -1.0, 1.0)
    wav_i16 = np.round(wav_np * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, wav_i16)


def _ffmpeg_available() -> bool:
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


def _ffmpeg_convert_to_wav_24k_mono(src: Path, dst: Path):
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
        str(TARGET_SR),
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed: {proc.stderr.strip()}")


def _ffmpeg_convert_wav_to_mp3(src: Path, dst: Path):
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "192k",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg wav->mp3 failed: {proc.stderr.strip()}")


def _check_gguf_metadata(gguf_path: Path):
    inspect_bin = Path(__file__).resolve().parents[1] / "build" / "inspect-codec"
    if not inspect_bin.exists():
        raise RuntimeError(f"inspect binary not found: {inspect_bin}")
    if not gguf_path.exists():
        raise RuntimeError(f"gguf model not found: {gguf_path}")

    proc = subprocess.run(
        [str(inspect_bin), str(gguf_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"inspect-codec failed: {proc.stderr.strip()}")

    lines = proc.stdout.splitlines()
    need = {
        "codec.sample_rate = 24000": False,
        "codec.has_encoder = true": False,
        "codec.has_decoder = true": False,
    }
    for line in lines:
        for key in list(need.keys()):
            if key in line:
                need[key] = True
    missing = [k for k, ok in need.items() if not ok]
    if missing:
        raise RuntimeError(f"gguf metadata missing/invalid: {missing}")


def load_model(config_path: str, ckpt_path: str):
    model = WavTokenizer.from_pretrained0802(config_path, ckpt_path)
    model.eval()
    return model


def _resolve_default_wav() -> Optional[Path]:
    candidates = [
        Path("/home/node/.linuxbrew/Homebrew/Library/Homebrew/test/support/fixtures/test.wav"),
        Path(__file__).resolve().parents[1] / "out_wavtokenizer" / "ref_24k.wav",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _to_bandwidth_tensor(bandwidth_id: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([bandwidth_id], dtype=torch.long, device=device)


def _bandwidth_info(model, bandwidth_id: int) -> Tuple[Optional[float], Optional[float]]:
    bandwidths = getattr(getattr(model, "feature_extractor", None), "bandwidths", None)
    if bandwidths is None or bandwidth_id < 0 or bandwidth_id >= len(bandwidths):
        return None, None
    bandwidth_kbps = float(bandwidths[bandwidth_id])
    return bandwidth_kbps, bandwidth_kbps


def encode_decode(model, wav_1ch: torch.Tensor, bandwidth_id: int, n_threads: int = 4):
    """wav_1ch: (1, T) float tensor in [-1, 1] at 24k"""
    # WavTokenizer API expects (B, T). Input from this script is already (B=1, T).
    if wav_1ch.dim() != 2:
        raise ValueError(f"Expected wav_1ch shape (B, T), got {tuple(wav_1ch.shape)}")
    wav_in = wav_1ch

    with torch.inference_mode():
        bandwidth = _to_bandwidth_tensor(bandwidth_id, wav_in.device)
        feats, tokens = model.encode(wav_in, bandwidth_id=bandwidth)
        # Keep decode() kwargs aligned with current API for conditioned backbones.
        recon = model.decode(feats, bandwidth_id=bandwidth, num_threads=n_threads)

    # recon: (B, C, T)
    return recon[0], tokens


def _validate_tokens(tokens: torch.Tensor):
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"tokens must be torch.Tensor, got {type(tokens)!r}")
    if tokens.numel() == 0:
        raise RuntimeError("tokens is empty")
    if tokens.dim() < 2:
        raise RuntimeError(f"tokens dim must be >= 2, got {tuple(tokens.shape)}")
    if 1 not in tokens.shape:
        raise RuntimeError(
            f"tokens should contain batch-size 1 in one axis for single input, got {tuple(tokens.shape)}"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("wav", nargs="?", help="input audio file")
    parser.add_argument(
        "--ckpt",
        default=os.path.expanduser(
            "~/.cache/huggingface/hub/models--novateur--WavTokenizer-medium-speech-75token/snapshots/8858552e69270816d6aeb37bfcf3b770769d4899/wavtokenizer_medium_speech_320_24k_v2.ckpt"
        ),
        help="path to wavtokenizer .ckpt",
    )
    parser.add_argument(
        "--config",
        default=str(
            REPO_ROOT
            / "WavTokenizer-source"
            / "configs"
            / "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        ),
        help="path to wavtokenizer config .yaml",
    )
    parser.add_argument("--seconds", type=float, default=5.0, help="trim to first N seconds")
    parser.add_argument("--out", default="out_wavtokenizer", help="output directory")
    parser.add_argument("--bandwidth-id", type=int, default=0, help="bandwidth index for encode/decode")
    parser.add_argument("--gguf-model", default="wavtokenizer_v2.gguf", help="GGUF model path for metadata checks")
    parser.add_argument(
        "--save-mp3",
        action="store_true",
        help="also save ref/recon as MP3 files when ffmpeg is available",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="number of threads for encoding/decoding",
    )

    args = parser.parse_args()

    wav_path: Optional[Path]
    if args.wav:
        wav_path = Path(args.wav)
    else:
        wav_path = _resolve_default_wav()
        if wav_path is None:
            parser.error(
                "no input wav provided and no default sample found; pass a wav path as positional argument"
            )

    if not wav_path.exists():
        parser.error(f"input wav not found: {wav_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _check_gguf_metadata(Path(args.gguf_model))

    model = load_model(args.config, args.ckpt)
    model = model.cpu()

    bandwidths = getattr(getattr(model, "feature_extractor", None), "bandwidths", None)
    if bandwidths is not None:
        if args.bandwidth_id < 0 or args.bandwidth_id >= len(bandwidths):
            parser.error(
                f"--bandwidth-id out of range: {args.bandwidth_id}, valid range is [0, {len(bandwidths) - 1}]"
            )

    converted_wav_path: Optional[Path] = None
    in_ext = wav_path.suffix.lower()
    if in_ext != ".wav":
        if not _ffmpeg_available():
            parser.error("input is not WAV and ffmpeg is unavailable for conversion")
        converted_wav_path = out_dir / "input_24k_mono.wav"
        _ffmpeg_convert_to_wav_24k_mono(wav_path, converted_wav_path)
        load_path = converted_wav_path
    else:
        load_path = wav_path

    sr, wav_np = wavfile.read(str(load_path))
    wav_np = _to_float_mono(wav_np)
    wav_np = _resample_to_24k(wav_np, sr)
    wav = torch.from_numpy(wav_np).unsqueeze(0)

    # trim
    n = int(args.seconds * TARGET_SR)
    if wav.shape[1] > n:
        wav = wav[:, :n]

    # Save reference (resampled/trimmed) and reconstruction
    ref_path = out_dir / "ref_24k.wav"
    _save_wav(ref_path, wav, TARGET_SR)

    recon, tokens = encode_decode(model, wav, bandwidth_id=args.bandwidth_id, n_threads=args.threads)
    _validate_tokens(tokens)
    recon = recon.clamp(-1, 1).cpu()

    recon_path = out_dir / "recon.wav"
    _save_wav(recon_path, recon, TARGET_SR)

    ref_mp3_path: Optional[Path] = None
    recon_mp3_path: Optional[Path] = None
    if args.save_mp3:
        if not _ffmpeg_available():
            parser.error("--save-mp3 requested but ffmpeg is unavailable")
        ref_mp3_path = out_dir / "ref_24k.mp3"
        recon_mp3_path = out_dir / "recon.mp3"
        _ffmpeg_convert_wav_to_mp3(ref_path, ref_mp3_path)
        _ffmpeg_convert_wav_to_mp3(recon_path, recon_mp3_path)

    # quick numeric sanity (not the main deliverable)
    ref_np = wav.squeeze(0).cpu().numpy()
    rec_np = recon.squeeze(0).cpu().numpy()
    min_len = min(ref_np.shape[-1], rec_np.shape[-1])
    ref_np = ref_np[:min_len]
    rec_np = rec_np[:min_len]
    noise = ref_np - rec_np
    snr = 10 * np.log10((np.mean(ref_np**2) + 1e-12) / (np.mean(noise**2) + 1e-12))

    bandwidth_kbps, bitrate_kbps = _bandwidth_info(model, args.bandwidth_id)

    print(f"Input: {wav_path}")
    print(f"GGUF model checked: {args.gguf_model}")
    if converted_wav_path is not None:
        print(f"Converted WAV (24k mono): {converted_wav_path}")
    print("Saved:", ref_path)
    print("Saved:", recon_path)
    if ref_mp3_path is not None and recon_mp3_path is not None:
        print("Saved:", ref_mp3_path)
        print("Saved:", recon_mp3_path)
    print(f"Input sample rate used by model: {TARGET_SR}")
    print(f"Tokens shape: {tuple(tokens.shape)}")
    if bandwidth_kbps is None:
        print(f"Bandwidth setting: id={args.bandwidth_id} (value unavailable)")
    else:
        print(
            f"Bandwidth setting: id={args.bandwidth_id}, bandwidth={bandwidth_kbps:.2f} kbps, bitrate={bitrate_kbps:.2f} kbps"
        )
    print(f"Reconstructed length: {recon.shape[-1]} samples ({recon.shape[-1] / TARGET_SR:.3f} s @ {TARGET_SR} Hz)")
    print(f"SNR(dB): {snr:.2f}")


if __name__ == "__main__":
    main()
