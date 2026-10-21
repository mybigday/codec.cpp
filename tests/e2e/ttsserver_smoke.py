"""End-to-end smoke test for `tts-server` (OpenAI-compatible /v1/audio/speech).

Starts the built server on KittenTTS 2 (builtin voices, backbone-driven) and
asserts over HTTP:

  * custom voices: POST /v1/audio/voices only with --voice-dir +
    --allow-voice-create (403 otherwise), listed under `custom`, used with
    "voice": {"id": ...}, and reloaded identically after a restart,
  * request validation: response_format required, unknown voice / format,
    speed != 1.0, instructions on a model without an instruct mode → 400 in
    OpenAI's error shape; a wrong API key → 401,
  * a wav response is 24 kHz mono and non-silent; pcm is its raw s16le body,
  * one session serves many requests: a repeat is byte-identical, and a
    request without `voice` after another voice gets the default speaker,
  * `input` is limited to 4096 characters, not bytes (CJK),
  * `--speaker-file` serves the speakers of a file saved by
    `tts-cli --save-speaker` (KittenTTS clone from reference audio),
  * every optional encoder (mp3 / opus / aac / flac) either answers with a
    stream ffmpeg decodes to the wav's length, or with the "not built" 400,
  * `stream_format: "sse"` deltas decode to the pcm response, and a
    two-paragraph input runs as two segments,
  * frame-level streaming (CSM, Mimi decoder): audio arrives in several
    pieces long before the end, and matches the per-segment decode,
  * a run that fails before any audio (Qwen3-TTS without a reference clip)
    still gets a 500 in OpenAI's error shape, streamed or not.

Missing assets → skipped.  Run from the repo root:
    .venv/bin/python tests/e2e/ttsserver_smoke.py
"""

import base64
import io
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

from _common import MODEL_SAMPLING, REPO, must, sampling_flags

SERVER = REPO / "build" / "tts-server"
TTS_CLI = REPO / "build" / "tts-cli"
KT = REPO / "models" / "kitten_tts_2"
CODEC = KT / "kitten_tts2.gguf"
BACKBONE = KT / "kitten_tts2_backbone-q4_0.gguf"
REF_WAV = REPO / "input_audio" / "10_2_24k.wav"
REF_TEXT = "这恶魔甩着头发喷着口沫张牙舞爪地扑在老妇人的镜子上扼住她的脆弱的喉管"
CSM_CODEC = REPO / "models" / "csm" / "csm.gguf"
CSM_BACKBONE = REPO / "models" / "csm" / "llama_backbone.gguf"
QWEN3_CODEC = REPO / "models" / "qwen3_tts" / "qwen3_tts_06b_base.gguf"
QWEN3_BACKBONE = REPO / "models" / "qwen3_tts" / "qwen3_tts_talker.gguf"
TEXT = "One day, a little girl named Lily found a needle in her room."
KEY = "smoke-key"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url, body, key=KEY):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


def get(url, key=KEY):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def post_stream(url, body, key=KEY):
    """POST and read the body as it arrives: (pieces, seconds to the first piece, total seconds)."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    pieces, first = [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        while True:
            b = r.read1(1 << 16)
            if not b:
                break
            if first is None:
                first = time.time() - t0
            pieces.append(b)
    return pieces, first, time.time() - t0


def wav_info(data: bytes):
    with wave.open(io.BytesIO(data)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        return w.getframerate(), w.getnchannels(), x


class Server:
    """tts-server on a free port for the duration of a `with` block."""

    def __init__(self, *extra, codec=CODEC, backbone=BACKBONE):
        self.extra = list(extra)
        self.codec, self.backbone = codec, backbone

    def __enter__(self):
        port = free_port()
        self.base = f"http://127.0.0.1:{port}"
        self.log = tempfile.TemporaryFile()
        # The model card's sampling and a fixed seed (llama.cpp's default is random).
        card = "csm" if "csm" in self.codec.name else "qwen3_tts" if "qwen3" in self.codec.name else "kitten_tts2"
        sampling = sampling_flags(dict(MODEL_SAMPLING[card], seed=0xC0DEC1AB))
        self.proc = subprocess.Popen([str(SERVER), "--model", str(self.codec), "--backbone", str(self.backbone),
                                      "--port", str(port), "--n-threads", "8", "--api-key", KEY, *sampling,
                                      *self.extra],
                                     stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(600):
            try:
                urllib.request.urlopen(self.base + "/health", timeout=1)
                return self
            except Exception:
                if self.proc.poll() is not None:
                    self.dump()
                    must(False, "tts-server exited during load")
                time.sleep(0.5)
        self.__exit__()
        must(False, "tts-server did not come up")

    def text(self):
        self.log.seek(0)
        return self.log.read().decode(errors="replace")

    def dump(self):
        sys.stdout.write(self.text()[-4000:])

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait(timeout=30)
        if self.proc.returncode not in (0, -15):
            self.dump()


def post_multipart(url, fields, files, key=KEY):
    boundary = "codeccppboundary7f3a"
    body = b""
    for k, v in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, (fname, data) in files.items():
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{fname}"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    r = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(r, timeout=600) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def custom_voices() -> None:
    """OpenAI custom voices: create from a sample (only with --voice-dir and
    --allow-voice-create), list, speak with {"id": ...}, reload after a restart."""
    ref = REPO / "input_audio" / "10_2_24k.wav"
    if not ref.exists():
        print("[SKIP] custom voices: reference clip missing")
        return
    body = {"input": TEXT, "response_format": "wav"}
    with tempfile.TemporaryDirectory() as d:
        with Server() as srv:
            code, data = post_multipart(srv.base + "/v1/audio/voices", {"name": "x"}, {"audio_sample": ("r.wav", ref.read_bytes())})
            must(code == 403, f"voice creation without --voice-dir: {code} {data[:200]}")
        with Server("--voice-dir", d, "--allow-voice-create") as srv:
            code, data = post_multipart(srv.base + "/v1/audio/voices", {"name": "narrator"},
                                        {"audio_sample": ("ref.wav", ref.read_bytes())})
            must(code == 200, f"create voice: {code} {data[:300]}")
            v = json.loads(data)
            must(v["object"] == "audio.voice" and v["name"] == "narrator" and v["id"].startswith("voice_"), f"voice: {v}")
            listed = get(srv.base + "/v1/audio/voices")["custom"]
            must([x["id"] for x in listed] == [v["id"]], f"custom list: {listed}")
            code, _, cloned = post(srv.base + "/v1/audio/speech", {**body, "voice": {"id": v["id"]}})
            code2, _, default = post(srv.base + "/v1/audio/speech", body)
            must(code == 200 and code2 == 200 and cloned != default, "the custom voice did not change the audio")
        with Server("--voice-dir", d) as srv:   # stored voices are served without creation rights
            code, _, again = post(srv.base + "/v1/audio/speech", {**body, "voice": {"id": v["id"]}})
            must(code == 200 and again == cloned, "the stored voice did not reload identically")
            code, _ = post_multipart(srv.base + "/v1/audio/voices", {"name": "y"}, {"audio_sample": ("r.wav", ref.read_bytes())})
            must(code == 403, f"creation without --allow-voice-create: {code}")
    print("[PASS] custom voices: gated create, list, speak by id, reload after restart")


def main() -> int:
    if not SERVER.exists():
        print(f"[SKIP] {SERVER} not built")
        return 0
    if not CODEC.exists() or not BACKBONE.exists():
        print(f"[SKIP] KittenTTS 2 assets missing under {KT}")
        return 0

    custom_voices()

    with Server() as srv:
        base = srv.base
        speech = base + "/v1/audio/speech"
        voices = get(base + "/v1/audio/voices")
        must(len(voices["voices"]) > 1 and voices["default"] in voices["voices"], f"voices: {voices}")
        default, other = voices["default"], next(v for v in voices["voices"] if v != voices["default"])
        must(get(base + "/v1/models")["data"][0]["id"] == "kitten_tts2", "model id")

        def bad(body, param, status=400, key=KEY):
            code, _, data = post(speech, body, key)
            err = json.loads(data)["error"]
            must(code == status and err["param"] == param, f"{body} -> {code} {err}")
            return err["message"]

        bad({"input": TEXT, "response_format": "wav"}, None, 401, key="wrong")
        must("required" in bad({"input": TEXT}, "response_format"), "missing response_format")
        bad({"input": TEXT, "response_format": "mp4"}, "response_format")
        bad({"input": TEXT, "response_format": "wav", "voice": "alloy"}, "voice")
        bad({"input": TEXT, "response_format": "wav", "speed": 1.25}, "speed")
        bad({"input": TEXT, "response_format": "wav", "instructions": "cheerful"}, "instructions")
        bad({"response_format": "wav"}, "input")
        # 3000 CJK characters are 9000 bytes: within the limit, so the missing
        # response_format is what fails; 4100 characters exceed it.
        bad({"input": "语" * 3000}, "response_format")
        bad({"input": "语" * 4100, "response_format": "wav"}, "input")
        print("[PASS] validation")

        req = {"model": "tts-1", "input": TEXT, "response_format": "wav"}
        code, ctype, wav_default = post(speech, {**req, "voice": default})
        must(code == 200 and ctype == "audio/wav", f"wav: {code} {ctype}")
        sr, ch, x = wav_info(wav_default)
        rms = float(np.sqrt((x ** 2).mean()))
        must(sr == 24000 and ch == 1 and 1.0 < len(x) / sr < 10.0 and rms > 0.01,
             f"wav: sr={sr} ch={ch} dur={len(x) / sr:.2f} rms={rms:.3f}")
        print(f"[PASS] wav: {len(x) / sr:.2f}s rms={rms:.3f}")

        code, _, wav_other = post(speech, {**req, "voice": other})
        must(code == 200 and wav_other != wav_default, f"voice {other} gave the default voice")
        code, _, wav_none = post(speech, req)
        must(code == 200 and wav_none == wav_default, "request without voice did not get the default speaker back")
        code, _, wav_again = post(speech, {**req, "voice": default})
        must(wav_again == wav_default, "repeat request is not byte-identical")
        code, _, wav_seed = post(speech, {**req, "voice": default, "seed": 7, "temperature": 0.9})
        must(code == 200 and wav_seed != wav_default, "request sampling fields (seed / temperature) had no effect")
        print(f"[PASS] session reuse: {other} differs, default restored, repeat identical, request sampling")

        code, ctype, pcm = post(speech, {**req, "response_format": "pcm"})
        must(code == 200 and ctype == "audio/pcm" and pcm == wav_default[44:], "pcm != wav body")
        print("[PASS] pcm")

        code, ctype, sse = post(speech, {**req, "response_format": "pcm", "stream_format": "sse"})
        events = [json.loads(line[6:]) for line in sse.decode().split("\n") if line.startswith("data: ")]
        must(code == 200 and ctype.startswith("text/event-stream"), f"sse: {code} {ctype}")
        must(events[-1]["type"] == "speech.audio.done" and
             all(e["type"] == "speech.audio.delta" for e in events[:-1]), f"sse events: {[e['type'] for e in events]}")
        must(b"".join(base64.b64decode(e["audio"]) for e in events[:-1]) == pcm, "sse deltas != pcm response")
        print(f"[PASS] sse: {len(events) - 1} deltas == pcm")

        code, _, two = post(speech, {**req, "input": TEXT + "\n\nThe end of the story.", "response_format": "pcm"})
        must(code == 200 and len(two) > len(pcm) and "segment 2/2" in srv.text(), "two paragraphs did not run as two segments")
        print("[PASS] segments: two paragraphs -> 2 runs")

        ffmpeg = shutil.which("ffmpeg")
        for fmt in ("mp3", "opus", "aac", "flac"):
            code, ctype, data = post(speech, {**req, "response_format": fmt})
            if code == 400:
                must("not built" in json.loads(data)["error"]["message"], f"{fmt}: {data[:200]}")
                print(f"[SKIP] {fmt}: not built")
                continue
            must(code == 200 and len(data) > 1000, f"{fmt}: {code} {len(data)} bytes")
            if ffmpeg:
                out = subprocess.run([ffmpeg, "-v", "error", "-i", "-", "-f", "s16le", "-ac", "1", "-ar", "24000", "-"],
                                     input=data, capture_output=True)
                dur = len(out.stdout) / 2 / 24000
                must(abs(dur - len(x) / sr) < 0.2, f"{fmt}: duration {dur:.2f} vs {len(x) / sr:.2f}")
            print(f"[PASS] {fmt}: {ctype} {len(data)} bytes")

    if CSM_CODEC.exists() and CSM_BACKBONE.exists():
        body = {"input": "Hello, this is a test of the emergency broadcast system. We are checking that "
                         "streaming audio sounds exactly like the whole clip.", "response_format": "pcm"}
        streamed = {}
        with Server("--max-frames", "400", codec=CSM_CODEC, backbone=CSM_BACKBONE) as srv:
            pieces, first, total = post_stream(srv.base + "/v1/audio/speech", body)
            for fmt in ("mp3", "opus", "aac", "flac", "wav"):
                code, _, data = post(srv.base + "/v1/audio/speech", {**body, "response_format": fmt})
                if code == 200:
                    streamed[fmt] = data
        with Server("--max-frames", "400", "--stream-chunk-ms", "0", codec=CSM_CODEC, backbone=CSM_BACKBONE) as srv:
            whole, _, _ = post_stream(srv.base + "/v1/audio/speech", body)
        a = np.frombuffer(b"".join(pieces), "<i2").astype(np.float32) / 32768
        b = np.frombuffer(b"".join(whole), "<i2").astype(np.float32) / 32768
        must(len(pieces) > 2 and first < total / 3, f"csm: {len(pieces)} pieces, first {first:.2f}s of {total:.2f}s")
        must(len(a) == len(b) and np.abs(a - b).max() < 2e-3, f"csm: streamed != per-segment ({len(a)} vs {len(b)})")
        print(f"[PASS] frame streaming (csm): {len(pieces)} pieces, first after {first:.2f}s of {total:.2f}s, "
              f"max|d| {np.abs(a - b).max():.1e}")
        ffmpeg = shutil.which("ffmpeg")
        for fmt, data in streamed.items():
            if not ffmpeg:
                break
            out = subprocess.run([ffmpeg, "-v", "error", "-i", "-", "-f", "s16le", "-ac", "1", "-ar", "24000", "-"],
                                 input=data, capture_output=True)
            dur = len(out.stdout) / 2 / 24000
            must(abs(dur - len(a) / 24000) < 0.2, f"streamed {fmt}: {dur:.2f}s vs {len(a) / 24000:.2f}s")
            print(f"[PASS] streamed {fmt}: {dur:.2f}s")
    else:
        print("[SKIP] frame streaming: CSM assets missing")

    if QWEN3_CODEC.exists() and QWEN3_BACKBONE.exists():
        with Server(codec=QWEN3_CODEC, backbone=QWEN3_BACKBONE) as srv:
            for extra in ({}, {"stream_format": "sse"}):
                code, ctype, data = post(srv.base + "/v1/audio/speech", {"input": "你好", "response_format": "wav", **extra})
                must(code == 500 and ctype == "application/json" and "ref-audio" in json.loads(data)["error"]["message"],
                     f"early failure {extra}: {code} {ctype} {data[:200]}")
        print("[PASS] early failure: 500 JSON (audio and sse)")
    else:
        print("[SKIP] early failure: Qwen3-TTS assets missing")

    if not TTS_CLI.exists() or not REF_WAV.exists():
        print("[SKIP] speaker file: tts-cli or reference audio missing")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        spk = Path(tmp) / "spk.gguf"
        r = subprocess.run([str(TTS_CLI), "synthesize", "--model", str(CODEC), "--backbone", str(BACKBONE),
                            "--n-threads", "8", "--text", "Hi.", *sampling_flags(MODEL_SAMPLING["kitten_tts2"]), "--ref-audio", str(REF_WAV), "--ref-text", REF_TEXT,
                            "--save-speaker", str(spk), "--output", str(Path(tmp) / "x.wav")], capture_output=True)
        must(r.returncode == 0 and spk.exists(), "tts-cli --save-speaker failed: " + r.stderr.decode()[-2000:])
        with Server("--speaker-file", str(spk)) as srv:
            voices = get(srv.base + "/v1/audio/voices")
            must(voices["voices"] == ["saved"] and voices["default"] == "saved", f"speaker-file voices: {voices}")
            req = {"input": TEXT, "response_format": "wav"}
            code, _, named = post(srv.base + "/v1/audio/speech", {**req, "voice": "saved"})
            code2, _, unnamed = post(srv.base + "/v1/audio/speech", req)
            must(code == 200 and code2 == 200 and named == unnamed, "saved speaker: named != default")
            must(named != wav_default, "speaker file served the builtin default voice")
            _, _, xs = wav_info(named)
            must(float(np.sqrt((xs ** 2).mean())) > 0.01, "speaker-file output is silent")
            print("[PASS] speaker file: 'saved' is the only and default voice")
    return 0


if __name__ == "__main__":
    sys.exit(main())
