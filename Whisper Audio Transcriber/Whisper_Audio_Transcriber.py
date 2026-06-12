"""
whisper_transcribe.py
─────────────────────
Batch audio transcriber using OpenAI Whisper Medium (via Hugging Face).

Features
  • Auto-downloads whisper-medium weights on first run (cached locally)
  • Accepts any audio format FFmpeg can decode (mp3, wav, m4a, ogg, flac, …)
  • Handles files of any length via chunked processing with overlap
  • Saves a .txt beside each audio file, same stem name
  • Progress bars, coloured console output, no external GUI needed

Requirements (install once):
  pip install transformers accelerate torch torchaudio soundfile librosa
  FFmpeg must be on PATH  →  https://ffmpeg.org/download.html
"""

import os
import sys
import time
import textwrap
import subprocess
import tempfile
from pathlib import Path

# ── Colour helpers ────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
DIM    = "\033[2m"

def c(text, colour): return f"{colour}{text}{RESET}"

# ── Dependency check ──────────────────────────────────────────────────────────
REQUIRED = {
    "transformers": "transformers",
    "torch":        "torch",
    "torchaudio":   "torchaudio",
    "librosa":      "librosa",
    "soundfile":    "soundfile",
}

def check_dependencies():
    missing = []
    for module, pkg in REQUIRED.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(c(f"\n[ERROR] Missing packages: {', '.join(missing)}", RED))
        print(c(f"  Run:  pip install {' '.join(missing)}", YELLOW))
        sys.exit(1)

check_dependencies()

# ── Imports (after dep check) ─────────────────────────────────────────────────
import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID        = "openai/whisper-medium"
SAMPLE_RATE     = 16_000          # Whisper expects 16 kHz mono
CHUNK_SECONDS   = 25              # seconds per chunk (< 30 s limit)
OVERLAP_SECONDS = 2               # overlap to avoid cut-word artefacts
AUDIO_EXTS      = {
    ".mp3", ".wav", ".m4a", ".ogg", ".flac",
    ".aac", ".wma", ".opus", ".webm", ".mp4",
    ".mov", ".mkv", ".avi",
}

# ── Model loader ──────────────────────────────────────────────────────────────
def load_model():
    print(c("\n● Loading Whisper Medium model …", CYAN))
    print(c(f"  Source : {MODEL_ID} (Hugging Face)", DIM))
    print(c("  Weights will be cached in ~/.cache/huggingface on first run.\n", DIM))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model     = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    ).to(device)
    model.eval()

    print(c(f"  ✓ Model ready  [{device.upper()}  |  {str(dtype).split('.')[-1]}]\n",
            GREEN))
    return processor, model, device, dtype

# ── Audio loader (supports any format via FFmpeg fallback) ────────────────────
def load_audio(path: Path) -> np.ndarray:
    """Return a 16 kHz mono float32 numpy array."""
    try:
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        return audio.astype(np.float32)
    except Exception:
        # Fallback: decode via FFmpeg → raw PCM
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(path),
                    "-ar", str(SAMPLE_RATE), "-ac", "1",
                    "-f", "wav", tmp_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
            return audio.astype(np.float32)
        finally:
            os.unlink(tmp_path)

# ── Transcriber ───────────────────────────────────────────────────────────────
def transcribe(audio: np.ndarray, processor, model, device, dtype) -> str:
    """
    Chunk the audio into CHUNK_SECONDS windows with OVERLAP_SECONDS overlap,
    transcribe each chunk, then stitch results together.
    """
    chunk_len   = CHUNK_SECONDS   * SAMPLE_RATE
    overlap_len = OVERLAP_SECONDS * SAMPLE_RATE
    step        = chunk_len - overlap_len
    total       = len(audio)
    segments    = []
    start       = 0
    chunk_index = 0

    total_chunks = max(1, int(np.ceil(total / step)))

    while start < total:
        end   = min(start + chunk_len, total)
        chunk = audio[start:end]

        # Progress
        chunk_index += 1
        elapsed_s = start / SAMPLE_RATE
        total_s   = total  / SAMPLE_RATE
        pct       = min(100, int(100 * start / total)) if total > 0 else 100
        bar_fill  = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(
            f"\r  [{bar_fill}] {pct:3d}%  chunk {chunk_index}/{total_chunks}"
            f"  ({_fmt_time(elapsed_s)} / {_fmt_time(total_s)})",
            end="", flush=True,
        )

        inputs = processor(
            chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_features.to(device, dtype=dtype)

        with torch.no_grad():
            predicted_ids = model.generate(inputs, language="en")

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        if text:
            segments.append(text)

        if end == total:
            break
        start += step

    print(f"\r  [{'█'*20}] 100%  done{' '*30}")
    return " ".join(segments)

# ── Output formatter ──────────────────────────────────────────────────────────
def format_transcript(raw: str, audio_name: str, duration_s: float) -> str:
    """Wrap text at 100 chars and add a header block."""
    wrapped = textwrap.fill(raw, width=100)
    header = (
        f"{'='*100}\n"
        f"  FILE     : {audio_name}\n"
        f"  DURATION : {_fmt_time(duration_s)}\n"
        f"  MODEL    : {MODEL_ID}\n"
        f"  CREATED  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'='*100}\n\n"
    )
    return header + wrapped + "\n"

def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

# ── File discovery ────────────────────────────────────────────────────────────
def find_audio_files(directory: Path) -> list[Path]:
    files = [
        p for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]
    return files

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(c("""
╔══════════════════════════════════════════════════════╗
║          Whisper Medium — Batch Transcriber          ║
║         openai/whisper-medium  ·  Hugging Face       ║
╚══════════════════════════════════════════════════════╝""", CYAN))

    # Get directory from user
    while True:
        raw = input(c("\n  Enter the directory path containing your audio file(s):\n  > ", BOLD)).strip()
        directory = Path(raw.strip('"').strip("'"))
        if directory.is_dir():
            break
        print(c(f"  [!] Not a valid directory: {directory}", RED))

    audio_files = find_audio_files(directory)

    if not audio_files:
        print(c(f"\n  No audio files found in: {directory}", YELLOW))
        print(c(f"  Supported extensions: {', '.join(sorted(AUDIO_EXTS))}", DIM))
        sys.exit(0)

    print(c(f"\n  Found {len(audio_files)} audio file(s):", GREEN))
    for i, f in enumerate(audio_files, 1):
        print(c(f"    {i:>3}. {f.name}", DIM))

    # Load model once, reuse for all files
    processor, model, device, dtype = load_model()

    success, failed = 0, []

    for idx, audio_path in enumerate(audio_files, 1):
        print(c(f"\n{'─'*60}", DIM))
        print(c(f"  [{idx}/{len(audio_files)}] {audio_path.name}", BOLD))

        try:
            t0 = time.time()

            print(c("  Loading audio …", DIM), end="\r")
            audio = load_audio(audio_path)
            duration_s = len(audio) / SAMPLE_RATE
            print(c(f"  Audio loaded   duration: {_fmt_time(duration_s)}", DIM))

            print(c("  Transcribing …", YELLOW))
            raw_text = transcribe(audio, processor, model, device, dtype)

            formatted = format_transcript(raw_text, audio_path.name, duration_s)

            out_path = audio_path.with_suffix(".txt")
            out_path.write_text(formatted, encoding="utf-8")

            elapsed = time.time() - t0
            print(c(f"  ✓ Saved  →  {out_path.name}  ({elapsed:.1f}s)", GREEN))
            success += 1

        except Exception as e:
            print(c(f"\n  [ERROR] {audio_path.name}: {e}", RED))
            failed.append(audio_path.name)

    # Summary
    print(c(f"\n{'═'*60}", CYAN))
    print(c(f"  Done.  {success} transcribed", GREEN), end="")
    if failed:
        print(c(f"  ·  {len(failed)} failed:", RED))
        for f in failed:
            print(c(f"      • {f}", RED))
    else:
        print()
    print(c(f"  Output folder: {directory}\n", DIM))


if __name__ == "__main__":
    main()