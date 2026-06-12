#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         GEMMA 4 VIDEO CAPTIONER  —  LTX 2.3 LoRA Training Tool            ║
║         Uses llama.cpp + mmproj vision model via OpenAI-compatible API     ║
║         Supports: gemma-4-E4B-it-GGUF  |  gemma-4-26B-A4B-it-GGUF         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Workflow:
  1. Auto-installs all Python dependencies (openai, opencv-python, etc.)
  2. Auto-downloads llama.cpp CUDA binaries from GitHub if not found on PATH
  3. Asks for your model folder (main .gguf + mmproj .gguf)
  4. Launches llama-server as a subprocess (CUDA / RTX GPU accelerated)
  5. Asks for your video folder
  6. Extracts frames from each video, sends them to the model
  7. Saves an extremely detailed .txt caption per video — on the fly

llama-server is downloaded automatically on first run. No manual install needed.
faster-whisper is auto-installed and used to transcribe audio/dialogue from each video.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0 ── Auto-install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
import sys
import subprocess
import importlib

REQUIRED_PACKAGES = {
    "openai": "openai>=1.0.0",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "tqdm": "tqdm",
    "requests": "requests",
    "faster_whisper": "faster-whisper[cuda]",
}

def ensure_packages():
    missing = []
    for module, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print("\n[SETUP] Installing missing Python packages...")
        for pkg in missing:
            print(f"  → Installing {pkg} ...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        print("[SETUP] All packages installed.\n")

ensure_packages()

# ─────────────────────────────────────────────────────────────────────────────
#  Now safe to import everything
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import sys
import time
import glob
import base64
import shutil
import signal
import textwrap
import threading
import subprocess
from pathlib import Path
from io import BytesIO

import cv2
import requests
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from faster_whisper import WhisperModel

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (tweak if needed)
# ─────────────────────────────────────────────────────────────────────────────

# llama-server listen port (change if 8788 is busy on your machine)
LLAMA_SERVER_PORT = 8788

# GPU layers — RTX 3090 has 24 GB; -1 = offload ALL layers to GPU
N_GPU_LAYERS = -1

# Context size — large so many frames fit comfortably
N_CTX = 32768

# How many frames to extract per video
# Gemma 4 supports up to 60 seconds @ 1 FPS = 60 frames max.
# For 3–5 second clips, 12–16 frames gives rich coverage without overloading context.
FRAMES_PER_VIDEO = 16

# JPEG quality for frames sent to the model (80 is a good balance)
FRAME_JPEG_QUALITY = 82

# Frame thumbnail max dimension (pixels) — keeps base64 payload manageable
FRAME_MAX_DIM = 768

# Max tokens the model may generate for each caption
MAX_CAPTION_TOKENS = 1024

# Generation temperature — 0.35 keeps output deterministic and professional
TEMPERATURE = 0.35

# Whisper model for audio transcription
# Options: "tiny", "base", "small", "medium", "large-v3"
# "small" is a great balance of speed vs accuracy on an RTX 3090
WHISPER_MODEL_SIZE = "small"

# Supported video extensions
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
    ".flv", ".wmv", ".m4v", ".3gp", ".ts",
    ".mts", ".m2ts", ".mpeg", ".mpg", ".vob",
}

# ─────────────────────────────────────────────────────────────────────────────
#  PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are writing training captions for a video-diffusion LoRA model (LTX-Video 2.3).
Your only job is to describe what is ACTUALLY visible in the provided frames.

STRICT RULES — follow these exactly:

1. CHARACTERS (most important):
   - Describe each visible person: gender, approximate age, skin tone, hair
     (colour, length, style), face features, exact clothing worn (garment type,
     colour, fit), accessories, footwear.
   - Describe their body language, posture, facial expression, and gaze direction.
   - Describe every action or movement they perform, frame by frame.

2. DIALOGUE (second most important):
   - If a transcript is provided, you MUST quote every line verbatim in
     double quotation marks, e.g.: She says, "I'll see you tomorrow."
   - Attribute each line to the person speaking based on visible mouth movement.
   - NEVER skip or paraphrase dialogue. Quote it exactly as given.

3. ENVIRONMENT (brief — 1–2 sentences maximum):
   - State only what is clearly visible in the background: room type, outdoor
     setting, key props. Do NOT infer or invent details not visible in the frames.

WHAT YOU MUST NEVER DO:
   - Do NOT invent colour grades, film aesthetics, sepia tones, or cinematic
     styles unless they are unmistakably obvious in the actual frames.
   - Do NOT describe sounds, music, or atmosphere you cannot see evidence of.
   - Do NOT use words like "dreamlike", "vintage", "ethereal", "muted", or any
     other subjective aesthetic language unless it is literally visible.
   - Do NOT pad the caption with assumptions — only describe what you can see.

Output ONLY the caption text. Present tense, third person, flowing prose.
Target length: 120–200 words.
""").strip()

USER_PROMPT_TEMPLATE = textwrap.dedent("""
Below are {n_frames} evenly-spaced frames extracted from a short video clip
({duration:.1f} seconds long). Frame timestamps are noted.
{transcript_block}
Analyse all frames holistically and produce a single, unified, extremely
detailed description of the entire video clip as instructed.
""").strip()

# ─────────────────────────────────────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────────────────────────────────────

BANNER = r"""
  ╔═══════════════════════════════════════════════════════════╗
  ║   GEMMA 4 VIDEO CAPTIONER  —  LTX 2.3 LoRA Dataset Tool  ║
  ║   llama.cpp · CUDA · RTX GPU · Whisper Audio + Vision     ║
  ╚═══════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_banner():
    print(BANNER)


def hr(char="─", width=65):
    print(char * width)


# ─────────────────────────────────────────────────────────────────────────────
#  FFMPEG AUTO-INSTALL  (needed for audio extraction from videos)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_ffmpeg() -> str:
    """
    Check if ffmpeg is available. If not, auto-install via winget (Windows).
    Returns the full path to ffmpeg.exe, or raises RuntimeError.
    """
    # 1. Already on PATH?
    found = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if found:
        return found

    # 2. Common LM Studio / manual install locations on Windows
    common = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        str(Path.home() / "ffmpeg" / "bin" / "ffmpeg.exe"),
    ]
    # Also search WinGet install locations
    winget_base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet"
    for candidate in (winget_base / "Links", winget_base / "Packages"):
        if candidate.exists():
            for exe in candidate.rglob("ffmpeg.exe"):
                common.append(str(exe))
                break

    for p in common:
        if Path(p).exists():
            os.environ["PATH"] = str(Path(p).parent) + os.pathsep + os.environ.get("PATH", "")
            return p

    # 3. Not found — attempt silent install via winget
    print("\n  [AUTO] ffmpeg not found. Installing via winget ...")
    print("  [AUTO] This may take a minute. Please wait ...")
    try:
        result = subprocess.run(
            [
                "winget", "install",
                "--id", "Gyan.FFmpeg",
                "--silent",
                "--accept-source-agreements",
                "--accept-package-agreements",
                "--disable-interactivity",
            ],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode not in (0, -1978335189):  # 0=ok, -1978335189=already installed
            raise RuntimeError(f"winget exited with code {result.returncode}:\n{result.stderr[-500:]}")
        print("  [AUTO] ✓ ffmpeg installed via winget.")
    except FileNotFoundError:
        raise RuntimeError(
            "winget is not available and ffmpeg was not found.\n"
            "Please install ffmpeg manually and add it to PATH:\n"
            "  https://ffmpeg.org/download.html"
        )

    # 4. Re-scan after winget install (it adds to PATH in new sessions but not current)
    found = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if found:
        return found

    # Search winget links dir which is where it actually lands
    winget_links = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links"
    if winget_links.exists():
        for exe in winget_links.rglob("ffmpeg.exe"):
            ffmpeg_dir = str(exe.parent)
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            print(f"  [AUTO] ffmpeg found at: {exe}")
            return str(exe)

    # Last resort: search all of Program Files
    for search_root in [r"C:\Program Files", r"C:\Program Files (x86)", str(Path.home())]:
        for exe in Path(search_root).rglob("ffmpeg.exe") if Path(search_root).exists() else []:
            os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ.get("PATH", "")
            return str(exe)

    raise RuntimeError(
        "ffmpeg was installed but could not be located automatically.\n"
        "Please restart this script — winget should have added it to PATH."
    )

# ─────────────────────────────────────────────────────────────────────────────
#  LLAMA.CPP AUTO-DOWNLOADER  (Windows CUDA builds from GitHub Releases)
# ─────────────────────────────────────────────────────────────────────────────

# Where we install llama.cpp if it isn't found on the system
LLAMA_INSTALL_DIR = Path.home() / ".gemma4_captioner" / "llama_cpp"

# GitHub API for latest release
LLAMA_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


def _detect_cuda_major() -> int:
    """
    Try to detect the installed CUDA major version.
    Falls back to 12 (safe default for RTX 30xx / 40xx / 50xx).
    Priority: nvcc → nvidia-smi driver version heuristic → default 12.
    """
    # 1. nvcc
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], stderr=subprocess.STDOUT, timeout=8
        ).decode(errors="replace")
        m = re.search(r"release\s+(\d+)\.\d+", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass

    # 2. nvidia-smi CUDA version line
    try:
        out = subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.STDOUT, timeout=8
        ).decode(errors="replace")
        m = re.search(r"CUDA Version:\s*(\d+)\.\d+", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass

    # 3. Windows registry — CUDA toolkit version
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA",
        )
        versions = []
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(key, i)
                versions.append(sub)
                i += 1
            except OSError:
                break
        if versions:
            versions.sort(reverse=True)
            m = re.match(r"v?(\d+)", versions[0])
            if m:
                return int(m.group(1))
    except Exception:
        pass

    print("  [AUTO] Could not detect CUDA version — assuming CUDA 12 (RTX 3090 compatible)")
    return 12


def _pick_release_assets(assets: list, cuda_major: int) -> tuple[str, str]:
    """
    From a list of GitHub release asset dicts, return:
      (main_zip_url, cudart_zip_url)

    Asset naming pattern (as of 2024-2025):
      llama-bXXXX-bin-win-cuda-12.4-x64.zip       <- main binaries
      cudart-llama-bin-win-cuda-12.4-x64.zip       <- bundled CUDA runtime DLLs

    We prefer an exact cuda_major match, then fall back to any cuda build.
    """
    names = {a["name"]: a["browser_download_url"] for a in assets}

    def score_asset(name: str) -> int:
        """Higher = better match."""
        if "win" not in name.lower():
            return -1
        if "x64" not in name.lower():
            return -1
        if "cuda" not in name.lower():
            return -1
        # Match exact major version
        m = re.search(r"cuda[_-](\d+)", name.lower())
        if m and int(m.group(1)) == cuda_major:
            return 2
        return 1  # any cuda build

    # --- main zip (starts with "llama-b") ---
    main_candidates = [
        (score_asset(n), n)
        for n in names
        if n.startswith("llama-b") and n.endswith(".zip")
    ]
    main_candidates = [(s, n) for s, n in main_candidates if s > 0]
    if not main_candidates:
        raise RuntimeError(
            "No Windows CUDA zip found in the latest llama.cpp release.\n"
            "Check https://github.com/ggml-org/llama.cpp/releases manually."
        )
    main_candidates.sort(reverse=True)
    main_zip_url = names[main_candidates[0][1]]
    main_zip_name = main_candidates[0][1]

    # --- cudart zip (optional, bundles CUDA .dlls so we don't need the toolkit) ---
    # Matches the same cuda version as the chosen main zip
    m = re.search(r"cuda[_-]([\d.]+)", main_zip_name.lower())
    cuda_ver_str = m.group(1) if m else str(cuda_major)

    cudart_candidates = [
        n for n in names
        if n.startswith("cudart-llama") and n.endswith(".zip")
        and cuda_ver_str in n.lower()
    ]
    # Fallback: any cudart zip
    if not cudart_candidates:
        cudart_candidates = [
            n for n in names
            if n.startswith("cudart-llama") and n.endswith(".zip")
        ]

    cudart_zip_url = names[cudart_candidates[0]] if cudart_candidates else ""

    return main_zip_url, cudart_zip_url


def _download_with_progress(url: str, dest: Path, label: str) -> None:
    """Download url → dest with a tqdm progress bar."""
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f, tqdm(
        desc=f"  {label}",
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        ncols=70,
        leave=True,
    ) as bar:
        for chunk in r.iter_content(chunk_size=1 << 17):
            f.write(chunk)
            bar.update(len(chunk))


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip archive, flattening one top-level folder if present."""
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        # Detect a single top-level folder
        top_dirs = {m.split("/")[0] for m in members if "/" in m}
        single_top = len(top_dirs) == 1 and list(top_dirs)[0]
        for member in members:
            if single_top and member.startswith(single_top + "/"):
                # Strip the top-level folder name
                rel = member[len(single_top) + 1:]
            else:
                rel = member
            if not rel:
                continue
            target = dest_dir / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())


def find_or_download_llama_server() -> str:
    """
    1. Search PATH and well-known install dirs.
    2. If not found, auto-download the latest CUDA build from GitHub Releases.
    3. Return the full path to llama-server.exe.
    """
    candidates  = ["llama-server.exe", "llama-server"]
    extra_dirs  = [
        str(LLAMA_INSTALL_DIR),                            # our own install
        r"C:\Program Files\llama.cpp",                     # winget default
        r"C:\llama.cpp",
        str(Path.home() / "llama.cpp"),
        str(Path.home() / "llama.cpp" / "build" / "bin"),
        str(Path.home() / "llama.cpp" / "build" / "bin" / "Release"),
    ]

    def _search() -> str:
        for name in candidates:
            found = shutil.which(name)
            if found:
                return found
        for d in extra_dirs:
            for name in candidates:
                p = Path(d) / name
                if p.exists():
                    return str(p)
        return ""

    found = _search()
    if found:
        return found

    # ── Not found — auto-download ────────────────────────────────────────────
    print("\n  [AUTO] llama-server not found on this system.")
    print("  [AUTO] Downloading the latest CUDA build from GitHub Releases ...")
    print(f"  [AUTO] Install location: {LLAMA_INSTALL_DIR}\n")

    cuda_major = _detect_cuda_major()
    print(f"  [AUTO] Detected CUDA major version: {cuda_major}")

    # Fetch release metadata
    print("  [AUTO] Fetching release metadata from GitHub ...")
    try:
        resp = requests.get(
            LLAMA_RELEASES_API,
            headers={"Accept": "application/vnd.github+json"},
            timeout=20,
        )
        resp.raise_for_status()
        release = resp.json()
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch llama.cpp release info from GitHub: {e}\n"
            "Check your internet connection and try again."
        )

    tag      = release.get("tag_name", "unknown")
    assets   = release.get("assets", [])
    print(f"  [AUTO] Latest release: {tag}  ({len(assets)} assets)")

    main_url, cudart_url = _pick_release_assets(assets, cuda_major)

    main_zip_name   = main_url.split("/")[-1]
    cudart_zip_name = cudart_url.split("/")[-1] if cudart_url else ""

    print(f"  [AUTO] Downloading: {main_zip_name}")
    tmp_dir   = LLAMA_INSTALL_DIR / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    main_zip  = tmp_dir / main_zip_name

    _download_with_progress(main_url, main_zip, main_zip_name)

    # Also grab cudart zip (contains CUDA runtime DLLs — avoids "DLL not found" errors)
    if cudart_url:
        print(f"  [AUTO] Downloading CUDA runtime DLLs: {cudart_zip_name}")
        cudart_zip = tmp_dir / cudart_zip_name
        _download_with_progress(cudart_url, cudart_zip, cudart_zip_name)
    else:
        cudart_zip = None

    # Extract
    print(f"\n  [AUTO] Extracting to {LLAMA_INSTALL_DIR} ...")
    LLAMA_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    _extract_zip(main_zip, LLAMA_INSTALL_DIR)
    if cudart_zip and cudart_zip.exists():
        _extract_zip(cudart_zip, LLAMA_INSTALL_DIR)

    # Clean up tmp zips
    try:
        import shutil as _sh
        _sh.rmtree(tmp_dir)
    except Exception:
        pass

    # Inject into PATH so shutil.which() and subprocess can find it
    os.environ["PATH"] = str(LLAMA_INSTALL_DIR) + os.pathsep + os.environ.get("PATH", "")

    # Try finding it now
    found = _search()
    if not found:
        # Last resort: scan extracted dir for the exe
        for exe in LLAMA_INSTALL_DIR.rglob("llama-server.exe"):
            found = str(exe)
            break
        if not found:
            for exe in LLAMA_INSTALL_DIR.rglob("llama-server"):
                found = str(exe)
                break

    if not found:
        raise RuntimeError(
            f"Extraction completed but llama-server.exe was not found inside:\n"
            f"  {LLAMA_INSTALL_DIR}\n"
            "Please check the folder manually."
        )

    print(f"  [AUTO] ✓ llama-server ready: {found}\n")
    return found


def browse_for_file(prompt_text: str, must_end_with: tuple = ()) -> str:
    """Simple text input; could be extended with tkinter if desired."""
    while True:
        path = input(f"\n{prompt_text}\n  > ").strip().strip('"').strip("'")
        if not path:
            print("  [!] Path cannot be empty. Try again.")
            continue
        p = Path(path)
        if not p.exists():
            print(f"  [!] Path not found: {path}")
            continue
        if must_end_with and p.is_file():
            if not path.lower().endswith(must_end_with):
                print(f"  [!] File must end with one of: {must_end_with}")
                continue
        return str(p)


def find_gguf_in_dir(model_dir: str) -> tuple[str, str]:
    """
    Scan model_dir for:
      - main model .gguf  (largest non-mmproj file)
      - mmproj .gguf      (file with 'mmproj' in the name)
    Returns (model_path, mmproj_path) or raises FileNotFoundError.
    """
    model_dir = Path(model_dir)
    all_ggufs = list(model_dir.glob("*.gguf"))
    if not all_ggufs:
        raise FileNotFoundError(f"No .gguf files found in: {model_dir}")

    mmproj_files = [f for f in all_ggufs if "mmproj" in f.name.lower()]
    main_files   = [f for f in all_ggufs if "mmproj" not in f.name.lower()]

    if not mmproj_files:
        raise FileNotFoundError(
            "No mmproj .gguf file found in the model directory.\n"
            "  Expected a file with 'mmproj' in its name, e.g.:\n"
            "    mmproj-BF16.gguf  /  mmproj-F16.gguf"
        )
    if not main_files:
        raise FileNotFoundError("No main (non-mmproj) .gguf file found.")

    # If multiple main .gguf files, let user pick
    if len(main_files) == 1:
        main_gguf = main_files[0]
    else:
        print("\n  Multiple main .gguf files found. Please choose one:")
        for i, f in enumerate(main_files):
            size_gb = f.stat().st_size / 1e9
            print(f"    [{i+1}] {f.name}  ({size_gb:.1f} GB)")
        while True:
            choice = input("  Enter number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(main_files):
                main_gguf = main_files[int(choice) - 1]
                break
            print("  Invalid choice.")

    # Same for mmproj
    if len(mmproj_files) == 1:
        mmproj_gguf = mmproj_files[0]
    else:
        print("\n  Multiple mmproj files found. Please choose one:")
        for i, f in enumerate(mmproj_files):
            print(f"    [{i+1}] {f.name}")
        while True:
            choice = input("  Enter number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(mmproj_files):
                mmproj_gguf = mmproj_files[int(choice) - 1]
                break
            print("  Invalid choice.")

    return str(main_gguf), str(mmproj_gguf)


# ─────────────────────────────────────────────────────────────────────────────
#  FRAME EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, n_frames: int = FRAMES_PER_VIDEO) -> tuple[list, float]:
    """
    Extract n_frames evenly-spaced frames from video_path.
    Returns (list_of_pil_images, duration_seconds).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 24.0
    duration     = total_frames / fps

    # Build evenly-spaced frame indices
    if total_frames <= n_frames:
        indices = list(range(total_frames))
    else:
        step = total_frames / n_frames
        indices = [int(i * step) for i in range(n_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb)

        # Resize so the longest side ≤ FRAME_MAX_DIM
        w, h = pil_img.size
        if max(w, h) > FRAME_MAX_DIM:
            ratio = FRAME_MAX_DIM / max(w, h)
            pil_img = pil_img.resize(
                (int(w * ratio), int(h * ratio)), Image.LANCZOS
            )
        frames.append(pil_img)

    cap.release()
    return frames, duration


def pil_to_base64(img: Image.Image, quality: int = FRAME_JPEG_QUALITY) -> str:
    """Encode a PIL image to a base64 JPEG data URI."""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ─────────────────────────────────────────────────────────────────────────────
#  LLAMA-SERVER  MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

_server_proc: subprocess.Popen | None = None
_whisper_model: object | None = None


def start_llama_server(
    llama_server_bin: str,
    model_path: str,
    mmproj_path: str,
    port: int = LLAMA_SERVER_PORT,
    n_gpu_layers: int = N_GPU_LAYERS,
    n_ctx: int = N_CTX,
) -> None:
    global _server_proc

    cmd = [
        llama_server_bin,
        "--model",        model_path,
        "--mmproj",       mmproj_path,
        "--port",         str(port),
        "--host",         "127.0.0.1",
        "--n-gpu-layers", str(n_gpu_layers),
        "--ctx-size",     str(n_ctx),
        "--temp",         str(TEMPERATURE),
        "--top-p",        "0.95",
        "--top-k",        "64",
        # Disable thinking/reasoning mode — cleaner captions, no <think> blocks
        "--reasoning", "off",
        # Flash attention — use auto mode (boolean flag, no value)
        "--flash-attn", "on",
        # Suppress verbose server logs
        "--log-verbosity", "-1",
    ]

    print(f"\n[SERVER] Launching llama-server on port {port} ...")
    print(f"  Model  : {Path(model_path).name}")
    print(f"  mmproj : {Path(mmproj_path).name}")
    print(f"  GPU layers: {'all' if n_gpu_layers == -1 else n_gpu_layers}")
    print(f"  Context: {n_ctx} tokens")

    # Redirect server stdout/stderr to a log file so it doesn't clutter our UI
    log_path = Path(model_path).parent / "llama_server.log"
    log_file = open(log_path, "w", encoding="utf-8")

    _server_proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        # Windows: CREATE_NEW_PROCESS_GROUP so Ctrl+C doesn't kill our script
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    # Wait until the server is ready (poll /health)
    base_url = f"http://127.0.0.1:{port}"
    print(f"  Waiting for server to be ready ", end="", flush=True)
    deadline = time.time() + 180  # 3-minute timeout for large models
    while time.time() < deadline:
        if _server_proc.poll() is not None:
            log_file.close()
            with open(log_path) as lf:
                tail = lf.read()[-3000:]
            raise RuntimeError(
                f"llama-server exited unexpectedly (code {_server_proc.returncode}).\n"
                f"Last log:\n{tail}"
            )
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print(" ✓ Ready!")
                return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(2)

    raise TimeoutError("llama-server did not become ready within 3 minutes.")


def stop_llama_server():
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        print("\n[SERVER] Shutting down llama-server ...")
        if sys.platform == "win32":
            _server_proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            _server_proc.terminate()
        try:
            _server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
        print("[SERVER] Server stopped.")
    _server_proc = None



# ─────────────────────────────────────────────────────────────────────────────
#  AUDIO TRANSCRIPTION  (faster-whisper → runs on RTX 3090 via CUDA)
# ─────────────────────────────────────────────────────────────────────────────

def check_ffmpeg() -> bool:
    """Return True if ffmpeg is on PATH, print a clear message if not."""
    if shutil.which("ffmpeg"):
        return True
    print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║  ffmpeg NOT FOUND — audio transcription will be DISABLED    ║
  ║                                                              ║
  ║  To enable dialogue/audio detection in your captions:        ║
  ║    1. Download from: https://www.gyan.dev/ffmpeg/builds/    ║
  ║       (grab the "release full" build)                        ║
  ║    2. Extract and add the /bin/ folder to your PATH          ║
  ║    3. Re-run this script                                     ║
  ╚══════════════════════════════════════════════════════════════╝""")
    return False


_FFMPEG_AVAILABLE: bool = check_ffmpeg()


def load_whisper_model() -> None:
    """Load the faster-whisper model once, onto the GPU."""
    global _whisper_model
    if _whisper_model is not None:
        return
    print(f"  [WHISPER] Loading Whisper '{WHISPER_MODEL_SIZE}' model on CUDA ...", end=" ", flush=True)
    try:
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cuda",
            compute_type="float16",   # FP16 is fine on RTX 3090
        )
        print("✓")
    except Exception as e:
        # Fall back to CPU if CUDA load fails (unlikely on RTX 3090)
        print(f"\n  [WHISPER] CUDA load failed ({e}), falling back to CPU ...")
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        print("  [WHISPER] Loaded on CPU.")


def transcribe_video_audio(video_path: str) -> str:
    """
    Extract audio from video_path and transcribe it with faster-whisper.
    Returns a formatted transcript string, or empty string if no speech detected.

    The transcript preserves:
    - Speaker segments with timestamps
    - All spoken words exactly as detected
    """
    global _whisper_model
    if _whisper_model is None:
        load_whisper_model()

    import tempfile
    tmp_wav = None
    try:
        # Extract audio to a temporary 16kHz mono WAV (Whisper's native format)
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()

        ffmpeg_exe = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe") or "ffmpeg"
        ffmpeg_cmd = [
            ffmpeg_exe, "-y",
            "-i", video_path,
            "-ar", "16000",        # 16kHz sample rate
            "-ac", "1",            # mono
            "-vn",                 # no video
            "-f", "wav",
            tmp_wav.name,
        ]
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        if result.returncode != 0:
            return ""   # ffmpeg failed (no audio track, etc.)

        # Transcribe
        segments, info = _whisper_model.transcribe(
            tmp_wav.name,
            beam_size=5,
            language=None,          # auto-detect language
            vad_filter=True,        # skip silence
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        lines = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                lines.append(f'[{seg.start:.1f}s] "{text}"')

        return "\n".join(lines)

    except FileNotFoundError:
        # ffmpeg not on PATH — already warned at startup
        return ""
    except Exception as _exc:
        # Unexpected error — don't crash the whole run, just skip audio
        print(f" [audio err: {_exc}]", end="")
        return ""
    finally:
        if tmp_wav and os.path.exists(tmp_wav.name):
            try:
                os.unlink(tmp_wav.name)
            except Exception:
                pass

# ─────────────────────────────────────────────────────────────────────────────
#  CAPTION GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_caption(
    client: OpenAI,
    frames: list,
    duration: float,
    video_name: str,
    transcript: str = "",
) -> str:
    """
    Send frames (+ optional Whisper transcript) to llama-server for captioning.
    Each frame is a PIL Image.
    """
    n_frames = len(frames)

    # Format the transcript block that gets injected into the prompt
    if transcript.strip():
        transcript_block = (
            "\n\nAUDIO TRANSCRIPT (from Whisper speech recognition — "
            "verbatim, timestamped):\n" + transcript +
            "\n\nIMPORTANT: The transcript above is ground-truth dialogue. "
            "You MUST incorporate every spoken line into your caption, "
            'enclosed in double quotation marks, attributed to the visible speaker.'
        )
    else:
        transcript_block = ""

    # Build the content list: text prompt + all frames as image_url blocks
    content = [
        {
            "type": "text",
            "text": USER_PROMPT_TEMPLATE.format(
                n_frames=n_frames,
                duration=duration,
                transcript_block=transcript_block,
            ),
        }
    ]

    # Calculate per-frame timestamp labels
    time_step = duration / max(n_frames - 1, 1)
    for i, frame in enumerate(frames):
        ts = i * time_step
        content.append({
            "type": "text",
            "text": f"[Frame {i+1}/{n_frames} — {ts:.2f}s]",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": pil_to_base64(frame),
            },
        })

    # Final instruction reinforcement
    content.append({
        "type": "text",
        "text": (
            "Describe this video clip accurately based ONLY on what you can see in "
            "the frames above. Focus on: (1) each character's appearance and exact "
            "actions, (2) any dialogue — quote every word verbatim in double quotes, "
            "(3) one brief sentence about the visible environment. "
            "Do NOT invent colour grades, moods, or sounds. Flowing prose only."
        ),
    })

    response = client.chat.completions.create(
        model="gemma4",          # model alias — llama-server ignores the name
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
        max_tokens=MAX_CAPTION_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
    )

    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  VIDEO DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_videos(video_dir: str) -> list[Path]:
    """Find all supported video files in video_dir (non-recursive by default)."""
    video_dir = Path(video_dir)
    found = []
    for ext in VIDEO_EXTENSIONS:
        # case-insensitive on Windows
        found.extend(video_dir.glob(f"*{ext}"))
        found.extend(video_dir.glob(f"*{ext.upper()}"))
    # Deduplicate and sort
    found = sorted(set(found))
    return found


# ─────────────────────────────────────────────────────────────────────────────
#  CAPTION FILE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def caption_path_for(video_path: Path) -> Path:
    return video_path.with_suffix(".txt")


def already_captioned(video_path: Path) -> bool:
    cp = caption_path_for(video_path)
    return cp.exists() and cp.stat().st_size > 0


def save_caption(video_path: Path, caption: str, trigger_word: str = "") -> Path:
    out = caption_path_for(video_path)
    final = f"{trigger_word}, {caption}" if trigger_word else caption
    out.write_text(final, encoding="utf-8")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print_banner()

    # ── 0. Ensure ffmpeg is available (needed for audio extraction) ──────────
    try:
        ffmpeg_bin = ensure_ffmpeg()
        print(f"  ✓ ffmpeg       : {ffmpeg_bin}")
    except Exception as e:
        print(f"\n  [WARNING] ffmpeg not available: {e}")
        print("  Audio transcription will be skipped for all videos.")
        ffmpeg_bin = None

    # ── 1. Locate (or auto-download) llama-server binary ─────────────────────
    hr()
    print("STEP 1 — Locating llama-server (CUDA build)")
    hr()
    try:
        llama_server_bin = find_or_download_llama_server()
        print(f"  ✓ llama-server : {llama_server_bin}")
    except Exception as e:
        print(f"\n  [ERROR] Could not obtain llama-server:\n  {e}")
        sys.exit(1)

    # ── 2. Model directory ───────────────────────────────────────────────────
    hr()
    print("STEP 2 — Gemma 4 model directory")
    hr()
    print(
        "  Point to the folder that contains BOTH files:\n"
        "    • The main .gguf   (e.g. gemma-4-E4B-it-Q4_K_M.gguf)\n"
        "    • The mmproj .gguf (e.g. mmproj-BF16.gguf)\n"
        "  Supported models: gemma-4-E4B-it-GGUF  |  gemma-4-26B-A4B-it-GGUF\n"
        "  (You can download them from: https://huggingface.co/ggml-org)"
    )

    while True:
        model_dir = input("\n  Model directory path:\n  > ").strip().strip('"').strip("'")
        if not model_dir or not Path(model_dir).is_dir():
            print("  [!] Not a valid directory. Try again.")
            continue
        try:
            main_gguf, mmproj_gguf = find_gguf_in_dir(model_dir)
            print(f"\n  ✓ Main model : {Path(main_gguf).name}")
            print(f"  ✓ mmproj     : {Path(mmproj_gguf).name}")
            break
        except FileNotFoundError as e:
            print(f"  [!] {e}")

    # ── 3. Video folder + Whisper transcription pass (before llama-server) ────
    hr()
    print("STEP 3 — Video dataset folder")
    hr()
    supported = ", ".join(sorted(VIDEO_EXTENSIONS))
    print(f"  Supported formats: {supported}")

    while True:
        video_dir = input("\n  Video folder path:\n  > ").strip().strip('"').strip("'")
        if not video_dir or not Path(video_dir).is_dir():
            print("  [!] Not a valid directory.")
            continue
        videos = discover_videos(video_dir)
        if not videos:
            print(f"  [!] No supported video files found in: {video_dir}")
            continue
        print(f"\n  ✓ Found {len(videos)} video(s) in: {video_dir}")
        break

    # ── Trigger word ─────────────────────────────────────────────────────────
    hr()
    print("STEP 4b — Trigger word (optional)")
    hr()
    print(
        "  A trigger word is prepended to every caption, e.g. 'alitastyle'\n"
        "  This trains the LoRA to activate on that keyword.\n"
        "  Press Enter to skip."
    )
    trigger_word = input("\n  Trigger word (or press Enter to skip):\n  > ").strip()
    if trigger_word:
        print(f"  ✓ Trigger word set: '{trigger_word}' — will be prepended to all captions.")
    else:
        print("  ⊘ No trigger word — captions will start directly with the description.")

    # Check for already-captioned files
    pending   = [v for v in videos if not already_captioned(v)]
    completed = len(videos) - len(pending)
    if completed:
        print(f"  ℹ  {completed} video(s) already have .txt captions — skipping.")

    if not pending:
        print("\n  All videos already captioned! Nothing to do.")
        return

    # ── Whisper pass: transcribe ALL pending videos now, before Gemma loads ──
    # This frees GPU VRAM completely before llama-server claims it.
    transcripts: dict[Path, str] = {}
    if _FFMPEG_AVAILABLE:
        hr()
        print(f"STEP 3b — Audio transcription pass ({len(pending)} video(s))")
        hr()
        print(f"  Whisper model : {WHISPER_MODEL_SIZE}  |  Device: CUDA (RTX 3090)")
        print(f"  Running Whisper BEFORE loading Gemma to avoid VRAM conflicts\n")
        load_whisper_model()
        for i, video_path in enumerate(pending, 1):
            print(f"  [{i}/{len(pending)}] {video_path.name} ... ", end="", flush=True)
            t = transcribe_video_audio(str(video_path))
            transcripts[video_path] = t
            if t:
                n_segs = t.count("\n") + 1
                print(f"✓ {n_segs} segment(s)")
                # Print first detected line as preview
                first_line = t.split("\n")[0]
                print(f"           {first_line}")
            else:
                print("– no speech")
        # Release Whisper from VRAM before Gemma loads
        global _whisper_model
        del _whisper_model
        _whisper_model = None
        import gc, torch
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        print(f"\n  [WHISPER] VRAM released. Starting Gemma now ...\n")
    else:
        for vp in pending:
            transcripts[vp] = ""

    # ── 4. Start llama-server (Gemma) ────────────────────────────────────────
    hr()
    print("STEP 4 — Starting llama-server (Gemma 4 E4B · CUDA · RTX 3090)")
    hr()
    try:
        start_llama_server(
            llama_server_bin,
            main_gguf,
            mmproj_gguf,
            port=LLAMA_SERVER_PORT,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
        )
    except Exception as e:
        print(f"\n  [ERROR] Failed to start llama-server:\n  {e}")
        sys.exit(1)

    # OpenAI client pointing at our local server
    client = OpenAI(
        base_url=f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1",
        api_key="not-needed",
    )

    # ── 5. Caption loop ──────────────────────────────────────────────────────
    hr()
    print(f"STEP 5 — Captioning {len(pending)} video(s)\n")
    hr()
    print(
        f"  Frames per video : {FRAMES_PER_VIDEO}\n"
        f"  Max frame dim    : {FRAME_MAX_DIM}px\n"
        f"  Max caption len  : {MAX_CAPTION_TOKENS} tokens\n"
        f"  Audio            : {'Whisper ' + WHISPER_MODEL_SIZE + ' (pre-transcribed)' if _FFMPEG_AVAILABLE else 'disabled (ffmpeg not found)'}\n"
        f"  Trigger word     : {trigger_word if trigger_word else '(none)'}\n"
        f"  Output           : .txt file next to each video (saved on the fly)\n"
    )

    success_count = 0
    fail_count    = 0

    for i, video_path in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] {video_path.name}")
        hr("·")

        try:
            # — Extract frames —
            print(f"  Extracting {FRAMES_PER_VIDEO} frames ...", end=" ", flush=True)
            frames, duration = extract_frames(str(video_path), FRAMES_PER_VIDEO)
            actual_frames = len(frames)
            print(f"✓ ({actual_frames} frames, {duration:.1f}s)")

            # — Transcribe audio with Whisper —
            # Retrieve pre-computed Whisper transcript for this video
            transcript = transcripts.get(video_path, "")
            if transcript:
                n_segs = transcript.count("\n") + 1
                print(f"  Audio: {n_segs} transcribed segment(s) → injecting into prompt")
            elif _FFMPEG_AVAILABLE:
                print(f"  Audio: no speech detected")

            # — Generate caption —
            print(f"  Generating caption  ...", end=" ", flush=True)
            t0      = time.time()
            caption = generate_caption(client, frames, duration, video_path.name, transcript)
            elapsed = time.time() - t0
            words   = len(caption.split())
            print(f"✓ ({words} words, {elapsed:.1f}s)")

            # — Save caption immediately —
            out_path = save_caption(video_path, caption, trigger_word)
            print(f"  Saved → {out_path.name}")

            # Print a short preview
            preview = " ".join(caption.split()[:30])
            print(f"  Preview: {preview}…")

            success_count += 1

        except KeyboardInterrupt:
            print("\n\n  [!] Interrupted by user.")
            break
        except Exception as e:
            fail_count += 1
            print(f"  [ERROR] {e}")
            # Write a placeholder so we know this failed
            error_txt = caption_path_for(video_path)
            error_txt.write_text(
                f"[CAPTIONING FAILED: {e}]", encoding="utf-8"
            )
            print(f"  Error note saved → {error_txt.name}")

    # ── 6. Summary ───────────────────────────────────────────────────────────
    hr("═")
    print("DONE")
    hr("═")
    print(f"  ✓ Successfully captioned : {success_count}")
    print(f"  ✗ Failed                 : {fail_count}")
    print(f"  ⊘ Skipped (existing)     : {completed}")
    print(f"\n  All .txt files are in: {video_dir}")

    stop_llama_server()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted. Shutting down...")
        stop_llama_server()
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        stop_llama_server()
        sys.exit(1)