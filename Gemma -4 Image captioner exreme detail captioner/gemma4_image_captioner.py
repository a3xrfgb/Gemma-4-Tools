#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          GEMMA 4 IMAGE CAPTIONER  —  Extreme-Detail Vision Tool             ║
║          Uses llama.cpp + mmproj vision model via OpenAI-compatible API     ║
║          Supports: gemma-4-E4B-it-GGUF  |  gemma-4-26B-A4B-it-GGUF          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Workflow:
  1. Auto-installs all Python dependencies (openai, Pillow, pillow-heif, etc.)
  2. Auto-downloads llama.cpp CUDA binaries from GitHub if not found on PATH
  3. Asks for your model folder (main .gguf + mmproj .gguf)
  4. Launches llama-server as a subprocess (CUDA / RTX GPU accelerated)
  5. Asks for a single image OR a folder of images
  6. Sends each image to the model and writes an extremely detailed .txt caption

Supports virtually any image format:
  JPG, JPEG, PNG, WEBP, BMP, TIFF, GIF, HEIC, HEIF, AVIF, ICO, PPM, TGA, JP2,
  and (via Pillow) most of what your system can decode.

llama-server is downloaded automatically on first run. No manual install needed.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0 ── Auto-install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
import sys
import subprocess
import importlib

REQUIRED_PACKAGES = {
    "openai":      "openai>=1.0.0",
    "PIL":         "Pillow",
    "pillow_heif": "pillow-heif",       # HEIC / HEIF / AVIF support
    "tqdm":        "tqdm",
    "requests":    "requests",
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
import time
import base64
import shutil
import signal
import textwrap
import subprocess
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image, ImageOps
from tqdm import tqdm
from openai import OpenAI

# Register HEIC/HEIF/AVIF opener with Pillow (iPhone photos, modern Android, etc.)
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    try:
        # AVIF support (newer pillow-heif builds)
        pillow_heif.register_avif_opener()
    except Exception:
        pass
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (tweak if needed)
# ─────────────────────────────────────────────────────────────────────────────

# llama-server listen port (change if 8788 is busy on your machine)
LLAMA_SERVER_PORT = 8788

# GPU layers — RTX 3090 has 24 GB; -1 = offload ALL layers to GPU
N_GPU_LAYERS = -1

# Context size — generous so the image + long output fit comfortably
N_CTX = 16384

# JPEG quality for the image sent to the model (90 is high for single-image work)
IMAGE_JPEG_QUALITY = 92

# Max long-edge of image sent to the model.
# Gemma 3/4 vision encoders work best around 768–1024px on the long edge.
# Larger = more detail but slower and more VRAM.
IMAGE_MAX_DIM = 1024

# Max tokens the model may generate for each caption.
# 1500 gives room for *very* long, exhaustive descriptions.
MAX_CAPTION_TOKENS = 1500

# Generation temperature — 0.4 keeps output grounded but not robotic
TEMPERATURE = 0.4

# Supported image extensions (lowercase, with dot)
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".jpe", ".jfif",
    ".png", ".apng",
    ".webp",
    ".bmp", ".dib",
    ".tif", ".tiff",
    ".gif",
    ".heic", ".heif",
    ".avif",
    ".ico",
    ".ppm", ".pgm", ".pbm", ".pnm",
    ".tga",
    ".jp2", ".j2k", ".jpx",
}

# ─────────────────────────────────────────────────────────────────────────────
#  PROMPTS — engineered for extreme detail on a single still image
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert image describer. Your task is to produce an EXTREMELY DETAILED
description of a single still image. Describe ONLY what is visible. Do not
invent, infer, or guess details that are not actually present in the image.

Cover, in this order, every aspect that is visible:

1. SUBJECT(S) — the main focus of the image:
   - For PEOPLE: gender presentation, approximate age range, skin tone, hair
     (color, length, texture, style), facial features, expression, gaze
     direction, posture, body language, every garment (type, color, fabric,
     fit, pattern, condition), every accessory, jewelry, footwear, makeup,
     tattoos, scars or distinguishing marks. Describe what they appear to be
     doing.
   - For ANIMALS: species or breed, color, markings, size, pose, expression,
     visible behavior.
   - For OBJECTS / PRODUCTS: type, material, color, finish, brand markings if
     legible, condition, scale, orientation. Be exhaustive about a hero
     product — describe every face, edge, surface, reflection, and texture.

2. SECONDARY ELEMENTS — anything else in the frame:
   - Other people, objects, props, vehicles, plants, food, text, signs,
     packaging, screens, etc. Describe each one with the same care as the
     main subject if it is clearly visible.

3. SETTING / ENVIRONMENT:
   - Indoor or outdoor; specific location type (kitchen, studio, forest,
     city street, beach, etc.); architectural details; furniture; floor
     and wall surfaces; visible weather, time of day, season — only if
     evidence is actually visible.

4. COMPOSITION & FRAMING:
   - Shot type (close-up, medium, wide, overhead, low angle, eye-level),
     where the subject sits in the frame, foreground/midground/background,
     depth of field (sharp vs. blurred areas), any leading lines or symmetry.

5. LIGHTING:
   - Direction (front, side, back, top), quality (hard / soft / diffused),
     color temperature (warm, cool, neutral, mixed), visible light sources,
     shadows, highlights, reflections, specular detail.

6. COLOR & TEXTURE:
   - Dominant palette, accent colors, contrast level, any obvious color
     grading or filter effects that are unmistakably present, and the
     texture of every major surface (smooth, rough, glossy, matte, woven,
     metallic, glass, etc.).

7. TEXT — if any text or numbers are legible anywhere in the image, quote
   them verbatim inside double quotation marks and state where they appear.

STRICT RULES:
   - Describe ONLY what is visible. If you are unsure, do not include it.
   - Do NOT invent moods, emotions, stories, or backstory.
   - Do NOT use vague aesthetic words like "beautiful", "stunning",
     "ethereal", "dreamlike", "vintage", "cinematic" unless that quality is
     literally and unmistakably present (e.g. visible film grain, obvious
     duotone). Even then, describe the concrete visual cause, not just the
     label.
   - Do NOT describe sounds, smells, or anything you cannot see.
   - Do NOT add a closing summary or interpretation. Just describe.

OUTPUT FORMAT:
   - Flowing prose, present tense, third person.
   - Multiple paragraphs are fine for very dense images.
   - No bullet lists, no headers, no markdown. Plain text only.
   - Aim for 250–500 words for typical images, more if the image is dense.
""").strip()

USER_PROMPT = textwrap.dedent("""
Below is a single still image. Analyse it carefully and produce one unified,
extremely detailed description following every rule in the system prompt.
Describe absolutely everything you can see — subjects, clothing, objects,
text, environment, composition, lighting, colors, textures — but never
invent details that are not visibly present.
""").strip()

# ─────────────────────────────────────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────────────────────────────────────

BANNER = r"""
  ╔═══════════════════════════════════════════════════════════╗
  ║   GEMMA 4 IMAGE CAPTIONER  —  Extreme-Detail Vision Tool ║
  ║   llama.cpp · CUDA · RTX GPU · mmproj vision encoder      ║
  ╚═══════════════════════════════════════════════════════════╝
"""


def print_banner():
    print(BANNER)


def hr(char="─", width=65):
    print(char * width)


# ─────────────────────────────────────────────────────────────────────────────
#  LLAMA.CPP AUTO-DOWNLOADER  (Windows CUDA builds from GitHub Releases)
# ─────────────────────────────────────────────────────────────────────────────

LLAMA_INSTALL_DIR = Path.home() / ".gemma4_captioner" / "llama_cpp"
LLAMA_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


def _detect_cuda_major() -> int:
    """
    Try to detect installed CUDA major version.
    Falls back to 12 (safe default for RTX 30xx / 40xx / 50xx).
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

    # 2. nvidia-smi
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

    print("  [AUTO] Could not detect CUDA version — assuming CUDA 12 (RTX 30xx/40xx/50xx)")
    return 12


def _pick_release_assets(assets: list, cuda_major: int) -> tuple[str, str]:
    """
    From a list of GitHub release asset dicts, return:
      (main_zip_url, cudart_zip_url)

    Asset naming pattern (as of 2024-2026):
      llama-bXXXX-bin-win-cuda-12.4-x64.zip       <- main binaries
      cudart-llama-bin-win-cuda-12.4-x64.zip      <- bundled CUDA runtime DLLs
    """
    names = {a["name"]: a["browser_download_url"] for a in assets}

    def score_asset(name: str) -> int:
        if "win" not in name.lower():
            return -1
        if "x64" not in name.lower():
            return -1
        if "cuda" not in name.lower():
            return -1
        m = re.search(r"cuda[_-](\d+)", name.lower())
        if m and int(m.group(1)) == cuda_major:
            return 2
        return 1

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
    main_zip_url  = names[main_candidates[0][1]]
    main_zip_name = main_candidates[0][1]

    m = re.search(r"cuda[_-]([\d.]+)", main_zip_name.lower())
    cuda_ver_str = m.group(1) if m else str(cuda_major)

    cudart_candidates = [
        n for n in names
        if n.startswith("cudart-llama") and n.endswith(".zip")
        and cuda_ver_str in n.lower()
    ]
    if not cudart_candidates:
        cudart_candidates = [
            n for n in names
            if n.startswith("cudart-llama") and n.endswith(".zip")
        ]

    cudart_zip_url = names[cudart_candidates[0]] if cudart_candidates else ""
    return main_zip_url, cudart_zip_url


def _download_with_progress(url: str, dest: Path, label: str) -> None:
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
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        top_dirs = {m.split("/")[0] for m in members if "/" in m}
        single_top = len(top_dirs) == 1 and list(top_dirs)[0]
        for member in members:
            if single_top and member.startswith(single_top + "/"):
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
        str(LLAMA_INSTALL_DIR),
        r"C:\Program Files\llama.cpp",
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

    print("\n  [AUTO] llama-server not found on this system.")
    print("  [AUTO] Downloading the latest CUDA build from GitHub Releases ...")
    print(f"  [AUTO] Install location: {LLAMA_INSTALL_DIR}\n")

    cuda_major = _detect_cuda_major()
    print(f"  [AUTO] Detected CUDA major version: {cuda_major}")

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

    tag    = release.get("tag_name", "unknown")
    assets = release.get("assets", [])
    print(f"  [AUTO] Latest release: {tag}  ({len(assets)} assets)")

    main_url, cudart_url = _pick_release_assets(assets, cuda_major)

    main_zip_name   = main_url.split("/")[-1]
    cudart_zip_name = cudart_url.split("/")[-1] if cudart_url else ""

    print(f"  [AUTO] Downloading: {main_zip_name}")
    tmp_dir   = LLAMA_INSTALL_DIR / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    main_zip  = tmp_dir / main_zip_name

    _download_with_progress(main_url, main_zip, main_zip_name)

    if cudart_url:
        print(f"  [AUTO] Downloading CUDA runtime DLLs: {cudart_zip_name}")
        cudart_zip = tmp_dir / cudart_zip_name
        _download_with_progress(cudart_url, cudart_zip, cudart_zip_name)
    else:
        cudart_zip = None

    print(f"\n  [AUTO] Extracting to {LLAMA_INSTALL_DIR} ...")
    LLAMA_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    _extract_zip(main_zip, LLAMA_INSTALL_DIR)
    if cudart_zip and cudart_zip.exists():
        _extract_zip(cudart_zip, LLAMA_INSTALL_DIR)

    try:
        import shutil as _sh
        _sh.rmtree(tmp_dir)
    except Exception:
        pass

    os.environ["PATH"] = str(LLAMA_INSTALL_DIR) + os.pathsep + os.environ.get("PATH", "")

    found = _search()
    if not found:
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


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL DIR DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def find_gguf_in_dir(model_dir: str) -> tuple[str, str]:
    """
    Scan model_dir for:
      - main model .gguf  (anything without 'mmproj' in the filename)
      - mmproj .gguf      (filename contains 'mmproj')
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
#  IMAGE LOADING + ENCODING  (handles ANY common image format)
# ─────────────────────────────────────────────────────────────────────────────

def load_image_any_format(image_path: str) -> Image.Image:
    """
    Robustly load an image of ANY supported format and return a clean PIL.Image
    in RGB mode. Handles:
      - Animated GIF/WebP (takes first frame)
      - PNG with alpha (composites onto white)
      - HEIC/HEIF/AVIF (via pillow-heif)
      - EXIF orientation (auto-rotates iPhone/Android shots)
      - 16-bit TIFFs, palette images, CMYK JPEGs, etc.
    """
    img = Image.open(image_path)

    # Auto-rotate based on EXIF orientation tag (iPhone photos especially)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # For animated formats, take the first frame
    try:
        img.seek(0)
    except Exception:
        pass

    # Convert palette / CMYK / 16-bit / alpha modes to clean RGB
    if img.mode == "RGBA" or img.mode == "LA" or "transparency" in img.info:
        # Composite onto white so transparency doesn't become black
        bg = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img


def prepare_image_for_model(img: Image.Image) -> Image.Image:
    """Resize so the long edge ≤ IMAGE_MAX_DIM, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) > IMAGE_MAX_DIM:
        ratio = IMAGE_MAX_DIM / max(w, h)
        img = img.resize(
            (max(int(w * ratio), 1), max(int(h * ratio), 1)),
            Image.LANCZOS,
        )
    return img


def pil_to_base64(img: Image.Image, quality: int = IMAGE_JPEG_QUALITY) -> str:
    """Encode a PIL image to a base64 JPEG data URI."""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ─────────────────────────────────────────────────────────────────────────────
#  LLAMA-SERVER  MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

_server_proc: subprocess.Popen | None = None


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
        "--reasoning",       "off",
        "--flash-attn",      "on",
        "--log-verbosity",   "-1",
    ]

    print(f"\n[SERVER] Launching llama-server on port {port} ...")
    print(f"  Model     : {Path(model_path).name}")
    print(f"  mmproj    : {Path(mmproj_path).name}")
    print(f"  GPU layers: {'all' if n_gpu_layers == -1 else n_gpu_layers}")
    print(f"  Context   : {n_ctx} tokens")

    log_path = Path(model_path).parent / "llama_server.log"
    log_file = open(log_path, "w", encoding="utf-8")

    _server_proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    base_url = f"http://127.0.0.1:{port}"
    print(f"  Waiting for server to be ready ", end="", flush=True)
    deadline = time.time() + 180
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
#  CAPTION GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_caption(client: OpenAI, img: Image.Image) -> str:
    """Send a single image to llama-server and return the caption text."""
    content = [
        {"type": "text",      "text": USER_PROMPT},
        {"type": "image_url", "image_url": {"url": pil_to_base64(img)}},
        {
            "type": "text",
            "text": (
                "Describe this image with extreme detail, following every rule "
                "in the system prompt. Plain prose only. Begin now."
            ),
        },
    ]

    response = client.chat.completions.create(
        model="gemma4",  # llama-server ignores the model alias
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
        max_tokens=MAX_CAPTION_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
    )

    text = response.choices[0].message.content.strip()
    # Strip any stray <think>...</think> blocks just in case the model emits them
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_images(image_dir: str, recursive: bool = True) -> list[Path]:
    """
    Find all supported image files in image_dir.

    With recursive=True (default), walks every subfolder at any depth using
    os.walk — fast, single-pass, case-insensitive on the file extension.

    Skips:
      - Hidden folders (names starting with '.')
      - Common system / build folders that should never contain dataset images
        (so we don't waste time descending into them on big drives).
    """
    image_dir = Path(image_dir)

    if not recursive:
        found = []
        for ext in IMAGE_EXTENSIONS:
            found.extend(image_dir.glob(f"*{ext}"))
            found.extend(image_dir.glob(f"*{ext.upper()}"))
        return sorted(set(found))

    # Folder names we always prune from the walk
    SKIP_DIRS = {
        "$recycle.bin", "system volume information",
        ".git", ".svn", ".hg",
        "node_modules", "__pycache__", ".venv", "venv", "env",
        ".idea", ".vscode",
        ".cache", ".thumbnails", ".trash",
    }

    found: list[Path] = []
    # os.walk is much faster than rglob on huge trees and lets us prune in-place
    for root, dirs, files in os.walk(image_dir, followlinks=False):
        # Prune unwanted/hidden dirs BEFORE descending into them
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d.lower() not in SKIP_DIRS
        ]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                found.append(Path(root) / fname)

    return sorted(set(found))


# ─────────────────────────────────────────────────────────────────────────────
#  CAPTION FILE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def caption_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def already_captioned(image_path: Path) -> bool:
    cp = caption_path_for(image_path)
    return cp.exists() and cp.stat().st_size > 0


def save_caption(image_path: Path, caption: str, trigger_word: str = "") -> Path:
    out = caption_path_for(image_path)
    final = f"{trigger_word}, {caption}" if trigger_word else caption
    out.write_text(final, encoding="utf-8")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print_banner()

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
        "  (Download from: https://huggingface.co/ggml-org)"
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

    # ── 3. Image source: single file or folder ───────────────────────────────
    hr()
    print("STEP 3 — Image input")
    hr()
    supported = ", ".join(sorted(IMAGE_EXTENSIONS))
    print(f"  Supported formats: {supported}")
    print(
        "\n  Enter EITHER the path to a single image\n"
        "  OR the path to a folder. Folders are scanned RECURSIVELY —\n"
        "  every subfolder at any depth is included automatically.\n"
        "  Captions are saved next to each image, preserving folder structure."
    )

    images: list[Path] = []
    image_root: Path | None = None
    while True:
        path_str = input("\n  Image file or folder path:\n  > ").strip().strip('"').strip("'")
        p = Path(path_str)
        if not p.exists():
            print(f"  [!] Path does not exist: {path_str}")
            continue
        if p.is_file():
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                print(f"  [!] Not a supported image format: {p.suffix}")
                continue
            images = [p]
            image_root = p.parent
            print(f"\n  ✓ Single image: {p.name}")
            break
        elif p.is_dir():
            print(f"\n  Scanning '{p}' recursively (this may take a moment for huge trees) ...")
            images = discover_images(str(p), recursive=True)
            if not images:
                print(f"  [!] No supported image files found anywhere under: {p}")
                continue
            image_root = p

            # Per-subfolder breakdown so the user can sanity-check the scan
            from collections import Counter
            subfolder_counts = Counter(
                str(im.parent.relative_to(p)) or "." for im in images
            )
            n_subs = len(subfolder_counts)
            print(f"\n  ✓ Found {len(images)} image(s) across {n_subs} folder(s)")
            # Show top 10 folders by image count, then a tail summary
            top = subfolder_counts.most_common(10)
            for rel, n in top:
                label = "(root)" if rel == "." else rel
                print(f"      {n:>5}  {label}")
            if n_subs > 10:
                hidden_total = sum(subfolder_counts.values()) - sum(n for _, n in top)
                print(f"      ... and {n_subs - 10} more folder(s) ({hidden_total} more image(s))")
            break
        else:
            print("  [!] Path is neither a file nor a directory.")

    # ── 4. Trigger word (optional) ───────────────────────────────────────────
    hr()
    print("STEP 4 — Trigger word (optional)")
    hr()
    print(
        "  A trigger word is prepended to every caption, e.g. 'alitastyle'\n"
        "  Useful when building a LoRA/finetune dataset.\n"
        "  Press Enter to skip."
    )
    trigger_word = input("\n  Trigger word (or press Enter to skip):\n  > ").strip()
    if trigger_word:
        print(f"  ✓ Trigger word set: '{trigger_word}'")
    else:
        print("  ⊘ No trigger word — captions will start directly with the description.")

    # ── 5. Skip already-captioned images ─────────────────────────────────────
    pending   = [im for im in images if not already_captioned(im)]
    completed = len(images) - len(pending)
    if completed:
        print(f"\n  ℹ  {completed} image(s) already have .txt captions — skipping.")
    if not pending:
        print("\n  All images already captioned! Nothing to do.")
        return

    # ── 6. Start llama-server ────────────────────────────────────────────────
    hr()
    print("STEP 5 — Starting llama-server (Gemma 4 · CUDA)")
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

    client = OpenAI(
        base_url=f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1",
        api_key="not-needed",
    )

    # ── 7. Caption loop ──────────────────────────────────────────────────────
    hr()
    print(f"STEP 6 — Captioning {len(pending)} image(s)\n")
    hr()
    print(
        f"  Max image dim    : {IMAGE_MAX_DIM}px (long edge)\n"
        f"  Max caption len  : {MAX_CAPTION_TOKENS} tokens\n"
        f"  Temperature      : {TEMPERATURE}\n"
        f"  Trigger word     : {trigger_word if trigger_word else '(none)'}\n"
        f"  Output           : .txt file next to each image (saved on the fly)\n"
    )

    success_count = 0
    fail_count    = 0

    for i, image_path in enumerate(pending, 1):
        # Show path relative to the root the user gave, so duplicate filenames
        # across subfolders (IMG_0001.jpg, etc.) don't all look the same.
        try:
            display_name = str(image_path.relative_to(image_root)) if image_root else image_path.name
        except (ValueError, TypeError):
            display_name = image_path.name
        print(f"\n[{i}/{len(pending)}] {display_name}")
        hr("·")

        try:
            # — Load image —
            print(f"  Loading image  ...", end=" ", flush=True)
            img = load_image_any_format(str(image_path))
            ow, oh = img.size
            img = prepare_image_for_model(img)
            nw, nh = img.size
            if (ow, oh) != (nw, nh):
                print(f"✓ ({ow}×{oh} → {nw}×{nh})")
            else:
                print(f"✓ ({nw}×{nh})")

            # — Generate caption —
            print(f"  Generating caption ...", end=" ", flush=True)
            t0      = time.time()
            caption = generate_caption(client, img)
            elapsed = time.time() - t0
            words   = len(caption.split())
            print(f"✓ ({words} words, {elapsed:.1f}s)")

            # — Save caption immediately —
            out_path = save_caption(image_path, caption, trigger_word)
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
            error_txt = caption_path_for(image_path)
            try:
                error_txt.write_text(f"[CAPTIONING FAILED: {e}]", encoding="utf-8")
                print(f"  Error note saved → {error_txt.name}")
            except Exception:
                pass

    # ── 8. Summary ───────────────────────────────────────────────────────────
    hr("═")
    print("DONE")
    hr("═")
    print(f"  ✓ Successfully captioned : {success_count}")
    print(f"  ✗ Failed                 : {fail_count}")
    print(f"  ⊘ Skipped (existing)     : {completed}")
    if len(images) == 1:
        print(f"\n  Caption file: {caption_path_for(images[0])}")
    else:
        print(f"\n  All .txt files are saved next to their images,")
        print(f"  preserving the original folder structure under:")
        print(f"    {image_root}")

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