#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          GEMMA 4 IMAGE CAPTIONER  —  Extreme-Detail Vision Tool             ║
║          Uses llama.cpp + mmproj vision model via OpenAI-compatible API     ║
║          Supports: gemma-4-E4B-it-GGUF  |  gemma-4-26B-A4B-it-GGUF          ║
║          Output: structured JSON schema (extreme-detail, vision-only)        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Workflow:
  1. Auto-installs all Python dependencies (openai, Pillow, pillow-heif, etc.)
  2. Auto-downloads llama.cpp CUDA binaries from GitHub if not found on PATH
  3. Asks for your model folder (main .gguf + mmproj .gguf)
  4. Launches llama-server as a subprocess (CUDA / RTX GPU accelerated)
  5. Asks for a single image OR a folder of images
  6. Sends each image to the model and writes TWO caption files:
       • .txt next to the image (same name as the image, JSON-formatted text)
       • .json under <image_root>/json/<...>/<name>.json (Hugging Face ready)

The JSON follows a compact compositional-deconstruction schema:
  high_level_description  — one detailed sentence summarising the image
  style_description       — aesthetics, lighting, photo style, medium, hex color palette
  compositional_deconstruction — background description + ordered elements array
    Each element carries: type ("obj"/"text"), bbox [y_min, x_min, y_max, x_max],
    desc, color_palette (hex values), and "text" (verbatim string, for text elements only).
Any field that is NOT visible in the image is set to JSON null — the model
is instructed to NEVER invent details that aren't actually there.

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
import json
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
# The new compositional schema is compact: one high-level sentence, a few
# style fields, and a variable-length elements array. A fully detailed response
# for a complex image with ~10–20 elements typically runs 800–2000 tokens.
# 3000 gives comfortable headroom for images with many labelled objects/text.
MAX_CAPTION_TOKENS = 3000

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
#  PROMPTS — engineered for compositional deconstruction of a single still image
# ─────────────────────────────────────────────────────────────────────────────

# The exact JSON schema the model must fill in for every image.
# Follows a compact, compositional-deconstruction style:
#   • high_level_description  — one-sentence overall caption
#   • style_description       — aesthetics, lighting, photo style, medium, color palette
#   • compositional_deconstruction — background + ordered list of elements (obj or text)
#     Each element carries: type, bbox [y_min, x_min, y_max, x_max], desc, color_palette
JSON_SCHEMA_TEMPLATE = textwrap.dedent("""
{
  "high_level_description": "...",
  "style_description": {
    "aesthetics": "...",
    "lighting": "...",
    "photo": "...",
    "medium": "...",
    "color_palette": ["#RRGGBB", "..."]
  },
  "compositional_deconstruction": {
    "background": "...",
    "elements": [
      {
        "type": "obj",
        "bbox": [0, 0, 0, 0],
        "desc": "...",
        "color_palette": ["#RRGGBB", "..."]
      }
    ]
  }
}
""").strip()

SYSTEM_PROMPT = textwrap.dedent(f"""
You are an expert image analyst that outputs ONLY a single valid JSON object
following an exact schema. Your task is to caption a single still image with
precise, structured compositional deconstruction.

  ★ DESCRIBE ONLY WHAT IS VISIBLE IN THE IMAGE. ★

  - If a field is NOT visible / NOT applicable, set its value to null.
  - Do NOT invent, guess, or hallucinate details not actually present.
  - Empty arrays are not allowed — use null for missing list fields.

────────────────────────────────────────────────────────────────
SCHEMA EXPLAINED
────────────────────────────────────────────────────────────────

"high_level_description"
  A single richly detailed sentence summarising the entire image — what it
  shows, its subject(s), setting, mood, and purpose.

"style_description"
  "aesthetics"    : Overall visual style / art direction (e.g. "minimalist
                    product photography", "cinematic dark fantasy", "editorial
                    fashion, high-contrast monochrome").
  "lighting"      : Describe the light sources, quality (hard/soft/diffuse),
                    direction, and any notable effects (e.g. "single overhead
                    softbox, soft shadows, subtle rim light on the right").
  "photo"         : Photography/rendering style (e.g. "shallow depth of field,
                    35 mm prime lens look", "flat lay, overhead shot",
                    "3D render, octane, physically-based").
  "medium"        : The medium or production method (e.g. "digital photograph",
                    "oil painting", "CGI render", "pencil sketch on paper").
  "color_palette" : Array of 3–6 dominant hex colour values sampled from the
                    image (e.g. ["#1A1A2E", "#E94560", "#F5F5F5"]).
                    Use accurate hexadecimal values — sample from the actual
                    pixels you observe.

"compositional_deconstruction"
  "background"    : Describe the background layer in detail — colour, texture,
                    environment, depth, or any backdrop elements that are NOT
                    a distinct foreground object.
  "elements"      : An ordered array of every distinct visual element in the
                    image, listed from back to front (farthest to nearest).
                    Each element is one of two types:

    type = "obj"   → a physical object, person, animal, shape, graphic, etc.
      "bbox"         : Bounding box [y_min, x_min, y_max, x_max] in PIXEL
                       COORDINATES relative to the full image dimensions.
                       y_min is the TOP edge, y_max is the BOTTOM edge.
                       x_min is the LEFT edge, x_max is the RIGHT edge.
                       Estimate as accurately as you can from the image.
      "desc"         : Detailed description of the object — what it is,
                       material/texture, condition, pose, orientation, and
                       any visually important details.
      "color_palette": Array of 2–4 hex colours dominant on this specific
                       element (e.g. ["#8B4513", "#D2691E"]).

    type = "text"  → any legible text rendered in the image.
      "text"         : The EXACT text string as it appears in the image
                       (verbatim transcription, preserve capitalisation).
      "bbox"         : Bounding box [y_min, x_min, y_max, x_max] in pixels
                       (same convention as "obj").
      "desc"         : Describe the text's visual treatment — font style
                       (serif/sans-serif/display/handwritten), weight, size
                       relative to image, colour, any effects (shadow, outline,
                       emboss), and its apparent purpose or hierarchy.
      "color_palette": Array of 1–3 hex colours of the text glyphs themselves.

────────────────────────────────────────────────────────────────
STRICT OUTPUT RULES
────────────────────────────────────────────────────────────────
  - Output ONLY the JSON object. No preamble, no explanation, no markdown
    fences (no ```json), no <think> blocks, no commentary after the closing }}.
  - Begin your response with '{{' and end it with '}}'. Nothing before or after.
  - The JSON MUST be syntactically valid: double-quoted keys and string
    values, commas between entries, NO trailing commas.
  - null (lowercase, no quotes) for any field not visible or not applicable.
  - color_palette values MUST be valid 7-character hex strings starting with #
    (e.g. "#A3C4BC"). Never use colour names like "red" or "blue".
  - bbox values MUST be integers (pixel coordinates), NOT percentages or floats.
  - List ALL distinct visual elements in "elements". Do not omit people,
    objects, logos, icons, UI components, decorations, or legible text.
  - Preserve the back-to-front (depth) ordering in the elements array.

SCHEMA:
{JSON_SCHEMA_TEMPLATE}
""").strip()

USER_PROMPT = textwrap.dedent("""
Below is a single still image. Carefully analyse every visual element it
contains and produce one JSON object that exactly matches the schema in the
system prompt.

For "high_level_description": write one rich sentence covering the full image.
For "style_description.color_palette": sample the 3–6 most dominant colours
  from the entire image as accurate hex values.
For each element in "compositional_deconstruction.elements":
  - Identify its type ("obj" or "text").
  - Estimate its bounding box in pixel coordinates [y_min, x_min, y_max, x_max].
  - Write a detailed "desc".
  - Sample 2–4 accurate hex colours for its "color_palette".
  - For "text" elements, transcribe the exact string in the "text" field.

Use null for any field that is not visible or not applicable.
Output ONLY the JSON object — no markdown, no commentary. Begin now.
""").strip()

# ─────────────────────────────────────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────────────────────────────────────

BANNER = r"""
  ╔═══════════════════════════════════════════════════════════╗
  ║   GEMMA 4 IMAGE CAPTIONER  —  Compositional Deconstruct  ║
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

def _extract_json_object(text: str) -> tuple[str, bool]:
    """
    Pull the first balanced top-level JSON object out of `text`.

    Returns (json_substring, truncated_flag).
    - truncated_flag is True if the JSON object opened but never closed
      (model output was cut off mid-generation).

    Robust against models that wrap output in ```json ... ``` fences,
    add stray prose before/after, or emit <think>...</think> blocks.
    """
    # Drop any <think> blocks first
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Strip ```json / ``` fences if present
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1), False

    # Otherwise, find the first '{' and scan forward for its matching '}'
    start = text.find("{")
    if start == -1:
        raise ValueError("No '{' found in model output — model did not return JSON.")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1], False

    # Reached end of text with depth still > 0 — output was truncated
    return text[start:], True


def _repair_truncated_json(partial: str) -> str:
    """
    Best-effort repair of a JSON object that the model truncated mid-stream.

    Strategy:
      1. If we're inside an unterminated string, close the string.
      2. Drop any trailing partial key/value (anything after the last
         comma or '{' that isn't a complete key:value pair).
      3. Strip trailing commas.
      4. Close every still-open '{' or '[' in correct order.

    The result is parseable JSON containing every COMPLETE field the model
    managed to write before being cut off. Any unwritten fields are simply
    absent from the dict (which is fine — the user can fill them as null
    later, or treat absence as null).
    """
    s = partial

    # Pass 1: walk the string, tracking strings vs. structure, and remember
    # the last "safe" cut point — i.e. a position where the JSON so far is
    # syntactically a valid prefix that ends right after a complete value.
    stack: list[str] = []  # holds '{' or '['
    in_string = False
    escape = False
    last_safe = -1  # index just past the last fully-written value/entry

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                # End of a string value or key — could be a safe cut point
                # only if it was a value, but we let pass 2 sort that out.
        else:
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()
                # closing a container is a clean cut point
                last_safe = i + 1
            elif ch == ",":
                # comma is the cleanest cut point — everything before it is
                # a complete entry inside the current container
                last_safe = i

    # If we never saw a safe cut, fall back to the whole string
    if last_safe < 0:
        last_safe = len(s)

    # Pass 2: if we're inside a string at last_safe, walk back to before
    # that string opened so we don't truncate mid-token.
    truncated = s[:last_safe]

    # Strip whitespace and any dangling comma
    truncated = truncated.rstrip()
    if truncated.endswith(","):
        truncated = truncated[:-1].rstrip()

    # Recompute the still-open container stack on the truncated prefix
    stack = []
    in_string = False
    escape = False
    for ch in truncated:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()

    # If we somehow ended up still inside a string, close it (rare after pass 1)
    if in_string:
        truncated += '"'

    # Close every still-open container, in the correct order
    closers = {"{": "}", "[": "]"}
    while stack:
        truncated += closers[stack.pop()]

    return truncated


# ─────────────────────────────────────────────────────────────────────────────
#  NULL PRUNING  —  strip empty/null fields so trained captions stay lean
# ─────────────────────────────────────────────────────────────────────────────

# Values treated as "the model didn't see anything here, drop the key"
_NULL_STRING_VALUES = {
    "", "null", "none", "n/a", "na", "not visible", "not applicable",
    "not present", "unknown", "...", "…",
}


def _is_empty(value) -> bool:
    """
    Return True when a value should be considered absent and pruned.

    - JSON null
    - Empty string, or one of the common 'no-value' placeholders the model
      sometimes emits despite instructions (case-insensitive)
    - Empty list / tuple
    - Empty dict (this is what causes parent sections like "foreground"
      to disappear entirely when every leaf inside them was null)
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _NULL_STRING_VALUES
    if isinstance(value, (list, tuple, dict)) and len(value) == 0:
        return True
    return False


def _prune_nulls(obj):
    """
    Recursively remove keys whose values are 'empty' (see _is_empty).

    - For dicts: walks every key, prunes children first, then drops the key
      if the (now possibly-pruned) value is empty. Parents that become
      empty as a result are themselves pruned by the caller's recursion.
    - For lists: prunes empty items and recurses into surviving items.
    - For everything else: returns the value unchanged.

    The schema's key order is preserved (Python dicts are insertion-ordered
    since 3.7), so the user's schema layout is still respected for any keys
    that survive.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            pruned_value = _prune_nulls(value)
            if not _is_empty(pruned_value):
                cleaned[key] = pruned_value
        return cleaned
    if isinstance(obj, list):
        cleaned_list = [_prune_nulls(item) for item in obj]
        return [item for item in cleaned_list if not _is_empty(item)]
    return obj


def generate_caption(client: OpenAI, img: Image.Image) -> tuple[dict, str, dict]:
    """
    Send a single image to llama-server and return:
      - parsed JSON dict matching the schema
      - pretty-printed JSON text (used for both .txt and .json files)
      - meta dict with debugging info: {finish_reason, truncated, repaired,
        raw_output} so the caller can log/preserve it on failure.
    """
    content = [
        {"type": "text",      "text": USER_PROMPT},
        {"type": "image_url", "image_url": {"url": pil_to_base64(img)}},
        {
            "type": "text",
            "text": (
                "Now output the JSON object for this image. Follow the schema "
                "exactly. Use null for any field that is not visible. Output "
                "ONLY the JSON — no markdown fences, no commentary. Begin now."
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

    choice         = response.choices[0]
    finish_reason  = getattr(choice, "finish_reason", None) or "unknown"
    raw            = (choice.message.content or "").strip()
    # Strip any stray <think>...</think> blocks just in case the model emits them
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    meta = {
        "finish_reason": finish_reason,
        "truncated":     False,
        "repaired":      False,
        "raw_output":    raw,
    }

    # Extract the JSON substring (may be truncated)
    json_str, was_truncated = _extract_json_object(raw)
    meta["truncated"] = was_truncated or finish_reason == "length"

    # First parse attempt
    data = None
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Common minor fix: trailing commas before } or ]
        cleaned = re.sub(r",(\s*[}\]])", r"\1", json_str)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = None

    # If the first attempts failed AND we know the output was truncated,
    # try to repair the partial JSON.
    if data is None and meta["truncated"]:
        try:
            repaired = _repair_truncated_json(json_str)
            data = json.loads(repaired)
            meta["repaired"] = True
        except Exception:
            data = None

    # As a last resort, even when not flagged as truncated, try repair —
    # some models truncate without setting finish_reason properly.
    if data is None:
        try:
            repaired = _repair_truncated_json(json_str)
            data = json.loads(repaired)
            meta["repaired"]  = True
            meta["truncated"] = True
        except Exception:
            err = ValueError(
                f"Model returned invalid JSON (finish_reason={finish_reason}, "
                f"truncated={meta['truncated']}). Raw output preserved in "
                f"<image>.raw.txt for inspection."
            )
            # Attach the raw model output so the caller can dump it for debugging
            err.raw_output = raw
            err.finish_reason = finish_reason
            raise err

    # Prune any null / empty / placeholder fields so the final JSON only
    # contains what the model actually saw. Parent sections that end up
    # entirely empty (e.g. "foreground" when no foreground was visible)
    # are dropped wholesale — keeps the captions lean for LoRA training.
    data = _prune_nulls(data)

    pretty = json.dumps(data, indent=2, ensure_ascii=False)
    return data, pretty, meta


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
        "json",  # this script's own .json output folder — skip on re-runs
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
    """Path of the .txt caption (saved next to the image)."""
    return image_path.with_suffix(".txt")


def json_path_for(image_path: Path, image_root: Path | None) -> Path:
    """
    Path of the .json caption.

    The .json files are NOT saved next to the images — they go into a
    parallel 'json/' folder placed directly under image_root, mirroring
    the image's relative subfolder structure so that duplicate filenames
    across subfolders never collide (important for HF dataset uploads).

      image_root / IMG_0001.jpg                -> image_root / json / IMG_0001.json
      image_root / sub_a / IMG_0001.jpg        -> image_root / json / sub_a / IMG_0001.json
      image_root / sub_a / sub_b / pic.png     -> image_root / json / sub_a / sub_b / pic.json

    If image_root is None (e.g. a single image was given with no parent
    context), the json folder is placed next to the image's parent.
    """
    if image_root is None:
        image_root = image_path.parent
    json_root = image_root / "json"
    try:
        rel = image_path.relative_to(image_root)
    except ValueError:
        # image_path isn't under image_root — fall back to flat layout
        rel = Path(image_path.name)
    return (json_root / rel).with_suffix(".json")


def already_captioned(image_path: Path, image_root: Path | None = None) -> bool:
    """An image is considered done only if BOTH the .txt and the .json exist."""
    txt = caption_path_for(image_path)
    js  = json_path_for(image_path, image_root)
    return (
        txt.exists() and txt.stat().st_size > 0
        and js.exists()  and js.stat().st_size > 0
    )


def save_caption(
    image_path: Path,
    caption_data: dict,
    caption_text: str,
    image_root: Path | None,
    trigger_word: str = "",
) -> tuple[Path, Path]:
    """
    Save the caption in two places:
      1. .txt file next to the image (same name as the image, .txt extension).
         Contents = pretty-printed JSON (the prompt is already in JSON form).
      2. .json file under <image_root>/json/<relative-path>/<name>.json
         Contents = the same parsed JSON object, pretty-printed.

    If a trigger_word is provided, it is stored under a top-level
    "trigger_word" key in BOTH files so the original schema stays untouched.

    Returns (txt_path, json_path).
    """
    # Build the final dict — original schema preserved, optional trigger word added
    if trigger_word:
        final_data = {"trigger_word": trigger_word, **caption_data}
        final_text = json.dumps(final_data, indent=2, ensure_ascii=False)
    else:
        final_data = caption_data
        final_text = caption_text

    # 1. .txt next to image
    txt_path = caption_path_for(image_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(final_text, encoding="utf-8")

    # 2. .json under <image_root>/json/...
    js_path = json_path_for(image_path, image_root)
    js_path.parent.mkdir(parents=True, exist_ok=True)
    js_path.write_text(final_text, encoding="utf-8")

    return txt_path, js_path


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
        "  A trigger word is stored under a top-level 'trigger_word' key in\n"
        "  both the .txt and .json output, e.g. 'alitastyle'.\n"
        "  Useful when building a LoRA/finetune dataset.\n"
        "  Press Enter to skip."
    )
    trigger_word = input("\n  Trigger word (or press Enter to skip):\n  > ").strip()
    if trigger_word:
        print(f"  ✓ Trigger word set: '{trigger_word}'")
    else:
        print("  ⊘ No trigger word — captions will start directly with the description.")

    # ── 5. Skip already-captioned images ─────────────────────────────────────
    pending   = [im for im in images if not already_captioned(im, image_root)]
    completed = len(images) - len(pending)
    if completed:
        print(f"\n  ℹ  {completed} image(s) already have BOTH .txt and .json captions — skipping.")
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
        f"  Output (.txt)    : next to each image (same name)\n"
        f"  Output (.json)   : under '{image_root / 'json'}' (mirrored subfolders)\n"
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
            t0 = time.time()
            caption_data, caption_text, meta = generate_caption(client, img)
            elapsed = time.time() - t0
            # After pruning every remaining leaf is a real description
            n_fields = len(re.findall(r'":\s*"', caption_text))
            flags = []
            if meta.get("truncated"):
                flags.append("TRUNCATED")
            if meta.get("repaired"):
                flags.append("repaired")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"✓ ({n_fields} fields filled, {elapsed:.1f}s){flag_str}")

            # — Save caption immediately (both .txt and .json) —
            txt_path, js_path = save_caption(
                image_path, caption_data, caption_text, image_root, trigger_word
            )
            try:
                rel_js = js_path.relative_to(image_root / 'json') if image_root else js_path.name
            except (ValueError, TypeError):
                rel_js = js_path.name
            print(f"  Saved → {txt_path.name}   +   json/{rel_js}")

            # Print a short preview (first non-null leaf field we can find)
            preview_match = re.search(r'"([^"]+)":\s*"([^"\\]{10,200})"', caption_text)
            if preview_match:
                print(f"  Preview: {preview_match.group(1)} → {preview_match.group(2)[:80]}…")

            success_count += 1

        except KeyboardInterrupt:
            print("\n\n  [!] Interrupted by user.")
            break
        except Exception as e:
            fail_count += 1
            print(f"  [ERROR] {e}")

            # Save error notes to both .txt and .json so re-runs see them
            error_txt = caption_path_for(image_path)
            error_js  = json_path_for(image_path, image_root)
            err_payload = json.dumps(
                {"error": f"CAPTIONING FAILED: {e}"}, indent=2, ensure_ascii=False
            )
            try:
                error_txt.parent.mkdir(parents=True, exist_ok=True)
                error_txt.write_text(err_payload, encoding="utf-8")
                error_js.parent.mkdir(parents=True, exist_ok=True)
                error_js.write_text(err_payload, encoding="utf-8")
                print(f"  Error note saved → {error_txt.name}  +  json/{error_js.name}")
            except Exception:
                pass

            # If the exception carries the raw model output, dump it for debugging
            raw_dump = getattr(e, "raw_output", None)
            if raw_dump:
                raw_path = image_path.with_suffix(".raw.txt")
                try:
                    raw_path.write_text(str(raw_dump), encoding="utf-8")
                    print(f"  Raw model output → {raw_path.name}")
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
        print(f"\n  .txt caption : {caption_path_for(images[0])}")
        print(f"  .json caption: {json_path_for(images[0], image_root)}")
    else:
        print(f"\n  .txt files saved next to their images, preserving folder structure under:")
        print(f"    {image_root}")
        print(f"\n  .json files (Hugging Face ready) saved under:")
        print(f"    {image_root / 'json'}")
        print(f"  (subfolder structure mirrors the original image tree)")

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