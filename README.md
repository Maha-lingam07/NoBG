<div align="center">

# NoBG

**Deep-learning passport photo background remover powered by BRIA RMBG 1.4.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Deploy to Render](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render&logoColor=000)](https://render.com/)

</div>

---

## Overview

**NoBG** is a fast, privacy-first web application for removing backgrounds from passport-size photos and portraits. Built with **FastAPI**, **PyTorch**, and the **BRIA RMBG 1.4** (`IS-Net`) segmentation architecture, it processes images completely in memory—no uploaded or generated files are ever written to disk.

Upload a photo, preview the cutout over a transparent or custom solid-color background in real time, and export in **PNG**, **JPEG**, **WebP**, or **GIF** at full original resolution.

---

## Key Features

- **High-Precision Segmentation** — Uses the BRIA RMBG 1.4 (`IS-Net` with `RSU7`–`RSU4F` encoder-decoder blocks) running at `1024 × 1024` inference resolution and upscaling the alpha matte cleanly back to your original image dimensions.
- **Privacy by Design** — All image decoding, inference, and format conversion happen strictly in memory with `Cache-Control: no-store`. Nothing is stored on the server.
- **Custom Background Studio** — Switch instantaneously between **Transparent**, preset passport tones (**White** `#FFFFFF`, **Light Blue** `#A8D2F4`, **Gray** `#C2C3CB`), a native color picker, or any `#RRGGBB` hex code.
- **Multi-Format Export** — Download as lossless **PNG**, **JPEG**, compact **WebP**, or still **GIF** (server-side 256-color median-cut quantization with binary transparency support via `/api/export-gif`).
- **Hardware-Aware Inference** — Automatically uses **CUDA** (with `torch.float16` autocast and `channels_last` memory format) when a GPU is available, or falls back smoothly to multi-threaded **CPU** execution.
- **Zero-Dependency Frontend** — Clean, responsive single-page UI (`index.html`) with drag-and-drop upload, EXIF auto-orientation, keyboard accessibility, and live status feedback.

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend** | FastAPI, Uvicorn, Python Multipart |
| **ML / Inference** | PyTorch (`RMBG` / `IS-Net` architecture) |
| **Image Processing** | Pillow (PIL), HTML5 Canvas |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript |
| **Deployment** | Render (`render.yaml` + `build.sh`), Git LFS |

---

## Project Structure

```text
NoBG/
├── app.py                     # FastAPI server, inference pipeline, and GIF export API
├── model.py                   # IS-Net (RMBG 1.4) PyTorch architecture definition
├── index.html                 # Single-page studio UI (upload, color picker, preview, export)
├── requirements.txt           # Python dependencies
├── rmbg14_model_weights.pth   # BRIA RMBG 1.4 pretrained weights (~176.6 MB, tracked via Git LFS)
├── render.yaml                # Render Blueprint configuration
├── build.sh                   # Render build script (CPU PyTorch + LFS weight fallback)
└── README.md
```

---

## Getting Started Locally

### Prerequisites

- **Python 3.10+**
- **Git LFS** (required to pull `rmbg14_model_weights.pth`)

### 1. Clone the Repository

```bash
git lfs install
git clone https://github.com/Maha-lingam07/NoBG.git
cd NoBG
```

### 2. Create a Virtual Environment & Install Dependencies

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip (CPU-only machines):** If you do not have an NVIDIA GPU and want a much smaller download (~200 MB instead of ~2.5 GB), install CPU PyTorch first:
> ```bash
> pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 3. Run the Server

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open **<http://127.0.0.1:8000>** in your browser. On startup, the server loads `rmbg14_model_weights.pth` into memory and exposes `/api/health`.

---

## Configuration

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `RMBG_WEIGHTS` | `./rmbg14_model_weights.pth` | Custom path to the `.pth` model weights file. |
| `RMBG_CPU_THREADS` | `min(8, cpu_count)` | Number of PyTorch CPU threads (`2` in `render.yaml`). |

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the NoBG web studio (`index.html`). |
| `GET` | `/api/health` | Returns `{"status": "ready" \| "loading", "device": "cuda" \| "cpu"}`. |
| `POST` | `/api/remove-bg` | Accepts `image` (`multipart/form-data`; JPEG/PNG/WebP, max 25 MB, max 40 MP) and returns a transparent `image/png`. |
| `POST` | `/api/export-gif` | Accepts `image` (cutout PNG) and `background` (`"transparent"` or `#RRGGBB`) and returns an optimized `image/gif`. |

---

## Limits & Notes

- **Accepted Inputs:** JPEG, PNG, and WebP up to **25 MB** and **40 megapixels**.
- **Concurrency:** Inference is serialized with a threading lock (`model_lock`) to keep memory usage predictable.
- **GIF Export:** Uses a standard 256-color palette and binary transparency (`alpha < 128`).
- **Model License:** The checkpoint architecture and weights correspond to **BRIA RMBG 1.4**. Please review BRIA AI's model license terms before commercial use or redistribution.

---

<div align="center">

Made with 💙 by [Mahalingam](https://github.com/Maha-lingam07)

</div>
