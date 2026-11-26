from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError

from model import RMBG

ROOT = Path(__file__).resolve().parent
WEIGHTS = Path(os.getenv("RMBG_WEIGHTS", ROOT / "rmbg14_model_weights.pth"))
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 40_000_000
INPUT_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, int(os.getenv("RMBG_CPU_THREADS", min(8, os.cpu_count() or 1)))))

app = FastAPI(title="Clearcut", docs_url=None, redoc_url=None)
model: RMBG | None = None
model_lock = threading.Lock()


@app.on_event("startup")
def load_model() -> None:
    global model
    if not WEIGHTS.is_file():
        raise RuntimeError(f"Model weights not found: {WEIGHTS}")
    network = RMBG()
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    network.load_state_dict(state, strict=True)
    network.eval().to(DEVICE)
    if DEVICE.type == "cuda":
        network = network.to(memory_format=torch.channels_last)
    model = network


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ready" if model is not None else "loading", "device": DEVICE.type}


@app.post("/api/remove-bg")
async def remove_background(image: UploadFile = File(...)) -> Response:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(415, "Choose an image file to continue.")
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "This image is larger than 25 MB.")
    try:
        source = Image.open(io.BytesIO(content))
        source = ImageOps.exif_transpose(source).convert("RGB")
        if source.width * source.height > MAX_PIXELS:
            raise HTTPException(413, "This image is larger than 40 megapixels.")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(415, "That image could not be opened. Try a JPEG, PNG, or WebP.")

    pixels = torch.frombuffer(bytearray(source.tobytes()), dtype=torch.uint8)
    tensor = pixels.reshape(source.height, source.width, 3).permute(2, 0, 1).unsqueeze(0).float()
    tensor = F.interpolate(tensor, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
    tensor = ((tensor / 255.0 - 0.5) / 1.0).to(DEVICE)
    try:
        with model_lock, torch.inference_mode():
            assert model is not None
            if DEVICE.type == "cuda":
                tensor = tensor.to(memory_format=torch.channels_last)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = model(tensor)[0]
            else:
                prediction = model(tensor)[0]
            mask = F.interpolate(prediction.float(), size=(source.height, source.width), mode="bilinear", align_corners=False)[0, 0]
            lo, hi = mask.amin(), mask.amax()
            mask = ((mask - lo) / (hi - lo).clamp_min(1e-6) * 255).round().clamp(0, 255).byte().cpu().numpy()
    except Exception as exc:
        raise HTTPException(500, "The image could not be processed. Please try another image.") from exc

    source.putalpha(Image.fromarray(mask, mode="L"))
    output = io.BytesIO()
    source.save(output, format="PNG", optimize=False)
    return Response(output.getvalue(), media_type="image/png", headers={"Content-Disposition": 'attachment; filename="clearcut.png"', "Cache-Control": "no-store"})


@app.post("/api/export-gif")
async def export_gif(image: UploadFile = File(...), background: str = Form("transparent")) -> Response:
    """Encode the already-cut-out PNG as a still GIF without re-running inference."""
    content = await image.read(80 * 1024 * 1024 + 1)
    if len(content) > 80 * 1024 * 1024:
        raise HTTPException(413, "The cutout is too large to export as GIF.")
    try:
        cutout = Image.open(io.BytesIO(content)).convert("RGBA")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(415, "The cutout could not be opened for GIF export.")
    if cutout.width * cutout.height > MAX_PIXELS:
        raise HTTPException(413, "This image is larger than 40 megapixels.")

    if background != "transparent":
        if len(background) != 7 or not background.startswith("#"):
            raise HTTPException(400, "Choose a valid background color.")
        try:
            color = tuple(int(background[i:i + 2], 16) for i in (1, 3, 5))
        except ValueError:
            raise HTTPException(400, "Choose a valid background color.")
        flattened = Image.new("RGB", cutout.size, color)
        flattened.paste(cutout, mask=cutout.getchannel("A"))
        indexed = flattened.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    else:
        alpha = cutout.getchannel("A")
        opaque = Image.new("RGB", cutout.size, (255, 255, 255))
        opaque.paste(cutout, mask=alpha)
        indexed = opaque.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        indexed.paste(255, mask=alpha.point(lambda value: 255 if value < 128 else 0))
        indexed.info["transparency"] = 255

    output = io.BytesIO()
    save_options = {"format": "GIF", "optimize": True}
    if "transparency" in indexed.info:
        save_options["transparency"] = indexed.info["transparency"]
    indexed.save(output, **save_options)
    return Response(output.getvalue(), media_type="image/gif", headers={"Content-Disposition": 'attachment; filename="nobg.gif"', "Cache-Control": "no-store"})


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((ROOT / "index.html").read_text(encoding="utf-8"))
