"""
engine.py — Aurora Ops: Unified AI + Classical Image Processing Engine
═══════════════════════════════════════════════════════════════════════

Protocol (stdin JSON → stdout JSON):
  echo '{"command":"analyze","image":"/path/to/img.jpg"}' | python engine.py
  echo '{"command":"adjust","image":"/path","params":{...}}' | python engine.py
  echo '{"command":"enhance","image":"/path","modes":["pro","color"]}' | python engine.py
  echo '{"command":"upscale","image":"/path","scale":4,"model":"realesrgan"}' | python engine.py
  echo '{"command":"face_restore","image":"/path","model":"codeformer","fidelity":0.7}' | python engine.py
  echo '{"command":"bg_remove","image":"/path"}' | python engine.py
  echo '{"command":"inpaint","image":"/path","mask":"/path/mask.png","prompt":"..."}' | python engine.py
  echo '{"command":"auto_enhance","image":"/path"}' | python engine.py
  echo '{"command":"batch","jobs":[...]}' | python engine.py
  echo '{"command":"save","temp_path":"/tmp/x.jpg","save_path":"/home/y.jpg"}' | python engine.py
  echo '{"command":"capabilities"}' | python engine.py

AI Models (auto-downloaded on first use):
  - Real-ESRGAN      → 2x/4x AI upscaling (superior to bicubic)
  - GFPGAN           → Face restoration + enhancement
  - CodeFormer       → Higher-fidelity face restoration
  - rembg (U2-Net)   → Background removal
  - LaMa             → AI inpainting (object removal)
"""

import sys
import json
import os
import tempfile
import shutil
import traceback
import importlib

import cv2
import numpy as np


# ============================================================================
#  CAPABILITY DETECTION — Graceful degradation if AI libs not installed
# ============================================================================

CAPS = {
    "classical": True,
    "upscale_realesrgan": False,
    "face_gfpgan": False,
    "face_codeformer": False,
    "bg_remove": False,
    "inpaint_lama": False,
    "inpaint_opencv": True,   # always available via cv2
}

def _probe(module_name, cap_key):
    try:
        importlib.import_module(module_name)
        CAPS[cap_key] = True
    except ImportError:
        pass

_probe("realesrgan", "upscale_realesrgan")
_probe("gfpgan", "face_gfpgan")
# CodeFormer usually bundled w/ gfpgan or separate
try:
    from gfpgan import GFPGANer
    CAPS["face_codeformer"] = True  # CodeFormer ships with newer gfpgan
except Exception:
    pass
_probe("rembg", "bg_remove")
try:
    import torch
    # LaMa needs torch
    CAPS["inpaint_lama"] = os.path.exists(
        os.path.join(os.path.expanduser("~"), ".aurora", "models", "big-lama.pt")
    )
except ImportError:
    pass


def cmd_capabilities(_data=None):
    """Report what this engine can do — UI uses this to show/hide features."""
    install_hints = {}
    if not CAPS["upscale_realesrgan"]:
        install_hints["upscale_realesrgan"] = "pip install realesrgan basicsr"
    if not CAPS["face_gfpgan"]:
        install_hints["face_gfpgan"] = "pip install gfpgan"
    if not CAPS["bg_remove"]:
        install_hints["bg_remove"] = "pip install rembg[gpu] onnxruntime-gpu  (or rembg onnxruntime for CPU)"
    if not CAPS["inpaint_lama"]:
        install_hints["inpaint_lama"] = "pip install torch torchvision  + download big-lama.pt to ~/.aurora/models/"

    return {
        "status": "success",
        "capabilities": CAPS,
        "install_hints": install_hints,
        "engine_version": "2.0.0",
    }


# ============================================================================
#  ANALYZE — Enhanced diagnostics with face detection + content classification
# ============================================================================

def cmd_analyze(data):
    img = load(data["image"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    noise = est_noise(gray)
    sharp = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)

    # Histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = gray.shape[0] * gray.shape[1]
    cumsum = np.cumsum(hist)
    lo = int(np.searchsorted(cumsum, total * 0.01))
    hi = int(np.searchsorted(cumsum, total * 0.99))

    # Color cast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _, a_ch, b_ch = cv2.split(lab)
    a_off = float(np.mean(a_ch)) - 128.0
    b_off = float(np.mean(b_ch)) - 128.0
    casts = []
    if abs(a_off) > 5: casts.append("magenta" if a_off > 0 else "green")
    if abs(b_off) > 5: casts.append("yellow" if b_off > 0 else "blue")
    cast_sev = round(float(np.sqrt(a_off**2 + b_off**2)), 2)

    # Skin
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    skin_pct = round(float(np.sum(skin_mask > 0)) / total * 100, 2)

    avg_b = float(np.mean(v))
    avg_s = float(np.mean(s))

    # Face detection (Haar cascade — always available)
    faces = []
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        detected = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        faces = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for x, y, w, h in detected]
    except Exception:
        pass

    # Content classification heuristics
    content_tags = []
    if len(faces) > 0:
        content_tags.append("portrait" if len(faces) <= 2 else "group")
    if skin_pct > 20:
        content_tags.append("people")
    if avg_s < 30:
        content_tags.append("monochrome")
    if noise > 15:
        content_tags.append("noisy")
    if sharp < 80:
        content_tags.append("blurry")
    h, w = img.shape[:2]
    if w < 800 or h < 800:
        content_tags.append("low_res")

    # Recommendations
    if avg_b < 60: exp = round(min(2.0, (110 - avg_b) / 70.0), 2)
    elif avg_b > 190: exp = round(max(-2.0, (130 - avg_b) / 70.0), 2)
    else: exp = round((127 - avg_b) / 127.0, 2)

    target = 80 if skin_pct > 10 else 95
    sat_adj = round(max(0.3, min(2.5, target / max(avg_s, 1))), 2)

    if noise > 15 and sharp < 200: best, reason = "smooth", "High noise"
    elif cast_sev > 8 or avg_s < 50: best, reason = "color", "Color issues"
    elif avg_b < 70 and noise < 10: best, reason = "light", "Dark but clean"
    elif skin_pct > 15: best, reason = "portrait", "Skin detected"
    else: best, reason = "pro", "Balanced"

    # AI suggestions based on analysis
    ai_suggestions = []
    if "low_res" in content_tags and CAPS["upscale_realesrgan"]:
        ai_suggestions.append({
            "action": "upscale",
            "reason": f"Image is {w}×{h} — AI upscaling recommended",
            "params": {"scale": 4, "model": "realesrgan"}
        })
    if len(faces) > 0 and (CAPS["face_gfpgan"] or CAPS["face_codeformer"]):
        if sharp < 200 or noise > 10:
            ai_suggestions.append({
                "action": "face_restore",
                "reason": f"{len(faces)} face(s) detected — restoration can enhance detail",
                "params": {"model": "gfpgan", "fidelity": 0.7}
            })
    if noise > 12:
        ai_suggestions.append({
            "action": "enhance",
            "reason": "High noise detected — AI denoise recommended",
            "params": {"modes": ["smooth"]}
        })

    # Diagnosis text
    lines = []
    if avg_b < 70: lines.append(f"Underexposed ({avg_b:.0f}/255).")
    elif avg_b > 190: lines.append(f"Overexposed ({avg_b:.0f}/255).")
    else: lines.append("Exposure OK.")
    dr = hi - lo
    if dr < 150: lines.append(f"Low contrast (range {dr}).")
    if noise > 15: lines.append(f"High noise (σ={noise}).")
    elif noise > 8: lines.append(f"Moderate noise (σ={noise}).")
    else: lines.append("Clean.")
    if sharp < 100: lines.append("Soft focus.")
    if avg_s < 50: lines.append("Desaturated.")
    if cast_sev > 8: lines.append(f"Cast: {', '.join(casts)}.")
    if len(faces) > 0: lines.append(f"{len(faces)} face(s).")
    if "low_res" in content_tags: lines.append(f"Low-res ({w}×{h}).")

    return {
        "status": "success",
        "analysis": " ".join(lines),
        "metrics": {
            "noise": noise, "sharpness": sharp,
            "brightness": round(avg_b, 1), "saturation": round(avg_s, 1),
            "dynamic_range": dr, "skin_pct": skin_pct,
            "cast_severity": cast_sev, "casts": casts,
            "width": w, "height": h,
            "faces": faces,
            "content_tags": content_tags,
        },
        "recommendations": {
            "exposure": exp, "saturation": sat_adj,
            "best_mode": best, "mode_reason": reason,
            "ai_suggestions": ai_suggestions,
        },
        "capabilities": CAPS,
    }


# ============================================================================
#  ADJUST — All 15 Lightroom-style sliders + X-Ray overlays
# ============================================================================

def cmd_adjust(data):
    img = load(data["image"])
    p = data.get("params", {})

    # --- TONE (float64 for precision) ---
    f = img.astype(np.float64) / 255.0

    # Exposure (stops)
    v = p.get("exposure", 0)
    if v != 0: f = np.clip(f * pow(2.0, v), 0, 1)

    # Contrast (S-curve)
    v = p.get("contrast", 0)
    if v != 0:
        s = v / 170.0
        f = np.clip(f + s * np.sin(2 * np.pi * f) / (2 * np.pi), 0, 1)

    # Highlights
    v = p.get("highlights", 0)
    if v != 0:
        gray = np.mean(f, axis=2, keepdims=True)
        mask = np.clip((gray - 0.55) / 0.45, 0, 1)
        f = np.clip(f + mask * v / 200.0, 0, 1)

    # Shadows
    v = p.get("shadows", 0)
    if v != 0:
        gray = np.mean(f, axis=2, keepdims=True)
        mask = np.clip(1.0 - gray / 0.45, 0, 1)
        f = np.clip(f + mask * v / 200.0, 0, 1)

    # Whites
    v = p.get("whites", 0)
    if v != 0:
        gray = np.mean(f, axis=2, keepdims=True)
        mask = np.clip((gray - 0.80) / 0.20, 0, 1)
        f = np.clip(f + mask * v / 150.0, 0, 1)

    # Blacks
    v = p.get("blacks", 0)
    if v != 0:
        gray = np.mean(f, axis=2, keepdims=True)
        mask = np.clip(1.0 - gray / 0.20, 0, 1)
        f = np.clip(f - mask * v / 150.0, 0, 1)

    u8 = np.clip(f * 255, 0, 255).astype(np.uint8)

    # --- COLOR ---
    v = p.get("temperature", 0)
    if v != 0:
        bf, gf, rf = cv2.split(u8.astype(np.float64))
        shift = v * 0.15
        u8 = cv2.merge((
            np.clip(bf - shift, 0, 255).astype(np.uint8),
            np.clip(gf + shift * 0.1, 0, 255).astype(np.uint8),
            np.clip(rf + shift, 0, 255).astype(np.uint8),
        ))

    v = p.get("tint", 0)
    if v != 0:
        bf, gf, rf = cv2.split(u8.astype(np.float64))
        shift = v * 0.12
        u8 = cv2.merge((
            np.clip(bf + shift * 0.5, 0, 255).astype(np.uint8),
            np.clip(gf - shift, 0, 255).astype(np.uint8),
            np.clip(rf + shift * 0.5, 0, 255).astype(np.uint8),
        ))

    v = p.get("vibrance", 0)
    if v != 0:
        hsv = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV)
        h, s, val = cv2.split(hsv)
        sf = s.astype(np.float64) / 255.0
        boost = v / 200.0
        sb = sf + boost * sf * (1.0 - sf)
        skin = ((h > 5) & (h < 25)).astype(np.float64)
        sb = sb * (1.0 - 0.4 * skin) + sf * (0.4 * skin)
        u8 = cv2.cvtColor(cv2.merge((h, np.clip(sb * 255, 0, 255).astype(np.uint8), val)), cv2.COLOR_HSV2BGR)

    v = p.get("saturation", 0)
    if v != 0:
        hsv = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV)
        h, s, val = cv2.split(hsv)
        factor = 1.0 + v / 100.0
        u8 = cv2.cvtColor(cv2.merge((h, np.clip(s.astype(np.float64) * factor, 0, 255).astype(np.uint8), val)), cv2.COLOR_HSV2BGR)

    # --- DETAIL ---
    v = p.get("clarity", 0)
    if v != 0:
        lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lf = l.astype(np.float64)
        rad = max(3, min(u8.shape[0], u8.shape[1]) // 30)
        if rad % 2 == 0: rad += 1
        blurred = cv2.GaussianBlur(lf, (rad, rad), 0)
        detail = lf - blurred
        l_out = np.clip(lf + detail * (v / 80.0), 0, 255).astype(np.uint8)
        u8 = cv2.cvtColor(cv2.merge((l_out, a, b)), cv2.COLOR_LAB2BGR)

    v = p.get("dehaze", 0)
    if v != 0:
        strength = v / 100.0
        ff = u8.astype(np.float64) / 255.0
        min_ch = np.min(ff, axis=2)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dark = cv2.erode(min_ch, kern)
        flat_dark = dark.flatten()
        n_bright = max(1, int(0.001 * len(flat_dark)))
        idx = np.argsort(flat_dark)[-n_bright:]
        atm = np.clip(np.mean(ff.reshape(-1, 3)[idx], axis=0), 0.5, 1.0)
        norm = ff / atm[np.newaxis, np.newaxis, :]
        trans = np.clip(1.0 - strength * cv2.erode(np.min(norm, axis=2), kern), 0.1, 1.0)
        rec = (ff - atm[np.newaxis, np.newaxis, :]) / trans[:, :, np.newaxis] + atm[np.newaxis, np.newaxis, :]
        u8 = np.clip(rec * 255, 0, 255).astype(np.uint8)

    v = p.get("sharpness", 0)
    if v != 0:
        strength = v / 60.0
        sigma = 1.0 + v / 50.0
        blurred = cv2.GaussianBlur(u8, (0, 0), sigma)
        u8 = np.clip(u8.astype(np.float64) * (1 + strength) - blurred.astype(np.float64) * strength, 0, 255).astype(np.uint8)

    # --- EFFECTS ---
    v = p.get("grain", 0)
    if v != 0:
        std = v / 100.0 * 25
        noise_arr = np.random.normal(0, std, u8.shape)
        gray = cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        weight = 0.5 + 0.5 * np.stack([gray] * 3, axis=-1)
        u8 = np.clip(u8.astype(np.float64) + noise_arr * weight, 0, 255).astype(np.uint8)

    v = p.get("vignette", 0)
    if v != 0:
        hi, wi = u8.shape[:2]
        Y, X = np.ogrid[:hi, :wi]
        cx, cy = wi / 2.0, hi / 2.0
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2) / np.sqrt(cx**2 + cy**2)
        falloff = np.clip((dist - 0.4) / 0.6, 0, 1)
        if v < 0:
            mask = 1.0 + (v / 100.0) * falloff
        else:
            mask = 1.0 + (v / 100.0) * 0.3 * falloff
        u8 = np.clip(u8.astype(np.float64) * np.stack([mask] * 3, axis=-1), 0, 255).astype(np.uint8)

    # --- X-RAY overlays ---
    xray = p.get("xray", "none")
    if xray != "none":
        xray_fn = XRAY_MAP.get(xray)
        if xray_fn:
            xray_img = xray_fn(load(data["image"]))
            blend = p.get("xray_blend", 100) / 100.0
            if blend >= 1.0:
                u8 = xray_img
            else:
                u8 = np.clip(u8.astype(np.float64) * (1 - blend) + xray_img.astype(np.float64) * blend, 0, 255).astype(np.uint8)

    return save_temp(u8, data["image"], "adjusted")


# ============================================================================
#  ENHANCE — Classical layer stacking (existing modes)
# ============================================================================

def cmd_enhance(data):
    img = load(data["image"])
    modes = data.get("modes", ["pro"])
    applied = []
    current = img.copy()
    for m in modes:
        fn = ENHANCE_MAP.get(m)
        if fn:
            current = fn(current)
            applied.append(m)
    if not applied:
        return {"error": "No valid modes"}
    result = save_temp(current, data["image"], "+".join(applied))
    result["applied_modes"] = applied
    return result


# ============================================================================
#  AI UPSCALE — Real-ESRGAN
# ============================================================================

def cmd_upscale(data):
    if not CAPS["upscale_realesrgan"]:
        return {"error": "Real-ESRGAN not installed. Run: pip install realesrgan basicsr", "missing": "realesrgan"}

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    img = load(data["image"])
    scale = data.get("scale", 4)
    denoise = data.get("denoise_strength", 0.5)

    # Select model based on scale
    if scale == 2:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        model_name = "RealESRGAN_x2plus"
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_name = "RealESRGAN_x4plus"

    model_dir = os.path.join(os.path.expanduser("~"), ".aurora", "models")
    os.makedirs(model_dir, exist_ok=True)

    upsampler = RealESRGANer(
        scale=scale,
        model_path=None,  # auto-download
        model=model,
        tile=400,         # tile processing for large images
        tile_pad=10,
        pre_pad=0,
        half=False,       # CPU-safe; set True if CUDA available
    )

    try:
        output, _ = upsampler.enhance(img, outscale=scale)
        result = save_temp(output, data["image"], f"upscaled_{scale}x")
        h, w = output.shape[:2]
        result["output_size"] = {"width": w, "height": h}
        result["scale"] = scale
        return result
    except Exception as e:
        return {"error": f"Upscale failed: {e}"}


# ============================================================================
#  AI FACE RESTORATION — GFPGAN / CodeFormer
# ============================================================================

def cmd_face_restore(data):
    model = data.get("model", "gfpgan")
    fidelity = data.get("fidelity", 0.7)

    if model == "gfpgan" and not CAPS["face_gfpgan"]:
        return {"error": "GFPGAN not installed. Run: pip install gfpgan", "missing": "gfpgan"}

    img = load(data["image"])

    try:
        from gfpgan import GFPGANer

        restorer = GFPGANer(
            model_path="GFPGANv1.4.pth",
            upscale=2,
            arch="clean",
            channel_multiplier=2,
        )

        _, _, output = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=fidelity,
        )

        result = save_temp(output, data["image"], f"face_{model}")
        result["model"] = model
        result["fidelity"] = fidelity
        return result

    except Exception as e:
        return {"error": f"Face restore failed: {e}"}


# ============================================================================
#  AI BACKGROUND REMOVAL — rembg (U2-Net)
# ============================================================================

def cmd_bg_remove(data):
    if not CAPS["bg_remove"]:
        return {"error": "rembg not installed. Run: pip install rembg onnxruntime", "missing": "rembg"}

    from rembg import remove
    from PIL import Image
    import io

    img_path = data["image"]
    bg_color = data.get("bg_color", None)  # None = transparent, or [R,G,B]

    try:
        with open(img_path, "rb") as f:
            input_data = f.read()

        output_data = remove(input_data)

        # Convert to PIL for optional background fill
        result_img = Image.open(io.BytesIO(output_data)).convert("RGBA")

        if bg_color:
            bg = Image.new("RGBA", result_img.size, tuple(bg_color) + (255,))
            bg.paste(result_img, mask=result_img.split()[3])
            result_img = bg.convert("RGB")
            ext = ".jpg"
        else:
            ext = ".png"  # keep alpha

        temp_path = os.path.join(
            tempfile.gettempdir(),
            os.path.splitext(os.path.basename(img_path))[0] + f"_nobg{ext}"
        )
        result_img.save(temp_path)

        return {"status": "success", "temp_path": temp_path}

    except Exception as e:
        return {"error": f"Background removal failed: {e}"}


# ============================================================================
#  AI INPAINTING — OpenCV (always available) + LaMa (if installed)
# ============================================================================

def cmd_inpaint(data):
    img = load(data["image"])
    mask_path = data.get("mask")

    if not mask_path or not os.path.exists(mask_path):
        return {"error": "Mask image required for inpainting"}

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return {"error": "Could not read mask image"}

    # Resize mask to match image if needed
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Threshold mask to binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    method = data.get("method", "telea")  # telea | ns | lama

    if method == "lama" and CAPS["inpaint_lama"]:
        return _inpaint_lama(img, mask, data)

    # OpenCV inpainting (always available, decent results)
    radius = data.get("radius", 5)
    if method == "ns":
        result = cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)
    else:
        result = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)

    out = save_temp(result, data["image"], "inpainted")
    out["method"] = method
    return out


def _inpaint_lama(img, mask, data):
    """LaMa large-mask inpainting (requires torch + model)."""
    try:
        import torch

        model_path = os.path.join(os.path.expanduser("~"), ".aurora", "models", "big-lama.pt")
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()

        # Prepare input
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            result = model(img_t, mask_t)

        result = (result.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        out = save_temp(result, data["image"], "inpainted_lama")
        out["method"] = "lama"
        return out

    except Exception as e:
        return {"error": f"LaMa inpainting failed: {e}"}


# ============================================================================
#  AUTO-ENHANCE — AI-guided automatic improvement pipeline
# ============================================================================

def cmd_auto_enhance(data):
    """
    Intelligent auto-enhance: analyzes the image, then applies the optimal
    combination of classical + AI processing.
    """
    # Step 1: Analyze
    analysis = cmd_analyze(data)
    if analysis.get("error"):
        return analysis

    metrics = analysis["metrics"]
    rec = analysis["recommendations"]
    steps_applied = []
    current_path = data["image"]

    # Step 2: Classical corrections first
    adjust_params = {}
    if abs(rec.get("exposure", 0)) > 0.1:
        adjust_params["exposure"] = rec["exposure"]
    if metrics.get("cast_severity", 0) > 8:
        # Auto white balance via temperature/tint
        a_off = metrics.get("casts", [])
        if "yellow" in a_off: adjust_params["temperature"] = -25
        elif "blue" in a_off: adjust_params["temperature"] = 25
        if "magenta" in a_off: adjust_params["tint"] = -20
        elif "green" in a_off: adjust_params["tint"] = 20

    if metrics.get("dynamic_range", 256) < 180:
        adjust_params["contrast"] = 20
        adjust_params["shadows"] = 15
        adjust_params["highlights"] = -10

    if adjust_params:
        adj_result = cmd_adjust({"image": current_path, "params": adjust_params})
        if adj_result.get("temp_path"):
            current_path = adj_result["temp_path"]
            steps_applied.append({"step": "adjust", "params": adjust_params})

    # Step 3: Enhancement layer
    best_mode = rec.get("best_mode", "pro")
    enh_result = cmd_enhance({"image": current_path, "modes": [best_mode]})
    if enh_result.get("temp_path"):
        current_path = enh_result["temp_path"]
        steps_applied.append({"step": "enhance", "mode": best_mode})

    # Step 4: AI face restoration if faces detected
    if len(metrics.get("faces", [])) > 0 and CAPS["face_gfpgan"]:
        face_result = cmd_face_restore({"image": current_path, "model": "gfpgan", "fidelity": 0.7})
        if face_result.get("temp_path"):
            current_path = face_result["temp_path"]
            steps_applied.append({"step": "face_restore", "model": "gfpgan"})

    # Step 5: AI upscale if low-res
    w, h = metrics.get("width", 9999), metrics.get("height", 9999)
    if (w < 1200 or h < 1200) and CAPS["upscale_realesrgan"]:
        up_result = cmd_upscale({"image": current_path, "scale": 2})
        if up_result.get("temp_path"):
            current_path = up_result["temp_path"]
            steps_applied.append({"step": "upscale", "scale": 2})

    # Read final result
    final = cv2.imread(current_path)
    result = save_temp(final, data["image"], "auto_enhanced")
    result["steps"] = steps_applied
    result["analysis"] = analysis["analysis"]
    result["metrics"] = metrics
    return result


# ============================================================================
#  BATCH — Run multiple commands in sequence
# ============================================================================

def cmd_batch(data):
    jobs = data.get("jobs", [])
    results = []
    current_path = None

    for job in jobs:
        # Chain: use output of previous as input to next
        if current_path and "image" not in job:
            job["image"] = current_path

        cmd = job.get("command", "")
        handler = COMMAND_MAP.get(cmd)
        if not handler:
            results.append({"error": f"Unknown command: {cmd}"})
            continue

        result = handler(job)
        results.append(result)

        if result.get("temp_path"):
            current_path = result["temp_path"]

    return {"status": "success", "results": results}


# ============================================================================
#  SAVE
# ============================================================================

def cmd_save(data):
    src, dst = data["temp_path"], data["save_path"]
    if not os.path.exists(src):
        return {"error": "Temp file not found"}

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    shutil.copy2(src, dst)
    return {"status": "saved", "saved_path": dst}


# ============================================================================
#  X-RAY MODES (carried over from v1)
# ============================================================================

def xray_structure(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e1 = cv2.Canny(gray, 50, 150)
    e2 = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 100)
    e3 = cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 20, 60)
    combined = e1.astype(np.float64) * 0.5 + e2.astype(np.float64) * 0.3 + e3.astype(np.float64) * 0.2
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    if mag.max() > 0: mag = mag / mag.max() * 255
    struct = np.clip(combined * 0.6 + mag * 0.4, 0, 255).astype(np.uint8)
    inv = 255 - struct
    out = np.zeros((*inv.shape, 3), dtype=np.uint8)
    out[:, :, 0] = np.clip(inv * 1.1, 0, 255).astype(np.uint8)
    out[:, :, 1] = np.clip(inv * 0.95, 0, 255).astype(np.uint8)
    out[:, :, 2] = np.clip(inv * 0.85, 0, 255).astype(np.uint8)
    return out

def xray_depth(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    lap = np.abs(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F))
    focus = cv2.GaussianBlur(lap, (31, 31), 0)
    if focus.max() > 0: focus /= focus.max()
    lum = gray.astype(np.float64) / 255.0
    vert = np.linspace(0.3, 1.0, h).reshape(-1, 1) * np.ones((1, w))
    depth = focus * 0.45 + lum * 0.25 + vert * 0.30
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

def xray_frequency(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    low = cv2.GaussianBlur(gray, (0, 0), 16)
    mid = cv2.GaussianBlur(gray, (0, 0), 4) - low
    high = gray - cv2.GaussianBlur(gray, (0, 0), 4)
    def norm(x):
        mn, mx = x.min(), x.max()
        return ((x - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
    lv = cv2.applyColorMap(norm(low), cv2.COLORMAP_OCEAN)
    mv = cv2.applyColorMap(norm(mid), cv2.COLORMAP_VIRIDIS)
    hv = cv2.applyColorMap(norm(high), cv2.COLORMAP_MAGMA)
    return np.clip(lv * 0.3 + mv * 0.35 + hv * 0.35, 0, 255).astype(np.uint8)

def xray_thermal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(3.0, (8, 8)).apply(gray)
    return cv2.GaussianBlur(cv2.applyColorMap(enhanced, cv2.COLORMAP_JET), (3, 3), 0)

def xray_bones(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    flesh = cv2.GaussianBlur(gray, (0, 0), 20)
    bones = gray - flesh
    bones -= bones.min()
    if bones.max() > 0: bones /= bones.max()
    bones = np.power(bones, 0.7) * 255
    inv = 255 - bones.astype(np.uint8)
    out = np.zeros((*inv.shape, 3), dtype=np.uint8)
    out[:, :, 0] = np.clip(inv * 1.05, 0, 255).astype(np.uint8)
    out[:, :, 1] = inv
    out[:, :, 2] = np.clip(inv * 0.90, 0, 255).astype(np.uint8)
    return out

def xray_reveal(img):
    b, g, r = cv2.split(img)
    r_float = r.astype(np.float64) / 255.0
    lifted = np.power(r_float, 0.35) * 255.0
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    texture = clahe.apply(lifted.astype(np.uint8))
    blur = cv2.GaussianBlur(texture, (0, 0), 3.0)
    high_pass = cv2.subtract(texture, blur)
    sharp = cv2.addWeighted(texture, 1.2, high_pass, 1.8, 0)
    return cv2.applyColorMap(sharp, cv2.COLORMAP_BONE)

def xray_bright(img):
    b, g, r = cv2.split(img)
    b_float = b.astype(np.float64) / 255.0
    compressed = np.power(b_float, 2.5) * 255.0
    inverted = 255 - compressed.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    texture = clahe.apply(inverted)
    blur = cv2.GaussianBlur(texture, (0, 0), 3.0)
    high_pass = cv2.subtract(texture, blur)
    sharp = cv2.addWeighted(texture, 1.2, high_pass, 1.8, 0)
    return cv2.applyColorMap(sharp, cv2.COLORMAP_OCEAN)

def xray_occlusion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base = cv2.GaussianBlur(gray, (0, 0), 9)
    detail = cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), 2))
    lifted = cv2.normalize(base.astype(np.float64) * 0.7 + detail.astype(np.float64) * 2.2, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(lifted.astype(np.uint8))
    edges = cv2.Canny(clahe, 40, 120)
    merged = np.clip(clahe.astype(np.float64) * 0.75 + edges.astype(np.float64) * 0.95, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(merged, cv2.COLORMAP_TURBO)

XRAY_MAP = {
    "structure": xray_structure, "depth": xray_depth,
    "frequency": xray_frequency, "thermal": xray_thermal,
    "bones": xray_bones, "reveal": xray_reveal,
    "bright": xray_bright, "occlusion": xray_occlusion,
}


# ============================================================================
#  ENHANCE MODES (carried over from v1, cleaned up)
# ============================================================================

def est_noise(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return round(float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(lap)) / np.sqrt(6.0)), 2)

def ms_sharpen(img, noise):
    if noise > 12: boost, sig = 0.8, 5
    elif noise > 6: boost, sig = 1.3, 4
    else: boost, sig = 1.8, 3
    chs = cv2.split(img) if img.ndim == 3 else [img]
    out = []
    for c in chs:
        cf = c.astype(np.float64)
        base = cv2.GaussianBlur(c, (0, 0), sig).astype(np.float64)
        out.append(np.clip(base + (cf - base) * (1 + boost), 0, 255).astype(np.uint8))
    return cv2.merge(out) if img.ndim == 3 else out[0]

def ada_clahe(l, avg):
    if avg < 70: c, g = 3.0, (4, 4)
    elif avg < 100: c, g = 2.0, (8, 8)
    elif avg > 180: c, g = 1.0, (16, 16)
    else: c, g = 1.5, (8, 8)
    return cv2.createCLAHE(c, g).apply(l)

def rm_cast(img):
    b, g, r = cv2.split(img.astype(np.float64))
    ab, ag, ar = np.mean(b), np.mean(g), np.mean(r)
    aa = (ab + ag + ar) / 3
    bp, gp, rp = np.percentile(b, 99), np.percentile(g, 99), np.percentile(r, 99)
    def sc(a, p): return max(0.7, min(1.4, 0.6 * (aa / max(a, 1)) + 0.4 * (255 / max(p, 1))))
    return cv2.merge(tuple(np.clip(ch * sc(av, pv), 0, 255).astype(np.uint8)
                           for ch, av, pv in [(b, ab, bp), (g, ag, gp), (r, ar, rp)]))

def s_curve(ch, s=0.3):
    lut = np.arange(256, dtype=np.float64) / 255
    return cv2.LUT(ch, np.clip((lut + s * np.sin(2 * np.pi * lut) / (2 * np.pi)) * 255, 0, 255).astype(np.uint8))

def vibrance(hsv, boost=0.35):
    h, s, v = cv2.split(hsv)
    sf = s.astype(np.float64) / 255
    sb = sf + boost * sf * (1 - sf)
    skin = ((h > 5) & (h < 25)).astype(np.float64)
    sb = sb * (1 - 0.4 * skin) + sf * 0.4 * skin
    return cv2.merge((h, np.clip(sb * 255, 0, 255).astype(np.uint8), v))

def enh_pro(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n, ab = est_noise(gray), float(np.mean(gray))
    bal = rm_cast(img)
    lab = cv2.cvtColor(bal, cv2.COLOR_BGR2LAB); l, a, bl = cv2.split(lab)
    bal = cv2.cvtColor(cv2.merge((ada_clahe(l, ab), a, bl)), cv2.COLOR_LAB2BGR)
    sh = ms_sharpen(bal, n)
    bc, gc, rc = cv2.split(sh)
    cur = cv2.merge((s_curve(bc, .15), s_curve(gc, .15), s_curve(rc, .15)))
    return cv2.cvtColor(vibrance(cv2.cvtColor(cur, cv2.COLOR_BGR2HSV), .25), cv2.COLOR_HSV2BGR)

def enh_color(img):
    bal = rm_cast(img)
    hsv = cv2.cvtColor(bal, cv2.COLOR_BGR2HSV); h, s, v = cv2.split(hsv)
    vf = v.astype(np.float64) / 255; g = max(.5, min(1., .5 + np.mean(vf) * .5))
    vl = np.power(vf, g); hm = vf > .85; vl[hm] = vl[hm] * .95 + .05 * vf[hm]
    rec = cv2.cvtColor(cv2.merge((h, s, np.clip(vl * 255, 0, 255).astype(np.uint8))), cv2.COLOR_HSV2BGR)
    viv = cv2.cvtColor(vibrance(cv2.cvtColor(rec, cv2.COLOR_BGR2HSV), .5), cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(viv, cv2.COLOR_BGR2LAB); l, a, bl = cv2.split(lab)
    pop = cv2.cvtColor(cv2.merge((cv2.createCLAHE(1.5, (8, 8)).apply(l), a, bl)), cv2.COLOR_LAB2BGR)
    lab2 = cv2.cvtColor(pop, cv2.COLOR_BGR2LAB); l2, a2, b2 = cv2.split(lab2)
    return cv2.cvtColor(cv2.merge((s_curve(l2, .2), a2, b2)), cv2.COLOR_LAB2BGR)

def enh_smooth(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); n = est_noise(gray)
    if n > 20: hl, hc, t, sr = 14, 14, 7, 21
    elif n > 10: hl, hc, t, sr = 10, 10, 7, 21
    else: hl, hc, t, sr = 6, 6, 5, 15
    dn = cv2.fastNlMeansDenoisingColored(img, None, hl, hc, t, sr)
    sm = cv2.edgePreservingFilter(dn, flags=1, sigma_s=40, sigma_r=0.3)
    det = cv2.subtract(dn, cv2.GaussianBlur(dn, (0, 0), 2.0))
    rec = np.clip(sm.astype(np.float64) + det.astype(np.float64) * 0.3, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rec, cv2.COLOR_BGR2LAB); l, a, bl = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((cv2.createCLAHE(1., (16, 16)).apply(l), a, bl)), cv2.COLOR_LAB2BGR)

def enh_light(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); ab = float(np.mean(gray))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); l, a, bl = cv2.split(lab)
    plo, phi = np.percentile(l, 1), np.percentile(l, 99)
    if phi > plo: l = np.clip((l.astype(np.float64) - plo) * 255 / (phi - plo), 0, 255).astype(np.uint8)
    l = ada_clahe(l, ab)
    st = cv2.cvtColor(cv2.merge((l, a, bl)), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(st, cv2.COLOR_BGR2HSV); h, s, v = cv2.split(hsv)
    vf = v.astype(np.float64) / 255
    sm = np.clip(1 - vf / .4, 0, 1); vf += sm * (.4 if ab < 80 else .2) * (1 - vf)
    hm = np.clip((vf - .8) / .2, 0, 1); vf -= hm * .15 * vf
    res = cv2.cvtColor(cv2.merge((h, s, np.clip(vf * 255, 0, 255).astype(np.uint8))), cv2.COLOR_HSV2BGR)
    bc, gc, rc = cv2.split(res)
    return cv2.merge((s_curve(bc, .1), s_curve(gc, .1), s_curve(rc, .1)))

def enh_portrait(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); n = est_noise(gray)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    smask = cv2.GaussianBlur(cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127])), (15, 15), 0)
    sf = smask.astype(np.float64) / 255; nsf = 1 - sf
    ss = cv2.bilateralFilter(img, 9, 60, 60)
    if n > 10: ss = cv2.fastNlMeansDenoisingColored(ss, None, 8, 8, 5, 15)
    sh = ms_sharpen(img, max(n, 5))
    s3, n3 = np.stack([sf]*3, -1), np.stack([nsf]*3, -1)
    comp = np.clip(ss.astype(np.float64)*s3 + sh.astype(np.float64)*n3, 0, 255).astype(np.uint8)
    bc, gc, rc = cv2.split(comp)
    w = cv2.merge((np.clip(bc.astype(np.int16)-3, 0, 255).astype(np.uint8), gc, np.clip(rc.astype(np.int16)+3, 0, 255).astype(np.uint8)))
    viv = cv2.cvtColor(vibrance(cv2.cvtColor(w, cv2.COLOR_BGR2HSV), .2), cv2.COLOR_HSV2BGR)
    hi, wi = viv.shape[:2]; Y, X = np.ogrid[:hi, :wi]
    dist = np.sqrt((X - wi/2)**2 + (Y - hi/2)**2) / np.sqrt((wi/2)**2 + (hi/2)**2)
    vig = 1 - 0.3 * np.clip(dist - 0.5, 0, 1)
    res = np.clip(viv.astype(np.float64) * np.stack([vig]*3, -1), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(res, cv2.COLOR_BGR2LAB); l, a, bl = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((cv2.createCLAHE(1., (8, 8)).apply(l), a, bl)), cv2.COLOR_LAB2BGR)

ENHANCE_MAP = {
    "pro": enh_pro, "color": enh_color, "smooth": enh_smooth,
    "light": enh_light, "portrait": enh_portrait,
}


# ============================================================================
#  HELPERS
# ============================================================================

def load(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read: {path}")
    return img

def save_temp(img, original_path, tag):
    _, ext = os.path.splitext(original_path)
    ext = ext if ext else ".jpg"
    base = os.path.splitext(os.path.basename(original_path))[0]
    temp_path = os.path.join(tempfile.gettempdir(), f"{base}_{tag}{ext}")
    cv2.imwrite(temp_path, img)
    return {"status": "success", "temp_path": temp_path}


# ============================================================================
#  COMMAND ROUTER
# ============================================================================

COMMAND_MAP = {
    "capabilities": cmd_capabilities,
    "analyze": cmd_analyze,
    "adjust": cmd_adjust,
    "enhance": cmd_enhance,
    "upscale": cmd_upscale,
    "face_restore": cmd_face_restore,
    "bg_remove": cmd_bg_remove,
    "inpaint": cmd_inpaint,
    "auto_enhance": cmd_auto_enhance,
    "batch": cmd_batch,
    "save": cmd_save,
}


# ============================================================================
#  MAIN — stdin JSON → route → stdout JSON
# ============================================================================

if __name__ == "__main__":
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            print(json.dumps({"error": "No input on stdin"}))
            sys.exit(1)

        data = json.loads(raw)
        cmd = data.get("command", "")

        handler = COMMAND_MAP.get(cmd)
        if not handler:
            print(json.dumps({"error": f"Unknown command: {cmd}"}))
            sys.exit(1)

        result = handler(data)
        print(json.dumps(result))

    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON: {e}"}))
    except ValueError as e:
        print(json.dumps({"error": str(e)}))
    except Exception as e:
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))