"""
adjust.py — Precision Adjustment Engine
Applies Lightroom-style adjustments via real pixel math (not CSS filters).

Usage:
  python adjust.py <image_path> '<json_params>'

JSON params (all optional, defaults shown):
{
  "exposure": 0,        # -3 to 3 (stops)
  "contrast": 0,        # -100 to 100
  "highlights": 0,      # -100 to 100
  "shadows": 0,         # -100 to 100
  "whites": 0,          # -100 to 100
  "blacks": 0,          # -100 to 100
  "temperature": 0,     # -100 (cool) to 100 (warm)
  "tint": 0,            # -100 (green) to 100 (magenta)
  "clarity": 0,         # -100 to 100
  "dehaze": 0,          # 0 to 100
  "vibrance": 0,        # -100 to 100
  "saturation": 0,      # -100 to 100
  "sharpness": 0,       # 0 to 100
  "grain": 0,           # 0 to 100
  "vignette": 0         # -100 (dark) to 0 to 100 (light)
}

Output: temp file path as JSON
"""

import sys
import json
import cv2
import numpy as np
import os
import tempfile


def apply_exposure(img_f, value):
    """Exposure in stops. Each stop = 2x light."""
    if value == 0:
        return img_f
    multiplier = pow(2.0, value)
    return np.clip(img_f * multiplier, 0, 1)


def apply_contrast(img_f, value):
    """S-curve contrast centered at midtones. -100 to 100."""
    if value == 0:
        return img_f
    # Strength from 0 to ~0.6
    strength = value / 170.0
    # S-curve via sine
    return np.clip(img_f + strength * np.sin(2.0 * np.pi * img_f) / (2.0 * np.pi), 0, 1)


def apply_highlights(img_f, value):
    """Recover or boost highlights only (top 30% of tonal range)."""
    if value == 0:
        return img_f
    gray = np.mean(img_f, axis=2, keepdims=True) if img_f.ndim == 3 else img_f
    # Soft mask that targets highlights
    mask = np.clip((gray - 0.55) / 0.45, 0, 1)
    shift = value / 200.0  # -0.5 to 0.5
    return np.clip(img_f + mask * shift, 0, 1)


def apply_shadows(img_f, value):
    """Lift or crush shadows (bottom 40% of tonal range)."""
    if value == 0:
        return img_f
    gray = np.mean(img_f, axis=2, keepdims=True) if img_f.ndim == 3 else img_f
    mask = np.clip(1.0 - gray / 0.45, 0, 1)
    shift = value / 200.0
    return np.clip(img_f + mask * shift, 0, 1)


def apply_whites(img_f, value):
    """Push the white point. Narrower than highlights — top 15%."""
    if value == 0:
        return img_f
    gray = np.mean(img_f, axis=2, keepdims=True) if img_f.ndim == 3 else img_f
    mask = np.clip((gray - 0.80) / 0.20, 0, 1)
    shift = value / 150.0
    return np.clip(img_f + mask * shift, 0, 1)


def apply_blacks(img_f, value):
    """Push the black point. Bottom 15%."""
    if value == 0:
        return img_f
    gray = np.mean(img_f, axis=2, keepdims=True) if img_f.ndim == 3 else img_f
    mask = np.clip(1.0 - gray / 0.20, 0, 1)
    shift = value / 150.0
    return np.clip(img_f - mask * shift, 0, 1)


def apply_temperature(img_u8, value):
    """Color temperature. Negative = cool (blue), positive = warm (yellow/orange)."""
    if value == 0:
        return img_u8
    img_f = img_u8.astype(np.float64)
    b, g, r = cv2.split(img_f)
    shift = value * 0.15  # scale to reasonable pixel range
    r = np.clip(r + shift, 0, 255)
    b = np.clip(b - shift, 0, 255)
    # Slight green adjustment to keep white balance natural
    g = np.clip(g + shift * 0.1, 0, 255)
    return cv2.merge((b, g, r)).astype(np.uint8)


def apply_tint(img_u8, value):
    """Tint. Negative = green, positive = magenta."""
    if value == 0:
        return img_u8
    img_f = img_u8.astype(np.float64)
    b, g, r = cv2.split(img_f)
    shift = value * 0.12
    g = np.clip(g - shift, 0, 255)
    r = np.clip(r + shift * 0.5, 0, 255)
    b = np.clip(b + shift * 0.5, 0, 255)
    return cv2.merge((b, g, r)).astype(np.uint8)


def apply_clarity(img_u8, value):
    """
    Clarity = midtone contrast via local contrast enhancement.
    Positive: crunchy detail. Negative: dreamy softness.
    Uses the difference between the image and a large-radius blur.
    """
    if value == 0:
        return img_u8
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_f = l.astype(np.float64)

    # Large-radius blur captures structure, difference = local detail
    blur_radius = max(3, min(img_u8.shape[0], img_u8.shape[1]) // 30)
    if blur_radius % 2 == 0:
        blur_radius += 1
    blurred = cv2.GaussianBlur(l_f, (blur_radius, blur_radius), 0)

    detail = l_f - blurred
    strength = value / 80.0  # -1.25 to 1.25

    l_out = np.clip(l_f + detail * strength, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l_out, a, b)), cv2.COLOR_LAB2BGR)


def apply_dehaze(img_u8, value):
    """
    Dehazing via dark channel prior (simplified He et al. method).
    Estimates atmospheric light and transmission, then recovers contrast.
    """
    if value == 0:
        return img_u8

    strength = value / 100.0  # 0 to 1
    img_f = img_u8.astype(np.float64) / 255.0

    # Dark channel: minimum across color channels in a local patch
    patch_size = 7
    min_channel = np.min(img_f, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    # Estimate atmospheric light from brightest dark channel pixels
    flat_dark = dark_channel.flatten()
    num_brightest = max(1, int(0.001 * len(flat_dark)))
    indices = np.argsort(flat_dark)[-num_brightest:]
    flat_img = img_f.reshape(-1, 3)
    atm_light = np.mean(flat_img[indices], axis=0)
    atm_light = np.clip(atm_light, 0.5, 1.0)

    # Transmission estimate
    normalized = img_f / atm_light[np.newaxis, np.newaxis, :]
    min_norm = np.min(normalized, axis=2)
    transmission = 1.0 - strength * cv2.erode(min_norm, kernel)
    transmission = np.clip(transmission, 0.1, 1.0)

    # Recover scene
    t_3ch = transmission[:, :, np.newaxis]
    recovered = (img_f - atm_light[np.newaxis, np.newaxis, :]) / t_3ch + atm_light[np.newaxis, np.newaxis, :]

    return np.clip(recovered * 255, 0, 255).astype(np.uint8)


def apply_vibrance(img_u8, value):
    """Non-linear vibrance. Boosts muted colors more than saturated ones."""
    if value == 0:
        return img_u8
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s_f = s.astype(np.float64) / 255.0
    boost = value / 200.0  # -0.5 to 0.5

    # Key formula: boost * s * (1 - s) peaks at mid-saturation
    s_out = s_f + boost * s_f * (1.0 - s_f)

    # Protect skin hues
    skin_mask = ((h > 5) & (h < 25)).astype(np.float64)
    s_out = s_out * (1.0 - 0.4 * skin_mask) + s_f * (0.4 * skin_mask)

    s_out = np.clip(s_out * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h, s_out, v)), cv2.COLOR_HSV2BGR)


def apply_saturation(img_u8, value):
    """Linear saturation. -100 = grayscale, 0 = no change, 100 = 2x."""
    if value == 0:
        return img_u8
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    factor = 1.0 + value / 100.0  # 0 to 2
    s_out = np.clip(s.astype(np.float64) * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h, s_out, v)), cv2.COLOR_HSV2BGR)


def apply_sharpness(img_u8, value):
    """Unsharp mask with adaptive radius."""
    if value == 0:
        return img_u8
    strength = value / 60.0  # 0 to ~1.7
    sigma = 1.0 + value / 50.0
    blurred = cv2.GaussianBlur(img_u8, (0, 0), sigma)
    return np.clip(
        img_u8.astype(np.float64) * (1.0 + strength) - blurred.astype(np.float64) * strength,
        0, 255
    ).astype(np.uint8)


def apply_grain(img_u8, value):
    """Film grain simulation. Luminance-weighted noise."""
    if value == 0:
        return img_u8
    strength = value / 100.0 * 25  # up to 25 pixel std dev
    noise = np.random.normal(0, strength, img_u8.shape).astype(np.float64)

    # Weight noise by luminance — less visible in shadows (like real film)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    weight = 0.5 + 0.5 * np.stack([gray] * 3, axis=-1)
    noise = noise * weight

    return np.clip(img_u8.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def apply_vignette(img_u8, value):
    """Vignette. Negative = darken edges, positive = lighten edges."""
    if value == 0:
        return img_u8
    h, w = img_u8.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2) / np.sqrt(cx**2 + cy**2)

    # Smooth falloff starting at 40% from center
    falloff = np.clip((dist - 0.4) / 0.6, 0, 1)
    strength = value / 100.0  # -1 to 1

    if strength < 0:
        # Darken edges
        mask = 1.0 + strength * falloff  # mask < 1 at edges
    else:
        # Lighten edges
        mask = 1.0 + strength * 0.3 * falloff

    mask_3ch = np.stack([mask] * 3, axis=-1)
    return np.clip(img_u8.astype(np.float64) * mask_3ch, 0, 255).astype(np.uint8)


def adjust_image(image_path, params):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(json.dumps({"error": "Could not read image."}))
            return

        # Apply adjustments in a proper order
        # Tone adjustments first (float space for precision)
        img_f = img.astype(np.float64) / 255.0

        img_f = apply_exposure(img_f, params.get("exposure", 0))
        img_f = apply_contrast(img_f, params.get("contrast", 0))
        img_f = apply_highlights(img_f, params.get("highlights", 0))
        img_f = apply_shadows(img_f, params.get("shadows", 0))
        img_f = apply_whites(img_f, params.get("whites", 0))
        img_f = apply_blacks(img_f, params.get("blacks", 0))

        # Convert back to uint8 for color/spatial operations
        img_u8 = np.clip(img_f * 255, 0, 255).astype(np.uint8)

        # Color adjustments
        img_u8 = apply_temperature(img_u8, params.get("temperature", 0))
        img_u8 = apply_tint(img_u8, params.get("tint", 0))

        # Local contrast
        img_u8 = apply_clarity(img_u8, params.get("clarity", 0))
        img_u8 = apply_dehaze(img_u8, params.get("dehaze", 0))

        # Color intensity
        img_u8 = apply_vibrance(img_u8, params.get("vibrance", 0))
        img_u8 = apply_saturation(img_u8, params.get("saturation", 0))

        # Detail
        img_u8 = apply_sharpness(img_u8, params.get("sharpness", 0))
        img_u8 = apply_grain(img_u8, params.get("grain", 0))

        # Effects
        img_u8 = apply_vignette(img_u8, params.get("vignette", 0))

        # Save to temp
        _, ext = os.path.splitext(image_path)
        ext = ext if ext else ".jpg"
        base = os.path.splitext(os.path.basename(image_path))[0]
        temp_path = os.path.join(tempfile.gettempdir(), f"{base}_adjusted{ext}")
        cv2.imwrite(temp_path, img_u8)

        print(json.dumps({"status": "success", "temp_path": temp_path}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        # CLI mode: python adjust.py <path> '<json>'
        try:
            adjust_image(sys.argv[1], json.loads(sys.argv[2]))
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Bad JSON in argv: {e}"}))
    elif len(sys.argv) == 2:
        # Stdin mode: echo '<json>' | python adjust.py <path>
        try:
            raw = sys.stdin.read()
            params = json.loads(raw) if raw.strip() else {}
            adjust_image(sys.argv[1], params)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Bad JSON on stdin: {e}"}))
    else:
        print(json.dumps({"error": "Usage: adjust.py <image_path> [json_params]"}))