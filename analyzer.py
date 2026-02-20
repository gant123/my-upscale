"""
analyzer.py — Pro Image Diagnostics Engine
Detects: exposure, contrast, noise, sharpness, color cast, dynamic range, skin presence.
"""

import sys
import json
import cv2
import numpy as np


def estimate_noise_level(img_gray):
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    return round(float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(laplacian)) / np.sqrt(6.0)), 2)

def measure_sharpness(img_gray):
    return round(float(cv2.Laplacian(img_gray, cv2.CV_64F).var()), 2)

def detect_color_cast(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a_cast, b_cast = float(np.mean(a)) - 128.0, float(np.mean(b)) - 128.0
    casts = []
    if abs(a_cast) > 5: casts.append("magenta" if a_cast > 0 else "green")
    if abs(b_cast) > 5: casts.append("yellow" if b_cast > 0 else "blue")
    return {"a_offset": round(a_cast, 2), "b_offset": round(b_cast, 2),
            "detected_casts": casts, "severity": round(float(np.sqrt(a_cast**2 + b_cast**2)), 2)}

def analyze_histogram(img_gray):
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    total = img_gray.shape[0] * img_gray.shape[1]
    cumsum = np.cumsum(hist)
    low_val, high_val = int(np.searchsorted(cumsum, total * 0.01)), int(np.searchsorted(cumsum, total * 0.99))
    return {
        "shadow_clip_pct": round(float(np.sum(hist[:5])) / total * 100, 2),
        "highlight_clip_pct": round(float(np.sum(hist[250:])) / total * 100, 2),
        "dynamic_range": high_val - low_val,
        "low_percentile": low_val, "high_percentile": high_val,
        "shadows_pct": round(float(np.sum(hist[:85])) / total * 100, 1),
        "mids_pct": round(float(np.sum(hist[85:170])) / total * 100, 1),
        "highlights_pct": round(float(np.sum(hist[170:])) / total * 100, 1),
    }

def detect_skin_presence(img_bgr):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    return round(float(np.sum(mask > 0)) / (mask.shape[0] * mask.shape[1]) * 100, 2)

def compute_recommendations(m):
    b, sat, noise, sharp = m["avg_brightness"], m["avg_saturation"], m["noise_level"], m["sharpness"]
    cast, skin = m["color_cast"], m["skin_pct"]
    if b < 60: exp = round(min(2.0, (110 - b) / 70.0), 2)
    elif b > 190: exp = round(max(-2.0, (130 - b) / 70.0), 2)
    else: exp = round((127 - b) / 127.0, 2)
    target = 80 if skin > 10 else 95
    sat_adj = round(max(0.3, min(2.5, target / max(sat, 1))), 2)
    if noise > 15 and sharp < 200: mode, reason = "smooth", "High noise with low detail"
    elif cast["severity"] > 8 or sat < 50: mode, reason = "color", "Color issues detected"
    elif b < 70 and noise < 10: mode, reason = "light", "Dark but clean"
    elif skin > 15: mode, reason = "portrait", "Skin tones detected"
    else: mode, reason = "pro", "Balanced enhancement"
    return {"exposure": exp, "saturation": sat_adj, "best_mode": mode, "mode_reason": reason}

def generate_diagnosis(m):
    lines = []
    b = m["avg_brightness"]
    if b < 70: lines.append(f"Underexposed ({b:.0f}/255).")
    elif b > 190: lines.append(f"Overexposed ({b:.0f}/255).")
    else: lines.append("Exposure balanced.")
    dr = m["histogram"]["dynamic_range"]
    if dr < 150: lines.append(f"Low contrast (range {dr}/255).")
    elif dr > 240: lines.append("Full dynamic range.")
    n = m["noise_level"]
    if n > 15: lines.append(f"High noise (sigma {n}).")
    elif n > 8: lines.append(f"Moderate noise (sigma {n}).")
    else: lines.append("Clean signal.")
    if m["sharpness"] < 100: lines.append("Soft focus.")
    elif m["sharpness"] > 500: lines.append("Sharp.")
    if m["avg_saturation"] < 50: lines.append("Desaturated.")
    c = m["color_cast"]
    if c["severity"] > 8: lines.append(f"Cast: {', '.join(c['detected_casts'])} ({c['severity']:.1f}).")
    h = m["histogram"]
    if h["shadow_clip_pct"] > 5: lines.append(f"Shadow clip: {h['shadow_clip_pct']:.1f}%.")
    if h["highlight_clip_pct"] > 5: lines.append(f"Highlight clip: {h['highlight_clip_pct']:.1f}%.")
    if m["skin_pct"] > 15: lines.append("Portrait detected.")
    return " ".join(lines)

def analyze_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(json.dumps({"error": "Could not read image."})); return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); _, s, v = cv2.split(hsv)
        metrics = {
            "avg_brightness": float(np.mean(v)), "avg_saturation": float(np.mean(s)),
            "noise_level": estimate_noise_level(gray), "sharpness": measure_sharpness(gray),
            "histogram": analyze_histogram(gray), "color_cast": detect_color_cast(img),
            "skin_pct": detect_skin_presence(img),
            "resolution": {"width": img.shape[1], "height": img.shape[0]}
        }
        print(json.dumps({
            "status": "success", "analysis": generate_diagnosis(metrics),
            "metrics": metrics, "recommendations": compute_recommendations(metrics),
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) > 1: analyze_image(sys.argv[1])
    else: print(json.dumps({"error": "No image provided"}))