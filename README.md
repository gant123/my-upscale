# Aurora Ops — AI-Powered Image Processing Engine

> Open-source alternative to Topaz Photo AI / Google Photos AI / Adobe Photoshop AI

## What This Is

Aurora Ops is a desktop image processing application built with **Electron + React + Python**. It combines **Lightroom-style manual controls** with **neural network AI processing** to deliver professional-grade image enhancement.

### Feature Comparison

| Feature | Aurora Ops | Topaz Photo AI | Google Photos | Photoshop |
|---|---|---|---|---|
| AI Upscaling (Real-ESRGAN) | ✅ | ✅ | ✅ | ✅ |
| AI Face Restoration (GFPGAN) | ✅ | ✅ | ✅ | ✅ |
| Background Removal | ✅ | ❌ | ✅ | ✅ |
| AI Inpainting | ✅ | ❌ | ✅ | ✅ |
| One-Click Auto Fix | ✅ | ✅ | ✅ | ❌ |
| 15 Manual Sliders | ✅ | ❌ | ❌ | ✅ |
| X-Ray Visualization (8 modes) | ✅ | ❌ | ❌ | ❌ |
| Enhancement Layers (stackable) | ✅ | ❌ | ❌ | ✅ |
| Before/After Compare | ✅ | ✅ | ✅ | ❌ |
| Batch Processing | ✅ | ✅ | ❌ | ✅ |
| Offline / Privacy | ✅ | ✅ | ❌ | ❌ |
| Open Source | ✅ | ❌ | ❌ | ❌ |
| **Price** | **Free** | $99/yr | $2.99/mo | $20/mo |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│ Electron Main Process (Node.js)                  │
│  ├─ IPC handlers for each command                │
│  ├─ File I/O, dialogs, validation                │
│  └─ Spawns Python engine via stdin JSON          │
├─────────────────────────────────────────────────┤
│ Python Engine (engine.py)                        │
│  ├─ Classical: 15 sliders, 5 enhance modes,      │
│  │   8 x-ray modes, analysis, diagnostics        │
│  ├─ AI: Real-ESRGAN, GFPGAN, rembg, inpaint     │
│  ├─ Auto-enhance: analyze → fix → enhance →      │
│  │   face restore → upscale (intelligent chain)   │
│  └─ Capability detection (graceful degradation)   │
├─────────────────────────────────────────────────┤
│ React Renderer                                   │
│  ├─ Adjust tab: live Lightroom sliders           │
│  ├─ Enhance tab: stackable layer cards           │
│  ├─ AI Tools tab: upscale, face, bg, inpaint     │
│  ├─ Analyze tab: metrics + AI suggestions        │
│  ├─ Export tab: save dialog                      │
│  └─ Canvas: split compare, zoom, hold-to-compare │
└─────────────────────────────────────────────────┘
```

**Key design decisions:**
- **stdin JSON protocol** — No shell escaping issues. All Python communication goes through `echo JSON | python engine.py`
- **Capability detection** — Engine reports what AI libs are installed. UI hides unavailable features and shows install hints
- **Graceful degradation** — Classical features always work. AI features light up as you install libs
- **Latest-wins debounce** — Live slider adjustments use sequence numbers to ignore stale responses

---

## Quick Start

### 1. Install Node dependencies
```bash
npm install
```

### 2. Install Python AI dependencies
```bash
# Core (required)
pip install opencv-python-headless numpy Pillow

# AI features (optional — install what you want)
pip install realesrgan basicsr          # AI upscaling
pip install gfpgan                       # Face restoration
pip install rembg onnxruntime            # Background removal

# Or run the setup script:
bash setup-ai.sh
```

### 3. Run
```bash
npm run dev
```

### GPU Acceleration (recommended for AI)
```bash
# NVIDIA GPU (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu

# Apple Silicon
pip install torch torchvision  # MPS support auto-detected
```

---

## Features in Detail

### 🎛 Manual Adjust (15 Sliders)
All Lightroom-style controls with real pixel math (not CSS filters):

**Tone:** Exposure, Contrast, Highlights, Shadows, Whites, Blacks
**Color:** Temperature, Tint, Vibrance, Saturation
**Detail:** Clarity, Dehaze, Sharpness, Grain, Vignette

Plus 8 X-Ray visualization modes with blend control.

### ✦ AI Auto-Fix (One Click)
Intelligent pipeline that:
1. Analyzes the image (exposure, noise, sharpness, faces, resolution)
2. Applies optimal tonal corrections
3. Runs best enhancement mode
4. Restores faces if detected (GFPGAN)
5. Upscales if low-res (Real-ESRGAN)

### 🔬 AI Upscale
Real-ESRGAN neural upscaling — generates realistic detail that doesn't exist in the original. 2× or 4× with tile processing for large images.

### 👤 Face Restoration
GFPGAN restores blurry, damaged, or low-quality faces. Adjustable fidelity slider.

### 🎨 Background Removal
U2-Net (via rembg) removes backgrounds automatically. Outputs transparent PNG.

### 🩹 Inpainting
Paint a mask → objects are intelligently removed. Uses OpenCV Telea (always available) or LaMa neural inpainting (with torch).

### 📊 Smart Analysis
- Noise estimation (Laplacian)
- Sharpness measurement
- Dynamic range analysis
- Color cast detection (LAB space)
- Face detection (Haar cascade)
- Content classification
- AI-powered fix suggestions

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Ctrl+O` | Open image |
| `Ctrl+S` | Save result |
| `Space` (hold) | Compare with original |
| `Esc` | Dismiss error/toast |
| Double-click slider | Reset to default |

---

## Build for Distribution

```bash
# Windows
npm run build:win

# macOS
npm run build:mac

# Linux
npm run build:linux
```

Bundle the `python/` directory with the app, or require users to have Python installed.

---

## Roadmap

- [ ] Draw-to-erase mask tool for inpainting
- [ ] Batch processing UI with progress
- [ ] Color grading LUTs
- [ ] Crop / rotate / transform tools
- [ ] AI colorization (for B&W photos)
- [ ] AI object detection + selective editing
- [ ] Plugin system for custom processing
- [ ] RAW file support (via rawpy)

---

## License

MIT — Free for personal and commercial use.