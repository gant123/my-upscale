# Aurora Ops — AI-Powered Image Processing Engine

> Open-source alternative to Topaz Photo AI / Google Photos AI / Adobe Photoshop AI

## What This Is

Aurora Ops is a desktop image processing application built with **Electron + React + Python**. It combines **Lightroom-style manual controls** with **neural network AI processing** to deliver professional-grade image enhancement.

---

## Quick Start (Recommended)

### Requirements
- Node.js 18+ (or 20 LTS)
- **Python 3.11.x (recommended)**  
  > Real-ESRGAN/Basicsr can fail on very new Python versions (3.13/3.14+). If AI deps fail, switch to Python 3.11.

---

### macOS setup (Homebrew)

#### 1) Install Node + Python 3.11
```bash
brew install node
brew install python@3.11
```

Verify:
```bash
node -v
python3.11 --version
```

#### 2) Install Node dependencies
```bash
npm install
```

#### 3) Create a Python venv (recommended) + install AI deps
From the project root:
```bash
python3.11 -m venv myenv
source myenv/bin/activate

# Install AI dependencies (recommended script)
bash setup-ai.sh
```

#### 4) Run the app
```bash
npm run dev
```

---

### Generic setup (Linux / Windows / other)

#### 1) Install Node dependencies
```bash
npm install
```

#### 2) Create a venv and install Python deps
```bash
python -m venv myenv
# mac/linux:
source myenv/bin/activate
# windows (powershell):
# .\myenv\Scripts\Activate.ps1

bash setup-ai.sh
```

#### 3) Run
```bash
npm run dev
```

---

## Python Dependencies (manual install)

> If you don’t want the script, install by hand inside your virtualenv.

```bash
# Core (required)
pip install -U pip setuptools wheel
pip install opencv-python-headless numpy Pillow

# AI features (optional)
pip install realesrgan basicsr     # AI upscaling
pip install gfpgan                 # Face restoration
pip install rembg onnxruntime      # Background removal
```

**Important:** Avoid forcing wheels with `--only-binary :all:` unless you know your environment has matching wheels.

---

## GPU Acceleration (recommended for AI)

### NVIDIA (CUDA)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu
```

### Apple Silicon (MPS)
```bash
pip install torch torchvision
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
