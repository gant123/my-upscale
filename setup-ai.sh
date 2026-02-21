#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  Aurora Ops — AI Dependencies Setup
#  Run: bash setup-ai.sh
# ═══════════════════════════════════════════════════════════

set -e

echo "╔═══════════════════════════════════════════╗"
echo "║  Aurora Ops — AI Module Installer         ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# Upgrade pip and install build tools
echo "→ Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel --quiet

# Core (always required)
echo "→ Installing core dependencies (opencv, numpy)..."
pip install opencv-python-headless numpy Pillow --quiet

# AI Upscaling
echo ""
echo "→ Installing Real-ESRGAN (AI Upscaling)..."
pip install realesrgan basicsr --only-binary :all: --quiet 2>/dev/null || {
  echo "  ⚠ realesrgan install failed. Try: pip install realesrgan basicsr --only-binary :all:"
  echo "  GPU users: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
}

# Face Restoration
echo ""
echo "→ Installing GFPGAN (Face Restoration)..."
pip install gfpgan --quiet 2>/dev/null || {
  echo "  ⚠ gfpgan install failed. Try: pip install gfpgan"
}

# Background Removal
echo ""
echo "→ Installing rembg (Background Removal)..."
pip install rembg onnxruntime --quiet 2>/dev/null || {
  echo "  ⚠ rembg install failed. Try: pip install rembg onnxruntime"
  echo "  GPU users: pip install rembg[gpu] onnxruntime-gpu"
}

# Model directory
echo ""
echo "→ Creating model cache directory..."
mkdir -p ~/.aurora/models

echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║  Setup complete! Run the app:             ║"
echo "║  npm run dev                              ║"
echo "║                                           ║"
echo "║  AI models download on first use.         ║"
echo "║  The app works without AI libs too —      ║"
echo "║  classical features always available.     ║"
echo "╚═══════════════════════════════════════════╝"