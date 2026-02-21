#!/bin/bash

echo "🚀 Starting Aurora Ops Model Downloader..."

# Define local models directory
MODELS_DIR="models"
mkdir -p "$MODELS_DIR"
echo "📁 Created local $MODELS_DIR/ directory..."

# 1. Setup GFPGAN
echo "📦 Downloading GFPGANv1.4..."
if [ ! -f "$MODELS_DIR/GFPGANv1.4.pth" ]; then
    curl -L -o "$MODELS_DIR/GFPGANv1.4.pth" "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    echo "✅ GFPGAN downloaded successfully."
else
    echo "⚡ GFPGANv1.4.pth already exists. Skipping."
fi

# 2. Download LaMa Inpainting model
echo "🎨 Downloading big-lama.pt..."
if [ ! -f "$MODELS_DIR/big-lama.pt" ]; then
    curl -L -o "$MODELS_DIR/big-lama.pt" "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
    echo "✅ LaMa downloaded successfully."
else
    echo "⚡ big-lama.pt already exists. Skipping."
fi

echo ""
echo "🎉 Download complete!"
echo "⚠️  ACTION REQUIRED: Please manually move your 'Nomos2.pth' file into the new '$MODELS_DIR/' folder located in your project root."
echo ""