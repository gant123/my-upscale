#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  Aurora Ops — AI Dependencies Setup
#  Run: bash setup-ai.sh
#
#  Notes:
#   - Real-ESRGAN/Basicsr often fails on very new Python versions (3.13/3.14+).
#   - Recommended: Python 3.11.x (macOS: brew install python@3.11)
# ═══════════════════════════════════════════════════════════

set -euo pipefail

banner() {
  echo "╔═══════════════════════════════════════════╗"
  echo "║  Aurora Ops — AI Module Installer         ║"
  echo "╚═══════════════════════════════════════════╝"
  echo ""
}

err() { echo "✖ $*" 1>&2; }
warn() { echo "⚠ $*" 1>&2; }
ok() { echo "✓ $*"; }

pick_python() {
  # Prefer python3.11 if available, else fallback to python3, else python
  if command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    err "Python not found. Install Python 3.11 and try again."
    exit 1
  fi
}

ensure_venv() {
  # If not already in a venv, create ./myenv and activate it.
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    warn "No virtualenv detected. Creating ./myenv ..."
    "$PY" -m venv myenv
    # shellcheck disable=SC1091
    source myenv/bin/activate
    ok "Activated virtualenv: $(pwd)/myenv"
  else
    ok "Using active virtualenv: $VIRTUAL_ENV"
  fi
}

check_python_version() {
  # Block 3.13+ because ML wheels lag behind, especially Basicsr/Real-ESRGAN.
  local ver major minor
  ver="$("$PY" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  major="$("$PY" -c 'import sys; print(sys.version_info.major)')"
  minor="$("$PY" -c 'import sys; print(sys.version_info.minor)')"

  echo "→ Python: $ver"

  if [[ "$major" -eq 3 && "$minor" -ge 13 ]]; then
    err "Python $major.$minor detected. Real-ESRGAN/Basicsr commonly fails on 3.13+."
    echo ""
    echo "Fix (macOS / Homebrew):"
    echo "  brew install python@3.11"
    echo "  python3.11 -m venv myenv"
    echo "  source myenv/bin/activate"
    echo "  bash setup-ai.sh"
    echo ""
    exit 1
  fi
}

pip_install() {
  # Args: packages...
  "$PY" -m pip install -U "$@" --quiet
}

pip_install_loud() {
  # For installs that may fail, keep output for debugging.
  "$PY" -m pip install -U "$@"
}

main() {
  banner

  PY="$(pick_python)"
  ok "Selected interpreter: $PY"

  check_python_version
  ensure_venv

  echo ""
  echo "→ Upgrading pip/build tools..."
  "$PY" -m pip install --upgrade pip setuptools wheel --quiet

  echo ""
  echo "→ Installing core dependencies..."
  pip_install opencv-python-headless numpy Pillow

  echo ""
  echo "→ Installing Real-ESRGAN (AI Upscaling)..."
  if ! pip_install_loud realesrgan basicsr; then
    warn "Real-ESRGAN install failed."
    echo "Try this:"
    echo "  $PY -m pip install -U pip setuptools wheel"
    echo "  $PY -m pip install -U realesrgan basicsr"
    echo ""
    echo "If you're on macOS and Python is too new, install 3.11:"
    echo "  brew install python@3.11"
    exit 1
  fi

  echo ""
  echo "→ Installing GFPGAN (Face Restoration)..."
  if ! pip_install_loud gfpgan; then
    warn "GFPGAN install failed. You can still run the app without it."
    echo "Try: $PY -m pip install -U gfpgan"
  fi

  echo ""
  echo "→ Installing rembg (Background Removal)..."
  if ! pip_install_loud rembg onnxruntime; then
    warn "rembg/onnxruntime install failed. You can still run the app without it."
    echo "Try: $PY -m pip install -U rembg onnxruntime"
  fi

  echo ""
  echo "→ Creating model cache directory..."
  mkdir -p ~/.aurora/models

  echo ""
  echo "╔═══════════════════════════════════════════╗"
  echo "║  Setup complete!                          ║"
  echo "║  Next:                                   ║"
  echo "║    npm install                            ║"
  echo "║    npm run dev                            ║"
  echo "║                                           ║"
  echo "║  Notes:                                   ║"
  echo "║  - AI models download on first use.       ║"
  echo "║  - App works without AI libs too.         ║"
  echo "╚═══════════════════════════════════════════╝"
}

main "$@"