#!/usr/bin/env bash
# PlantDiseaseAI — Raspberry Pi setup helper
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_PATH="${PLANT_DISEASE_VENV:-$HOME/plantdisease-venv}"

echo "=== PlantDiseaseAI Raspberry Pi Setup ==="
echo "Project: $PROJECT_ROOT"
echo "Virtualenv: $VENV_PATH"

if ! grep -qi "raspberry pi" /proc/device-tree/model 2>/dev/null; then
  echo "Warning: Raspberry Pi not detected. Continuing anyway..."
fi

echo
echo "[1/7] Free apt cache (helps if SD card is low on space)..."
sudo apt clean || true

echo
echo "[2/7] System packages (camera + GUI deps)..."
sudo apt update
sudo apt install -y \
  python3-venv python3-pip python3-dev \
  python3-picamera2 python3-opencv libcamera-apps \
  libatlas-base-dev libgl1 libxcb-xinerama0 libxkbcommon0 \
  v4l-utils

echo
echo "[3/7] Add current user to video group..."
sudo usermod -aG video "$USER" || true

echo
echo "[4/7] Python virtual environment in home folder..."
python3 -m venv "$VENV_PATH"
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip wheel

echo
echo "[5/7] Install PyTorch for aarch64 (CPU)..."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo
echo "[6/7] Install application requirements..."
python -m pip install -r Documentation/docs/requirements-raspberry-pi.txt

echo
echo "[7/7] Verify camera..."
python App/test_pi_camera.py || true

echo
echo "=== Setup complete ==="
echo
echo "Before first run:"
echo "  1. Enable camera: sudo raspi-config -> Interface Options -> Camera -> Enable"
echo "  2. Reboot if you were added to the video group"
echo "  3. Optional Groq AI tips: copy .env.example to .env and set GROQ_API_KEY"
echo
echo "Run the app:"
echo "  cd \"$PROJECT_ROOT\""
echo "  source \"$VENV_PATH/bin/activate\""
echo "  python App/run_app.py"
echo
echo "Optional: enable Grad-CAM on Pi (slower):"
echo "  export PLANT_DISEASE_ENABLE_GRADCAM=1"
