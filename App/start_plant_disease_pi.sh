#!/usr/bin/env bash
# Launch PlantDiseaseAI on Raspberry Pi (used by desktop autostart).
set -euo pipefail

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Wait until the graphical session is ready (avoids crash on early boot).
for _ in $(seq 1 60); do
  if [ -n "${DISPLAY:-}" ] && command -v xset >/dev/null 2>&1; then
    if xset q >/dev/null 2>&1; then
      break
    fi
  fi
  sleep 2
done
sleep 3

# Candidate project locations (home clone first, then common USB mount).
PROJECT_CANDIDATES=(
  "$HOME/Plant_DiseaseAI"
  "/media/$USER/152E-F056/Plant_DiseaseAI"
  "/media/smartlabsu/152E-F056/Plant_DiseaseAI"
)

PROJECT_ROOT=""
for candidate in "${PROJECT_CANDIDATES[@]}"; do
  if [ -f "$candidate/App/run_app.py" ]; then
    PROJECT_ROOT="$candidate"
    break
  fi
done

if [ -z "$PROJECT_ROOT" ]; then
  echo "PlantDiseaseAI project not found. Tried: ${PROJECT_CANDIDATES[*]}" >&2
  exit 1
fi

# Candidate virtualenvs.
VENV_CANDIDATES=(
  "$PROJECT_ROOT/.venv"
  "$HOME/plantdisease-venv"
  "/media/$USER/New Volume/plantdisease-venv"
  "/media/smartlabsu/New Volume/plantdisease-venv"
)

VENV_DIR=""
for candidate in "${VENV_CANDIDATES[@]}"; do
  if [ -f "$candidate/bin/activate" ]; then
    VENV_DIR="$candidate"
    break
  fi
done

if [ -z "$VENV_DIR" ]; then
  echo "No virtualenv found for PlantDiseaseAI." >&2
  exit 1
fi

# Optional: mount New Volume if venv lives there and is missing.
if [ ! -f "/media/$USER/New Volume/plantdisease-venv/bin/activate" ] \
  && [ ! -f "/media/smartlabsu/New Volume/plantdisease-venv/bin/activate" ] \
  && [ -b /dev/mmcblk0p3 ]; then
  mkdir -p "/media/$USER/New Volume" 2>/dev/null || true
  mount /dev/mmcblk0p3 "/media/$USER/New Volume" 2>/dev/null || true
fi

cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

LOG_DIR="$PROJECT_ROOT/Outputs/runtime/logs/desktop"
mkdir -p "$LOG_DIR"
exec python App/run_app.py >>"$LOG_DIR/autostart.log" 2>&1
