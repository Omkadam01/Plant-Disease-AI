#!/usr/bin/env bash
# Install:
#  1) Boot autostart (app opens when Pi desktop starts)
#  2) Desktop icon (double-click like Windows .exe — no terminal)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$PROJECT_ROOT/App/start_plant_disease_pi.sh"
AUTOSTART_DIR="$HOME/.config/autostart"
AUTOSTART_FILE="$AUTOSTART_DIR/plantdiseaseai.desktop"
DESKTOP_DIR="$HOME/Desktop"
DESKTOP_ICON="$DESKTOP_DIR/PlantDiseaseAI.desktop"
ICON_SRC="$PROJECT_ROOT/App/resources/app_icon.ico"
ICON_PNG="$PROJECT_ROOT/App/resources/app_icon.png"

chmod +x "$LAUNCHER" "$PROJECT_ROOT/App/install_pi_autostart.sh" 2>/dev/null || true
mkdir -p "$AUTOSTART_DIR" "$DESKTOP_DIR"

# Optional PNG icon for desktop (from ico if ImageMagick/Pillow available).
if [ ! -f "$ICON_PNG" ] && [ -f "$ICON_SRC" ]; then
  if command -v convert >/dev/null 2>&1; then
    convert "$ICON_SRC" "$ICON_PNG" 2>/dev/null || true
  elif command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY' 2>/dev/null || true
from pathlib import Path
try:
    from PIL import Image
    root = Path(__file__).resolve().parent if False else Path.cwd()
except Exception:
    raise SystemExit(0)
src = Path("App/resources/app_icon.ico")
dst = Path("App/resources/app_icon.png")
if src.exists():
    Image.open(src).save(dst)
PY
  fi
fi

ICON_LINE=""
if [ -f "$ICON_PNG" ]; then
  ICON_LINE="Icon=$ICON_PNG"
elif [ -f "$ICON_SRC" ]; then
  ICON_LINE="Icon=$ICON_SRC"
fi

# Shared desktop entry body
read -r -d '' ENTRY <<EOF || true
[Desktop Entry]
Type=Application
Name=PlantDiseaseAI
Comment=Plant Disease Detection
Exec=$LAUNCHER
Path=$PROJECT_ROOT
Terminal=false
Categories=Education;Science;
StartupNotify=true
X-GNOME-Autostart-enabled=true
$ICON_LINE
EOF

printf '%s\n' "$ENTRY" >"$AUTOSTART_FILE"
printf '%s\n' "$ENTRY" >"$DESKTOP_ICON"
chmod +x "$DESKTOP_ICON"

# Allow double-click launch (mark as trusted) on Pi OS / LXDE / Wayfire.
if command -v gio >/dev/null 2>&1; then
  gio set "$DESKTOP_ICON" metadata::trusted true 2>/dev/null || true
fi

echo
echo "Done."
echo "  Boot autostart : $AUTOSTART_FILE"
echo "  Desktop icon   : $DESKTOP_ICON"
echo
echo "Next:"
echo "  1) Enable Desktop Autologin: sudo raspi-config"
echo "     System Options -> Boot / Auto Login -> Desktop Autologin"
echo "  2) Reboot: sudo reboot"
echo
echo "After reboot the app starts itself."
echo "Or double-click PlantDiseaseAI on the Desktop (no Terminal)."
echo
echo "Disable boot start later:"
echo "  rm \"$AUTOSTART_FILE\""
