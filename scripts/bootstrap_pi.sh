#!/usr/bin/env bash
# One-time setup ON THE PI. Run once after the first deploy.
#   ssh pi 'cd pi-edge-ai && bash scripts/bootstrap_pi.sh'
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Warning: $(uname -m) detected. ai-edge-litert has no 32-bit wheels." >&2
fi

sudo apt-get update
sudo apt-get install -y python3-venv python3-opencv v4l-utils \
  mosquitto mosquitto-clients   # MQTT broker + CLI, see docs/mqtt.md

# --system-site-packages is what makes the apt build of OpenCV (and picamera2)
# visible inside the venv. Without it you would rebuild OpenCV via pip.
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements-pi.txt
.venv/bin/pip install -e . --no-deps        # makes `python -m edge.…` work
.venv/bin/pip install pytest                # for `make test-pi`

.venv/bin/python -c "import cv2, numpy; print('OpenCV', cv2.__version__)"
.venv/bin/python -c "from ai_edge_litert.interpreter import Interpreter; print('LiteRT ready')"

echo "Bootstrap complete."
