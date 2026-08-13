#!/usr/bin/env bash
# Sync the runtime subset of the repo to the Pi.
# Anything not matched by an --include stays on the Mac.
#
# tests/ IS deployed: `make test-pi` runs the hardware-marked tests on the
# device. They are the only tests that cannot run on the Mac, so shipping the
# suite is the point rather than an exception to it.
#
# Rule order matters: rsync uses the FIRST matching rule, so the excludes for
# build artefacts must come before the includes, otherwise 'src/***' would
# happily drag __pycache__ and *.egg-info along.
set -euo pipefail
 
HOST="${PI_HOST:-pi}"                 # matches the Host entry in ~/.ssh/config
REMOTE="${PI_PATH:-pi-edge-ai}"
 
echo "Deploying to ${HOST}:${REMOTE}"
 
rsync -az --delete \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.egg-info/' \
  --include='src/***' \
  --include='models/' \
  --include='models/*.tflite' \
  --include='models/*.txt' \
  --include='samples/' \
  --include='samples/*.jpg' \
  --include='samples/*.bmp' \
  --include='samples/*.png' \
  --include='tests/***' \
  --include='scripts/' \
  --include='scripts/bootstrap_pi.sh' \
  --include='requirements-pi.txt' \
  --include='pyproject.toml' \
  --exclude='*' \
  "$@" \
  ./ "${HOST}:${REMOTE}/"
 
echo
echo "Next on the Pi:"
echo "  ssh ${HOST} 'cd ${REMOTE} && .venv/bin/python -m edge.vision.classify IMAGE.jpg'"