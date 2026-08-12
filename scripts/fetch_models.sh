#!/usr/bin/env bash
# Download model weights. Runs on the Mac AND on the Pi — weights are never
# committed to git, so both sides fetch them from the same manifest.
set -euo pipefail

cd "$(dirname "$0")/../models"

while IFS=' ' read -r name url; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  if [[ -f "$name" ]]; then
    echo "skip  $name"
  else
    echo "fetch $name"
    curl -fsSL -o "$name" "$url"
  fi
done < manifest.txt

echo "Models present:"
ls -lh -- *.tflite *.txt 2>/dev/null || true
