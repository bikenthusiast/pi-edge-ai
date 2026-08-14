#!/usr/bin/env bash
# Download model weights. Runs on the Mac AND on the Pi — weights are never
# committed to git, so both sides fetch them from the same manifest.

# -e	cancel: script execution on any error
# -u	cancel: script execution on any unset variable
# -o pipefail:	any command in a pipeline fails, the entire pipeline fails

set -euo pipefail

# "$(dirname "$0"):path to this script, so we can cd to the models directory
cd "$(dirname "$0")/../models"

# Read the manifest file line by line, splitting each line into two variables: 'name' and 'url', 
# -r prevents backslash escapes from being interpreted, and IFS=' ' sets the internal field separator to a space, allowing us to split the line into two parts.
while IFS=' ' read -r name url; do
# Skip empty lines and comments in the manifest file. Lines starting with '#' are treated as comments.
 if [[ -z "$name" || "$name" == \#* ]]; then continue; fi
  # Checks if the file already exists, and skips downloading if it does. This is useful
  # for running the script multiple times without re-downloading the same files.
  if [[ -f "$name" ]]; then
    echo "skip  $name"
  else
    echo "fetch $name"
    # Download the file using curl. The -f option makes curl fail silently on server errors, -s makes it silent, 
    # -S shows errors if they occur, and -L follows redirects. The -o option specifies the output filename.
    curl -fsSL -o "$name" "$url"
  fi  
done < manifest.txt

echo "Models present:"
# List the downloaded model files in a human-readable format. The 'ls -lh' command lists files with their sizes in a human-readable format
# and the '--' option prevents issues with filenames starting with a dash. The '2>/dev/null || true' part suppresses error messages if no matching files are found
ls -lh -- *.tflite *.txt 2>/dev/null || true
