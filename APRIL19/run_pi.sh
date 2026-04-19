#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing pip dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo
echo "Done."
echo "Manual steps may still be required. See MANUAL_INSTALL_NOTES.txt"
echo
echo "Examples:"
echo "  python3 inference.py --initialize"
echo "  python3 inference.py --mode picamera"
echo "  python3 inference.py --mode webcam"
echo "  python3 inference.py --mode video --source sample.mp4"
