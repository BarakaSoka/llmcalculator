#!/bin/bash
# Double-click this file to open llmcalculator in your browser.
# macOS may ask you to allow it the first time: right-click > Open.
cd "$(dirname "$0")" || exit 1

PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done

if [ -z "$PY" ]; then
  echo "Python is not installed."
  echo "Get it from https://www.python.org/downloads/ then double-click this file again."
  read -r -p "Press Return to close."
  exit 1
fi

if ! "$PY" -c "import llmcalculator" >/dev/null 2>&1; then
  echo "Installing llmcalculator (one time only)..."
  "$PY" -m pip install --user --quiet llmcalculator || {
    echo "Install failed. Try running:  $PY -m pip install llmcalculator"
    read -r -p "Press Return to close."
    exit 1
  }
fi

echo "Starting llmcalculator..."
exec "$PY" -m llmcalculator.cli app
