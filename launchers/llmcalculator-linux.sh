#!/bin/bash
# Run this file to open llmcalculator in your browser:  ./llmcalculator-linux.sh
cd "$(dirname "$0")" || exit 1

PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done

if [ -z "$PY" ]; then
  echo "Python is not installed. Install it with your package manager, for example:"
  echo "  sudo apt install python3 python3-pip"
  exit 1
fi

if ! "$PY" -c "import llmcalculator" >/dev/null 2>&1; then
  echo "Installing llmcalculator (one time only)..."
  "$PY" -m pip install --user --quiet llmcalculator || {
    echo "Install failed. Try:  $PY -m pip install --user llmcalculator"
    exit 1
  }
fi

echo "Starting llmcalculator..."
exec "$PY" -m llmcalculator.cli app
