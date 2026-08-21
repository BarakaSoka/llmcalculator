#!/bin/bash
# Build a single-file executable that needs no Python installed.
# Produces dist/llmcalculator (or dist\llmcalculator.exe on Windows).
#
# Run this on each OS you want to ship for: PyInstaller does not cross-compile.
set -e
cd "$(dirname "$0")/.."

python3 -m pip install --quiet --upgrade pyinstaller
python3 -m PyInstaller \
  --onefile \
  --name llmcalculator \
  --add-data "src/llmcalculator/models/catalog.json:llmcalculator/models" \
  --add-data "src/llmcalculator/ui/app.html:llmcalculator/ui" \
  --collect-all rich \
  --console \
  launchers/entrypoint.py

echo
echo "Built: dist/llmcalculator"
echo "Users can double-click it; it opens the app in their browser."
