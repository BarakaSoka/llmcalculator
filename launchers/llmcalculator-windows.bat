@echo off
REM Double-click this file to open llmcalculator in your browser.
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python is not installed.
  echo Get it from https://www.python.org/downloads/ then double-click this file again.
  echo Tick "Add Python to PATH" during installation.
  pause
  exit /b 1
)

python -c "import llmcalculator" >nul 2>&1
if errorlevel 1 (
  echo Installing llmcalculator ^(one time only^)...
  python -m pip install --user --quiet llmcalculator
  if errorlevel 1 (
    echo Install failed. Try running:  python -m pip install llmcalculator
    pause
    exit /b 1
  )
)

echo Starting llmcalculator...
python -m llmcalculator.cli app
pause
