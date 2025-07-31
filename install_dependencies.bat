@echo off

REM Check for Python
where python >NUL 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python 3 is required but was not found in your PATH.
    echo Please install Python 3.10 or later from https://www.python.org/ and try again.
    pause
    exit /b 1
)

REM Install ffmpeg and espeak using Chocolatey if available
if defined ChocolateyInstall (
    echo Installing ffmpeg and espeak with Chocolatey...
    choco install -y ffmpeg espeak
) else (
    echo If you have Chocolatey installed, run this script from an elevated shell to automatically install ffmpeg and espeak.
    echo Otherwise please install ffmpeg and espeak manually.
)

echo Installing required Python packages...
python -m pip install --upgrade pip
pip install numpy streamlit soundfile gTTS scipy psutil pydub pyttsx3

echo All necessary dependencies have been installed.
pause
