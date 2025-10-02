#!/usr/bin/env bash
set -euo pipefail

# Determine whether sudo is available and required
SUDO=""
if command -v sudo >/dev/null 2>&1; then
  if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    SUDO="sudo"
  fi
fi

# Helper to run a command and show a friendly message
run_cmd() {
  local message="$1"
  shift
  echo "$message"
  "$@"
}

if command -v apt-get >/dev/null 2>&1; then
  run_cmd "Updating package list..." ${SUDO} apt-get update
  run_cmd "Installing Python3 and pip3..." ${SUDO} apt-get install -y python3 python3-pip
  run_cmd "Installing ffmpeg and audio libraries..." ${SUDO} apt-get install -y ffmpeg espeak libespeak1 libsndfile1
else
  echo "Unsupported operating system: please install Python 3, pip, ffmpeg, and espeak manually."
  exit 1
fi

run_cmd "Upgrading pip..." python3 -m pip install --upgrade pip
run_cmd "Installing Python packages..." python3 -m pip install --upgrade pydub pyttsx3 numpy gTTS streamlit soundfile

echo "All necessary dependencies have been installed."
