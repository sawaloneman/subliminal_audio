#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

if command -v apt-get >/dev/null 2>&1; then
  PKG_MGR="apt-get"
elif command -v apt >/dev/null 2>&1; then
  PKG_MGR="apt"
else
  echo "This script currently supports Debian/Ubuntu distributions with apt/apt-get."
  echo "Please install the required packages manually if you are using another distribution."
  exit 1
fi

echo "Updating package list..."
${SUDO} ${PKG_MGR} update

echo "Installing Python3 and pip3..."
${SUDO} ${PKG_MGR} install -y python3 python3-pip

echo "Installing ffmpeg..."
${SUDO} ${PKG_MGR} install -y ffmpeg

echo "Installing espeak and libespeak1..."
${SUDO} ${PKG_MGR} install -y espeak libespeak1

if command -v chromium-browser >/dev/null 2>&1; then
  echo "Chromium browser already installed."
else
  echo "Installing Chromium browser dependencies..."
  if ${SUDO} ${PKG_MGR} install -y chromium-browser; then
    echo "Chromium browser installed."
  else
    echo "Chromium browser package unavailable; attempting to install chromium."
    ${SUDO} ${PKG_MGR} install -y chromium || echo "Chromium browser could not be installed automatically."
  fi
fi

if command -v chromedriver >/dev/null 2>&1; then
  echo "Chromedriver already installed."
else
  echo "Installing Chromium driver dependencies..."
  ${SUDO} ${PKG_MGR} install -y chromium-chromedriver || \
    ${SUDO} ${PKG_MGR} install -y chromium-driver || \
    echo "Chromedriver package could not be installed automatically."
fi

echo "Upgrading pip..."
python3 -m pip install --upgrade pip

echo "Installing Python dependencies from requirements.txt..."
python3 -m pip install -r "${ROOT_DIR}/requirements.txt"

echo "All necessary dependencies have been installed."
