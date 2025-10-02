#!/bin/bash
set -euo pipefail

# Updating the package list
echo "Updating package list..."
sudo apt update

# Installing Python3 and pip3 if not already installed
echo "Installing Python3 and pip3..."
sudo apt install -y python3 python3-pip

# Installing ffmpeg for audio processing
echo "Installing ffmpeg..."
sudo apt install -y ffmpeg

# Installing espeak and its library
echo "Installing espeak and libespeak1..."
sudo apt install -y espeak libespeak1

# Installing system dependencies required by Selenium (Chrome/Chromium)
echo "Installing Chromium browser and driver dependencies..."
sudo apt install -y chromium-browser chromium-chromedriver || sudo apt install -y chromium-driver

# Installing Python packages from requirements.txt
echo "Installing Python dependencies from requirements.txt..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo "All necessary dependencies have been installed."
