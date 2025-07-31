
#!/bin/bash

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

# Installing Python packages
echo "Installing Python packages..."
pip3 install numpy streamlit soundfile gTTS scipy psutil pydub pyttsx3

echo "All necessary dependencies have been installed."
