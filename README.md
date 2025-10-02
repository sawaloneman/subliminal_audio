
# Subliminal Audio Application

This repository contains a Python script for creating subliminal audio tracks with adjustable features such as speed, pitch, and volume. Additionally, it supports multiple layers of audio and text-to-speech capabilities.

## Installation

Run the following script to install all necessary dependencies. This script installs Python 3, pip, ffmpeg, espeak, and the required Python libraries.

```bash
./install_dependencies.sh
```

## Usage

### Streamlit generator

To use the classic standalone script, run:

```bash
python3 subliminal.py
```

### Automation agent

The repository now includes `automation_agent.py`, an AI-assisted workflow for
automating subliminal audio video creation and publishing. The agent expects an
existing Streamlit app that exposes a `generate_subliminal_audio(...)`
function. Configure the required API credentials (OpenAI, YouTube, Selenium)
via environment variables, then launch the agent:

```bash
python3 automation_agent.py
```

You can choose between manual mode (step-by-step configuration for each run)
and auto mode (randomized settings every five minutes).

## Features

- Text to speech conversion
- Adjust playback speed
- Combine multiple audio tracks
- Export audio to a file

## Requirements

- Python 3
- pip
- ffmpeg
- espeak
- pydub
- pyttsx3
- google-auth
- google-auth-oauthlib
- google-api-python-client
