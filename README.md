# Subliminal Audio Application

This repository contains a Python script for creating subliminal audio tracks with adjustable features such as speed, pitch, and
 volume. Additionally, it supports multiple layers of audio and text-to-speech capabilities.

## Installation

Run the following script to install all necessary system and Python dependencies. This script installs Python 3, pip, ffmpeg,
espeak, Chromium (for Selenium automation), and every required Python library.

```bash
./install_dependencies.sh
```

Alternatively, if you only need the Python packages you can install them directly:

```bash
pip install -r requirements.txt
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

If the `OPENAI_API_KEY` environment variable is not set, the agent will prompt
you to securely enter the key at runtime. When a `GOOGLE_CLIENT_SECRETS` path
is supplied, the script launches the standard YouTube OAuth consent flow so you
can log in and approve uploads before the automation proceeds.

If your Streamlit module or generator function are named differently, export
`STREAMLIT_APP_MODULE` and/or `STREAMLIT_GENERATOR_NAME` so the automation agent
can import the correct callable. When unset, it searches common module names
such as `streamlit_app`, `app`, `main`, and `subliminal` for a
`generate_subliminal_audio` function.

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
- Chromium browser and chromedriver (for Selenium automation)
- pydub
- pyttsx3
- openai
- streamlit
- selenium
- google-auth
- google-auth-oauthlib
- google-api-python-client
