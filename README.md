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

### Streamlit application

Launch the Streamlit interface to experiment with the generator in your
browser. The app provides two tabs: a fully manual workflow and an AI-assisted
experience that uses OpenAI to suggest affirmations and metadata. The manual
tab does not require any API keys. Both modes now expose the recursive
auto-layering controls so you can stack subtly different ambience beds without
leaving the UI.

```bash
streamlit run streamlit_app.py
```

### Classic script

If you prefer the original sample script, execute:

```bash
python3 subliminal.py
```

### Automation agent

The repository now includes `automation_agent.py`, an AI-assisted workflow for
automating subliminal audio video creation and publishing. Configure the
required API credentials (OpenAI, YouTube, Selenium) via environment variables,
then launch the agent:

```bash
python3 automation_agent.py
```

On startup the script offers three options:

1. **Manual generator (no AI agent)** – craft affirmations yourself, optionally
   convert to video, and upload to YouTube with your own metadata.
2. **AI agent – manual confirmation** – review each automated run before video
   conversion and publishing.
3. **AI agent – automated schedule** – run every five minutes with randomized
   settings.

If the `OPENAI_API_KEY` environment variable is not set, the agent will prompt
you to securely enter the key at runtime. When a `GOOGLE_CLIENT_SECRETS` path
is supplied, the script launches the standard YouTube OAuth consent flow so you
can log in and approve uploads before the automation proceeds.

When using either AI-driven option you can now set an affirmation theme. The
agent threads this theme through the OpenAI prompts so the generated
affirmations, metadata, and thumbnail direction align with the creative goal.

If your Streamlit module or generator function are named differently, export
`STREAMLIT_APP_MODULE` and/or `STREAMLIT_GENERATOR_NAME` so the automation agent
can import the correct callable. When unset, it searches common module names
such as `streamlit_app`, `app`, `main`, and `subliminal` for a
`generate_subliminal_audio` function.

## Features

- Text to speech conversion
- Adjust playback speed
- Recursive auto-layered ambience with per-layer variation controls
- AI-assisted affirmation, metadata, and thumbnail generation driven by a
  user-specified theme
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
