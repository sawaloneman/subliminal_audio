
# Superliminal Audio Application

This repository provides a Streamlit based tool for creating subliminal audio tracks with advanced sound design features.  Users can apply multiple audio effects, generate binaural or isochronic tones and mix in noise or text-to-speech layers.

## Installation

Run the appropriate script to install all necessary dependencies.  The script installs Python 3, pip, ffmpeg, espeak and the required Python libraries.

On Linux or WSL:

```bash
./install_dependencies.sh
```

On Windows 11:

```bat
install_dependencies.bat
```

## Usage

### Streamlit Interface

Launch the graphical interface with:

```bash
streamlit run superliminal_streamlit.py
```

### Command Line Interface

Automation tools such as n8n should call the dedicated CLI script:

```bash
python superliminal_cli.py --input input.wav --output out.wav --fx "Micro Delay,Phase Flips" --speed 1.2
```

## Features

- Text to speech conversion
- Multiple noise generators
- Binaural and isochronic beats
- Extensive audio effects
- Benchmarking of processing speed
- Adjustable playback speed

## Requirements

- Python 3
- pip
- ffmpeg
- espeak
- numpy
- streamlit
- soundfile
- gTTS
- scipy
- psutil
- pydub
- pyttsx3
