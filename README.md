# Aura Forge Subliminal Audio Suite

Aura Forge is a research-grade subliminal audio generator that blends text-to-speech
affirmations, precision binaural beat entrainment, morphic field sound design,
and coloured-noise atmospherics.  It runs headlessly (CLI / Google Colab) or as
a Streamlit experience that echoes Spotify's polished dark aesthetic.

## Highlights

- **Unlimited layering pipeline** – stack as many affirmations, binaural
  programs, ambient beds, and morphic textures as you want.
- **Science-aligned controls** – dial in carrier sweeps, beat schedules,
  harmonic spreads, and ADSR envelopes instead of vague presets.
- **Cloud-ready text-to-speech** – automatically chooses `pyttsx3` locally or
  `gTTS` in hosted notebooks such as Google Colab.
- **Streamlit UI** – craft sessions visually with an Aura/Spotify inspired look
  and one-click export to WAV, MP3, or FLAC.

## Installation

### One-line bootstrap (Debian/Ubuntu)

```bash
./install_dependencies.sh
```

The script installs Python 3, ffmpeg, espeak (for local TTS voices), libsndfile
(optional high-fidelity export), and all Python packages used by the toolkit.

### Manual / cross-platform setup

1. Install Python 3.9+ and ffmpeg.
2. Install Python dependencies:

   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install pydub numpy streamlit gTTS pyttsx3 soundfile
   ```

3. (Optional) install `espeak` or another speech engine when using `pyttsx3`.

## Google Colab quick start

1. Upload this repository or clone it inside Colab.
2. Run the following cell to install requirements and render a demo mix:

   ```python
   !pip install pydub numpy gTTS soundfile streamlit pyttsx3
   !apt-get install -y ffmpeg espeak

   from subliminal import demo_session, SubliminalSession
   from subliminal import TextToSpeechEngine

   session = SubliminalSession(demo_session(), tts_engine=TextToSpeechEngine("gtts"))
   audio = session.render()
   audio.export("aura_forge_demo.mp3", format="mp3")
   ```

3. Download `aura_forge_demo.mp3` from the Colab file browser.

## Command line usage

Render the built-in five minute experience:

```bash
python3 subliminal.py --output aura_forge.wav
```

Customise rendering with a JSON file:

```json
{
  "sample_rate": 48000,
  "duration_ms": 420000,
  "layers": [
    {
      "name": "Evening affirmations",
      "type": "tts",
      "gain_db": -6,
      "adsr": {"attack": 120, "decay": 320, "sustain": 0.8, "release": 700},
      "params": {
        "text": "My body harmonises with rejuvenating frequencies.",
        "rate": 150,
        "pitch": 35,
        "duration_ms": 420000
      }
    },
    {
      "name": "Gamma burst",
      "type": "binaural",
      "params": {
        "duration_ms": 420000,
        "carrier_start": 180.0,
        "carrier_end": 360.0,
        "beat_schedule": [[0.0, 12.0], [0.4, 40.0], [1.0, 18.0]],
        "amplitude": 0.75,
        "waveform": "triangle"
      }
    }
  ]
}
```

Save the JSON as `session.json` and run:

```bash
python3 subliminal.py --config session.json --output night_session.mp3
```

### Useful CLI flags

- `--preview` – limit the render to 30 seconds for quick iteration.
- `--backend gtts` – force Google TTS in environments without `pyttsx3`.
- `--format flac` – override the file format without changing the filename.

## Streamlit interface

1. Install dependencies as shown above.
2. Launch the UI:

   ```bash
   streamlit run streamlit_app.py
   ```

3. Use the sidebar to set session length, sample rate, and rendering backend.
4. Add unlimited layers – each layer exposes controls for gain, pan, ADSR,
   binaural beat schedules, morphic textures, and coloured noise beds.
5. Click **Render session** to preview the mix and download the master.

> Tip: For cloud notebooks, pair Streamlit with `pip install pyngrok` to tunnel
the UI.

## Project structure

- `subliminal.py` – core rendering engine plus CLI.
- `streamlit_app.py` – Aura Forge Streamlit UI.
- `install_dependencies.sh` – convenience installer for Debian/Ubuntu hosts.

## License

MIT
