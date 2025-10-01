"""Streamlit dashboard for sculpting advanced subliminal audio sessions.

The UI is inspired by Spotify's dark mode aesthetic with aurora-like accents to
match the "Aura Phonk" visual requested by the user.  Every control maps onto
the signal-generation primitives implemented in ``subliminal.py`` so advanced
users can dial-in precise, science-backed sonic rituals.
"""

from __future__ import annotations

import dataclasses
import io
from typing import Any, Dict, List

import streamlit as st

from subliminal import (
    LayerConfig,
    SessionConfig,
    SubliminalSession,
    TextToSpeechEngine,
    demo_session,
)

AUDIO_FORMATS = {
    "High Fidelity WAV": ("wav", "audio/wav"),
    "Universal MP3": ("mp3", "audio/mpeg"),
    "Lossless FLAC": ("flac", "audio/flac"),
}

LAYER_TYPES = [
    "Text to Speech",
    "Binaural Beats",
    "Morphic Field",
    "Ambient Noise",
    "Upload",
]

THEME_CSS = """
<style>
:root {
    --spotify-black: #121212;
    --spotify-green: #1db954;
    --aura-purple: #7f5af0;
    --aura-cyan: #2cb1bc;
    --card-bg: rgba(255, 255, 255, 0.04);
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 20% 20%, rgba(32, 189, 255, 0.18), transparent 55%),
                radial-gradient(circle at 80% 10%, rgba(127, 90, 240, 0.28), transparent 45%),
                radial-gradient(circle at 50% 80%, rgba(15, 207, 155, 0.25), transparent 60%),
                var(--spotify-black);
    color: #f1f1f1;
}

h1, h2, h3, h4, h5, h6 {
    font-family: "Montserrat", sans-serif;
    letter-spacing: 0.04em;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(18, 18, 18, 0.95) 0%, rgba(18, 18, 18, 0.6) 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

.sidebar .sidebar-content {
    color: #f8f8f8;
}

.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, var(--spotify-green), var(--aura-purple));
    color: #121212;
    border: none;
    border-radius: 999px;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.8rem;
    box-shadow: 0 12px 22px rgba(0, 0, 0, 0.35);
}

.stButton>button:hover, .stDownloadButton>button:hover {
    filter: brightness(1.1);
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    letter-spacing: 0.05em;
}

.stExpander {
    background-color: var(--card-bg) !important;
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.stSlider>div>div>div>div {
    background: linear-gradient(90deg, var(--aura-cyan), var(--aura-purple));
}
</style>
"""


def _init_state() -> None:
    if "layer_configs" not in st.session_state:
        demo = demo_session()
        st.session_state.layer_configs = [dataclasses.asdict(layer) for layer in demo.layers]
        st.session_state.global_settings = {
            "sample_rate": demo.sample_rate,
            "duration_ms": demo.duration_ms,
            "target_level_db": demo.target_level_db,
            "normalize_output": demo.normalize_output,
        }
        st.session_state.backend = "auto"
        st.session_state.output_format_label = "High Fidelity WAV"


def _adsr_controls(layer: Dict[str, Any], key_prefix: str) -> Dict[str, float]:
    enabled = st.checkbox("Enable ADSR shaping", value=layer.get("adsr") is not None, key=f"adsr_enable_{key_prefix}")
    if not enabled:
        return {}
    defaults = layer.get("adsr") or {"attack": 120.0, "decay": 300.0, "sustain": 0.75, "release": 600.0}
    attack = st.slider("Attack (ms)", 0, 4000, int(defaults.get("attack", 120)), key=f"adsr_attack_{key_prefix}")
    decay = st.slider("Decay (ms)", 0, 4000, int(defaults.get("decay", 300)), key=f"adsr_decay_{key_prefix}")
    sustain = st.slider("Sustain level", 0.0, 1.0, float(defaults.get("sustain", 0.75)), key=f"adsr_sustain_{key_prefix}")
    release = st.slider("Release (ms)", 0, 4000, int(defaults.get("release", 600)), key=f"adsr_release_{key_prefix}")
    return {"attack": float(attack), "decay": float(decay), "sustain": float(sustain), "release": float(release)}


def _layer_controls(index: int, layer: Dict[str, Any]) -> Dict[str, Any]:
    key_prefix = f"layer_{index}"
    with st.expander(f"Layer {index + 1}: {layer['name']}", expanded=True):
        col_a, col_b, col_c = st.columns([3, 2, 2])
        with col_a:
            name = st.text_input("Layer name", value=layer.get("name", f"Layer {index + 1}"), key=f"name_{key_prefix}")
        with col_b:
            current_label = layer.get("type_readable", "Text to Speech")
            if current_label not in LAYER_TYPES:
                current_label = "Text to Speech"
            layer_type_label = st.selectbox(
                "Layer type",
                LAYER_TYPES,
                index=LAYER_TYPES.index(current_label),
                key=f"type_{key_prefix}",
            )
        with col_c:
            gain = st.slider("Gain (dB)", -36, 12, int(layer.get("gain_db", 0)), key=f"gain_{key_prefix}")

        pan = st.slider("Stereo pan", -1.0, 1.0, float(layer.get("pan", 0.0)), key=f"pan_{key_prefix}")
        adsr = _adsr_controls(layer, key_prefix)

        params: Dict[str, Any] = {}
        duration_default = int(layer.get("params", {}).get("duration_ms", st.session_state.global_settings["duration_ms"]))

        if layer_type_label == "Text to Speech":
            params["text"] = st.text_area(
                "Affirmation script",
                value=layer.get("params", {}).get(
                    "text",
                    "I effortlessly embody confidence, clarity, and creative flow.",
                ),
                key=f"text_{key_prefix}",
            )
            params["rate"] = st.slider("Speech rate", 90, 240, int(layer.get("params", {}).get("rate", 150)), key=f"rate_{key_prefix}")
            params["pitch"] = st.slider("Pitch offset", -600, 600, int(layer.get("params", {}).get("pitch", 0)), key=f"pitch_{key_prefix}")
            params["volume"] = st.slider("Speech volume", 0.1, 1.2, float(layer.get("params", {}).get("volume", 0.9)), key=f"volume_{key_prefix}")
            params["language"] = st.selectbox("Language", ["en", "es", "fr", "de", "it", "pt"], key=f"lang_{key_prefix}")
            params["duration_ms"] = st.slider("Loop to duration (ms)", 1000, 600_000, duration_default, step=1000, key=f"dur_{key_prefix}")
            layer_type = "tts"
        elif layer_type_label == "Binaural Beats":
            params["duration_ms"] = st.slider("Duration (ms)", 10_000, 900_000, duration_default, step=1000, key=f"dur_{key_prefix}")
            start_carrier = st.number_input("Carrier start (Hz)", 20.0, 2000.0, float(layer.get("params", {}).get("carrier_start", 110.0)), key=f"carrier_start_{key_prefix}")
            end_carrier = st.number_input("Carrier end (Hz)", 20.0, 2000.0, float(layer.get("params", {}).get("carrier_end", 174.0)), key=f"carrier_end_{key_prefix}")
            start_beat = st.number_input("Beat start (Hz)", 0.1, 40.0, float(layer.get("params", {}).get("beat_start", 7.0)), key=f"beat_start_{key_prefix}")
            mid_beat = st.number_input("Mid-session beat (Hz)", 0.1, 40.0, float(layer.get("params", {}).get("beat_mid", 4.5)), key=f"beat_mid_{key_prefix}")
            end_beat = st.number_input("Beat end (Hz)", 0.1, 40.0, float(layer.get("params", {}).get("beat_end", 12.0)), key=f"beat_end_{key_prefix}")
            params["carrier_start"] = start_carrier
            params["carrier_end"] = end_carrier
            params["beat_schedule"] = [(0.0, start_beat), (0.5, mid_beat), (1.0, end_beat)]
            params["amplitude"] = st.slider("Amplitude", 0.05, 1.5, float(layer.get("params", {}).get("amplitude", 0.8)), key=f"amp_{key_prefix}")
            params["tremolo_hz"] = st.slider("Tremolo rate (Hz)", 0.0, 8.0, float(layer.get("params", {}).get("tremolo_hz", 0.4)), key=f"trem_{key_prefix}")
            params["tremolo_depth"] = st.slider("Tremolo depth", 0.0, 1.0, float(layer.get("params", {}).get("tremolo_depth", 0.35)), key=f"trem_depth_{key_prefix}")
            params["waveform"] = st.selectbox("Waveform", ["sine", "triangle", "square"], key=f"waveform_{key_prefix}")
            layer_type = "binaural"
        elif layer_type_label == "Morphic Field":
            params["duration_ms"] = st.slider("Duration (ms)", 10_000, 900_000, duration_default, step=1000, key=f"dur_{key_prefix}")
            frequencies = st.text_input(
                "Carrier frequencies (Hz, comma separated)",
                value=", ".join(str(f) for f in layer.get("params", {}).get("carrier_frequencies", [174.0, 285.0, 396.0, 528.0])),
                key=f"freqs_{key_prefix}",
            )
            params["carrier_frequencies"] = [float(f.strip()) for f in frequencies.split(",") if f.strip()]
            params["harmonic_spread"] = st.slider("Harmonic spread", 1.0, 4.0, float(layer.get("params", {}).get("harmonic_spread", 2.5)), key=f"spread_{key_prefix}")
            params["modulation_rate_hz"] = st.slider("Modulation rate (Hz)", 0.05, 2.0, float(layer.get("params", {}).get("modulation_rate_hz", 0.25)), key=f"mod_rate_{key_prefix}")
            params["modulation_depth"] = st.slider("Modulation depth", 0.0, 1.0, float(layer.get("params", {}).get("modulation_depth", 0.45)), key=f"mod_depth_{key_prefix}")
            params["noise_colour"] = st.selectbox("Noise colour", ["white", "pink", "brown"], index=["white", "pink", "brown"].index(layer.get("params", {}).get("noise_colour", "pink")), key=f"noise_colour_{key_prefix}")
            params["noise_level"] = st.slider("Noise blend", 0.0, 1.0, float(layer.get("params", {}).get("noise_level", 0.3)), key=f"noise_level_{key_prefix}")
            params["amplitude"] = st.slider("Field amplitude", 0.05, 1.5, float(layer.get("params", {}).get("amplitude", 0.6)), key=f"amp_{key_prefix}")
            layer_type = "morphic_field"
        elif layer_type_label == "Ambient Noise":
            params["duration_ms"] = st.slider("Duration (ms)", 10_000, 900_000, duration_default, step=1000, key=f"dur_{key_prefix}")
            params["colour"] = st.selectbox("Noise colour", ["white", "pink", "brown"], index=["white", "pink", "brown"].index(layer.get("params", {}).get("colour", "brown")), key=f"noise_colour_{key_prefix}")
            params["amplitude"] = st.slider("Amplitude", 0.05, 1.5, float(layer.get("params", {}).get("amplitude", 0.45)), key=f"amp_{key_prefix}")
            layer_type = "noise"
        else:  # Upload
            params["duration_ms"] = st.slider("Duration (ms)", 5_000, 900_000, duration_default, step=1000, key=f"dur_{key_prefix}")
            upload = st.file_uploader("Upload audio (mp3/wav/flac)", type=["wav", "mp3", "flac", "ogg"], key=f"upload_{key_prefix}")
            if upload is not None:
                params["path"] = {"name": upload.name, "data": upload.getvalue()}
            else:
                existing = layer.get("params", {}).get("path")
                if isinstance(existing, dict) and existing.get("data"):
                    st.caption(f"Reusing uploaded asset: {existing.get('name', 'custom audio')}")
                    params["path"] = existing
            layer_type = "file"

        col_remove, col_blank = st.columns([1, 3])
        with col_remove:
            if st.button("Remove layer", key=f"remove_{key_prefix}"):
                st.session_state.layer_configs.pop(index)
                st.experimental_rerun()

    return {
        "name": name,
        "type": layer_type,
        "gain_db": float(gain),
        "pan": float(pan),
        "adsr": adsr or None,
        "params": params,
        "type_readable": layer_type_label,
    }


def _build_session_config() -> SessionConfig:
    layer_dicts = st.session_state.layer_configs
    layers = [LayerConfig(name=layer["name"], type=layer["type"], gain_db=layer["gain_db"], pan=layer.get("pan", 0.0), adsr=layer.get("adsr"), params=layer.get("params", {})) for layer in layer_dicts]
    globals_cfg = st.session_state.global_settings
    return SessionConfig(
        layers=layers,
        sample_rate=int(globals_cfg["sample_rate"]),
        duration_ms=int(globals_cfg["duration_ms"]),
        target_level_db=float(globals_cfg["target_level_db"]),
        normalize_output=bool(globals_cfg["normalize_output"]),
    )


def main() -> None:
    st.set_page_config(page_title="Aura Forge | Subliminal Architect", page_icon="🎧", layout="wide")
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    _init_state()

    st.title("Aura Forge")
    st.subheader("Design intensely potent subliminal fields with surgical precision.")

    with st.sidebar:
        st.header("Session blueprint")
        duration_minutes = st.slider(
            "Session length (minutes)",
            1,
            90,
            int(st.session_state.global_settings["duration_ms"] // 60000),
        )
        st.session_state.global_settings["duration_ms"] = duration_minutes * 60_000
        st.session_state.global_settings["sample_rate"] = st.selectbox("Sample rate", [44100, 48000, 88200], index=[44100, 48000, 88200].index(st.session_state.global_settings.get("sample_rate", 48000)))
        st.session_state.global_settings["target_level_db"] = st.slider("Master peak (dBFS)", -12.0, -0.1, float(st.session_state.global_settings.get("target_level_db", -1.0)), step=0.1)
        st.session_state.global_settings["normalize_output"] = st.toggle("Normalise master", value=st.session_state.global_settings.get("normalize_output", True))
        st.session_state.backend = st.selectbox("TTS backend", ["auto", "pyttsx3", "gtts"], index=["auto", "pyttsx3", "gtts"].index(st.session_state.backend))
        st.session_state.output_format_label = st.selectbox("Render format", list(AUDIO_FORMATS.keys()), index=list(AUDIO_FORMATS.keys()).index(st.session_state.output_format_label))
        if st.button("Add new layer", use_container_width=True):
            st.session_state.layer_configs.append(
                {
                    "name": f"Layer {len(st.session_state.layer_configs) + 1}",
                    "type": "tts",
                    "type_readable": "Text to Speech",
                    "gain_db": -6.0,
                    "pan": 0.0,
                    "adsr": {"attack": 120.0, "decay": 300.0, "sustain": 0.7, "release": 700.0},
                    "params": {
                        "text": "My energy resonates with abundance.",
                        "rate": 150,
                        "pitch": 20,
                        "volume": 0.9,
                        "duration_ms": st.session_state.global_settings["duration_ms"],
                    },
                }
            )

    updated_layers: List[Dict[str, Any]] = []
    for idx, layer in enumerate(st.session_state.layer_configs):
        if "type_readable" not in layer:
            mapping = {
                "tts": "Text to Speech",
                "binaural": "Binaural Beats",
                "morphic_field": "Morphic Field",
                "noise": "Ambient Noise",
                "file": "Upload",
            }
            layer["type_readable"] = mapping.get(layer["type"], "Text to Speech")
        updated_layers.append(_layer_controls(idx, layer))

    st.session_state.layer_configs = updated_layers

    st.divider()
    st.markdown("### Render preview")
    if st.button("Render session", type="primary"):
        with st.spinner("Sculpting resonant field..."):
            session_cfg = _build_session_config()
            backend = st.session_state.backend
            engine = TextToSpeechEngine(backend=backend)
            audio = SubliminalSession(session_cfg, tts_engine=engine).render()
            fmt_label = st.session_state.output_format_label
            fmt, mime = AUDIO_FORMATS[fmt_label]
            buffer = io.BytesIO()
            audio.export(buffer, format=fmt)
            buffer.seek(0)
            payload = buffer.getvalue()
        st.audio(payload, format=mime)
        st.download_button(
            "Download master",
            data=payload,
            file_name=f"aura_forge_session.{fmt}",
            mime=mime,
        )

    st.caption(
        "Crafted with psychoacoustic principles, harmonic entrainment, and fractal sound design to amplify intent without pseudo-science fluff."
    )


if __name__ == "__main__":
    main()
