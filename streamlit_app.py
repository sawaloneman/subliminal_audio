"""Streamlit front-end for the subliminal audio generator and AI agent."""

from __future__ import annotations

import importlib
import os
import random
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import streamlit as st
from pydub import AudioSegment, effects
from pydub.generators import Sine, WhiteNoise

AVAILABLE_NOISE_TYPES: Sequence[str] = (
    "purple",
    "brown",
    "pink",
    "center",
    "binaural",
    "white",
    "none",
)


def _sanitize_filename(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def _synthesize_affirmations(
    affirmations: Sequence[str],
    rate: int = 160,
    volume: float = 1.0,
) -> AudioSegment:
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", max(0.0, min(volume, 1.0)))

    text = " \n".join(affirmations)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        temp_path = Path(tmp_file.name)

    try:
        engine.save_to_file(text, str(temp_path))
        engine.runAndWait()
        segment = AudioSegment.from_file(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return segment


def _change_speed(segment: AudioSegment, speed: float) -> AudioSegment:
    if speed == 1.0:
        return segment
    return segment._spawn(segment.raw_data, overrides={"frame_rate": int(segment.frame_rate * speed)}).set_frame_rate(
        segment.frame_rate
    )


def _build_noise(noise_type: str, duration_ms: int) -> AudioSegment:
    base = WhiteNoise().to_audio_segment(duration=duration_ms)

    if noise_type == "pink":
        return base.low_pass_filter(1200).apply_gain(-8)
    if noise_type == "brown":
        return base.low_pass_filter(800).apply_gain(-10)
    if noise_type == "purple":
        return base.high_pass_filter(1800).apply_gain(-10)
    if noise_type == "center":
        tone = Sine(432).to_audio_segment(duration=duration_ms).apply_gain(-9)
        return base.overlay(tone)
    if noise_type == "binaural":
        left = Sine(210).to_audio_segment(duration=duration_ms).apply_gain(-12)
        right = Sine(214).to_audio_segment(duration=duration_ms).apply_gain(-12)
        return AudioSegment.from_mono_audiosegments(left, right)
    if noise_type == "none":
        return AudioSegment.silent(duration=duration_ms)
    return base.apply_gain(-8)


def _ensure_duration(track: AudioSegment, target_ms: int) -> AudioSegment:
    if len(track) >= target_ms:
        return track[:target_ms]
    loops = (target_ms // len(track)) + 1
    extended = track * loops
    return extended[:target_ms]


def _apply_layer_variation(
    segment: AudioSegment,
    *,
    variation_strength: float,
    rng: random.Random,
    layer_index: int,
    target_duration: int,
) -> AudioSegment:
    """Apply subtle, layer-specific tweaks to the noise bed."""

    if variation_strength <= 0:
        return segment

    base_offset = (layer_index % 3 - 1) * 1.2 * variation_strength
    segment = segment.apply_gain(base_offset)

    gain_shift = rng.uniform(-6.0, 2.5) * variation_strength
    segment = segment.apply_gain(gain_shift)

    if rng.random() < 0.5 * variation_strength:
        cutoff = rng.randint(600, 1800)
        segment = segment.low_pass_filter(cutoff)
    else:
        cutoff = rng.randint(1200, 4200)
        segment = segment.high_pass_filter(cutoff)

    if segment.channels == 1:
        pan_amount = max(-1.0, min(1.0, rng.uniform(-0.7, 0.7) * variation_strength))
        segment = segment.pan(pan_amount)

    stretch = 1.0 + rng.uniform(-0.12, 0.12) * variation_strength
    if stretch != 1.0:
        segment = _change_speed(segment, stretch)

    segment = _ensure_duration(segment, target_duration)

    return segment


def auto_layer_noise(
    noise_type: str,
    duration_ms: int,
    *,
    layer_count: int,
    variation_strength: float,
    seed: Optional[int] = None,
) -> AudioSegment:
    """Recursively build a layered ambient bed with subtle variation."""

    layer_count = max(1, layer_count)
    rng = random.Random(seed)

    def _build_variant(index: int) -> AudioSegment:
        layer = _ensure_duration(_build_noise(noise_type, duration_ms), duration_ms)
        return _apply_layer_variation(
            layer,
            variation_strength=variation_strength,
            rng=rng,
            layer_index=index,
            target_duration=duration_ms,
        )

    def _recursive_overlay(current_index: int, acc: AudioSegment) -> AudioSegment:
        if current_index >= layer_count:
            return acc
        next_layer = _build_variant(current_index)
        combined = acc.overlay(next_layer, gain_during_overlay=-1)
        return _recursive_overlay(current_index + 1, combined)

    base_layer = _build_variant(0)
    return _recursive_overlay(1, base_layer)


def generate_subliminal_audio(
    *,
    noise_type: str,
    affirmations: Sequence[str],
    playback_speed: float,
    output_dir: str,
    voice_rate: int = 160,
    voice_volume: float = 1.0,
    base_filename: str = "subliminal_mix",
    auto_layer: bool = True,
    layer_count: int = 3,
    layer_variation: float = 0.4,
    layer_seed: Optional[int] = None,
) -> Path:
    """Create an audio file containing the affirmations mixed with ambient noise."""

    if not affirmations:
        raise ValueError("At least one affirmation is required.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    speech = _synthesize_affirmations(affirmations, rate=voice_rate, volume=voice_volume)
    speech = _change_speed(speech, playback_speed)

    if auto_layer:
        noise = auto_layer_noise(
            noise_type,
            len(speech),
            layer_count=layer_count,
            variation_strength=max(0.0, min(layer_variation, 1.0)),
            seed=layer_seed,
        )
    else:
        noise = _build_noise(noise_type, len(speech))
    if noise.channels != speech.channels:
        speech = speech.set_channels(noise.channels)

    balanced_noise = _ensure_duration(noise, len(speech))
    mixed = balanced_noise.overlay(speech + 6)
    final_track = effects.normalize(mixed)

    filename = f"{_sanitize_filename(base_filename)}.mp3"
    final_path = output_path / filename
    final_track.export(final_path, format="mp3")
    return final_path


def _load_openai_helpers():
    automation_agent = importlib.import_module("automation_agent")
    OpenAI = automation_agent._load_openai_client()  # type: ignore[attr-defined]
    return {
        "OpenAI": OpenAI,
        "generate_affirmations": automation_agent.generate_affirmations,
        "generate_metadata": automation_agent.generate_metadata,
        "generate_thumbnail": automation_agent.generate_thumbnail,
    }


def _render_manual_tab():
    st.subheader("Manual Generator (No AI Agent)")

    with st.form("manual-generator"):
        noise_type = st.selectbox("Noise profile", AVAILABLE_NOISE_TYPES, index=0)
        playback_speed = st.slider("Playback speed", min_value=0.5, max_value=2.0, value=1.0, step=0.05)
        voice_rate = st.slider("Voice rate", min_value=120, max_value=220, value=160, step=5)
        voice_volume = st.slider("Voice volume", min_value=0.3, max_value=1.0, value=0.9, step=0.05)
        filename = st.text_input("Base file name", "subliminal_mix")
        auto_layer = st.checkbox("Auto-layer ambient bed", value=True)
        layer_count = st.slider("Layer count", min_value=1, max_value=8, value=3)
        layer_variation = st.slider("Layer variation", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        layer_seed_input = st.text_input("Layer seed (optional)", "")
        affirmations_text = st.text_area(
            "Affirmations",
            "I am calm and focused.\nI attract positive energy.\nMy mind and body are aligned.",
            height=160,
        )
        submitted = st.form_submit_button("Generate audio")

    if not submitted:
        return

    affirmations = [line.strip() for line in affirmations_text.splitlines() if line.strip()]
    if not affirmations:
        st.error("Please provide at least one affirmation.")
        return

    layer_seed: Optional[int]
    if layer_seed_input.strip():
        try:
            layer_seed = int(layer_seed_input.strip())
        except ValueError:
            st.error("Layer seed must be an integer if provided.")
            return
    else:
        layer_seed = None

    tmp_dir = Path(tempfile.mkdtemp(prefix="subliminal_manual_"))
    audio_path = generate_subliminal_audio(
        noise_type=noise_type,
        affirmations=affirmations,
        playback_speed=playback_speed,
        output_dir=str(tmp_dir),
        voice_rate=voice_rate,
        voice_volume=voice_volume,
        base_filename=filename,
        auto_layer=auto_layer,
        layer_count=layer_count,
        layer_variation=layer_variation,
        layer_seed=layer_seed,
    )

    st.success(f"Audio generated: {audio_path.name}")
    st.audio(str(audio_path))
    with audio_path.open("rb") as audio_file:
        st.download_button("Download MP3", data=audio_file, file_name=audio_path.name, mime="audio/mpeg")


def _render_ai_tab():
    st.subheader("AI Agent Assistance")
    st.write(
        "Use OpenAI to generate affirmations, metadata, and an optional thumbnail. "
        "Your API key is used only for the current session."
    )

    with st.form("ai-agent"):
        api_key = st.text_input("OpenAI API key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
        theme = st.text_input("Affirmation theme", "Calm confidence")
        count = st.slider("Number of affirmations", min_value=3, max_value=20, value=8)
        noise_type = st.selectbox("Noise profile", AVAILABLE_NOISE_TYPES[:-1], index=0)
        playback_speed = st.slider("Playback speed", min_value=0.5, max_value=2.0, value=1.0, step=0.05)
        auto_layer = st.checkbox("Auto-layer ambient bed", value=True)
        layer_count = st.slider("Layer count", min_value=1, max_value=8, value=4)
        layer_variation = st.slider("Layer variation", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
        layer_seed_input = st.text_input("Layer seed (optional)", "")
        generate_thumbnail = st.checkbox("Generate thumbnail", value=True)
        submitted = st.form_submit_button("Run AI workflow")

    if not submitted:
        return

    if not api_key:
        st.error("An OpenAI API key is required for the AI workflow.")
        return

    layer_seed: Optional[int]
    if layer_seed_input.strip():
        try:
            layer_seed = int(layer_seed_input.strip())
        except ValueError:
            st.error("Layer seed must be an integer if provided.")
            return
    else:
        layer_seed = None

    try:
        helpers = _load_openai_helpers()
        OpenAI = helpers["OpenAI"]
        client = OpenAI(api_key=api_key)

        affirmations = helpers["generate_affirmations"](client, theme, count)
        title, description, thumbnail_prompt = helpers["generate_metadata"](
            client, affirmations, noise_type, theme
        )

        tmp_dir = Path(tempfile.mkdtemp(prefix="subliminal_ai_"))
        audio_path = generate_subliminal_audio(
            noise_type=noise_type,
            affirmations=affirmations,
            playback_speed=playback_speed,
            output_dir=str(tmp_dir),
            base_filename="ai_subliminal_mix",
            auto_layer=auto_layer,
            layer_count=layer_count,
            layer_variation=layer_variation,
            layer_seed=layer_seed,
        )

        thumbnail_path: Path | None = None
        if generate_thumbnail:
            thumbnail_path = tmp_dir / "thumbnail.png"
            helpers["generate_thumbnail"](client, thumbnail_prompt, thumbnail_path)

    except Exception as exc:  # noqa: BLE001
        st.error(f"AI workflow failed: {exc}")
        return

    st.success("AI workflow completed.")
    st.write(f"**Theme:** {theme}")
    st.write(f"**Title:** {title}")
    st.write("**Description:**")
    st.write(description)
    st.write("**Affirmations:**")
    for affirmation in affirmations:
        st.write(f"- {affirmation}")

    st.audio(str(audio_path))
    with audio_path.open("rb") as audio_file:
        st.download_button("Download MP3", data=audio_file, file_name=audio_path.name, mime="audio/mpeg")

    if thumbnail_path and thumbnail_path.exists():
        st.image(str(thumbnail_path), caption="AI-generated thumbnail")

    st.info(
        "For full automation (video conversion and YouTube upload), run `python automation_agent.py` "
        "and choose one of the AI agent options."
    )


def run() -> None:
    st.set_page_config(page_title="Subliminal Audio Studio", page_icon="🎧")
    st.title("Subliminal Audio Studio")
    st.write(
        "Create custom subliminal audio manually or let the AI agent suggest "
        "affirmations and metadata."
    )

    tabs = st.tabs(["Manual", "AI Agent"])
    with tabs[0]:
        _render_manual_tab()
    with tabs[1]:
        _render_ai_tab()


if __name__ == "__main__":
    run()

