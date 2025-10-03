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


def _inject_app_styles() -> None:
    """Apply a dark-aura Streamlit theme with flame gradients and neon panels."""

    st.markdown(
        """
        <style>
        :root {
            --aura-bg-0: #05010f;
            --aura-bg-1: #0d0124;
            --aura-bg-2: #130033;
            --aura-bg-3: #2a033d;
            --aura-accent-0: #ff5f6d;
            --aura-accent-1: #ffc371;
            --aura-accent-2: #9b5dff;
            --aura-accent-3: #0ff0fc;
            --aura-surface: rgba(10, 3, 28, 0.65);
            --aura-border: rgba(255, 111, 145, 0.35);
        }

        .stApp {
            background: radial-gradient(circle at 10% 15%, rgba(255,95,109,0.18), transparent 48%),
                        radial-gradient(circle at 85% 10%, rgba(79,173,255,0.15), transparent 55%),
                        radial-gradient(circle at 50% 120%, rgba(118,75,162,0.22), transparent 58%),
                        linear-gradient(135deg, var(--aura-bg-0) 0%, var(--aura-bg-1) 40%, var(--aura-bg-2) 75%, #010106 100%) !important;
            color: #f5f7ff;
        }

        .stApp::before,
        .stApp::after {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: -1;
        }

        .stApp::before {
            background: radial-gradient(circle at 20% 20%, rgba(255, 125, 67, 0.28), transparent 55%),
                        radial-gradient(circle at 80% 80%, rgba(123, 31, 162, 0.32), transparent 58%);
            filter: blur(12px);
            animation: auraDrift 18s ease-in-out infinite alternate;
        }

        .stApp::after {
            background: conic-gradient(from 120deg at 50% 50%, rgba(255, 95, 109, 0.12), rgba(79, 173, 255, 0.08), rgba(255, 195, 113, 0.12), rgba(0, 255, 240, 0.08));
            mix-blend-mode: screen;
            opacity: 0.65;
            animation: flamePulse 6.5s ease-in-out infinite alternate;
        }

        @keyframes auraDrift {
            0% { transform: scale(1) translate3d(0, 0, 0); }
            50% { transform: scale(1.05) translate3d(-1%, -1%, 0); }
            100% { transform: scale(1.08) translate3d(1%, 2%, 0); }
        }

        @keyframes flamePulse {
            0% { opacity: 0.5; filter: hue-rotate(0deg) saturate(110%); }
            100% { opacity: 0.85; filter: hue-rotate(25deg) saturate(145%); }
        }

        .hero-banner {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            padding: 3.2rem 3rem;
            margin-bottom: 2.5rem;
            background: linear-gradient(145deg, rgba(12, 3, 30, 0.95) 0%, rgba(36, 6, 46, 0.88) 60%, rgba(5, 0, 20, 0.92) 100%);
            border: 1px solid rgba(255, 111, 145, 0.2);
            box-shadow: 0 0 45px rgba(255, 95, 109, 0.28), 0 0 120px rgba(79, 173, 255, 0.12);
        }

        .hero-banner::before,
        .hero-banner::after {
            content: "";
            position: absolute;
            inset: -30%;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255, 111, 145, 0.18) 0%, transparent 60%);
            filter: blur(24px);
            animation: flameFlicker 12s ease-in-out infinite;
        }

        .hero-banner::after {
            inset: -20% -40% -10% -15%;
            background: radial-gradient(circle, rgba(0, 255, 240, 0.14) 0%, transparent 60%);
            animation-duration: 9s;
            animation-direction: alternate-reverse;
        }

        @keyframes flameFlicker {
            0%, 100% { transform: rotate(0deg) scale(1); opacity: 0.55; }
            40% { transform: rotate(3deg) scale(1.05); opacity: 0.85; }
            70% { transform: rotate(-2deg) scale(0.98); opacity: 0.65; }
        }

        .hero-content {
            position: relative;
            z-index: 1;
            max-width: 720px;
        }

        .hero-label {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-size: 0.78rem;
            letter-spacing: 0.16rem;
            text-transform: uppercase;
            background: linear-gradient(120deg, rgba(255, 95, 109, 0.72), rgba(155, 93, 255, 0.72));
            color: #0a031c;
            font-weight: 700;
            margin-bottom: 1.4rem;
            box-shadow: 0 0 18px rgba(255, 95, 109, 0.45);
        }

        .hero-banner h1 {
            font-size: clamp(2.7rem, 6vw, 3.6rem);
            font-weight: 800;
            margin-bottom: 1rem;
            line-height: 1.1;
            color: #f8f4ff;
            text-shadow: 0 0 18px rgba(255, 95, 109, 0.75), 0 0 35px rgba(79, 173, 255, 0.45);
        }

        .hero-banner p {
            font-size: 1.08rem;
            color: rgba(240, 241, 255, 0.82);
            margin-bottom: 0;
        }

        .pulse-divider {
            width: 100%;
            height: 1px;
            margin: 1.8rem 0 1.4rem;
            background: linear-gradient(90deg, transparent, rgba(255, 111, 145, 0.6), rgba(0, 255, 240, 0.6), transparent);
            box-shadow: 0 0 18px rgba(255, 111, 145, 0.35);
        }

        .stTabs [data-baseweb="tab"] {
            font-weight: 700;
            letter-spacing: 0.03em;
            color: rgba(245, 247, 255, 0.55);
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: rgba(255, 195, 113, 0.9);
        }

        .stTabs [aria-selected="true"] {
            color: #fdfcff !important;
            background: linear-gradient(120deg, rgba(255, 95, 109, 0.35), rgba(155, 93, 255, 0.35));
            border-radius: 18px 18px 0 0;
            box-shadow: 0 12px 26px rgba(255, 95, 109, 0.18);
        }

        [data-testid="stForm"] {
            background: linear-gradient(155deg, rgba(9, 3, 26, 0.82), rgba(32, 6, 48, 0.78));
            border: 1px solid rgba(255, 111, 145, 0.18);
            border-radius: 24px;
            padding: 2.4rem 2rem 2.1rem;
            box-shadow: 0 0 36px rgba(0, 255, 240, 0.18), inset 0 0 18px rgba(255, 95, 109, 0.15);
        }

        [data-testid="stForm"] label {
            font-weight: 600;
            color: rgba(244, 244, 255, 0.9);
        }

        [data-testid="stWidgetLabel"] p {
            color: rgba(244, 244, 255, 0.86) !important;
        }

        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        .stSlider > div > div > div > div {
            background: rgba(14, 5, 30, 0.78) !important;
            border: 1px solid rgba(0, 255, 240, 0.28) !important;
            color: #f8f8ff !important;
            border-radius: 14px !important;
        }

        .stSlider [data-baseweb="slider"] {
            background: linear-gradient(90deg, rgba(255, 95, 109, 0.65), rgba(0, 255, 240, 0.65)) !important;
        }

        .stCheckbox > label > div[data-testid="stMarkdownContainer"] p {
            color: rgba(244, 244, 255, 0.85);
        }

        .stButton > button,
        .stDownloadButton button,
        [data-testid="stFormSubmitButton"] button {
            background: linear-gradient(120deg, rgba(255, 95, 109, 0.92), rgba(155, 93, 255, 0.95));
            color: #0c021f;
            font-weight: 800;
            letter-spacing: 0.04em;
            border: none;
            border-radius: 999px;
            padding: 0.65rem 1.6rem;
            box-shadow: 0 0 18px rgba(255, 95, 109, 0.55), 0 0 45px rgba(79, 173, 255, 0.35);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        .stButton > button:hover,
        .stDownloadButton button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 0 25px rgba(255, 95, 109, 0.72), 0 0 65px rgba(79, 173, 255, 0.45);
        }

        .stButton > button:focus,
        .stDownloadButton button:focus,
        [data-testid="stFormSubmitButton"] button:focus {
            outline: none;
            box-shadow: 0 0 0 3px rgba(0, 255, 240, 0.5);
        }

        div.stAlert {
            border-radius: 18px;
            border: 1px solid rgba(155, 93, 255, 0.35);
            background: rgba(18, 5, 45, 0.85);
            box-shadow: 0 0 24px rgba(79, 173, 255, 0.25);
        }

        div.stSuccess {
            border-color: rgba(0, 255, 240, 0.45) !important;
            background: rgba(6, 30, 35, 0.78) !important;
        }

        div[data-testid="stMarkdownContainer"] > p,
        div[data-testid="stMarkdownContainer"] > span,
        div[data-testid="stMarkdownContainer"] li {
            color: rgba(245, 246, 255, 0.86);
        }

        audio {
            filter: drop-shadow(0 0 12px rgba(0, 255, 240, 0.35));
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero_section() -> None:
    """Render a hero banner that sets the "dark mogger" aesthetic tone."""

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-content">
                <div class="hero-label">Flame Aura Studio</div>
                <h1>Dark Aura Subliminal Forge</h1>
                <p>
                    Sculpt hypnotic mixes with recursive layers, center-channel mastery, and
                    subconscious override tech. Choose manual precision or enlist the AI agent
                    for mind-hacking affirmation alchemy.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

AVAILABLE_NOISE_TYPES: Sequence[str] = (
    "rainbow",
    "white",
    "gray",
    "black",
    "pink",
    "brown",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "indigo",
    "violet",
    "purple",
    "center",
    "center-pulse",
    "subconscious-hack",
    "conscious-hack",
    "mind-hijack",
    "binaural-delta",
    "binaural-theta",
    "binaural-alpha",
    "binaural-beta",
    "binaural-gamma",
    "binaural-epsilon",
    "binaural-lambda",
    "binaural-mu",
    "morphic-field",
    "metaliminal",
    "supraliminal",
    "phantomliminal",
    "binaural",
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


def _white_noise(duration_ms: int, *, gain: float = -8) -> AudioSegment:
    return WhiteNoise().to_audio_segment(duration=duration_ms).apply_gain(gain)


def _color_noise(
    duration_ms: int,
    *,
    low_pass: int | None = None,
    high_pass: int | None = None,
    gain: float = -10,
) -> AudioSegment:
    segment = WhiteNoise().to_audio_segment(duration=duration_ms)
    if high_pass is not None:
        segment = segment.high_pass_filter(high_pass)
    if low_pass is not None:
        segment = segment.low_pass_filter(low_pass)
    return segment.apply_gain(gain)


BINAURAL_BEAT_PRESETS: dict[str, tuple[int, int]] = {
    "binaural": (210, 214),
    "binaural-delta": (100, 104),
    "binaural-theta": (160, 166),
    "binaural-alpha": (220, 228),
    "binaural-beta": (300, 314),
    "binaural-gamma": (420, 444),
    "binaural-epsilon": (40, 44),
    "binaural-lambda": (80, 87),
    "binaural-mu": (520, 540),
}


def _build_binaural(noise_type: str, duration_ms: int) -> AudioSegment:
    pair = BINAURAL_BEAT_PRESETS.get(noise_type, BINAURAL_BEAT_PRESETS["binaural"])
    left = Sine(pair[0]).to_audio_segment(duration=duration_ms).apply_gain(-12)
    right = Sine(pair[1]).to_audio_segment(duration=duration_ms).apply_gain(-12)
    bed = AudioSegment.from_mono_audiosegments(left, right)
    pink = _color_noise(duration_ms, low_pass=1400, gain=-18)
    return bed.overlay(pink)


def _build_rainbow(duration_ms: int) -> AudioSegment:
    layers = [
        _color_noise(duration_ms, low_pass=500, gain=-16),  # red foundation
        _color_noise(duration_ms, low_pass=900, gain=-15),  # orange
        _color_noise(duration_ms, low_pass=1400, gain=-14),  # yellow
        _color_noise(duration_ms, low_pass=2200, high_pass=300, gain=-13),  # green
        _color_noise(duration_ms, high_pass=2100, gain=-12),  # blue
        _color_noise(duration_ms, high_pass=2600, gain=-11),  # indigo
        _color_noise(duration_ms, high_pass=3200, gain=-10),  # violet
    ]
    combined = layers[0]
    for idx, layer in enumerate(layers[1:], start=1):
        pan_amount = (-0.6 + (idx / (len(layers) - 1)) * 1.2) if len(layers) > 1 else 0.0
        combined = combined.overlay(layer.pan(pan_amount), gain_during_overlay=-1.5)
    shimmer = Sine(963).to_audio_segment(duration=duration_ms).apply_gain(-24)
    return combined.overlay(shimmer)


def _build_morphic_field(duration_ms: int) -> AudioSegment:
    base = _color_noise(duration_ms, low_pass=900, gain=-14)
    solfeggio = [174, 285, 396, 417, 528, 639, 741, 852, 963]
    layered = base
    for index, freq in enumerate(solfeggio[:4]):
        tone = Sine(freq).to_audio_segment(duration=duration_ms).apply_gain(-22 + index * 2)
        layered = layered.overlay(tone.fade_in(2000).fade_out(2000))
    pulsar = Sine(40).to_audio_segment(duration=duration_ms).apply_gain(-26)
    return layered.overlay(pulsar)


def _build_metaliminal(duration_ms: int) -> AudioSegment:
    foundation = _color_noise(duration_ms, low_pass=1500, gain=-12)
    reverse_layer = foundation.reverse().apply_gain(-6)
    high_whisper = _color_noise(duration_ms, high_pass=3200, gain=-20)
    carrier = Sine(432).to_audio_segment(duration=duration_ms).apply_gain(-18)
    combined = foundation.overlay(reverse_layer, gain_during_overlay=-2)
    combined = combined.overlay(high_whisper)
    return combined.overlay(carrier)


def _build_supraliminal(duration_ms: int) -> AudioSegment:
    airy = _color_noise(duration_ms, high_pass=2500, gain=-18)
    bright = _color_noise(duration_ms, low_pass=1800, gain=-20)
    shimmer = Sine(777).to_audio_segment(duration=duration_ms).apply_gain(-24)
    return airy.overlay(bright).overlay(shimmer)


def _build_phantomliminal(duration_ms: int) -> AudioSegment:
    hush = _color_noise(duration_ms, low_pass=600, gain=-26)
    hiss = _color_noise(duration_ms, high_pass=3200, gain=-32)
    phantom = hush.overlay(hiss, gain_during_overlay=-3)
    return phantom.fade_in(1500).fade_out(1500)


def _build_center_pulse(duration_ms: int, *, focus: float = 0.6) -> AudioSegment:
    base = _color_noise(duration_ms, low_pass=1600, gain=-13)
    heart = Sine(432).to_audio_segment(duration=duration_ms).apply_gain(-9 + focus * 3)
    breath = Sine(64).to_audio_segment(duration=duration_ms).apply_gain(-30 + focus * 6)
    shimmer = Sine(963).to_audio_segment(duration=duration_ms).apply_gain(-28 + focus * 4)
    combined = base.overlay(heart).overlay(breath.fade_in(800).fade_out(800))
    return combined.overlay(shimmer.pan(-0.15))


def _build_subconscious_hack(duration_ms: int) -> AudioSegment:
    low_bed = _color_noise(duration_ms, low_pass=700, gain=-18)
    infra = Sine(12).to_audio_segment(duration=duration_ms).apply_gain(-25)
    reversed_chime = Sine(528).to_audio_segment(duration=duration_ms).reverse().apply_gain(-26)
    pulses = Sine(96).to_audio_segment(duration=duration_ms).apply_gain(-24).pan(-0.3)
    return low_bed.overlay(infra).overlay(reversed_chime).overlay(pulses)


def _build_conscious_hack(duration_ms: int) -> AudioSegment:
    airy = _color_noise(duration_ms, high_pass=2800, gain=-22)
    sparkle = _color_noise(duration_ms, high_pass=4200, gain=-26)
    focus = Sine(963).to_audio_segment(duration=duration_ms).apply_gain(-20)
    strobe = Sine(1470).to_audio_segment(duration=duration_ms).apply_gain(-24)
    return airy.overlay(sparkle).overlay(focus.pan(0.2)).overlay(strobe.pan(-0.2))


def _build_mind_hijack(duration_ms: int) -> AudioSegment:
    base = _build_metaliminal(duration_ms)
    phantom = _build_phantomliminal(duration_ms).apply_gain(-6)
    binaural = _build_binaural("binaural-theta", duration_ms).apply_gain(-8)
    return base.overlay(phantom, gain_during_overlay=-2).overlay(binaural)


COLOR_NOISE_PROFILES: dict[str, dict[str, int | float | None]] = {
    "white": {"gain": -8},
    "gray": {"gain": -11},
    "black": {"gain": -30},
    "pink": {"low_pass": 1200, "gain": -9},
    "brown": {"low_pass": 800, "gain": -12},
    "red": {"low_pass": 500, "gain": -14},
    "orange": {"low_pass": 900, "gain": -13},
    "yellow": {"low_pass": 1400, "gain": -12},
    "green": {"low_pass": 2200, "high_pass": 250, "gain": -11},
    "blue": {"high_pass": 2100, "gain": -10},
    "indigo": {"high_pass": 2600, "gain": -9},
    "violet": {"high_pass": 3200, "gain": -9},
    "purple": {"high_pass": 1800, "gain": -10},
}


def _build_noise(noise_type: str, duration_ms: int) -> AudioSegment:
    key = noise_type.lower()
    if key in COLOR_NOISE_PROFILES:
        return _color_noise(duration_ms, **COLOR_NOISE_PROFILES[key])
    if key == "rainbow":
        return _build_rainbow(duration_ms)
    if key == "center":
        base = _color_noise(duration_ms, low_pass=1600, gain=-12)
        tone = Sine(432).to_audio_segment(duration=duration_ms).apply_gain(-9)
        return base.overlay(tone)
    if key == "center-pulse":
        return _build_center_pulse(duration_ms)
    if key == "subconscious-hack":
        return _build_subconscious_hack(duration_ms)
    if key == "conscious-hack":
        return _build_conscious_hack(duration_ms)
    if key == "mind-hijack":
        return _build_mind_hijack(duration_ms)
    if key in BINAURAL_BEAT_PRESETS:
        return _build_binaural(key, duration_ms)
    if key == "morphic-field":
        return _build_morphic_field(duration_ms)
    if key == "metaliminal":
        return _build_metaliminal(duration_ms)
    if key == "supraliminal":
        return _build_supraliminal(duration_ms)
    if key == "phantomliminal":
        return _build_phantomliminal(duration_ms)
    if key == "none":
        return AudioSegment.silent(duration=duration_ms)
    return _white_noise(duration_ms)


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


def _apply_center_focus(segment: AudioSegment, *, center_focus: float, duration_ms: int) -> AudioSegment:
    focus = max(0.0, min(center_focus, 1.0))
    if focus <= 0:
        return segment
    carrier = Sine(432).to_audio_segment(duration=duration_ms).apply_gain(-12 + focus * 6)
    harmonic = Sine(864).to_audio_segment(duration=duration_ms).apply_gain(-26 + focus * 8)
    pulse = Sine(8).to_audio_segment(duration=duration_ms).apply_gain(-32 + focus * 10)
    centered = segment.overlay(carrier)
    centered = centered.overlay(harmonic.pan(0.25))
    return centered.overlay(pulse.fade_in(1000).fade_out(1000))


def _apply_cognitive_hacks(
    segment: AudioSegment,
    *,
    subconscious_intensity: float,
    conscious_intensity: float,
    override_intensity: float,
    duration_ms: int,
) -> AudioSegment:
    def _scale(value: float) -> float:
        return max(0.0, min(value, 1.0))

    subconscious = _scale(subconscious_intensity)
    conscious = _scale(conscious_intensity)
    override = _scale(override_intensity)

    processed = segment

    if subconscious > 0:
        rumble = Sine(5).to_audio_segment(duration=duration_ms).apply_gain(-35 + subconscious * 12)
        whisper = segment.reverse().apply_gain(-18 + subconscious * 4).low_pass_filter(1500)
        processed = processed.overlay(rumble)
        processed = processed.overlay(whisper, gain_during_overlay=-2)

    if conscious > 0:
        shimmer = _color_noise(duration_ms, high_pass=3600, gain=-30 + conscious * 8)
        ping = Sine(1111).to_audio_segment(duration=duration_ms).apply_gain(-28 + conscious * 6)
        processed = processed.overlay(shimmer)
        processed = processed.overlay(ping.pan(-0.2))

    if override > 0:
        driver = Sine(222).to_audio_segment(duration=duration_ms).apply_gain(-24 + override * 8)
        gating = processed.invert_phase().apply_gain(-32 + override * 6)
        processed = processed.overlay(driver)
        processed = processed.overlay(gating, gain_during_overlay=-3)

    return processed


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
    center_focus: float = 0.5,
    subconscious_intensity: float = 0.35,
    conscious_intensity: float = 0.2,
    override_intensity: float = 0.3,
) -> Path:
    """Create an audio file containing the affirmations mixed with ambient noise.

    The new ``center_focus`` and hack intensity parameters control how strongly
    the mix emphasises the center technique, subconscious entrainment pulses,
    conscious shimmer cues, and the mind override layer that blends inversion
    and harmonic drivers.
    """

    if not affirmations:
        raise ValueError("At least one affirmation is required.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    speech = _synthesize_affirmations(affirmations, rate=voice_rate, volume=voice_volume)
    speech = _change_speed(speech, playback_speed)

    duration_ms = len(speech)

    if auto_layer:
        noise = auto_layer_noise(
            noise_type,
            duration_ms,
            layer_count=layer_count,
            variation_strength=max(0.0, min(layer_variation, 1.0)),
            seed=layer_seed,
        )
    else:
        noise = _build_noise(noise_type, duration_ms)

    noise = _apply_center_focus(noise, center_focus=center_focus, duration_ms=duration_ms)
    noise = _apply_cognitive_hacks(
        noise,
        subconscious_intensity=subconscious_intensity,
        conscious_intensity=conscious_intensity,
        override_intensity=override_intensity,
        duration_ms=duration_ms,
    )
    if noise.channels != speech.channels:
        speech = speech.set_channels(noise.channels)

    balanced_noise = _ensure_duration(noise, duration_ms)
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
        center_focus = st.slider("Center technique focus", min_value=0.0, max_value=1.0, value=0.55, step=0.05)
        subconscious_intensity = st.slider(
            "Subconscious hack intensity", min_value=0.0, max_value=1.0, value=0.4, step=0.05
        )
        conscious_intensity = st.slider(
            "Conscious hack intensity", min_value=0.0, max_value=1.0, value=0.25, step=0.05
        )
        override_intensity = st.slider(
            "Mind override layering", min_value=0.0, max_value=1.0, value=0.35, step=0.05
        )
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
        center_focus=center_focus,
        subconscious_intensity=subconscious_intensity,
        conscious_intensity=conscious_intensity,
        override_intensity=override_intensity,
    )

    st.success(f"Audio generated: {audio_path.name}")
    st.write(
        f"Center focus: {center_focus:.2f} | Subconscious hack: {subconscious_intensity:.2f} | "
        f"Conscious hack: {conscious_intensity:.2f} | Override: {override_intensity:.2f}"
    )
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
        center_focus = st.slider("Center technique focus", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        subconscious_intensity = st.slider(
            "Subconscious hack intensity", min_value=0.0, max_value=1.0, value=0.45, step=0.05
        )
        conscious_intensity = st.slider(
            "Conscious hack intensity", min_value=0.0, max_value=1.0, value=0.3, step=0.05
        )
        override_intensity = st.slider(
            "Mind override layering", min_value=0.0, max_value=1.0, value=0.4, step=0.05
        )
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
            center_focus=center_focus,
            subconscious_intensity=subconscious_intensity,
            conscious_intensity=conscious_intensity,
            override_intensity=override_intensity,
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
    st.write(
        f"**Center focus:** {center_focus:.2f} | Subconscious hack: {subconscious_intensity:.2f} | "
        f"Conscious hack: {conscious_intensity:.2f} | Override: {override_intensity:.2f}"
    )
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
    st.set_page_config(page_title="Subliminal Audio Studio", page_icon="🎧", layout="wide")
    _inject_app_styles()
    _render_hero_section()
    st.caption(
        "Forge layered entrainment soundscapes with surgical manual control or AI-guided flow."
    )
    st.markdown('<div class="pulse-divider"></div>', unsafe_allow_html=True)

    tabs = st.tabs(["Manual", "AI Agent"])
    with tabs[0]:
        _render_manual_tab()
    with tabs[1]:
        _render_ai_tab()


if __name__ == "__main__":
    run()

