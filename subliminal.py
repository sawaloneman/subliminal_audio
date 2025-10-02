"""Advanced subliminal audio synthesis toolkit.

This module provides building blocks for crafting multi-layered subliminal
soundscapes that combine text-to-speech affirmations, binaural beat
programming, morphic field textures, and a variety of ambient layers.  The
implementation is intentionally modular so it can power both automated
pipelines (Google Colab, CI environments) and rich interactive experiences such
as the accompanying Streamlit dashboard.

The design goals for this rewrite are:

* Scientific grounding – signal-generation routines expose precise frequency,
  amplitude, and modulation parameters instead of opaque presets.
* Unlimited layering – the mixer accepts any number of tracks and gracefully
  normalises the final master.
* Cloud friendly – the text-to-speech subsystem falls back to gTTS whenever
  offline engines such as ``pyttsx3`` are unavailable, which is the case on
  Google Colab.
* Reusability – both the CLI entry point and the Streamlit UI share the same
  rendering engine so improvements propagate everywhere.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydub import AudioSegment

# Optional dependency used for higher quality exports when available.
try:  # pragma: no cover - optional dependency that may be missing on CI
    import soundfile as sf  # type: ignore

    _HAVE_SOUNDFILE = True
except Exception:  # pragma: no cover - see above
    _HAVE_SOUNDFILE = False

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 48_000
TWOPI = 2 * math.pi


def ms_to_samples(duration_ms: float, sample_rate: int) -> int:
    """Convert milliseconds to integer sample count."""

    return max(1, int(round(duration_ms * sample_rate / 1000.0)))


def db_to_amplitude(db: float) -> float:
    """Convert decibels relative to full scale into a linear amplitude."""

    return 10 ** (db / 20.0)


def ensure_directory(path: Union[str, os.PathLike]) -> None:
    """Create the directory that will contain ``path`` if required."""

    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def array_to_segment(
    array: np.ndarray,
    sample_rate: int,
    channels: int = 1,
) -> AudioSegment:
    """Convert a floating-point numpy array ``[-1, 1]`` into an AudioSegment."""

    if array.ndim == 1:
        array = array[:, np.newaxis]
    if channels not in (1, 2):
        raise ValueError("Only mono or stereo arrays are supported")
    if channels == 2 and array.shape[1] == 1:
        array = np.repeat(array, 2, axis=1)
    elif channels == 1 and array.shape[1] == 2:
        array = array.mean(axis=1, keepdims=True)

    array = np.clip(array, -1.0, 1.0)
    int_samples = (array * 32767).astype(np.int16)
    return AudioSegment(
        int_samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=channels,
    )


def segment_to_array(segment: AudioSegment) -> np.ndarray:
    """Convert an AudioSegment into a floating-point numpy array."""

    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    samples = samples.reshape((-1, segment.channels))
    return samples / 32768.0


def linear_envelope(
    duration_ms: float,
    keyframes: Sequence[Tuple[float, float]],
    sample_rate: int,
) -> np.ndarray:
    """Generate a piecewise linear envelope.

    ``keyframes`` are pairs of ``(time_ratio, amplitude)`` where time_ratio is in
    ``[0, 1]``.  Missing endpoints are automatically filled.
    """

    if not keyframes:
        return np.ones(ms_to_samples(duration_ms, sample_rate))

    keyframes = sorted(keyframes, key=lambda x: x[0])
    if keyframes[0][0] > 0.0:
        keyframes = [(0.0, keyframes[0][1])] + list(keyframes)
    if keyframes[-1][0] < 1.0:
        keyframes = list(keyframes) + [(1.0, keyframes[-1][1])]

    total_samples = ms_to_samples(duration_ms, sample_rate)
    envelope = np.zeros(total_samples)
    positions = [int(round(ratio * (total_samples - 1))) for ratio, _ in keyframes]
    values = [amp for _, amp in keyframes]
    last_idx = positions[-1]
    for idx in range(len(positions) - 1):
        start_idx, end_idx = positions[idx], positions[idx + 1]
        start_val, end_val = values[idx], values[idx + 1]
        length = max(1, end_idx - start_idx)
        envelope[start_idx:end_idx] = np.linspace(start_val, end_val, length, endpoint=False)
    envelope[last_idx:] = values[-1]
    return envelope


def adsr_envelope(
    duration_ms: float,
    attack_ms: float,
    decay_ms: float,
    sustain_level: float,
    release_ms: float,
    sample_rate: int,
) -> np.ndarray:
    """Create an ADSR envelope."""

    total_samples = ms_to_samples(duration_ms, sample_rate)
    attack = ms_to_samples(attack_ms, sample_rate)
    decay = ms_to_samples(decay_ms, sample_rate)
    release = ms_to_samples(release_ms, sample_rate)
    sustain = max(0, total_samples - (attack + decay + release))

    env = np.empty(total_samples)
    idx = 0
    if attack:
        env[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False)
        idx += attack
    if decay:
        env[idx : idx + decay] = np.linspace(1.0, sustain_level, decay, endpoint=False)
        idx += decay
    if sustain:
        env[idx : idx + sustain] = sustain_level
        idx += sustain
    if release:
        env[idx:] = np.linspace(env[idx - 1] if idx else sustain_level, 0.0, total_samples - idx)
    else:
        env[idx:] = env[idx - 1] if idx else sustain_level
    return env


def apply_envelope(segment: AudioSegment, envelope: np.ndarray) -> AudioSegment:
    """Apply an envelope to an audio segment."""

    data = segment_to_array(segment)
    if envelope.ndim == 1:
        envelope = envelope[:, np.newaxis]
    if envelope.shape[0] != data.shape[0]:
        raise ValueError("Envelope and audio length mismatch")
    scaled = np.clip(data * envelope, -1.0, 1.0)
    return array_to_segment(scaled, segment.frame_rate, segment.channels)


def normalize(segment: AudioSegment, target_db: float = -1.0) -> AudioSegment:
    """Normalise an audio segment to the specified peak level."""

    if segment.rms == 0:
        return segment
    change = target_db - segment.max_dBFS
    return segment.apply_gain(change)


def ensure_length(segment: AudioSegment, duration_ms: int) -> AudioSegment:
    """Pad or loop a segment so it spans ``duration_ms`` milliseconds."""

    if len(segment) >= duration_ms:
        return segment[:duration_ms]
    loops = (duration_ms + len(segment) - 1) // len(segment)
    extended = segment * loops
    return extended[:duration_ms]


def coloured_noise(
    duration_ms: float,
    sample_rate: int,
    colour: str = "white",
) -> np.ndarray:
    """Generate white, pink or brownian noise."""

    samples = ms_to_samples(duration_ms, sample_rate)
    colour = colour.lower()
    if colour == "white":
        noise = np.random.normal(0.0, 1.0, samples)
    elif colour == "pink":
        # Voss-McCartney algorithm for 1/f noise
        num_rows = 16
        array = np.zeros(samples)
        random_rows = np.random.random((num_rows, samples))
        running_sums = np.zeros(samples)
        for i in range(num_rows):
            step = 2 ** i
            repeated = np.repeat(random_rows[i, ::step][: (samples + step - 1) // step], step)[:samples]
            running_sums += repeated
        noise = running_sums / num_rows
        noise -= noise.mean()
    elif colour in {"brown", "brownian"}:
        noise = np.cumsum(np.random.normal(0.0, 1.0, samples))
        noise -= noise.mean()
        noise /= np.max(np.abs(noise))
    else:
        raise ValueError(f"Unsupported noise colour: {colour}")
    noise /= max(np.max(np.abs(noise)), 1e-6)
    return noise


def smoothstep(x: np.ndarray) -> np.ndarray:
    """Cubic smoothstep used for morphic field modulation."""

    return x * x * (3 - 2 * x)


# -----------------------------------------------------------------------------
# Layer configuration dataclasses
# -----------------------------------------------------------------------------


@dataclass
class BinauralBeatConfig:
    duration_ms: int
    carrier_hz: Union[float, Tuple[float, float]] = 220.0
    beat_schedule: Sequence[Tuple[float, float]] = dataclasses.field(
        default_factory=lambda: [(0.0, 8.0), (1.0, 12.0)]
    )
    amplitude: float = 0.6
    tremolo_hz: Optional[float] = 0.5
    tremolo_depth: float = 0.25
    waveform: str = "sine"


@dataclass
class MorphicFieldConfig:
    duration_ms: int
    carrier_frequencies: Sequence[float] = dataclasses.field(default_factory=lambda: [110.0, 220.0, 440.0])
    harmonic_spread: float = 1.5
    modulation_rate_hz: float = 0.2
    modulation_depth: float = 0.4
    noise_colour: str = "pink"
    noise_level: float = 0.35
    amplitude: float = 0.5


@dataclass
class LayerConfig:
    name: str
    type: str
    gain_db: float = 0.0
    pan: float = 0.0
    adsr: Optional[Dict[str, float]] = None
    params: Dict[str, Union[str, float, int, Sequence[float], Dict[str, float]]] = field(default_factory=dict)


@dataclass
class SessionConfig:
    layers: List[LayerConfig]
    sample_rate: int = DEFAULT_SAMPLE_RATE
    duration_ms: Optional[int] = None
    target_level_db: float = -1.0
    normalize_output: bool = True


# -----------------------------------------------------------------------------
# Text-to-speech backend
# -----------------------------------------------------------------------------


class TextToSpeechEngine:
    """Adaptive text-to-speech helper compatible with Google Colab."""

    def __init__(self, backend: str = "auto", voice: Optional[str] = None):
        self.voice = voice
        self.backend = backend
        self._pyttsx3_engine = None
        self._gtts_cls = None

        pyttsx3_exc: Optional[Exception] = None
        gtts_exc: Optional[Exception] = None

        if backend in {"auto", "pyttsx3"}:
            try:  # pragma: no cover - depends on system packages
                import pyttsx3

                self._pyttsx3_engine = pyttsx3.init()
                self._pyttsx3_engine.setProperty("rate", 160)
                self._pyttsx3_engine.setProperty("volume", 1.0)
                if voice:
                    self._pyttsx3_engine.setProperty("voice", voice)
                if backend == "auto":
                    self.backend = "pyttsx3"
            except Exception as exc:  # pragma: no cover
                pyttsx3_exc = exc
                if backend == "pyttsx3":
                    raise

        if backend in {"auto", "gtts"} and self.backend == "auto":
            try:
                from gtts import gTTS

                self._gtts_cls = gTTS
                self.backend = "gtts"
            except Exception as exc:  # pragma: no cover - optional dependency
                gtts_exc = exc
                if backend == "gtts":
                    raise

        if self.backend == "auto":
            raise RuntimeError(
                "No text-to-speech backend available. pyttsx3 error: "
                f"{pyttsx3_exc}; gTTS error: {gtts_exc}"
            )

    # Public API -------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        rate: int = 160,
        volume: float = 1.0,
        pitch: int = 0,
        language: str = "en",
    ) -> AudioSegment:
        """Return an ``AudioSegment`` containing the spoken ``text``."""

        if self.backend == "pyttsx3":  # pragma: no cover - depends on engine
            assert self._pyttsx3_engine is not None
            engine = self._pyttsx3_engine
            engine.setProperty("rate", rate)
            engine.setProperty("volume", max(0.0, min(1.0, volume)))
            # Not all pyttsx3 drivers expose pitch, so guard it.
            try:
                engine.setProperty("pitch", 100 + pitch)
            except Exception:
                pass
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            try:
                return AudioSegment.from_file(tmp_path)
            finally:
                os.unlink(tmp_path)

        # gTTS fallback – works reliably on Google Colab
        assert self._gtts_cls is not None
        tts = self._gtts_cls(text=text, lang=language, slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        segment = AudioSegment.from_file(buffer, format="mp3")
        if rate != 160:
            segment = speed_change(segment, rate / 160.0)
        if volume != 1.0:
            segment += 20 * math.log10(max(volume, 1e-4))
        if pitch:
            semitones = pitch / 100.0
            segment = pitch_shift(segment, semitones)
        return segment


# -----------------------------------------------------------------------------
# Generators
# -----------------------------------------------------------------------------


def speed_change(segment: AudioSegment, speed: float) -> AudioSegment:
    """Return a new segment played back at a different speed."""

    new_frame_rate = int(segment.frame_rate * speed)
    changed = segment._spawn(segment.raw_data, overrides={"frame_rate": new_frame_rate})
    return changed.set_frame_rate(segment.frame_rate)


def pitch_shift(segment: AudioSegment, semitones: float) -> AudioSegment:
    """Pitch shift without altering duration using resampling."""

    factor = 2 ** (semitones / 12.0)
    return speed_change(segment, factor)


def _coerce_time_series(values: Sequence[Tuple[float, float]]) -> Sequence[Tuple[float, float]]:
    """Ensure time/value pairs are clamped to ``[0, 1]`` and sorted."""

    if not values:
        return [(0.0, 0.0), (1.0, 0.0)]
    clamped = []
    for time_ratio, freq in values:
        clamped.append((max(0.0, min(1.0, float(time_ratio))), float(freq)))
    return sorted(clamped, key=lambda item: item[0])


def generate_binaural(config: BinauralBeatConfig, sample_rate: int) -> AudioSegment:
    """Generate a binaural beat program with optional tremolo."""

    duration_ms = config.duration_ms
    samples = ms_to_samples(duration_ms, sample_rate)
    times = np.linspace(0.0, duration_ms / 1000.0, samples, endpoint=False)

    if isinstance(config.carrier_hz, (tuple, list)):
        start, end = float(config.carrier_hz[0]), float(config.carrier_hz[-1])
        carrier = np.linspace(start, end, samples)
    else:
        carrier = np.full(samples, float(config.carrier_hz))

    beat_env = linear_envelope(duration_ms, config.beat_schedule, sample_rate)

    if config.waveform == "triangle":
        waveform = lambda freq: 2 * np.abs((freq * times) % 1 - 0.5) - 1
    elif config.waveform == "square":
        waveform = lambda freq: np.sign(np.sin(TWOPI * freq * times))
    else:
        waveform = lambda freq: np.sin(TWOPI * freq * times)

    left = waveform(carrier - beat_env / 2.0)
    right = waveform(carrier + beat_env / 2.0)

    if config.tremolo_hz and config.tremolo_depth:
        tremolo = 1.0 - config.tremolo_depth + config.tremolo_depth * np.sin(
            TWOPI * config.tremolo_hz * times
        )
        left *= tremolo
        right *= tremolo

    stereo = np.stack([left, right], axis=1) * config.amplitude
    return array_to_segment(stereo, sample_rate, channels=2)


def generate_morphic_field(config: MorphicFieldConfig, sample_rate: int) -> AudioSegment:
    """Create a morphic field pad using layered harmonic sweeps and coloured noise."""

    duration_ms = config.duration_ms
    samples = ms_to_samples(duration_ms, sample_rate)
    times = np.linspace(0.0, duration_ms / 1000.0, samples, endpoint=False)

    lfo = smoothstep((np.sin(TWOPI * config.modulation_rate_hz * times) + 1) / 2)
    spectrum = np.zeros(samples)
    for base in config.carrier_frequencies:
        base = float(base)
        for harmonic in np.linspace(1.0, config.harmonic_spread, 4):
            freq = base * harmonic
            phase = np.random.rand() * TWOPI
            sweep = np.sin(TWOPI * freq * times + phase)
            modulation = 1.0 + config.modulation_depth * (lfo - 0.5)
            spectrum += sweep * modulation
    spectrum /= max(np.max(np.abs(spectrum)), 1e-6)

    noise = coloured_noise(duration_ms, sample_rate, config.noise_colour) * config.noise_level
    pad = (spectrum * (1 - config.noise_level)) + noise
    stereo = np.stack([pad, pad], axis=1) * config.amplitude
    return array_to_segment(stereo, sample_rate, channels=2)


def generate_noise_layer(
    duration_ms: int,
    sample_rate: int,
    colour: str = "white",
    amplitude: float = 0.5,
) -> AudioSegment:
    noise = coloured_noise(duration_ms, sample_rate, colour) * amplitude
    stereo = np.stack([noise, noise], axis=1)
    return array_to_segment(stereo, sample_rate, channels=2)


# -----------------------------------------------------------------------------
# Compatibility helpers
# -----------------------------------------------------------------------------


def binaural(
    duration_ms: float,
    sample_rate: int,
    carrier_hz: Union[float, Sequence[float], Tuple[float, float]],
    beat_hz: Union[
        float,
        Sequence[float],
        Sequence[Tuple[float, float]],
    ],
    *,
    amplitude: float = 0.6,
    waveform: str = "sine",
    tremolo_hz: Optional[float] = None,
    tremolo_depth: float = 0.0,
) -> AudioSegment:
    """Backwards-compatible binaural generator used by legacy scripts.

    The historical public API exposed a ``binaural`` function that accepted
    primitive frequency arguments.  Modern callers should instantiate
    ``BinauralBeatConfig`` directly, but this shim keeps older notebooks and
    automation working by translating into the richer configuration object.
    """

    beat_sequence: Optional[Sequence] = None
    if isinstance(beat_hz, Sequence) and not isinstance(beat_hz, (str, bytes)):
        beat_sequence = list(beat_hz)

    if beat_sequence and isinstance(beat_sequence[0], (tuple, list)):
        schedule = _coerce_time_series([(float(t), float(v)) for t, v in beat_sequence])
    else:
        if beat_sequence is not None:
            beat_values = [float(val) for val in beat_sequence]
        else:
            beat_values = [float(beat_hz)]
        if len(beat_values) == 1:
            schedule = [(0.0, beat_values[0]), (1.0, beat_values[0])]
        else:
            total = max(1, len(beat_values) - 1)
            schedule = [(idx / total, value) for idx, value in enumerate(beat_values)]

    if isinstance(carrier_hz, Sequence) and not isinstance(carrier_hz, (str, bytes)):
        carrier_values = list(carrier_hz)
        carrier = tuple(float(value) for value in carrier_values)
        if len(carrier) == 1:
            carrier = carrier[0]
    else:
        carrier = float(carrier_hz)

    config = BinauralBeatConfig(
        duration_ms=int(round(duration_ms)),
        carrier_hz=carrier,
        beat_schedule=schedule,
        amplitude=float(amplitude),
        tremolo_hz=tremolo_hz,
        tremolo_depth=float(tremolo_depth) if tremolo_depth is not None else 0.0,
        waveform=waveform,
    )
    return generate_binaural(config, sample_rate)


def overlay(segments: Sequence[AudioSegment]) -> AudioSegment:
    """Overlay multiple segments, matching the behaviour of legacy helpers."""

    segments = [segment for segment in segments if segment is not None]
    if not segments:
        raise ValueError("No audio segments provided")

    base = segments[0]
    for segment in segments[1:]:
        base = base.overlay(segment)

    target_duration = max(len(segment) for segment in segments)
    if len(base) < target_duration:
        base = ensure_length(base, target_duration)
    return base


# -----------------------------------------------------------------------------
# Rendering engine
# -----------------------------------------------------------------------------


class SubliminalSession:
    """Render layered subliminal sessions."""

    def __init__(self, config: SessionConfig, tts_engine: Optional[TextToSpeechEngine] = None):
        self.config = config
        self.tts_engine = tts_engine

    # API -------------------------------------------------------------------

    def render(self) -> AudioSegment:
        segments: List[AudioSegment] = []
        for layer in self.config.layers:
            segment = self._render_layer(layer)
            if layer.gain_db:
                segment = segment.apply_gain(layer.gain_db)
            if layer.pan:
                segment = segment.pan(layer.pan)
            if layer.adsr:
                adsr_cfg = layer.adsr
                segment = apply_envelope(
                    segment,
                    adsr_envelope(
                        duration_ms=len(segment),
                        attack_ms=float(adsr_cfg.get("attack", 50.0)),
                        decay_ms=float(adsr_cfg.get("decay", 200.0)),
                        sustain_level=float(adsr_cfg.get("sustain", 0.7)),
                        release_ms=float(adsr_cfg.get("release", 500.0)),
                        sample_rate=segment.frame_rate,
                    ),
                )
            segments.append(segment)

        if not segments:
            return AudioSegment.silent(duration=1000, frame_rate=self.config.sample_rate)

        max_duration = max(len(seg) for seg in segments)
        base = AudioSegment.silent(duration=max_duration, frame_rate=self.config.sample_rate)
        mix = base
        for seg in segments:
            if seg.frame_rate != self.config.sample_rate:
                seg = seg.set_frame_rate(self.config.sample_rate)
            seg = ensure_length(seg, max_duration)
            mix = mix.overlay(seg)

        if self.config.normalize_output:
            mix = normalize(mix, self.config.target_level_db)
        return mix

    # Internal ---------------------------------------------------------------

    def _render_layer(self, layer: LayerConfig) -> AudioSegment:
        layer_type = layer.type.lower()
        params = layer.params
        sample_rate = self.config.sample_rate

        if layer_type == "file":
            source = params.get("path")
            if source is None:
                raise ValueError(f"Layer '{layer.name}' is missing an audio source")
            if isinstance(source, dict) and "data" in source:
                data = io.BytesIO(source["data"])
                format_hint = Path(source.get("name", "")).suffix.lstrip(".") or None
                segment = AudioSegment.from_file(data, format=format_hint)
            elif hasattr(source, "read"):
                stream = source
                if hasattr(stream, "seek"):
                    stream.seek(0)
                segment = AudioSegment.from_file(stream)
            else:
                path = Path(str(source)).expanduser()
                if not path.exists():
                    raise FileNotFoundError(f"Audio file not found: {path}")
                segment = AudioSegment.from_file(path)
            duration_ms = int(params.get("duration_ms", len(segment)))
            return ensure_length(segment, duration_ms)

        if layer_type == "tts":
            text = str(params.get("text", ""))
            if not text.strip():
                raise ValueError(f"Layer '{layer.name}' has no text")
            rate = int(params.get("rate", 160))
            volume = float(params.get("volume", 1.0))
            pitch = int(params.get("pitch", 0))
            language = str(params.get("language", "en"))
            loop_ms = int(params.get("duration_ms", 0))
            if self.tts_engine is None:
                self.tts_engine = TextToSpeechEngine(backend="auto")
            segment = self.tts_engine.synthesize(text, rate=rate, volume=volume, pitch=pitch, language=language)
            if loop_ms:
                segment = ensure_length(segment, loop_ms)
            return segment

        if layer_type == "binaural":
            cfg = BinauralBeatConfig(
                duration_ms=int(params.get("duration_ms", self.config.duration_ms or 300_000)),
                carrier_hz=params.get("carrier_hz", (params.get("carrier_start", 220.0), params.get("carrier_end", 220.0))),
                beat_schedule=params.get(
                    "beat_schedule",
                    [
                        (0.0, float(params.get("beat_start", 8.0))),
                        (1.0, float(params.get("beat_end", 12.0))),
                    ],
                ),
                amplitude=float(params.get("amplitude", 0.6)),
                tremolo_hz=params.get("tremolo_hz", 0.5),
                tremolo_depth=float(params.get("tremolo_depth", 0.25)),
                waveform=str(params.get("waveform", "sine")),
            )
            return generate_binaural(cfg, sample_rate)

        if layer_type == "noise":
            duration = int(params.get("duration_ms", self.config.duration_ms or 300_000))
            colour = str(params.get("colour", "white"))
            amplitude = float(params.get("amplitude", 0.4))
            return generate_noise_layer(duration, sample_rate, colour, amplitude)

        if layer_type == "morphic_field":
            cfg = MorphicFieldConfig(
                duration_ms=int(params.get("duration_ms", self.config.duration_ms or 300_000)),
                carrier_frequencies=params.get("carrier_frequencies", [110.0, 174.0, 432.0]),
                harmonic_spread=float(params.get("harmonic_spread", 2.0)),
                modulation_rate_hz=float(params.get("modulation_rate_hz", 0.3)),
                modulation_depth=float(params.get("modulation_depth", 0.5)),
                noise_colour=str(params.get("noise_colour", "pink")),
                noise_level=float(params.get("noise_level", 0.35)),
                amplitude=float(params.get("amplitude", 0.5)),
            )
            return generate_morphic_field(cfg, sample_rate)

        raise ValueError(f"Unsupported layer type: {layer.type}")


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------


def load_session_from_json(path: Union[str, os.PathLike]) -> SessionConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    layers = [LayerConfig(**layer) for layer in raw.get("layers", [])]
    return SessionConfig(
        layers=layers,
        sample_rate=int(raw.get("sample_rate", DEFAULT_SAMPLE_RATE)),
        duration_ms=raw.get("duration_ms"),
        target_level_db=float(raw.get("target_level_db", -1.0)),
        normalize_output=bool(raw.get("normalize_output", True)),
    )


def demo_session() -> SessionConfig:
    """Provide a richly layered default session for quick experimentation."""

    duration_ms = 5 * 60 * 1000  # 5 minutes
    return SessionConfig(
        sample_rate=DEFAULT_SAMPLE_RATE,
        duration_ms=duration_ms,
        target_level_db=-1.0,
        normalize_output=True,
        layers=[
            LayerConfig(
                name="Affirmations",
                type="tts",
                gain_db=-4.0,
                pan=-0.15,
                adsr={"attack": 250.0, "decay": 400.0, "sustain": 0.85, "release": 800.0},
                params={
                    "text": "I am focused, grounded, and aligned with my highest potential.",
                    "rate": 150,
                    "pitch": 40,
                    "volume": 0.9,
                    "duration_ms": duration_ms,
                },
            ),
            LayerConfig(
                name="Deep Theta Binaural",
                type="binaural",
                gain_db=-6.0,
                pan=0.0,
                params={
                    "duration_ms": duration_ms,
                    "carrier_start": 110.0,
                    "carrier_end": 174.0,
                    "beat_schedule": [(0.0, 7.0), (0.5, 4.5), (1.0, 12.0)],
                    "amplitude": 0.8,
                    "tremolo_hz": 0.4,
                    "tremolo_depth": 0.35,
                },
            ),
            LayerConfig(
                name="Morphic Resonance",
                type="morphic_field",
                gain_db=-8.0,
                pan=0.1,
                params={
                    "duration_ms": duration_ms,
                    "carrier_frequencies": [174.0, 285.0, 396.0, 528.0],
                    "harmonic_spread": 2.5,
                    "modulation_rate_hz": 0.25,
                    "modulation_depth": 0.45,
                    "noise_colour": "pink",
                    "noise_level": 0.3,
                    "amplitude": 0.6,
                },
            ),
            LayerConfig(
                name="Aurora Texture",
                type="noise",
                gain_db=-14.0,
                pan=0.0,
                params={
                    "duration_ms": duration_ms,
                    "colour": "brown",
                    "amplitude": 0.45,
                },
            ),
        ],
    )


# -----------------------------------------------------------------------------
# Command line interface
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render multi-layer subliminal audio")
    parser.add_argument("--config", type=str, help="Path to a JSON session configuration")
    parser.add_argument(
        "--output",
        type=str,
        default="final_output.wav",
        help="Destination file (extension determines format)",
    )
    parser.add_argument("--format", type=str, default=None, help="Explicit export format")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Render a 30-second preview instead of the full duration",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "pyttsx3", "gtts"],
        default="auto",
        help="Text-to-speech backend to use",
    )
    return parser.parse_args(argv)


def render_to_file(session_cfg: SessionConfig, output_path: Union[str, os.PathLike], fmt: Optional[str] = None) -> None:
    audio = SubliminalSession(session_cfg).render()
    ensure_directory(output_path)
    export_kwargs = {"format": fmt or Path(output_path).suffix.lstrip(".") or "wav"}
    if _HAVE_SOUNDFILE and export_kwargs["format"] in {"wav", "flac", "ogg"}:
        samples = segment_to_array(audio)
        sf.write(output_path, samples, audio.frame_rate)  # type: ignore[arg-type]
    else:
        audio.export(output_path, **export_kwargs)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.config:
        session_cfg = load_session_from_json(args.config)
    else:
        session_cfg = demo_session()

    if args.preview:
        preview_ms = min(30_000, session_cfg.duration_ms or 30_000)
        session_cfg = dataclasses.replace(
            session_cfg,
            duration_ms=preview_ms,
            layers=[
                dataclasses.replace(layer, params={**layer.params, "duration_ms": preview_ms})
                for layer in session_cfg.layers
            ],
        )

    output_path = args.output
    fmt = args.format
    ensure_directory(output_path)
    session = SubliminalSession(session_cfg, TextToSpeechEngine(backend=args.backend))
    audio = session.render()
    format_hint = fmt or Path(output_path).suffix.lstrip(".") or "wav"
    audio.export(output_path, format=format_hint)
    print(f"Rendered subliminal session to {output_path}")


if __name__ == "__main__":
    main()
