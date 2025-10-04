from __future__ import annotations

import contextlib
import io
import importlib.util
import math
import os
import tempfile
from pathlib import Path

from pydub import AudioSegment

_FFMPEG_NOTE = ""


def _configure_audio_backend() -> None:
    global _FFMPEG_NOTE

    converter = AudioSegment.converter
    if converter:
        path = Path(converter)
        if path.exists():
            _FFMPEG_NOTE = f"Using ffmpeg at {path}"
            return

    try:
        import imageio_ffmpeg
    except Exception as exc:  # noqa: BLE001
        _FFMPEG_NOTE = f"imageio-ffmpeg unavailable: {exc}"
        return

    try:
        ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # noqa: BLE001
        _FFMPEG_NOTE = f"Could not resolve bundled ffmpeg: {exc}"
        return

    AudioSegment.converter = str(ffmpeg_path)
    AudioSegment.ffmpeg = str(ffmpeg_path)
    if hasattr(AudioSegment, "ffprobe"):
        AudioSegment.ffprobe = str(ffmpeg_path)
    os.environ.setdefault("FFMPEG_BINARY", str(ffmpeg_path))
    _FFMPEG_NOTE = f"Configured bundled ffmpeg at {ffmpeg_path}"


_configure_audio_backend()


def _synthesize_with_gtts(text: str, rate: int, volume: float):
    if importlib.util.find_spec("gtts") is None:
        raise ModuleNotFoundError("gTTS not installed")

    from gtts import gTTS

    buffer = io.BytesIO()
    gTTS(text=text, lang="en").write_to_fp(buffer)
    if buffer.tell() <= 0:
        raise RuntimeError("gTTS returned no audio")
    buffer.seek(0)
    segment = AudioSegment.from_file(buffer, format="mp3")

    reference = 160
    if rate > 0 and rate != reference:
        speed = max(rate / float(reference), 0.25)
        segment = segment._spawn(
            segment.raw_data, overrides={"frame_rate": int(segment.frame_rate * speed)}
        ).set_frame_rate(segment.frame_rate)

    if volume <= 0:
        segment = segment - 120
    elif volume != 1.0:
        segment = segment.apply_gain(20 * math.log10(volume))

    return segment


def _synthesize_with_pyttsx3(text: str, rate: int, volume: float, pitch: int):
    if importlib.util.find_spec("pyttsx3") is None:
        raise ModuleNotFoundError("pyttsx3 not installed")

    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    if pitch is not None:
        with contextlib.suppress(Exception):
            engine.setProperty("pitch", pitch)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        temp_path = Path(tmp_file.name)

    try:
        engine.save_to_file(text, str(temp_path))
        engine.runAndWait()
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError("pyttsx3 produced no audio")
        segment = AudioSegment.from_file(temp_path, format="wav")
    finally:
        engine.stop()
        temp_path.unlink(missing_ok=True)

    return segment


def text_to_speech(text, file_name="tts_output.wav", rate=150, volume=1.0, pitch=50):
    try:
        segment = _synthesize_with_gtts(text, rate, volume)
    except Exception:  # noqa: BLE001
        segment = _synthesize_with_pyttsx3(text, rate, volume, pitch)

    segment.export(file_name, format="wav")
    return segment

def adjust_speed(audio_segment, playback_speed=1.0):
    new_frame_rate = int(audio_segment.frame_rate * playback_speed)
    return audio_segment._spawn(audio_segment.raw_data, overrides={'frame_rate': new_frame_rate})

def combine_audio_tracks(tracks):
    combined_track = tracks[0]
    for track in tracks[1:]:
        combined_track = combined_track.overlay(track)
    return combined_track

def save_audio(audio_segment, file_path="output.mp3"):
    audio_segment.export(file_path, format="mp3")

if __name__ == "__main__":
    # Example usage
    tts_audio = text_to_speech(
        "Hello, this is an example of text to speech conversion.", rate=120, volume=0.8, pitch=70
    )
    faster_audio = adjust_speed(tts_audio, playback_speed=1.5)
    combined_audio = combine_audio_tracks([tts_audio, faster_audio])
    save_audio(combined_audio, "final_output.mp3")
    