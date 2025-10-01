
from pydub import AudioSegment
from pydub.generators import Sine
import pyttsx3
import os

# Setup Text-to-Speech engine
engine = pyttsx3.init()

def text_to_speech(text, file_name="tts_output.mp3", rate=150, volume=1.0, pitch=50):
    engine.setProperty('rate', rate)  # Speech speed
    engine.setProperty('volume', volume)  # Volume
    engine.setProperty('pitch', pitch)  # Pitch

    engine.save_to_file(text, file_name)
    engine.runAndWait()
    
    return AudioSegment.from_file(file_name)

def adjust_speed(audio_segment, playback_speed=1.0):
    new_frame_rate = int(audio_segment.frame_rate * playback_speed)
    return audio_segment._spawn(audio_segment.raw_data, overrides={'frame_rate': new_frame_rate})

def binaural(duration, sample_rate, carrier_freq, beat_freq, volume=0):
    """Generate a binaural beat as a stereo AudioSegment.

    Args:
        duration (float): Length of the tone in seconds.
        sample_rate (int): Samples per second.
        carrier_freq (float): Base frequency in Hz.
        beat_freq (float): Desired beat frequency in Hz.
        volume (float): Gain adjustment in dB.

    Returns:
        AudioSegment: Stereo segment containing the binaural beat.
    """
    left_freq = carrier_freq - beat_freq / 2.0
    right_freq = carrier_freq + beat_freq / 2.0

    left = Sine(left_freq, sample_rate=sample_rate).to_audio_segment(duration=duration * 1000)
    right = Sine(right_freq, sample_rate=sample_rate).to_audio_segment(duration=duration * 1000)

    if volume:
        left = left.apply_gain(volume)
        right = right.apply_gain(volume)

    return AudioSegment.from_mono_audiosegments(left, right)

def combine_audio_tracks(tracks):
    combined_track = tracks[0]
    for track in tracks[1:]:
        combined_track = combined_track.overlay(track)
    return combined_track

def save_audio(audio_segment, file_path="output.mp3"):
    audio_segment.export(file_path, format="mp3")

# Example usage
tts_audio = text_to_speech("Hello, this is an example of text to speech conversion.", rate=120, volume=0.8, pitch=70)
faster_audio = adjust_speed(tts_audio, playback_speed=1.5)
binaural_audio = binaural(duration=len(tts_audio) / 1000, sample_rate=44100,
                          carrier_freq=440, beat_freq=10, volume=-10)
combined_audio = combine_audio_tracks([tts_audio, faster_audio, binaural_audio])
save_audio(combined_audio, "final_output.mp3")
    