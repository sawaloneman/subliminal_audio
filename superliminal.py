import os
import tempfile
import random
import math
import numpy as np
try:
    import streamlit as st
except Exception:
    st = None  # Allows CLI use without Streamlit installed
import soundfile as sf
from gtts import gTTS
from scipy.signal import resample_poly
from functools import lru_cache
import concurrent.futures
import logging
import time
import statistics
import psutil
import pickle

try:
    import cupy as cp
    use_gpu = True
except ImportError:
    cp = np
    use_gpu = False

SR = 44100

logging.basicConfig(
    filename='superliminal.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

durations = []


def resample_to_length(x, new_len):
    """Resample ``x`` to ``new_len`` samples using polyphase filtering."""
    if len(x) == new_len:
        return x
    up = new_len
    down = len(x)
    g = math.gcd(up, down)
    up //= g
    down //= g
    arr = cp.asnumpy(x) if use_gpu else x
    res = resample_poly(arr, up, down, axis=0)
    return cp.asarray(res) if use_gpu else res


if st is not None:
    @st.cache_data
    def cache_tts(txt):
        """Cache TTS audio to avoid repeated generation."""
        if not txt:
            return None
        tts = gTTS(txt)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        try:
            tts.save(tmp)
            s, tts_sr = sf.read(tmp)
            if s.ndim == 1:
                s = np.column_stack((s, s))
            if tts_sr != SR:
                s = resample_poly(s, SR, tts_sr)
            return s
        finally:
            os.unlink(tmp)
else:
    def cache_tts(txt):
        """Cache TTS audio to avoid repeated generation (no Streamlit)."""
        if not txt:
            return None
        tts = gTTS(txt)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        try:
            tts.save(tmp)
            s, tts_sr = sf.read(tmp)
            if s.ndim == 1:
                s = np.column_stack((s, s))
            if tts_sr != SR:
                s = resample_poly(s, SR, tts_sr)
            return s
        finally:
            os.unlink(tmp)


@lru_cache(maxsize=32)
def cache_noise(noise_type, duration, sr, level):
    return NOISE[noise_type](duration, sr, level)


@lru_cache(maxsize=32)
def cache_binaural(bina_mode, duration, sr, fc, fb):
    if bina_mode == "Single":
        return binaural(duration, sr, fc, fb)
    elif bina_mode == "Chord":
        return binaural_chord(duration, sr, fc, [1, 1.26, 1.5], fb)
    elif bina_mode == "Sweep":
        return binaural_sweep(duration, sr, 3, 40)
    elif bina_mode == "Prism":
        return binaural_prism(duration, sr, [150, 200, 300], [4, 6, 8])
    elif bina_mode == "Morph":
        return binaural_morph(duration, sr, morphs[0] if morphs else "")
    elif bina_mode == "Quantum":
        return binaural_quantum(duration, sr, 8)
    return np.zeros((int(duration * sr), 2))


def hue_noise(duration, sr, hue, amp=0.1):
    exp = (hue - 0.5) * 4
    n = int(sr * duration)
    white = cp.random.randn(n) if use_gpu else np.random.randn(n)
    f = cp.fft.rfft(white) if use_gpu else np.fft.rfft(white)
    freq = cp.fft.rfftfreq(n, 1 / sr) if use_gpu else np.fft.rfftfreq(n, 1 / sr)
    filt = 1 / cp.where(freq == 0, 1, freq ** exp) if use_gpu else 1 / np.where(freq == 0, 1, freq ** exp)
    sig = cp.fft.irfft(f * filt, n=n) if use_gpu else np.fft.irfft(f * filt, n=n)
    max_abs = cp.max(cp.abs(sig)) + 1e-9 if use_gpu else np.max(np.abs(sig)) + 1e-9
    sig = sig / max_abs * amp
    return cp.asnumpy(sig) if use_gpu else sig


def binaural(duration, sr, fc, fb):
    t = cp.linspace(0, duration, int(sr * duration)) if use_gpu else np.linspace(0, duration, int(sr * duration))
    sig = cp.column_stack([
        cp.sin(2 * cp.pi * (fc - fb / 2) * t),
        cp.sin(2 * cp.pi * (fc + fb / 2) * t)
    ]) if use_gpu else np.column_stack([
        np.sin(2 * np.pi * (fc - fb / 2) * t),
        np.sin(2 * np.pi * (fc + fb / 2) * t)
    ])
    max_abs = cp.max(cp.abs(sig)) + 1e-9 if use_gpu else np.max(np.abs(sig)) + 1e-9
    sig = sig / max_abs * 0.5
    return cp.asnumpy(sig) if use_gpu else sig


def binaural_chord(duration, sr, root, chord, beat):
    t = cp.linspace(0, duration, int(sr * duration)) if use_gpu else np.linspace(0, duration, int(sr * duration))
    L = cp.zeros_like(t) if use_gpu else np.zeros_like(t)
    R = cp.zeros_like(t) if use_gpu else np.zeros_like(t)
    for r in chord:
        L += cp.sin(2 * cp.pi * (root * r - beat / 2) * t) if use_gpu else np.sin(2 * np.pi * (root * r - beat / 2) * t)
        R += cp.sin(2 * cp.pi * (root * r + beat / 2) * t) if use_gpu else np.sin(2 * np.pi * (root * r + beat / 2) * t)
    sig = cp.column_stack([L / len(chord), R / len(chord)]) if use_gpu else np.column_stack([L / len(chord), R / len(chord)])
    max_abs = cp.max(cp.abs(sig)) + 1e-9 if use_gpu else np.max(np.abs(sig)) + 1e-9
    sig = sig / max_abs * 0.5
    return cp.asnumpy(sig) if use_gpu else sig


def binaural_sweep(duration, sr, start, end):
    t = cp.linspace(0, duration, int(sr * duration)) if use_gpu else np.linspace(0, duration, int(sr * duration))
    f = cp.linspace(start, end, len(t)) if use_gpu else np.linspace(start, end, len(t))
    sig = cp.column_stack([
        cp.sin(2 * cp.pi * (150 - f / 2) * t),
        cp.sin(2 * cp.pi * (150 + f / 2) * t)
    ]) if use_gpu else np.column_stack([
        np.sin(2 * np.pi * (150 - f / 2) * t),
        np.sin(2 * np.pi * (150 + f / 2) * t)
    ])
    max_abs = cp.max(cp.abs(sig)) + 1e-9 if use_gpu else np.max(np.abs(sig)) + 1e-9
    sig = sig / max_abs * 0.5
    return cp.asnumpy(sig) if use_gpu else sig


def binaural_prism(duration, sr, carriers, beats):
    t = cp.linspace(0, duration, int(sr * duration)) if use_gpu else np.linspace(0, duration, int(sr * duration))
    L = cp.zeros_like(t) if use_gpu else np.zeros_like(t)
    R = cp.zeros_like(t) if use_gpu else np.zeros_like(t)
    for c, b in zip(carriers, beats):
        L += cp.sin(2 * cp.pi * (c - b / 2) * t) if use_gpu else np.sin(2 * np.pi * (c - b / 2) * t)
        R += cp.sin(2 * cp.pi * (c + b / 2) * t) if use_gpu else np.sin(2 * np.pi * (c + b / 2) * t)
    sig = cp.column_stack([L / len(carriers), R / len(carriers)]) if use_gpu else np.column_stack([L / len(carriers), R / len(carriers)])
    max_abs = cp.max(cp.abs(sig)) + 1e-9 if use_gpu else np.max(np.abs(sig)) + 1e-9
    sig = sig / max_abs * 0.5
    return cp.asnumpy(sig) if use_gpu else sig


def binaural_morph(duration, sr, text):
    seed = sum(ord(c) for c in text) if text else 0
    beat = (seed % 30) + 0.5
    t = cp.linspace(0, duration, int(sr * duration)) if use_gpu else np.linspace(0, duration, int(sr * duration))
    sig = cp.column_stack([
        cp.sin(2 * cp.pi * (150 - beat / 2) * t),
        cp.sin(2 * cp.pi * (150 + beat / 2) * t)
    ]) if use_gpu else np.column_stack([
        np.sin(2 * np.pi * (150 - beat / 2) * t),
        np.sin(2 * np.pi * (150 + beat / 2) * t)
    ])
    max_abs = cp.max(cp.abs(sig)) + 1e-9 if use_gpu else np.max(np.abs(sig)) + 1e-9
    sig = sig / max_abs * 0.5
    return cp.asnumpy(sig) if use_gpu else sig


def binaural_quantum(duration, sr, steps):
    seg = int(duration * sr / steps)
    out = cp.zeros((int(duration * sr), 2)) if use_gpu else np.zeros((int(duration * sr), 2))
    idx = 0
    for _ in range(steps):
        beat = random.uniform(0.5, 30)
        seg_len = min(seg, len(out) - idx)
        t = cp.linspace(0, seg_len / sr, seg_len) if use_gpu else np.linspace(0, seg_len / sr, seg_len)
        chunk = cp.column_stack([
            cp.sin(2 * cp.pi * (150 - beat / 2) * t),
            cp.sin(2 * cp.pi * (150 + beat / 2) * t)
        ]) if use_gpu else np.column_stack([
            np.sin(2 * np.pi * (150 - beat / 2) * t),
            np.sin(2 * np.pi * (150 + beat / 2) * t)
        ])
        out[idx:idx + seg_len] = chunk
        idx += seg_len
    max_abs = cp.max(cp.abs(out)) + 1e-9 if use_gpu else np.max(np.abs(out)) + 1e-9
    out = out / max_abs * 0.5
    return cp.asnumpy(out) if use_gpu else out


def isochronic(duration, sr, fc, fb):
    t = cp.linspace(0, duration, int(sr * duration)) if use_gpu else np.linspace(0, duration, int(sr * duration))
    carrier = cp.sin(2 * cp.pi * fc * t) if use_gpu else np.sin(2 * np.pi * fc * t)
    mod = 0.5 * (1 + cp.sign(cp.sin(2 * cp.pi * fb * t))) if use_gpu else 0.5 * (1 + np.sign(np.sin(2 * np.pi * fb * t)))
    sig = carrier * mod
    sig = cp.column_stack([sig, sig]) if use_gpu else np.column_stack([sig, sig])
    max_abs = cp.max(cp.abs(sig)) + 1e-9 if use_gpu else np.max(np.abs(sig)) + 1e-9
    sig = sig / max_abs * 0.5
    return cp.asnumpy(sig) if use_gpu else sig


def add_micro_delay(a, sr):
    d = int(0.005 * sr)
    out = cp.zeros_like(a) if use_gpu else np.zeros_like(a)
    out[d:] = a[:-d]
    result = (a + out) / 2
    return cp.asnumpy(result) if use_gpu else result


def add_phase_flips(a, sr):
    dur = len(a) / sr
    mask = cp.ones(len(a)) if use_gpu else np.ones(len(a))
    for _ in range(int(dur * 10)):
        if random.random() < 0.5:
            i = random.randint(0, len(a) - int(0.01 * sr))
            mask[i:i + int(0.01 * sr)] *= -1
    a[:, 0] *= mask
    a[:, 1] *= mask
    return cp.asnumpy(a) if use_gpu else a


def add_glitch(a, sr):
    a = cp.array(a) if use_gpu else a
    for _ in range(10):
        i = random.randint(0, len(a) - int(0.005 * sr))
        a[i:i + int(0.005 * sr)] = 0
    return cp.asnumpy(a) if use_gpu else a


def add_pan(a, sr):
    n = len(a)
    t = cp.linspace(0, 2 * cp.pi, n) if use_gpu else np.linspace(0, 2 * np.pi, n)
    pan = (cp.sin(t) + 1) / 2 if use_gpu else (np.sin(t) + 1) / 2
    a[:, 0] *= pan
    a[:, 1] *= (1 - pan)
    return cp.asnumpy(a) if use_gpu else a


def add_breath(a, sr):
    t = cp.linspace(0, len(a) / sr, len(a)) if use_gpu else np.linspace(0, len(a) / sr, len(a))
    env = (cp.sin(2 * cp.pi * 0.3 * t) + 1) / 2 if use_gpu else (np.sin(2 * np.pi * 0.3 * t) + 1) / 2
    a[:, 0] *= env
    a[:, 1] *= env
    return cp.asnumpy(a) if use_gpu else a


def add_infra(a, sr):
    t = cp.linspace(0, len(a) / sr, len(a)) if use_gpu else np.linspace(0, len(a) / sr, len(a))
    env = (cp.sin(2 * cp.pi * 0.1 * t) + 1) / 2 if use_gpu else (np.sin(2 * np.pi * 0.1 * t) + 1) / 2
    a[:, 0] *= env
    a[:, 1] *= env
    return cp.asnumpy(a) if use_gpu else a


def add_static(a, sr):
    noise = cp.random.randn(*a.shape) * 0.01 if use_gpu else np.random.randn(*a.shape) * 0.01
    result = a + noise
    return cp.asnumpy(result) if use_gpu else result


def add_dropout(a, sr):
    a = cp.array(a) if use_gpu else a
    for _ in range(5):
        i = random.randint(0, len(a) - int(0.05 * sr))
        a[i:i + int(0.05 * sr)] = 0
    return cp.asnumpy(a) if use_gpu else a


def add_shepard_riser(a, sr):
    t = cp.linspace(0, len(a) / sr, len(a)) if use_gpu else np.linspace(0, len(a) / sr, len(a))
    riser = cp.sin(2 * cp.pi * cp.logspace(0, 2, len(a)) * t) * 0.1 if use_gpu else np.sin(2 * np.pi * np.logspace(0, 2, len(a)) * t) * 0.1
    a[:, 0] += riser
    a[:, 1] += riser
    max_abs = cp.max(cp.abs(a)) + 1e-9 if use_gpu else np.max(np.abs(a)) + 1e-9
    a = a / max_abs
    return cp.asnumpy(a) if use_gpu else a


def add_click_train(a, sr):
    click = cp.zeros_like(a) if use_gpu else np.zeros_like(a)
    interval = int(0.5 * sr)
    for i in range(0, len(a), interval):
        click[i:i + 10] = 0.5
    result = a + click
    return cp.asnumpy(result) if use_gpu else result


def add_tremor(a, sr):
    t = cp.linspace(0, len(a) / sr, len(a)) if use_gpu else np.linspace(0, len(a) / sr, len(a))
    env = 1 + 0.1 * cp.sin(2 * cp.pi * 1 * t) if use_gpu else 1 + 0.1 * np.sin(2 * np.pi * 1 * t)
    a[:, 0] *= env
    a[:, 1] *= env
    return cp.asnumpy(a) if use_gpu else a


def add_phantom_words(a, sr):
    t = cp.linspace(0, len(a) / sr, len(a)) if use_gpu else np.linspace(0, len(a) / sr, len(a))
    noise = cp.random.randn(len(a)) * 0.01 if use_gpu else np.random.randn(len(a)) * 0.01
    a[:, 0] += noise
    a[:, 1] += noise
    return cp.asnumpy(a) if use_gpu else a


def add_hyper_spikes(a, sr):
    a = cp.array(a) if use_gpu else a
    for _ in range(20):
        idx = random.randint(0, len(a) - 1)
        a[idx, :] += 0.5
    return cp.asnumpy(a) if use_gpu else a


def add_time_flutter(a, sr):
    t = cp.linspace(0, len(a) / sr, len(a)) if use_gpu else np.linspace(0, len(a) / sr, len(a))
    flutter = cp.sin(2 * cp.pi * 5 * t) * 0.05 if use_gpu else np.sin(2 * np.pi * 5 * t) * 0.05
    a[:, 0] += flutter
    a[:, 1] += flutter
    return cp.asnumpy(a) if use_gpu else a


def add_doppler_pan(a, sr):
    t = cp.linspace(0, len(a) / sr, len(a)) if use_gpu else np.linspace(0, len(a) / sr, len(a))
    pan = (1 + cp.cos(2 * cp.pi * 0.5 * t)) / 2 if use_gpu else (1 + np.cos(2 * np.pi * 0.5 * t)) / 2
    a[:, 0] *= pan
    a[:, 1] *= (1 - pan)
    return cp.asnumpy(a) if use_gpu else a


def add_ultrasonic(a, sr):
    t = cp.linspace(0, len(a) / sr, len(a)) if use_gpu else np.linspace(0, len(a) / sr, len(a))
    hf = cp.sin(2 * cp.pi * 18000 * t) * 0.001 if use_gpu else np.sin(2 * np.pi * 18000 * t) * 0.001
    a[:, 0] += hf
    a[:, 1] += hf
    return cp.asnumpy(a) if use_gpu else a


def add_inverse_gate(a, sr):
    a = cp.array(a) if use_gpu else a
    thr = cp.mean(cp.abs(a)) if use_gpu else np.mean(np.abs(a))
    result = cp.where(cp.abs(a) < thr, a, 0) if use_gpu else np.where(np.abs(a) < thr, a, 0)
    return cp.asnumpy(result) if use_gpu else result


def add_keyed_echo(a, sr):
    d = int(0.2 * sr)
    echo = cp.zeros_like(a) if use_gpu else np.zeros_like(a)
    echo[d:] = a[:-d] * 0.6
    result = a + echo
    return cp.asnumpy(result) if use_gpu else result


def add_hue_layer(a, sr, hue):
    if hue is None:
        return a
    dur = len(a) / sr
    n = hue_noise(dur, sr, hue / 360, amp=0.1)
    n = resample_to_length(n, len(a))
    if n.ndim == 1:
        n = np.column_stack((n, n))
    a[:, 0] += n[:, 0]
    a[:, 1] += n[:, 1]
    max_abs = cp.max(cp.abs(a)) + 1e-9 if use_gpu else np.max(np.abs(a)) + 1e-9
    a = a / max_abs
    return cp.asnumpy(a) if use_gpu else a


def add_supraliminal(a, sr, txt):
    tts_audio = cache_tts(txt)
    if tts_audio is None:
        return a
    s = tts_audio
    s = resample_to_length(s, len(a))
    result = (a + dbfs(s, -18)) / (cp.max(cp.abs(a + dbfs(s, -18))) + 1e-9 if use_gpu else np.max(np.abs(a + dbfs(s, -18))) + 1e-9)
    return cp.asnumpy(result) if use_gpu else result


NOISE = {
    "White": lambda d, sr, l: cp.random.randn(int(sr * d)) * l if use_gpu else np.random.randn(int(sr * d)) * l,
    "Pink": lambda d, sr, l: hue_noise(d, sr, 0.8, amp=l),
    "Brown": lambda d, sr, l: cp.cumsum(cp.random.randn(int(sr * d))) * (l / (cp.max(cp.abs(cp.cumsum(cp.random.randn(int(sr * d))))) + 1e-9)) if use_gpu else np.cumsum(np.random.randn(int(sr * d))) * (l / (np.max(np.abs(np.cumsum(np.random.randn(int(sr * d))))) + 1e-9)),
    "Red": lambda d, sr, l: hue_noise(d, sr, 0.0, amp=l),
    "Orange": lambda d, sr, l: hue_noise(d, sr, 0.1, amp=l),
    "Yellow": lambda d, sr, l: hue_noise(d, sr, 0.166, amp=l),
    "Green": lambda d, sr, l: hue_noise(d, sr, 0.333, amp=l),
    "Blue": lambda d, sr, l: hue_noise(d, sr, 0.666, amp=l),
    "Violet": lambda d, sr, l: hue_noise(d, sr, 0.75, amp=l),
    "Black": lambda d, sr, l: cp.zeros(int(sr * d)) if use_gpu else np.zeros(int(sr * d))
}


FX = {
    "Micro Delay": add_micro_delay,
    "Phase Flips": add_phase_flips,
    "Glitch Spikes": add_glitch,
    "Rotating Pan": add_pan,
    "Breath Pulse": add_breath,
    "Infrasound LFO": add_infra,
    "Confusion Static": add_static,
    "Temporal Dropout": add_dropout,
    "Shepard Riser": add_shepard_riser,
    "Click Train": add_click_train,
    "Sub Tremor": add_tremor,
    "Phantom Words": add_phantom_words,
    "Hyper Spikes": add_hyper_spikes,
    "Time Flutter": add_time_flutter,
    "Doppler Pan": add_doppler_pan,
    "Ultrasonic": add_ultrasonic,
    "Inverse Gate": add_inverse_gate,
    "Keyed Echo": add_keyed_echo,
    "Hue Noise": lambda a, sr, hue: add_hue_layer(a, sr, hue),
    "Supraliminal": lambda a, sr, txt: add_supraliminal(a, sr, txt)
}


def dbfs(a, target=-25):
    a = cp.array(a) if use_gpu else a
    rms = cp.sqrt(cp.mean(a ** 2)) if use_gpu else np.sqrt(np.mean(a ** 2))
    gain = 10 ** ((target - 20 * cp.log10(rms + 1e-9)) / 20) if use_gpu else 10 ** ((target - 20 * np.log10(rms + 1e-9)) / 20)
    result = a * gain
    return cp.asnumpy(result) if use_gpu else result


def overlay(segs):
    """Efficiently overlay multiple segments with normalization."""
    if not segs:
        return np.zeros((1, 2))
    max_len = max(len(s) for s in segs)
    padded = np.zeros((len(segs), max_len, 2))
    for i, s in enumerate(segs):
        if s.ndim == 1:
            s = np.column_stack((s, s))
        padded[i, : len(s)] = s
    mix = padded.sum(axis=0)
    max_abs = np.max(np.abs(mix)) + 1e-9
    mix = mix / max_abs / len(segs)
    return mix


def get_unique_filename(fn):
    base, ext = os.path.splitext(fn)
    counter = 1
    while os.path.exists(fn):
        fn = f"{base}_{counter}{ext}"
        counter += 1
    return fn


def proc(a, fx_vals, noise_type, hue, supra_txt, bina_toggle, bina_mode, fc, fb,
         iso_toggle, vol, time_stretch=True, speed=1.0):
    start = time.time()
    start_mem = psutil.Process().memory_info().rss
    try:
        a = cp.array(a) if use_gpu else a
        if time_stretch:
            new_len = int(len(a) * random.uniform(0.9, 1.1))
            a = resample_to_length(a, new_len)
        if speed != 1.0:
            new_len = int(len(a) / speed)
            a = resample_to_length(a, new_len)

        for k, fn in FX.items():
            if fx_vals.get(k):
                try:
                    if k == "Hue Noise":
                        a = fn(a, SR, hue)
                    elif k == "Supraliminal":
                        a = fn(a, SR, supra_txt)
                    else:
                        a = fn(a, SR)
                    logging.info(f"Applied FX {k}")
                except Exception as e:
                    logging.error(f"FX {k} error: {e}")
                    return None, f"FX {k} error: {e}"

        if noise_type != "None":
            try:
                n = cache_noise(noise_type, len(a) / SR, SR, 0.1)
                if n.ndim == 1:
                    n = np.column_stack((n, n))
                n = n[:len(a)]
                a += n / (np.max(np.abs(n)) + 1e-9) * 0.5
                logging.info(f"Applied noise {noise_type}")
            except Exception as e:
                logging.error(f"Noise error: {e}")
                return None, f"Noise error: {e}"

        if bina_toggle:
            try:
                bb = cache_binaural(bina_mode, len(a) / SR, SR, fc, fb)
                bb = bb[:len(a)]
                a += bb / (np.max(np.abs(bb)) + 1e-9) * 0.5
                if iso_toggle:
                    iso = isochronic(len(a) / SR, SR, fc, fb)[:len(a)]
                    a += iso / (np.max(np.abs(iso)) + 1e-9) * 0.5
                logging.info(f"Applied binaural mode {bina_mode}")
            except Exception as e:
                logging.error(f"Binaural error: {e}")
                return None, f"Binaural error: {e}"

        elapsed = time.time() - start
        mem_used = (psutil.Process().memory_info().rss - start_mem) / 1024 / 1024
        durations.append((elapsed, mem_used))
        result = dbfs(a, vol)
        return cp.asnumpy(result) if use_gpu else result, None
    except Exception as e:
        logging.error(f"Processing error: {e}")
        return None, f"Processing error: {e}"


def proc_wrapper(serialized_args):
    args = pickle.loads(serialized_args)
    return proc(*args)


def generate_layers(
    base,
    fx_vals,
    noise_type,
    hue,
    supra_txt,
    bina_toggle,
    bina_mode,
    fc,
    fb,
    iso_toggle,
    vol,
    layers,
    rec,
    time_stretch,
    speed,
    progress=None,
):
    """Optimized layered processing with recursion."""
    current = base
    total = rec + 1
    for i in range(rec + 1):
        proc_args = [
            pickle.dumps(
                (
                    current,
                    fx_vals,
                    noise_type,
                    hue,
                    supra_txt,
                    bina_toggle,
                    bina_mode,
                    fc,
                    fb,
                    iso_toggle,
                    vol,
                    time_stretch,
                    speed,
                )
            )
            for _ in range(layers)
        ]
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(layers, os.cpu_count() or 1)
            ) as ex:
                results = list(ex.map(proc_wrapper, proc_args))
        except Exception as e:
            logging.warning(
                f"Process pool failed ({e}), falling back to sequential processing"
            )
            results = [proc_wrapper(a) for a in proc_args]
        processed = []
        for res, err in results:
            if err:
                raise RuntimeError(err)
            processed.append(res)
        layer_mix = overlay(processed)
        current = overlay([current, layer_mix])
        if progress:
            progress((i + 1) / total)
    return current


def run_streamlit():
    """Launch the Streamlit user interface."""
    if st is None:
        raise RuntimeError("Streamlit is not installed")

    st.set_page_config(page_title="Superliminal", layout="wide")
    st.title("🧬 Superliminal – Optimized Benchmark")

    fx_vals = {k: st.sidebar.checkbox(k, k in ("Micro Delay", "Phase Flips")) for k in FX}

    bina_toggle = st.sidebar.checkbox("Enable Binaural Beats", True)
    time_stretch_toggle = st.sidebar.checkbox("Enable Time Stretch", False)

    st.sidebar.subheader("Morphic Fields")
    m_count = st.sidebar.number_input("Count", 1, 20, 1)
    morphs = [st.sidebar.text_area(f"Morphic {i+1}") for i in range(m_count)]

    noise_type = st.sidebar.selectbox("Noise", list(NOISE.keys()) + ["None"])
    hue = None
    if noise_type == "Hue":
        hue = st.sidebar.slider("Hue", 0, 360, 180)
    n_lvl = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)

    bina_mode = st.sidebar.selectbox("Bina Mode", ["Single", "Chord", "Sweep", "Prism", "Morph", "Quantum"])
    fc = st.sidebar.slider("Carrier Hz", 20, 2000, 150)
    fb = st.sidebar.slider("Beat Hz", 0.1, 40.0, 6.0)
    iso_toggle = st.sidebar.checkbox("Enable Isochronic", False)

    if fc < 20 or fc > 20000:
        st.error("Carrier frequency must be between 20 Hz and 20,000 Hz")
        st.stop()
    if fb <= 0 or fb > 100:
        st.error("Beat frequency must be between 0.1 Hz and 100 Hz")
        st.stop()

    st.sidebar.subheader("Supraliminal")
    supra_txt = st.sidebar.text_area("Text")
    if supra_txt and not supra_txt.strip():
        st.warning("Supraliminal text is empty")
        supra_txt = None

    st.sidebar.subheader("TTS")
    tts_txt = st.sidebar.text_area("TTS Text")
    tts_fn = st.sidebar.text_input("TTS File", "tts.mp3")
    if st.sidebar.button("Generate TTS") and tts_txt:
        if not tts_txt.strip():
            st.error("TTS text cannot be empty")
            st.stop()
        tts_fn = get_unique_filename(tts_fn)
        gTTS(tts_txt).save(tts_fn)
        st.sidebar.audio(tts_fn)
        st.sidebar.download_button("Download TTS", open(tts_fn, "rb"), tts_fn)

    up = st.file_uploader("Upload audio(s)", accept_multiple_files=True)
    out_fn = st.text_input("Output WAV", "out.wav")
    vol = st.slider("dBFS", -60.0, 0.0, -25.0)
    speed = st.sidebar.slider("Speed", 0.5, 2.0, 1.0, 0.05)
    layers = st.number_input("Layers", min_value=1, value=2, step=1)
    rec = st.number_input("Recursions", min_value=1, value=1, step=1)
    mono_output = st.sidebar.checkbox("Mono Output", False)

    if st.button("Preview"):
        if not up:
            st.error("Upload files first")
            st.stop()
        segs = []
        for f in up:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]).name
            try:
                open(tmp, 'wb').write(f.read())
                a, file_sr = sf.read(tmp)
                if file_sr != SR:
                    a = resample_poly(a, SR, file_sr)
                if a.ndim == 1:
                    a = np.column_stack((a, a))
                segs.append(a)
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")
                st.stop()
            finally:
                os.unlink(tmp)
        base = overlay(segs)
        preview_base = resample_poly(base[:int(SR * 10)], SR // 4, SR)
        preview, error = proc(
            preview_base,
            fx_vals,
            noise_type,
            hue,
            supra_txt,
            bina_toggle,
            bina_mode,
            fc,
            fb,
            iso_toggle,
            vol,
            time_stretch_toggle,
            speed,
        )
        if error:
            st.error(error)
            st.stop()
        preview = resample_poly(preview, SR, SR // 4)
        sf.write("preview.wav", preview, SR)
        st.audio("preview.wav")

    if st.button("Run"):
        if not up:
            st.error("Upload files first")
            st.stop()
        segs = []
        for f in up:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]).name
            try:
                open(tmp, 'wb').write(f.read())
                a, file_sr = sf.read(tmp)
                if file_sr != SR:
                    a = resample_poly(a, SR, file_sr)
                if a.ndim == 1:
                    a = np.column_stack((a, a))
                segs.append(a)
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")
                st.stop()
            finally:
                os.unlink(tmp)

        base = overlay(segs)
        with st.spinner("Processing layers and recursions..."):
            progress_bar = st.progress(0)

            def update(p):
                progress_bar.progress(p)

            try:
                auto = generate_layers(
                    base,
                    fx_vals,
                    noise_type,
                    hue,
                    supra_txt,
                    bina_toggle,
                    bina_mode,
                    fc,
                    fb,
                    iso_toggle,
                    vol,
                    layers,
                    rec,
                    time_stretch_toggle,
                    speed,
                    progress=update,
                )
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

        if mono_output:
            auto = np.mean(auto, axis=1)[:, np.newaxis]

        sf.write(out_fn, auto, SR)
        st.success("Done!")
        st.audio(out_fn)
        st.download_button("Download WAV", open(out_fn, "rb"), out_fn)

    if st.sidebar.button("Run Benchmark"):
        if not up:
            st.sidebar.error("Upload files first")
        else:
            durations.clear()
            segs = []
            for f in up:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]).name
                try:
                    open(tmp, 'wb').write(f.read())
                    a, file_sr = sf.read(tmp)
                    if file_sr != SR:
                        a = resample_poly(a, SR, file_sr)
                    if a.ndim == 1:
                        a = np.column_stack((a, a))
                    segs.append(a)
                except Exception as e:
                    st.sidebar.error(f"Failed to read {f.name}: {e}")
                    st.stop()
                finally:
                    os.unlink(tmp)
            base = overlay(segs)
            with st.spinner("Benchmarking..."):
                progress_bar = st.progress(0)
                errors = []
                for i in range(layers * rec):
                    result, error = proc(
                        base,
                        fx_vals,
                        noise_type,
                        hue,
                        supra_txt,
                        bina_toggle,
                        bina_mode,
                        fc,
                        fb,
                        iso_toggle,
                        vol,
                        time_stretch_toggle,
                        speed,
                    )
                    if error:
                        errors.append(error)
                    progress_bar.progress((i + 1) / (layers * rec))
                if errors:
                    for error in errors:
                        st.sidebar.error(error)
                    st.stop()
            st.sidebar.subheader("Benchmark Results")
            st.sidebar.write(f"Calls: {len(durations)}")
            st.sidebar.write(f"Avg Time: {statistics.mean([d[0] for d in durations]):.3f}s")
            st.sidebar.write(f"Min Time: {min([d[0] for d in durations]):.3f}s")
            st.sidebar.write(f"Max Time: {max([d[0] for d in durations]):.3f}s")
            st.sidebar.write(f"Avg Memory: {statistics.mean([d[1] for d in durations]):.3f} MB")

    st.sidebar.subheader("Logs")
    st.sidebar.download_button("Download log", open("superliminal.log", "rb"), "superliminal.log")


def run_cli(argv=None):
    """Command line interface for automation tools."""
    import argparse

    parser = argparse.ArgumentParser(description="Superliminal Audio CLI")
    parser.add_argument('--streamlit', action='store_true', help='Launch Streamlit UI')
    parser.add_argument('--input', nargs='+', help='Input audio files')
    parser.add_argument('--output', default='out.wav', help='Output WAV file')
    parser.add_argument('--fx', default='', help='Comma separated list of FX names')
    parser.add_argument('--noise', default='None', choices=list(NOISE.keys()) + ['None'])
    parser.add_argument('--hue', type=int)
    parser.add_argument('--binaural', action='store_true')
    parser.add_argument('--bina-mode', default='Single')
    parser.add_argument('--fc', type=int, default=150)
    parser.add_argument('--fb', type=float, default=6.0)
    parser.add_argument('--iso', action='store_true')
    parser.add_argument('--vol', type=float, default=-25.0)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--rec', type=int, default=0)
    parser.add_argument('--time-stretch', action='store_true')
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--mono-output', action='store_true')
    parser.add_argument('--supra', default=None, help='Supraliminal text')
    args = parser.parse_args(argv)

    if args.streamlit or not args.input:
        run_streamlit()
        return

    fx_vals = {k: False for k in FX}
    if args.fx:
        for name in args.fx.split(','):
            name = name.strip()
            if name in FX:
                fx_vals[name] = True

    segs = []
    for fn in args.input:
        a, file_sr = sf.read(fn)
        if file_sr != SR:
            a = resample_poly(a, SR, file_sr)
        if a.ndim == 1:
            a = np.column_stack((a, a))
        segs.append(a)
    base = overlay(segs)

    try:
        auto = generate_layers(
            base,
            fx_vals,
            args.noise,
            args.hue,
            args.supra,
            args.binaural,
            args.bina_mode,
            args.fc,
            args.fb,
            args.iso,
            args.vol,
            args.layers,
            args.rec,
            args.time_stretch,
            args.speed,
        )
    except RuntimeError as e:
        raise SystemExit(str(e))
    if args.mono_output:
        auto = np.mean(auto, axis=1)[:, np.newaxis]
    sf.write(args.output, auto, SR)
    print(f"Wrote {args.output}")


# This file provides reusable functions for both the CLI and Streamlit
# interfaces.  Use ``superliminal_cli.py`` or ``superliminal_streamlit.py``
# to run the application.

if __name__ == '__main__':
    # Default to running the CLI when executed directly
    run_cli()



