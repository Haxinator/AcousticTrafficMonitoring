# sudo apt update
# sudo apt install portaudio19-dev

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import wave

SAMPLE_RATE = 16000
DURATION = 10  # seconds to record
SAMPLE_FILE = "./CustomDatasetRaw/cv/Fraunhofer-IDMT_train_2019-10-22-08-40_Fraunhofer-IDMT_30Kmh_2565293_M_D_TL_ME_CH12.wav_peak.wav"

def find_device_by_name(keyword):
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if keyword.lower() in d['name'].lower():
            return i
    return None

def load_audio_sample(path):
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        data = wf.readframes(wf.getnframes())
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr

def record_audio(duration, sample_rate, device=None):
    try:
        if device is not None:
            print(f"Recording {duration} seconds using device index {device}...")
        else:
            print(f"Recording {duration} seconds using default microphone...")

        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            device=device
        )
        sd.wait()
        print("Recording complete.")
        return audio.flatten()

    except Exception as e:
        print(f"Microphone recording failed: {e}")
        return None

def plot_audio(title, audio, sr, color):
    plt.figure(figsize=(10, 4))
    t = np.linspace(0, len(audio) / sr, len(audio))
    plt.plot(t, audio, color=color)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sample_audio, sample_sr = load_audio_sample(SAMPLE_FILE)
    plot_audio("Reference Sample Audio", sample_audio, sample_sr, "blue")

    xylo_device = find_device_by_name("xylo") or find_device_by_name("mems")

    if xylo_device is not None:
        print(f"Found MEMS microphone device index: {xylo_device}")
    else:
        print("No MEMS microphone found — using system default input.")

    mic_audio = record_audio(DURATION, SAMPLE_RATE, device=xylo_device)

    if mic_audio is not None:
        plot_audio("Live Microphone Audio (Xylo MEMS Mic)", mic_audio, SAMPLE_RATE, "orange")
    else:
        print("Skipping mic plot — no recording available.")
