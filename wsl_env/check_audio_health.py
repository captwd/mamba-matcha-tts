import glob
import os

import numpy as np
import soundfile as sf

BASE = "/mnt/d/PycharmProjects/PythonProject9/Matcha-TTS/test_audio/sway_steps"
print(f"{'config':<14}{'dur(s)':>8}{'RMS':>8}{'peak':>8}{'clipping%':>11}")
print("-" * 50)
for cfg in sorted(os.listdir(BASE)):
    wavs = sorted(glob.glob(f"{BASE}/{cfg}/*.wav"))
    if not wavs:
        continue
    durs, rmss, peaks, clips = [], [], [], []
    for w in wavs:
        x, sr = sf.read(w, dtype="float32")
        durs.append(len(x) / sr)
        rmss.append(float(np.sqrt(np.mean(x**2))))
        peaks.append(float(np.max(np.abs(x))))
        clips.append(float(np.mean(np.abs(x) > 0.99)) * 100)
    print(f"{cfg:<14}{np.mean(durs):>8.2f}{np.mean(rmss):>8.4f}{np.mean(peaks):>8.3f}{np.mean(clips):>11.2f}")
print("\n(5 条句子均值；RMS 过小=接近静音，peak 接近 1 且 clipping% 高=削波)")
