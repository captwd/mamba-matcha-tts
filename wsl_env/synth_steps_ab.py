# -*- coding: utf-8 -*-
"""生成"步数/sway"对比音频，供人耳试听。
配置：1步 / 2步(关) / 2步+sway / 4步+sway / 10步(关)
输出：/mnt/d/PycharmProjects/PythonProject9/Matcha-TTS/test_audio/sway_steps/<cfg>/sentence_i.wav
"""
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, "/home/zkw/projects/Matcha-TTS")

from matcha.cli import load_matcha, load_vocoder, process_text  # noqa: E402
from matcha.utils.utils import get_user_data_dir  # noqa: E402

CKPT = "/mnt/d/PycharmProjects/PythonProject9/Matcha-TTS/logs/train/ljspeech_min/runs/2026-09-07_16-11-41/checkpoints/checkpoint_epoch=134.ckpt"
VAL = Path("/home/zkw/projects/Matcha-TTS/data/LJSpeech-1.1/val.txt")
OUT = Path("/mnt/d/PycharmProjects/PythonProject9/Matcha-TTS/test_audio/sway_steps")
UTTS = ["LJ050-0184", "LJ009-0077", "LJ016-0002", "LJ010-0097", "LJ003-0136"]

CFGS = [
    ("1step", 1, None),
    ("2step", 2, None),
    ("2step_sway", 2, -1.0),
    ("4step_sway", 4, -1.0),
    ("10step", 10, None),
]

# 读文本
texts = {}
for line in VAL.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    parts = line.strip().split("|")
    uid = Path(parts[0]).stem
    if uid in UTTS:
        texts[uid] = parts[2] if len(parts) >= 3 else parts[1]
assert len(texts) == len(UTTS), texts.keys()

device = torch.device("cuda")
model = load_matcha("custom_model", CKPT, device)
voc_path = get_user_data_dir() / "hifigan_T2_v1"
vocoder, denoiser = load_vocoder("hifigan_T2_v1", voc_path, device)

with torch.inference_mode():
    for name, steps, sway in CFGS:
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        model.decoder.sway_sampling_coef = sway
        for i, uid in enumerate(UTTS, start=1):
            torch.manual_seed(1234 + i)
            torch.cuda.manual_seed_all(1234 + i)
            p = process_text(i, texts[uid], device, ("english_cleaners2",))
            out = model.synthesise(p["x"], p["x_lengths"], n_timesteps=steps,
                                   temperature=0.667, length_scale=1.0)
            wav = vocoder(out["mel"]).clamp(-1, 1)
            if denoiser is not None:
                wav = denoiser(wav.squeeze(), strength=0.00025).unsqueeze(0)
            sf.write(d / f"sentence_{i}.wav", wav.squeeze().cpu().numpy(), 22050, "PCM_16")
        print(f"[+] {name}: steps={steps} sway={sway} -> {d}")
print("DONE")
