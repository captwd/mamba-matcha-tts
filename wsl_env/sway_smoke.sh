#!/bin/bash
cd ~/projects/Matcha-TTS
P=~/miniconda3/envs/matcha/bin/python
CKPT=/mnt/d/PycharmProjects/PythonProject9/Matcha-TTS/logs/train/ljspeech_min/runs/2026-09-07_16-11-41/checkpoints/checkpoint_epoch=134.ckpt

echo "=== sway OFF (steps=4, 3 utts) ==="
$P scripts/evaluate.py --checkpoint_path $CKPT --filelist data/LJSpeech-1.1/val.txt \
  --output_folder results/sway_smoke_off --vocoder hifigan_T2_v1 --steps 4 --max_utts 3 --seed 1234 2>&1 | tail -6

echo "=== sway -1.0 (steps=4, 3 utts) ==="
$P scripts/evaluate.py --checkpoint_path $CKPT --filelist data/LJSpeech-1.1/val.txt \
  --output_folder results/sway_smoke_on --vocoder hifigan_T2_v1 --steps 4 --max_utts 3 --seed 1234 --sway_sampling_coef -1.0 2>&1 | tail -6
