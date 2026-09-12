#!/bin/bash
cd ~/projects/Matcha-TTS
CKPT=logs/train/ljspeech_min/runs/2026-09-08_13-30-03/checkpoints/checkpoint_epoch=139.ckpt
~/miniconda3/envs/matcha/bin/python scripts/evaluate.py \
  --checkpoint_path $CKPT \
  --filelist data/LJSpeech-1.1/val.txt \
  --output_folder results/eval_mamba_ep139 \
  --steps 10
