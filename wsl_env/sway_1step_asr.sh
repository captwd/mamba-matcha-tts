#!/bin/bash
cd ~/projects/Matcha-TTS
P=~/miniconda3/envs/matcha/bin/python
W=/mnt/d/PycharmProjects/PythonProject9/Matcha-TTS/logs/train/ljspeech_min/runs
OUT=results/sway_curve
CN="$W/2026-09-07_16-11-41/checkpoints/checkpoint_epoch=134.ckpt"
OFF="$W/matcha_ljspeech.ckpt"

for pair in "convnext_decay134:$CN" "official:$OFF"; do
  name=${pair%%:*}; ckpt=${pair#*:}
  echo "--- $name steps=1 (with ASR) ---"
  $P scripts/evaluate.py --checkpoint_path "$ckpt" --filelist data/LJSpeech-1.1/val.txt \
    --output_folder $OUT/${name}_asr/s1_off --vocoder hifigan_T2_v1 \
    --steps 1 --seed 1234 2>&1 | grep -E "MCD \(dtw\)|WER :|CER :"
done
echo "SWAY_1STEP_ASR_DONE"
