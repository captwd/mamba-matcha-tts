#!/bin/bash
cd ~/projects/Matcha-TTS
P=~/miniconda3/envs/matcha/bin/python
CKPT=/mnt/d/PycharmProjects/PythonProject9/Matcha-TTS/logs/train/ljspeech_min/runs/2026-09-07_16-11-41/checkpoints/checkpoint_epoch=134.ckpt

for steps in 4 6 10; do
  for sway in off m1; do
    if [ "$sway" = "off" ]; then EXTRA=""; else EXTRA="--sway_sampling_coef -1.0"; fi
    echo "########## steps=$steps sway=$sway ##########"
    $P scripts/evaluate.py --checkpoint_path $CKPT \
      --filelist data/LJSpeech-1.1/val.txt \
      --output_folder results/sway_ab/s${steps}_${sway} \
      --vocoder hifigan_T2_v1 --steps $steps --seed 1234 $EXTRA 2>&1 \
      | grep -E "sway_sampling_coef =|MCD \(dtw\)|WER :|CER :"
  done
done
echo "SWAY_GRID_DONE"
