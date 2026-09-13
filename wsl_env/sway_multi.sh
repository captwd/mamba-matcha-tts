#!/bin/bash
cd ~/projects/Matcha-TTS
P=~/miniconda3/envs/matcha/bin/python
W=/mnt/d/PycharmProjects/PythonProject9/Matcha-TTS/logs/train/ljspeech_min/runs
OUT=results/sway_multi

declare -A CKPTS=(
  [unet139]="$W/2026-09-05_12-02-02/checkpoints/checkpoint_epoch=139.ckpt"
  [convnext139_const]="$W/2026-09-06_19-53-10/checkpoints/checkpoint_epoch=139.ckpt"
  [mamba139]="$HOME/projects/Matcha-TTS/logs/train/ljspeech_min/runs/2026-09-08_13-30-03/checkpoints/checkpoint_epoch=139.ckpt"
  [official]="$W/matcha_ljspeech.ckpt"
)

for name in unet139 convnext139_const mamba139 official; do
  ckpt=${CKPTS[$name]}
  for steps in 4 6 10; do
    for sway in off m1; do
      if [ "$sway" = "off" ]; then EXTRA=""; else EXTRA="--sway_sampling_coef -1.0"; fi
      echo "########## $name | steps=$steps sway=$sway ##########"
      $P scripts/evaluate.py --checkpoint_path "$ckpt" \
        --filelist data/LJSpeech-1.1/val.txt \
        --output_folder $OUT/${name}/s${steps}_${sway} \
        --vocoder hifigan_T2_v1 --steps $steps --seed 1234 --asr none $EXTRA 2>&1 \
        | grep -E "MCD \(dtw\)|sway_sampling_coef ="
    done
  done
done
echo "SWAY_MULTI_DONE"
