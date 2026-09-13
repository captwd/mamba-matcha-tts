#!/bin/bash
cd ~/projects/Matcha-TTS
P=~/miniconda3/envs/matcha/bin/python
W=/mnt/d/PycharmProjects/PythonProject9/Matcha-TTS/logs/train/ljspeech_min/runs
OUT=results/sway_curve

declare -A CKPTS=(
  [convnext_decay134]="$W/2026-09-07_16-11-41/checkpoints/checkpoint_epoch=134.ckpt"
  [official]="$W/matcha_ljspeech.ckpt"
)

for name in convnext_decay134 official; do
  ckpt=${CKPTS[$name]}
  for steps in 2 3 4 6 10; do
    for tag in off m05 m10; do
      case $tag in
        off) EXTRA="";;
        m05) EXTRA="--sway_sampling_coef -0.5";;
        m10) EXTRA="--sway_sampling_coef -1.0";;
      esac
      echo "########## $name | steps=$steps sway=$tag ##########"
      $P scripts/evaluate.py --checkpoint_path "$ckpt" \
        --filelist data/LJSpeech-1.1/val.txt \
        --output_folder $OUT/${name}/s${steps}_${tag} \
        --vocoder hifigan_T2_v1 --steps $steps --seed 1234 --asr none $EXTRA 2>&1 \
        | grep -E "MCD \(dtw\)"
    done
  done
done
echo "SWAY_CURVE_DONE"
