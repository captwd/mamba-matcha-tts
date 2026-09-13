#!/bin/bash
cd ~/projects/Matcha-TTS
P=~/miniconda3/envs/matcha/bin/python
W=/mnt/d/PycharmProjects/PythonProject9/Matcha-TTS/logs/train/ljspeech_min/runs
OUT=results/sway_curve
CN="$W/2026-09-07_16-11-41/checkpoints/checkpoint_epoch=134.ckpt"
OFF="$W/matcha_ljspeech.ckpt"

echo "===== 1 步（sway 对 1 步无影响，端点不变）====="
for pair in "convnext_decay134:$CN" "official:$OFF"; do
  name=${pair%%:*}; ckpt=${pair#*:}
  echo "--- $name steps=1 ---"
  $P scripts/evaluate.py --checkpoint_path "$ckpt" --filelist data/LJSpeech-1.1/val.txt \
    --output_folder $OUT/${name}/s1_off --vocoder hifigan_T2_v1 --steps 1 --seed 1234 --asr none 2>&1 | grep "MCD (dtw)"
done

echo "===== 带 ASR：2 步(关/开) vs 10 步(关) ====="
for pair in "convnext_decay134:$CN" "official:$OFF"; do
  name=${pair%%:*}; ckpt=${pair#*:}
  for spec in "2:off:" "2:m10:--sway_sampling_coef -1.0" "10:off:"; do
    steps=${spec%%:*}; rest=${spec#*:}; tag=${rest%%:*}; extra=${rest#*:}
    echo "--- $name steps=$steps sway=$tag ---"
    $P scripts/evaluate.py --checkpoint_path "$ckpt" --filelist data/LJSpeech-1.1/val.txt \
      --output_folder $OUT/${name}_asr/s${steps}_${tag} --vocoder hifigan_T2_v1 \
      --steps $steps --seed 1234 $extra 2>&1 | grep -E "MCD \(dtw\)|WER :|CER :"
  done
done
echo "SWAY_CURVE_ASR_DONE"
