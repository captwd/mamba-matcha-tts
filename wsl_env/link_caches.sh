#!/bin/bash
set -e
mkdir -p ~/.cache/torch/hub
# symlink HF cache (BigVGAN weights) and torchaudio ASR checkpoint from Windows cache
[ -e ~/.cache/huggingface ] || ln -s /mnt/c/Users/zkw34/.cache/huggingface ~/.cache/huggingface
[ -e ~/.cache/torch/hub/checkpoints ] || ln -s /mnt/c/Users/zkw34/.cache/torch/hub/checkpoints ~/.cache/torch/hub/checkpoints
echo "--- HF hub contents ---"
ls ~/.cache/huggingface/hub/ 2>/dev/null
echo "--- torch hub checkpoints ---"
ls -lh ~/.cache/torch/hub/checkpoints/
echo LINKED
