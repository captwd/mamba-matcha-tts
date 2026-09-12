#!/bin/bash
E=$HOME/miniconda3/envs/matcha
PIP=$E/bin/pip
W=$HOME/mamba_build/wheels
mkdir -p $W && cd $W

fetch() {  # fetch <url>
  wget -c -q --timeout=30 --tries=2 "$1" && return 0
  echo "direct download failed, trying gh-proxy..."
  wget -c -q --timeout=60 --tries=3 "https://gh-proxy.com/$1" && return 0
  return 1
}

CC_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
MM_URL="https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"

echo "=== download causal-conv1d 1.7.0 wheel ==="
fetch "$CC_URL" || { echo DL_CC_FAIL; exit 1; }
echo "=== download mamba-ssm 2.2.5 wheel ==="
fetch "$MM_URL" || { echo DL_MM_FAIL; exit 1; }
ls -lh *.whl

echo "=== install ==="
$PIP install --no-deps --force-reinstall causal_conv1d-1.7.0+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl || { echo INST_CC_FAIL; exit 1; }
$PIP install --no-deps --force-reinstall mamba_ssm-2.2.5+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl || { echo INST_MM_FAIL; exit 1; }
echo WHEEL_INSTALL_DONE
