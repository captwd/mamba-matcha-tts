# WSL2 / Linux 环境搭建与验证脚本

本项目在 Windows 上训练，但 Mamba 骨干依赖 `mamba-ssm`（无 Windows 官方 wheel），因此迁移到 WSL2 Ubuntu 训练。本目录收录迁移与验证用到的脚本。

## 环境要点

| 项 | 版本 |
|---|---|
| Python | 3.11 |
| torch | 2.8.0+cu128（nvcc 12.8，GPU 为 sm_120/Blackwell） |
| mamba-ssm | 2.2.5（官方预编译 wheel） |
| causal-conv1d | 1.7.0（官方预编译 wheel） |
| transformers | <5（mamba 运行依赖） |

## 安装步骤

1. **系统依赖**

   ```bash
   sudo apt-get install -y espeak-ng
   ```

2. **安装 mamba 相关 wheel**

   选 wheel 的四个要素：CUDA 主版本、torch 版本、C++ ABI、Python/平台。ABI 用
   `python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"` 查。

   ```bash
   bash install_mamba_wheels.sh
   ```

   > 注意：不要装 mamba-ssm 2.3.x —— 其 metadata 会拉入 triton>=3.5 / tilelang /
   > quack-kernels，可能强制升级 torch。2.2.5 依赖干净且自带 `Mamba` 与 `Mamba2`。

3. **复用已有权重缓存（可选，省去重复下载）**

   ```bash
   bash link_caches.sh
   ```

   软链 Windows 侧的 HuggingFace 缓存（BigVGAN）、torchaudio 检查点（wav2vec2 ASR）、
   以及 `matcha_tts` 声码器权重目录。

## 验证

```bash
python smoke.py            # torch/CUDA + matcha 全模块导入
python test_mamba_gpu.py   # causal_conv1d / Mamba / Mamba2 的 GPU 前向+反向
python test_mamba_block.py # MambaBlock1D 单元/集成/初始化保护/hydra 端到端（15 项）
python check_frontend.py   # 文本前端（phonemizer/espeak）+ 对齐算子
```

## 评估协议（示例）

统一协议：**HiFi-GAN T2 声码器 + Denoiser、LJSpeech val 全量 100 句、ODE 10 步**。

```bash
bash eval_full_ep139.sh    # 单检查点全量评估 → results/<name>/results.csv
bash compare_final.sh      # 与基线逐句配对 Wilcoxon 检验
```

评估脚本中 `--no_denoiser` 可关闭去噪器；`--vocoder hifigan_T2_v1|bigvgan` 切换声码器。
