# Mamba-Matcha-TTS

在 [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)（ICASSP 2024，基于条件流匹配的非自回归 TTS）基础上做的**骨干 / 算子替换实验**仓库：

- 把 U-Net 声学骨干的**全局混合器**（Self-Attention）替换为**双向 Mamba2**
- 把**局部特征块**替换为 **ConvNeXt V2** 单元
- 配套：声码器注册表（HiFi-GAN / BigVGAN）、MCD/WER/CER 评估体系、离线预处理缓存

所有变体通过配置开关隔离，同一份代码可跑全部实验。上游原版说明与论文引用见文末[关于上游](#关于上游)。

## 本仓库改动

| 模块 | 改动 | 入口 |
|---|---|---|
| **双向 Mamba2 骨干** | U-Net 每级的全局混合器（Self-Attention）可替换为双向 Mamba2 + LayerScale 零初始化；含 Mamba2 特殊初始化的保护逻辑 | `matcha/models/components/mamba_block.py` · `model/decoder=mamba` |
| **ConvNeXt V2 局部算子** | 局部特征块（ResnetBlock1D）可替换为 ConvNeXt V2 单元（深度卷积 k7 + GRN + 倒瓶颈） | `matcha/models/components/convnext_v2.py` · `+model.decoder.resnet_type=convnext_v2` |
| **声码器注册表** | 声码器加载从 if/else 重构为装饰器注册表，新增声码器只需加文件；接入 HiFi-GAN（T2 / universal）与 BigVGAN | `matcha/vocoders/` |
| **评估体系** | 零新依赖的 MCD（DTW + DCT-II）与 WER/CER；批量评估 CLI；训练中 `val_mcd` 曲线 | `matcha/utils/metrics.py` · `scripts/evaluate.py` |
| **Sway Sampling（推理期）** | 推理时对 ODE 时间轴做非均匀重参数化，少步数下显著改善 MCD；不改模型、不重训（5 个模型 × 3 个步数全部 p<0.0001） | `matcha/models/components/flow_matching.py` · `--sway_sampling_coef` |
| **离线预处理缓存** | mel / 音素离线算一次存盘，训练只读缓存 | `scripts/preprocess_dataset.py` |
| **文档** | 每日工作总结、改动清单、硬件 / 参数图解文档 | `docs/` |
| **静态 Demo** | 多系统试听页与一键音频生成脚本 | `demo/` · `scripts/prepare_demo_audio.py` |

## 环境安装

```bash
conda create -n matcha python=3.11 -y
conda activate matcha
pip install -e .
```

文本前端依赖系统级 `espeak-ng`（pip 装不了）：

```bash
# Ubuntu / WSL
sudo apt-get install -y espeak-ng
# Windows
winget install --id eSpeak-NG.eSpeak-NG --silent
```

Mamba 变体需要 Linux / WSL + `mamba-ssm`（Windows 无官方 wheel）。安装与验证脚本见 [`wsl_env/`](wsl_env/)：

```bash
bash wsl_env/install_mamba_wheels.sh   # 官方预编译 wheel（torch2.8 + cu128 + cp311）
python wsl_env/smoke.py                # 环境自检（torch/CUDA/matcha 导入）
python wsl_env/test_mamba_gpu.py       # Mamba / causal_conv1d GPU 前向+反向
```

## 数据准备（以 LJSpeech 为例）

1. 下载 [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/)，解压到 `data/LJSpeech-1.1`
2. 生成训练 / 验证文件列表：

   ```bash
   python scripts/prepare_ljspeech_filelists.py
   ```

3. 计算归一化统计量，写入 `configs/data/ljspeech.yaml` 的 `data_statistics`：

   ```bash
   matcha-data-stats -i ljspeech.yaml
   ```

4. （可选，推荐）离线预处理缓存：

   ```bash
   python scripts/preprocess_dataset.py --data_config configs/data/ljspeech.yaml --workers 8
   ```

## 训练

```bash
# 基线（原版 Transformer + U-Net）
python matcha/train.py experiment=ljspeech_min_memory

# 双向 Mamba2 全局混合器（需 Linux/WSL）
python matcha/train.py experiment=ljspeech_min_memory model/decoder=mamba

# ConvNeXt V2 局部算子
python matcha/train.py experiment=ljspeech_min_memory +model.decoder.resnet_type=convnext_v2

# 只替换中间段，做小步验证
python matcha/train.py experiment=ljspeech_min_memory model/decoder.mid_block_type=mamba
```

常用覆盖项：

```bash
data.batch_size=32 +trainer.accumulate_grad_batches=2 \
trainer.max_epochs=140 trainer.check_val_every_n_epoch=5 \
callbacks.model_checkpoint.every_n_epochs=5 callbacks.model_checkpoint.save_top_k=8
```

## 合成 / 推理

```bash
# 命令行合成（需指定自己的 checkpoint；声码器可选 hifigan_T2_v1 / hifigan_univ_v1 / bigvgan）
matcha-tts --text "<文本>" --checkpoint_path <ckpt> --vocoder hifigan_T2_v1

# Gradio 界面
matcha-tts-app
```

## 评估

统一协议：**HiFi-GAN T2 + Denoiser、val 全量 100 句、ODE 10 步**（`--no_denoiser` 可关闭去噪器）。

```bash
python scripts/evaluate.py \
    --checkpoint_path <ckpt> \
    --filelist data/LJSpeech-1.1/val.txt \
    --output_folder results/eval_xxx \
    --vocoder hifigan_T2_v1 --steps 10
```

输出逐句 `results.csv`（MCD / WER / CER），可与其他系统做**逐句配对 Wilcoxon 检验**（示例：`wsl_env/compare_final.sh`）。

指标口径（重要）：

- MCD 使用本仓库自研提取器（log-mel 80 + DCT-II + DTW），**绝对值只做同体系横向比较**。参考刻度：声码器往返 ≈ 3.65 dB（链路下限）、已训模型 ≈ 51–53 dB、跨语句自然语音 ≈ 55–90 dB
- WER / CER 由 wav2vec2 ASR 计算，存在 ASR 上限（真实语音自身 WER ≈ 7.6%），接近上限时差异会被压缩

## 其它功能

- **ONNX 导出 / 推理**：`python3 -m matcha.onnx.export` / `python3 -m matcha.onnx.infer`（上游功能，已保留）
- **从训练模型提取音素时长**：`matcha-tts-get-durations -i ljspeech.yaml -c <ckpt>`
- **静态试听 Demo**：在线试听 <https://captwd.github.io/demo/> ｜ 源码 [`demo/`](demo/)，`scripts/prepare_demo_audio.py` 可一键重新生成全部音频

## 文档与实验记录

- 改动清单：[`docs/modifications.md`](docs/modifications.md)
- 工作总结：
  - [`docs/2026-09-08_summary.md`](docs/2026-09-08_summary.md)：WSL 迁移、mamba-ssm 安装、Mamba 对照实验与统计结论
  - [`docs/2026-09-06_summary.md`](docs/2026-09-06_summary.md)：评估体系、ConvNeXt V2、LR 衰减干预与震荡机制
- 训练日志：[`docs/worklog.md`](docs/worklog.md)
- 系列博客：<https://captwd.github.io/series/>
- WSL / mamba 环境脚本：[`wsl_env/`](wsl_env/)

> 仓库**不含**数据集与模型权重 / checkpoint（`data/`、`logs/` 已忽略）；`results/` 仅保留逐句指标 CSV，评估音频由 `demo/` 提供试听样本。

## 关于上游

本仓库基于 [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) 修改，保留其 [MIT License](LICENSE)。

Matcha-TTS 提出了一种基于条件流匹配（conditional flow matching）的非自回归 TTS 架构，特点是概率化建模、内存占用小、合成速度快、音质自然。原论文与官方资源：

- 论文：<https://arxiv.org/abs/2309.03199>（ICASSP 2024）
- 官方 demo：<https://shivammehta25.github.io/Matcha-TTS>
- 预训练模型 / HuggingFace Space：见上游 README

### 引用

如果你使用了上游代码或论文，请引用：

```text
@inproceedings{mehta2024matcha,
  title={Matcha-{TTS}: A fast {TTS} architecture with conditional flow matching},
  author={Mehta, Shivam and Tu, Ruibo and Beskow, Jonas and Sz{\'e}kely, {\'E}va and Henter, Gustav Eje},
  booktitle={Proc. ICASSP},
  year={2024}
}
```

### 致谢

- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)：训练框架基础
- [Coqui-TTS](https://github.com/coqui-ai/TTS/tree/dev)：Cython 二进制打包思路
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)：Transformer 组件与 BigVGAN 生态
- [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)：单调对齐搜索源码
- [torchdyn](https://github.com/DiffEqML/torchdyn)、[labml.ai](https://nn.labml.ai/transformers/rope/index.html)：ODE 求解器与 RoPE 实现参考
