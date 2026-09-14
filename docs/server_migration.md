# 服务器迁移与情感模型续训指南

> 目标：把情感可控合成（ESD + emotion2vec）整体搬到 Linux 服务器，
> 从本地 ep127 的 checkpoint 续训 50 epoch（带 LR 衰减），并在服务器上做情感推理。

## 1. 本地打包（Windows，Git Bash）

代码走 git（先确认情感模块已提交）：

```bash
cd /d/PycharmProjects/PythonProject9/Matcha-TTS
git log --oneline -3   # 确认情感模块 commit 已在
```

数据打包（约 2.6GB，不含原始数据集与 emotion2vec 权重）：

```bash
tar -czf esd_emo_transfer.tar.gz \
    data/ESD/train.txt data/ESD/val.txt data/ESD/wav \
    data/ESD/emo_feat data/ESD/emo_labels.json esd.json \
    logs/train/esd_emo/runs/2026-09-14_10-11-20/checkpoints/last.ckpt
```

> last.ckpt（258MB）= 全局 ep127，续训起点。若服务器带宽紧张可单独传。

## 2. 服务器环境

```bash
# 代码
git clone <你的仓库地址> Matcha-TTS && cd Matcha-TTS
# （或本地 rsync/scp 整个目录，排除 data/ logs/ results/）

# 环境
conda create -n matcha python=3.11 -y && conda activate matcha
pip install -e .
sudo apt-get install -y espeak-ng        # 文本前端，漏装会在预处理时报错
# 情感推理若要在服务器现场抽参考音频特征：pip install funasr
```

数据解压到仓库根目录（保持 `data/ESD/...` 相对路径）：

```bash
tar -xzf esd_emo_transfer.tar.gz
```

## 3. 冒烟验证（各 ~1 分钟）

```bash
# 3.1 配置可组合（含 ljspeech 回归修复、衰减调度器）
python matcha/train.py --cfg job experiment=esd_emo_long | grep -E "emo_dim|milestones|max_epochs"
#    期望看到 emo_dim: 768 / milestones: [600, 900, 1050, 1150] / max_epochs: 1200

# 3.2 一个 epoch 的试跑（确认数据加载 / 特征 / 标签全通）
python matcha/train.py experiment=esd_emo_finetune \
    ckpt_path=/data/ckpts/esd_emo_ep127_last.ckpt \
    trainer.max_epochs=128 trainer.fast_dev_run=false data.num_workers=8
#    跑起来 1~2 个 epoch 后 Ctrl+C，检查 logs/train/esd_emo_finetune/runs/<ts>/train.log
```

## 4. 正式训练（tmux 里跑）

**主路径：长跑 1200 epoch**（62 万步，超过官方 500k updates 量级；按本地 3.7 min/epoch
约 74 小时，服务器 GPU 按比例缩短）：

```bash
tmux new -s emo
conda activate matcha
cd Matcha-TTS
python matcha/train.py experiment=esd_emo_long \
    ckpt_path=/data/ckpts/esd_emo_ep127_last.ckpt \
    data.num_workers=8 2>&1 | tee train_emo.log
# Ctrl+B, D 脱离；tmux attach -t emo 回来
```

要点：

- **LR 衰减已配置好**：`model/scheduler=multistep_long`，里程碑写在全局 epoch 轴
  `[600, 900, 1050, 1150]`、gamma 0.5：前 600 epoch 恒定 1e-4，之后 5e-5 → 2.5e-5 →
  1.25e-5 → 6.25e-6（终点 LR 与 LJSpeech 衰减配方一致）。从 ep127 续训则实际
  还要跑 1073 epoch。恢复后抽查 ckpt 内 lr 确认调度生效（09-07 的教训）
- **checkpoint 策略**：每 epoch 存编号 ckpt 但 `save_top_k=3` + `last.ckpt`，磁盘占用 ≈ 1GB；
  中断后用同一个命令重跑（`ckpt_path` 指向新 run 的 last.ckpt）即断点续训
- 多卡：加 `trainer.devices=2`（官方即 2 卡 batch 32 训练）
- 监控：`tensorboard --logdir logs/train/esd_emo_long --port 6006`，
  关注 `val_mcd/mean`（长恒定段允许震荡，ep600 衰减后应下探并走平）
  与 `sub_loss/val_emo_loss`（应维持 ~0.05）

**备选：先来个 50 epoch 快赢**（不进 tmux 长跑，先快速验证衰减收益、出可听版本）：

```bash
python matcha/train.py experiment=esd_emo_finetune \
    ckpt_path=/data/ckpts/esd_emo_ep127_last.ckpt   # ep127 -> ep177，衰减 [140,155,167,175]
```

注意：跑完这个再进 `esd_emo_long` 会从 6.25e-6 回跳 1e-4（warm restart，通常无害
但要有预期）。要长跑就直接用上面的主路径，二选一。

## 5. 情感推理（训练完就能听）

```bash
# 5.1 一次性构建 5 类情感中心（angry/happy/neutral/sad/surprise）
python scripts/synthesize_emo.py --checkpoint_path <新ckpt> --build_centroids

# 5.2 合成（emo_scale 建议扫 {1.0, 1.5, 2.0, 3.0} 对比听）
python scripts/synthesize_emo.py --checkpoint_path <新ckpt> \
    --text "I can't believe you did that!" --emotion surprise --emo_scale 2.0 --spk 0

# 5.3 用任意参考音频的语气合成（需 funasr + emo2vec_base_funasr 模型目录）
python scripts/synthesize_emo.py --checkpoint_path <新ckpt> \
    --text "..." --emo_wav ref.wav --emo_model /data/models/emo2vec_base_funasr --emo_scale 1.5
```

输出在 `results/synth_emo/`（wav + mel npy）。

## 6. 统一评估（可选，与 LJSpeech 协议对齐）

```bash
python scripts/evaluate.py --checkpoint_path <新ckpt> \
    --filelist data/ESD/val.txt --output_folder results/eval_emo_ft \
    --vocoder hifigan_T2_v1 --steps 10
```

对照基线：本地 ep126 模型（同协议）可先在服务器评一遍作 baseline，续训后对比逐句 Wilcoxon。

## 备注

- `esd.json` / `ravdess.json`（仓库根目录的 mel 统计）只是 `generate_data_statistics.py`
  的落盘副本，统计已内嵌在 `configs/data/*.yaml`，服务器上不需要
- MCD 绝对值跨数据集不可比：ESD 是 10 说话人情感语料，val_mcd 与 LJSpeech 的 51~53dB
  不可直接对表，看相对改善即可
