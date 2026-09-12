# Matcha-TTS 训练工作日志

> 记录环境搭建、训练过程中遇到的问题及解决方案，供日后查阅。

## 项目背景

- 项目：Matcha-TTS（基于条件流匹配 Conditional Flow Matching 的快速 TTS）
- 数据集：LJSpeech-1.1（13100 条，~24 小时单说话人英文语音）
- 硬件：RTX 5060 Laptop GPU（8GB 显存）+ AMD R9 8940HX + 16GB DDR5
- 环境：conda env `myagent`（Python 3.10 / torch 2.7.1+cu128 / Lightning 2.6.5 / matplotlib 3.10.9）

---

## 2026-09-04

### 1. 代码学习与中文注释
- 对项目约 30 个核心文件添加了模块级中文注释（入口脚本、模型、声码器、文本前端、数据模块、工具集、配置文件）。

### 2. 安装方式选择
- 代码已 clone 到本地，无需 `pip install matcha-tts` 或 `pip install git+...`；
- 只需 `pip install -e .`：安装依赖 + 编译 Cython 对齐模块 + 注册命令行工具。

### 3. ❌ pip 构建依赖 SSL 连接失败
- **现象**：`installing build dependencies did not run successfully`，USTC 镜像 SSL EOF。
- **原因**：VPN/代理拦截或镜像瞬时故障。
- **解决**：重试即恢复；必要时 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 换源或 `--no-build-isolation`。

### 4. ❌ 缺少 MSVC 编译器（重要）
- **现象**：`error: Microsoft Visual C++ 14.0 or greater is required`（编译 Cython 模块 `matcha.utils.monotonic_align.core` 失败）。
- **官方解法**：装几个 GB 的 VS Build Tools（不划算）。
- **实际解法（免编译）**：
  1. `matcha/utils/monotonic_align/__init__.py`：编译版导入失败时自动回退到纯 Python/Numpy 实现（200 组随机用例验证与 Cython 语义逐位一致）；
  2. `setup.py`：自定义 `TolerantBuildExt`，编译失败打印警告并跳过，安装不中断。
- **代价**：对齐计算稍慢（推理无感知）；以后装了 VS Build Tools 可自动切回编译版。

### 5. ❌ LJSpeech 缺少训练/验证文件列表
- **现象**：配置里 `train_filelist_path` 指向的 train.txt/val.txt 不存在，只有原始 metadata.csv。
- **解决**：新写 `scripts/prepare_ljspeech_filelists.py`，把 metadata.csv（id|原文|规范化文本）转换为训练格式（`wav相对路径|规范化文本`），随机划分 13000 训练 / 100 验证（种子 1234），并校验 wav 路径存在。

### 6. ❌ espeak not installed on your system
- **现象**：实例化 datamodule 时 `RuntimeError`。
- **原因**：pip 包 `phonemizer` 依赖系统级原生程序 espeak-ng（pip 装不了）。
- **解决**：
  ```powershell
  winget install --id eSpeak-NG.eSpeak-NG --silent
  setx PHONEMIZER_ESPEAK_LIBRARY "C:\Program Files\eSpeak NG\libespeak-ng.dll"
  setx PHONEMIZER_ESPEAK_PATH "C:\Program Files\eSpeak NG\espeak-ng.exe"
  ```
- **注意**：phonemizer 3.4.0 不会自动发现 winget 安装路径，环境变量必须设。

### 7. ❌ matplotlib 3.10 移除旧 API
- **现象**：Sanity Check 后崩溃 `'FigureCanvasTkAgg' object has no attribute 'tostring_rgb'`。
- **原因**：matplotlib 3.10 删除了 `tostring_rgb()`；且默认用了 TkAgg 后端（画图会偷偷创建窗口）。
- **解决**：`matcha/utils/utils.py` 的 `save_figure_to_numpy()` 改用 `buffer_rgba()`，并在模块顶部强制 `matplotlib.use("Agg")`。

### 8. ❌ 训练 9 轮后没有任何 checkpoint
- **现象**：想断点续训，发现 checkpoints 文件夹根本没建出来。
- **原因**：配置 `every_n_epochs: 100`（编号 checkpoint 每 100 轮存一次）；且新版 Lightning（2.5+）规定 **last.ckpt 只在"本轮存过编号 checkpoint"时才跟着写**。
- **解决**：`configs/callbacks/model_checkpoint.yaml` 改为 `every_n_epochs: 1`、`save_top_k: 3`（每个 checkpoint 含优化器状态约 209MB，控制磁盘）。
- **教训**：9 轮进度丢失，只能重训。改完配置先确认第 1 轮能产出 last.ckpt 再长挂。

### 9. ✅ 提速补丁（训练入口 + dataloader）
- `matcha/train.py`：`torch.set_float32_matmul_precision("high")` 开启 TF32（RTX Tensor Core）。
- `matcha/data/text_mel_datamodule.py`：`persistent_workers=True` + `prefetch_factor=3`（Windows 上每轮重启 worker 进程开销大）。

### 10. ⚠️ 常驻 worker 吃内存（8 个进程 ~6.2GB）
- **原因**：train/val 两个 loader 各 4 个 persistent worker，每个进程 ~780MB（torch 运行时）。
- **解决**：离线缓存生效时，训练集 worker 最多 2 个、验证集 worker 0 个（验证集仅 100 条，主进程直接读缓存），省约 4.6GB。

### 11. ✅ 离线预处理缓存（本轮最大优化，自己设计的方案）
- **思路**：训练时每轮都重复"读 wav→FFT→梅尔谱"和"espeak 音素化"，纯属浪费。改为离线算一次存盘，训练时直接读。
- **实现**：
  - 新增 `scripts/preprocess_dataset.py`（多进程并行、断点续跑、meta.json 参数校验）；
  - `TextMelDataset` 加缓存命中检测：命中读 `.npy`/`phonemes.json`，未命中自动回退现场计算；配置与缓存参数不一致时自动禁用缓存并告警。
- **效果**：13100 条预处理仅 47 秒（8 进程），缓存 2.2GB；6 个抽查样本验证缓存与现场计算逐位一致；每轮训练大约快 30 秒。
- **注意**：运行脚本前必须先设置 espeak 环境变量（脚本 import cleaners 时就需要）。

### 12. ❌ torch 2.6 checkpoint 加载失败（weights_only）
- **现象**：断点续训报 `UnpicklingError ... omegaconf.dictconfig.DictConfig was not an allowed global`。
- **原因**：PyTorch 2.6 起 `torch.load` 默认 `weights_only=True`（安全模式），Lightning 传 `weights_only=None` 也会按新默认处理；checkpoint 里存了 OmegaConf 对象被拒。
- **解决**：`matcha/utils/utils.py` 模块级补丁 `_torch_load_compat`：未显式指定时把 `weights_only=None` 恢复为旧行为 `False`（本地文件来源可信）。放在公共模块，训练/推理两条路径都生效。

### 13. ❌ 系统睡眠杀死训练
- **现象**：训练"暂停"（GPU 1%，进程全没了）。
- **原因**：系统日志显示 21:04 笔记本进入睡眠（Connected Standby），唤醒后 CUDA 上下文损坏，进程死亡。
- **解决**：
  ```powershell
  powercfg /change standby-timeout-ac 0
  powercfg /change hibernate-timeout-ac 0
  ```
- **教训**：笔记本挂训练必须禁止系统睡眠（屏幕可关，系统不能睡）。

### 14. ❌ Hydra 参数解析报错（mismatched input '='）
- **现象**：`ckpt_path=...\checkpoint_epoch=071.ckpt` 让 Hydra 把文件名里的 `=` 当成键值对分隔符。
- **解决**：给值加引号 `"ckpt_path='...\checkpoint_epoch=071.ckpt'"`；或复制改名成无 `=` 的文件（如 resume.ckpt）。这也是 `last.ckpt` 好用的原因（文件名无特殊字符）。

### 15. batch_size 与显存调优记录
- 32 → 48：每轮略快，显存 6GB+；
- 64：GPU 利用率最高（60%+），但显存 7.7/8GB 贴上限，有概率性 OOM 风险（PyTorch 缓存分配器稳态复用显存，长音频批次+碎片化才会触发）；
- 缓解：`$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"`（减少碎片化）；
- 每轮固定开销优化：`trainer.check_val_every_n_epoch=5`（验证+画图摊薄）。
- **重要认知**：PyTorch 没有自动防 OOM 机制，炸了靠每轮 checkpoint 兜底。

### 16. PyCharm 索引占资源
- **现象**：CPU 55%、内存膨胀到 5GB，"什么都没干"。
- **原因**：新增 1.3 万缓存文件 + 拉取新项目（dsharness）触发全量索引；排除目录不中断进行中的索引，且 PyCharm 内存只增不减。
- **解决**：data/logs 目录标记"排除"；重启 PyCharm 释放内存；训练期间尽量不开 PyCharm。

### 17. 其他记录
- Lightning 的几类黄字警告（val_dataloader 建议 31 workers、batch_size 推断歧义）均为机械误报，可无视；
- 开机内存 7GB 偏高：自启软件过多（Wallpaper Engine/QQ/NVIDIA App/Armoury Crate），可按需禁用自启；
- 16GB 内存机器训练纪律：关浏览器/PyCharm，内存低于 2GB 空闲会触发页面交换导致全面卡顿。

---

## 当前训练状态（截至 2026-09-04 晚）

- 运行目录：`logs/train/ljspeech_min/runs/2026-09-04_18-39-13`
- 配置：batch_size=64、check_val_every_n_epoch=5、每轮存 checkpoint
- 进度：epoch 77+，loss 2.37 → 1.72（持续下降中）
- 每轮耗时：~2.3 分钟（batch 64）

---

# 2026-09-05（第二天：续训至 epoch 140 + 🎉 首次成功合成语音）

## 一、查明训练"暂停"真凶：系统睡眠

- **现象**：训练停在 epoch 71，GPU 1%，所有 python 进程死亡，checkpoints 停在 21:06。
- **诊断**：系统事件日志显示 21:04 笔记本进入睡眠（Connected Standby），唤醒后 CUDA 上下文损坏。
- **解决**：
  ```powershell
  powercfg /change standby-timeout-ac 0
  powercfg /change hibernate-timeout-ac 0
  ```
- **心得**：笔记本挂训练三铁律——禁系统睡眠、插电源、控温度（<80℃，高了会降频偷速度）。

## 二、三个新报错及解决

1. **Hydra 参数解析报错（mismatched input '='）**：checkpoint 文件名里的 `=` 被当成键值对分隔符。解决：`"ckpt_path='完整路径'"`（双引号包单引号）。
2. **torch 2.6 checkpoint 加载失败（UnpicklingError）**：`torch.load` 新默认 `weights_only=True` 拒绝加载含 OmegaConf 的 checkpoint。解决：`matcha/utils/utils.py` 加 `_torch_load_compat` 补丁（None → False，仅恢复旧默认，本地文件可信）。
3. **合成时 CLI 强制下载官方 218MB 模型**：官方 `cli.py` 逻辑 bug（`not hasattr(args,...)` 恒为 False，永远走下载分支）。解决：改为"传了 `--checkpoint_path` 就直接用"。

## 三、续训至 epoch 140

- 运行目录：`logs/train/ljspeech_min/runs/2026-09-05_12-02-02`
- batch_size=64，204 步/轮，~2.3 分钟/轮；loss 1.72 → 持续下降。
- GPU 利用率讨论：合成时 RTF 0.07，训练时 ~50-60%（框架调度地板，非瓶颈）。

## 四、PyCharm 性能连环案（耗时最多的支线）

- **问题链**：训练写文件 → "保存设置"后台任务循环 → 卡顿；索引两个项目（含新拉的 dsharness）→ 堆内存耗尽 → "内存不足"警告。
- **解决**：① data 目录标记排除 ② 堆 2048→4096MB（Help → Change Memory Settings）③ 删除 dsharness 项目 ④ 计划 Invalidate Caches 清掉 6.5GB 臃肿索引库 ⑤ 训练期间关 PyCharm。
- **看法**：PyCharm 的 JVM 架构在"16GB 内存 + 训练负载"场景天然吃力；日常工作流可切换 VS Code（Electron，占用 ~1GB，无堆概念）。

## 五、磁盘占用分析

- 项目共 **7.49GB**：wavs 3.54 + 缓存 2.22 + checkpoints 1.67 + 代码 ~10MB——大头全是训练资产。
- **纠偏**：任务管理器里 SSD"活动时间 100%"是误导指标（NVMe 有请求即 100%），真实压力看吞吐（当时才 70MB/s，上限 3GB/s+）。磁盘 62MB/s 读取 = Defender 异步扫描新写的 checkpoint。
- **虚惊一场**：剪切 cache 文件夹到项目外导致训练找不到缓存，已从 `D:\PycharmProjects\cache` 移回原位，13100 个文件无损。教训：同盘移动不省空间，缓存是训练必需品。

## 六、🎉 里程碑：首次成功合成语音

**排障链**：合成报错 → 发现 CLI 强制下载官方模型（修 bug）→ 声码器需手动下载 → 直连 GitHub 被重置 → 开梯子走本地代理（127.0.0.1:7897）→ `generator_v1` 下载成功（53.2MB，验证 PyTorch 魔数 `80 02 8A 0A...`）→ 改名 `hifigan_T2_v1` 放入 `%LOCALAPPDATA%\matcha_tts\`。

**结果**：

```
[🍵-1] Matcha-TTS RTF: 0.0705
[🍵-1] Matcha-TTS + VOCODER RTF: 0.0796    ← 比实时快 12.5 倍
[+] Waveform saved: ...test_audio\utterance_001.wav
```

140 轮模型清晰合成完整句子，音色像 LJ 女声。

## 七、与官方模型对比 + 领域认知

- **自己的模型**：感情更平淡、"trained"一词不清晰——恰好对应 TTS 两大核心难题：韵律表现力 + 发音鲁棒性，均为训练量问题，继续训会改善。
- **评价**：18M 小模型、batch 64、~140 轮（约 2.9 万步）就有模有样，验证了架构与数据管线的正确性。
- **领域焦点**：零样本声音克隆、情感/风格控制（瓶颈在数据）、LLM+codec 架构、长文本稳定性、流式低延迟；普通单句朗读已趋同（MOS 4.5+）。
- **声码器认知**：解耦训练是工业主流；换声码器的硬约束是 mel 参数对齐（采样率/hop/mel 维数），软约束是写适配器并收敛改动到 load/调用两点。

## 今日心得

1. 训练稳定性三板斧：每轮 checkpoint、禁睡眠、内存纪律（PyCharm 和训练错峰）。
2. GPU 利用率 = 计算量 ÷（计算量 + 等待时间）；小模型的框架调度开销是刚性地板，别硬抠。
3. 速算：1M 参数 ≈ 16MB 训练显存（权重+梯度+Adam）；激活值随 batch×序列长度涨，才是大头。
4. RTF（实时率）= 合成耗时 ÷ 音频时长，<1 即快于实时，是衡量合成速度的核心指标。
5. 遇事先量化（进程/内存/IO 采样）再下结论——今天多次避免了误判（磁盘、CPU、内存三案）。

## 八、🔧 框架改造实验：接入 BigVGAN 声码器（免重训）

### 动机
- 目标不是复现基线，而是**改框架**：在不动声学模型（不重训）的前提下替换声码器。
- 选型过程：对照 BigVGAN 官方型号表逐参数核对，**fmax 一致性是关键**——
  - `bigvgan_v2_22khz_80band_256x`：fmax=11025 ❌（与配置的 8000 不符，mel 滤波器组不同）
  - `bigvgan_v2_22khz_80band_fmax8k_256x`：fmax=8000 ✅ 但 112M
  - **最终选用 `nvidia/bigvgan_base_22khz_80band`**：22kHz/80band/fmax8000/hop256 全匹配，14M（与 HiFi-GAN T2 的 14.2M 同量级，对比更公平），训练集含 LJSpeech。

### 实现（声码器适配层，最小侵入）
- 新增 `matcha/vocoders/`（适配层）+ `matcha/vocoders/bigvgan.py`（BigVGAN 适配器）；
- `matcha/cli.py` 四处小改：常量 `BIGVGAN_VOCODER_NAME`、`load_vocoder` 分支（无 Denoiser）、`--vocoder` choices、校验豁免；
- HiFi-GAN 路径原样保留，随时可对照/回退。

### 排障记录（4 连坑）
1. bigvgan 包的 `from_pretrained` 依赖旧版 huggingface_hub 的 `PyTorchModelHubMixin._from_pretrained(proxies, resume_download)` 签名，hub 1.x 不再传参 → **改为手动加载**：config.json → AttrDict → 构建 → 加载 state_dict；
2. `snapshot_download` 卡在 140MB 的 `bigvgan_discriminator_optimizer.pt`（训练残留，推理无用）→ `allow_patterns` 只取 `config.json + bigvgan_generator.pt`，并加本地缓存离线回退；
3. 权重文件格式是 `{"generator": state_dict}`（与 HiFi-GAN 习惯一致，不是裸 state_dict）→ 解包；
4. GBK 控制台打不出 🍵 emoji → `PYTHONUTF8=1`。

### 实测结果
- 冒烟测试：mel [1,80,200] → 波形 [1,1,51200]（200 帧 × hop 256 = 2.32s @ 22050Hz）✅
- 端到端：全链路 RTF **0.235**（HiFi-GAN 为 0.128，BigVGAN 略慢但仍快 4 倍实时）✅

### 👂 A/B 对听结论（主观，两轮实验）

**第一轮（不公平对照）**：HiFi-GAN 有去噪器 vs BigVGAN 裸奔 → 听感"BigVGAN 电音稍重"。

**第二轮（修正实验，给 BigVGAN 也挂上 Denoiser）**：去噪后电音**确实减轻，但未完全消除**。

**最终结论（两个因素各占部分）**：
1. ~~去噪器不对称~~（已排除大半）：确有贡献，属实验设计缺陷，已修复；
2. **架构呈现差异（残余因素）**：Snake 周期偏置 + 抗混叠设计对 mel 瑕疵"如实还原"，而 HiFi-GAN 的卷积堆叠天然带平滑美颜；叠加 BigVGAN 混合训练中 LJSpeech 仅占 ~19% 的分布折中。

**决策**：日常合成继续用 **HiFi-GAN T2**；BigVGAN 适配层保留（实验基座 + 备选）。
**后续可选**：试 112M 的 `fmax8k` 版（容量换电音）、或将来用 LJSpeech 微调 BigVGAN。

### 心得
- 换声码器的硬约束是 **mel 参数对齐（采样率/hop/mel 维数/fmax）**，本次 fmax 差异就是靠官方型号表人工核对出来的；
- "同参数量 ≠ 同效果"：专用小声码器在匹配数据上可以打赢通用声码器；
- 适配层模式的价值：新声码器 = 新增一个文件 + cli 两行分支，主框架不脏；
- **对照实验要检查隐藏混淆变量**——本次"电音"结论的一半其实是去噪器不对称造成的，用户用听感质疑、修正实验后得到干净结论。第一轮的结论差点被当成事实写进日志。

## 常用命令速查

```powershell
# 断点续训（当前进度 epoch 140+，新终端先设环境变量）
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
python matcha/train.py experiment=ljspeech_min_memory data.batch_size=64 trainer.check_val_every_n_epoch=5 ckpt_path=logs\train\ljspeech_min\runs\2026-09-05_12-02-02\checkpoints\last.ckpt

# 试听合成（已验证可用）
matcha-tts --text "任意英文文本" --checkpoint_path logs\train\ljspeech_min\runs\2026-09-05_12-02-02\checkpoints\last.ckpt --vocoder hifigan_T2_v1 --output_folder test_audio

# 试听合成（BigVGAN 适配版，含去噪器的公平对照版，见第八节两轮实验）
matcha-tts --text "任意英文文本" --checkpoint_path logs\train\ljspeech_min\runs\2026-09-05_12-02-02\checkpoints\last.ckpt --vocoder bigvgan_base_22khz_80band --output_folder test_audio\bigvgan_denoised

# 与官方模型对比（官方 ckpt 放在 runs 目录）
matcha-tts --text "同一句话" --checkpoint_path logs\train\ljspeech_min\runs\matcha_ljspeech.ckpt --vocoder hifigan_T2_v1 --output_folder test_audio\official

# TensorBoard 看曲线
tensorboard --logdir logs

# 重新生成数据缓存（换数据集时）
python scripts/preprocess_dataset.py --data_config configs/data/ljspeech.yaml --workers 8

# 重新划分文件列表
python scripts/prepare_ljspeech_filelists.py
```

---

## 2026-09-06

当日完成五件事：声码器注册表重构、模型框架逐段中文注释、评价指标体系（MCD/WER/CER + 批量评估脚本 + tensorboard val/MCD）、静态演示页（demo/）、ConvNeXt V2 局部算子替换（代码就绪待训练）。

140ep 基线客观成绩（val 100 条）：WER 9.5%（ASR 上限 7.6%）/ CER 2.8% / MCD(DTW) 52.92±17.20 / 声码器往返下限 3.65。

⚠ 已有权重未被触碰（ConvNeXt 走开关默认关闭；训练自动建新时间戳目录）。

**详细记录见 [`2026-09-06_summary.md`](2026-09-06_summary.md)。**

---

## 2026-09-07

LR 衰减干预实验日（衔接 09-06 的震荡发现）：

1. **首次衰减重训静默失效**：里程碑按本 run 局部 epoch 写（[5,15,25,35]），但仓库恢复 ckpt 时把 scheduler 计数器拨到恢复 epoch-1（99），里程碑永远够不到 → 全程恒定 1e-4。教训：**MultiStepLR 恢复训练时里程碑必须写在全局 epoch 轴上**；训练中途抽查 ckpt 内 lr 验证调度生效。
2. **修复后重训成功**（13-00-43 + 16-11-41 两个会话拼接，lr 1e-4 → 6.25e-6 阶梯衰减）：衰减轨迹 ep120 后平稳于 51~52.6dB，同 epoch 配对全面优于恒定 LR（ep135 Δ−3.37、ep140 Δ−3.78，p<0.0001），**剂量-反应关系成立** → "恒定 lr 扰动 → MCD 震荡"因果确认。
3. **新最佳系统**：ConvNeXt V2 + LR 衰减（ep135）WER 10.1% / MCD 51.02 —— MCD 显著优于基线（p=0.0024），WER 打平；骨干替换正式获得正面结论。
4. Demo 更新：试听表扩到 6 系统（新增 ConvNeXt ep140 两组）、指标表加衰减行、专题章节加第三条曲线（衰减轨迹）。

**详细记录见 [`2026-09-06_summary.md`](2026-09-06_summary.md)（LR 衰减干预实验 + 震荡研究章节）。**
下一步：机制归因（为什么恒定 LR 会伤细节）——分析脚手架已备好 `scripts/analyze_drift.py`，路线图见 summary 文档"机制分析路线图"。

---

## 2026-09-08

Mamba 骨干迁移日（整体搬迁到 WSL2 Linux + 双向 Mamba2 全局混合器落地 + 完整评估）：

1. **训练环境整体迁移至 WSL2 Ubuntu**：数据（5.9GB LJSpeech + mel 缓存）拷入 ext4；matcha-tts editable 安装；espeak-ng / wav2vec2 / BigVGAN / HiFi-GAN 权重缓存软链复用 Windows 侧，零重下。
2. **mamba-ssm 2.2.5 + causal-conv1d 1.7.0 安装成功**（官方预编译 wheel，torch2.8+cu128+cxx11abiTRUE+cp311 完全匹配）。教训两条：① PyPI sdist 缺 csrc 源文件（打包 bug），源码编译还撞上 nvcc 12.8 与 Ubuntu 25.10 glibc 2.42 的 `cospi/sinpi` 头冲突——**直接用官方 GitHub releases 的对应矩阵 wheel 绕开全部编译问题**；② mamba-ssm 2.3.x 的 metadata 会强升 triton≥3.5/tilelang/quack，威胁 torch 2.8 栈，勿碰。
3. **MambaBlock1D 集成**：U-Net 每级 down/mid/up 的 Transformer 子块 → 双向 Mamba2（正/反序独立实例求和）+ LayerScale(1e-6) 零初始化，签名与 BasicTransformerBlock 对齐；`decoder.py` 工厂加 `"mamba"` 开关（惰性导入，Windows 兼容）；`initialize_weights` **跳过 Mamba 子树**（保护 A_log/D/dt/conv 特殊初始化）。15 项验证全过（含 hydra 组合 + CFM 真实路径 loss+backward）。
4. **训练**：从头 140ep（32×2 有效 64 + multistep_late，与 09-07 修复配置一致）。mamba 版显存比 transformer 基线低 ~0.6GB（O(T) vs O(T²) + 分配器行为差异）。
5. **统一评估协议定版：HiFi-GAN T2 + Denoiser（原论文配方），val 100 句**。成绩：Mamba ep139 **WER 8.7%（历史最佳，ASR 上限 7.6%）/ CER 2.3% / MCD 51.70±17.36**。逐句配对 Wilcoxon：vs U-Net ep140（恒定LR）MCD −1.16dB **p=0.0015 显著**；vs 衰减 ConvNeXt ep134（同协议 HiFi+Deno 重评）**MCD 打平（+0.82, p=0.23），WER −1.7pt 显著更优（p=0.023）**。

**详细记录见 [`2026-09-08_summary.md`](2026-09-08_summary.md)。**
下一步：mamba 恒定 LR 对照 run（把调度因素彻底剥离）；analyze_drift.py 查 mamba 块 LayerScale gamma 漂移；demo 页加 mamba 试听列。
