# 代码改动清单

> 相对官方原版 Matcha-TTS 的所有本地改动。按文件组织，注明改动内容与原因。

## 一、为"免编译安装"打的补丁（机器无 MSVC 编译器）

| 文件 | 改动 | 原因 |
|---|---|---|
| `matcha/utils/monotonic_align/__init__.py` | 尝试导入 Cython 编译的 `maximum_path_c`；失败则使用文件内新增的纯 Numpy 实现 `_maximum_path_each` / `_maximum_path_c_fallback`（逐行复刻 core.pyx 逻辑） | 没装 VS Build Tools 时装不上包；回退实现经 200 组随机用例验证与 Cython 版逐位一致 |
| `setup.py` | 新增 `TolerantBuildExt`（继承 build_ext，编译失败打印中文警告并跳过），`cmdclass={"build_ext": TolerantBuildExt}` | 缺编译器时 `pip install -e .` 不再中断 |

> 备注：若日后安装了 Microsoft C++ Build Tools，Cython 版会自动编译并被优先使用，无需改代码。

## 二、torch / matplotlib 版本兼容补丁

| 文件 | 改动 | 原因 |
|---|---|---|
| `matcha/utils/utils.py` | 模块级 `_torch_load_compat` 包装 `torch.load`：`weights_only=None` → `False` | torch 2.6+ 默认安全模式拒绝加载含 OmegaConf 对象的 checkpoint |
| `matcha/utils/utils.py` | `save_figure_to_numpy()` 改用 `fig.canvas.buffer_rgba()`；顶部加 `matplotlib.use("Agg")` | matplotlib 3.10 移除了 `tostring_rgb()`；Agg 后端避免训练时创建 Tk 窗口 |

## 三、性能优化

| 文件 | 改动 | 原因 |
|---|---|---|
| `matcha/train.py` | 顶部加 `torch.set_float32_matmul_precision("high")` | 开启 TF32，用 RTX Tensor Core 加速 fp32 矩阵运算 |
| `matcha/data/text_mel_datamodule.py` | train_dataloader：缓存生效时 `num_workers = min(原值, 2)`；val_dataloader：缓存生效时 `num_workers = 0`；均带 `persistent_workers` + `prefetch_factor=3` | Windows 上每轮重启 worker 开销大；worker 进程每个常驻 ~780MB，验证集仅 100 条无需 worker；缓解 16GB 内存压力 |

## 四、离线预处理缓存（新功能）

| 文件 | 类型 | 说明 |
|---|---|---|
| `scripts/preprocess_dataset.py` | 新增 | 离线预处理：梅尔谱存 `data/<数据集>/cache/mel/*.npy`，音素序列存 `cache/phonemes.json`，参数存 `cache/meta.json`；多进程、断点续跑、meta 参数校验 |
| `matcha/data/text_mel_datamodule.py` | 修改 | `TextMelDataset` 新增 `_load_preprocess_cache()`（校验 meta 与当前配置一致才启用）；`get_mel()` 优先读缓存 .npy；`get_datapoint()` 优先读缓存的音素序列；未命中自动回退原计算逻辑 |

正确性验证：6 个抽查样本，缓存 vs 现场计算逐位一致（音素/梅尔/清洗文本全同）。

## 五、训练配置修改

| 文件 | 改动 | 原因 |
|---|---|---|
| `configs/callbacks/model_checkpoint.yaml` | `every_n_epochs: 100 → 1`；`save_top_k: 10 → 3`（含中文注释说明） | 新版 Lightning 中 last.ckpt 只在"本轮存过编号 checkpoint"时写入；每 100 轮存一次导致断点续训全部落空 |
| `configs/data/ljspeech.yaml` | `num_workers: 20 → 4` | Windows 多进程数据加载开销大，20 容易报错 |

## 六、新增工具脚本

| 文件 | 说明 |
|---|---|
| `scripts/prepare_ljspeech_filelists.py` | metadata.csv → train.txt（13000 条）/ val.txt（100 条），固定种子 1234，格式 `wav相对路径\|规范化文本` |
| `scripts/preprocess_dataset.py` | 见第四节 |

## 七、中文注释（不影响逻辑）

以下文件添加了模块级/行内中文注释，仅注释，代码逻辑未动：

- 入口：`matcha/train.py`、`matcha/app.py`、`matcha/cli.py`
- 模型：`matcha/models/matcha_tts.py`、`baselightningmodule.py`、`components/{flow_matching,text_encoder,decoder,transformer}.py`
- 声码器：`matcha/hifigan/{models,denoiser,config,env,meldataset,xutils}.py`
- 导出：`matcha/onnx/{export,infer}.py`
- 文本前端：`matcha/text/{__init__,cleaners,numbers,symbols}.py`
- 数据与工具：`matcha/data/text_mel_datamodule.py`、`matcha/utils/{audio,utils,model,monotonic_align/__init__,instantiators,logging_utils,generate_data_statistics,get_durations_from_trained_model}.py`
- 配置：`configs/train.yaml`、`scripts/schedule.sh`

## 八、未改动但需记住的外部依赖

| 项 | 内容 |
|---|---|
| espeak-ng 1.52 | winget 安装（系统级），环境变量 `PHONEMIZER_ESPEAK_LIBRARY` / `PHONEMIZER_ESPEAK_PATH` 已 setx 持久化 |
| 数据缓存 | `data/LJSpeech-1.1/cache/`（2.2GB，可删可重建） |
| 训练产物 | `logs/train/ljspeech_min/runs/<时间戳>/checkpoints/`（每个 ckpt 约 209MB） |

## 九、Mamba 双向骨干集成（2026-09-08，WSL/Linux-only）

| 文件 | 类型 | 说明 |
|---|---|---|
| `matcha/models/components/mamba_block.py` | 新增 | `MambaBlock1D`：双向 Mamba2（正/反序独立实例求和）+ `LayerScale(1e-6)` 零初始化输出分支；签名与 `BasicTransformerBlock` 对齐（`(B,T,C)` 进出，attention_mask/timestep 收下不用）；`mamba_ssm` 惰性导入保证 Windows 可导入 |
| `matcha/models/components/decoder.py` | 修改 | 工厂函数 `get_block` 新增 `"mamba"` 分支（惰性导入）；`initialize_weights` **跳过 MambaBlock1D 子树**（Mamba2 的 A_log/D/dt_bias 特殊初始化不可被 Kaiming 覆盖，测试已验证）；注释更新 |
| `configs/model/decoder/mamba.yaml` | 新增 | down/mid/up 三段 `block_type: mamba`，与 `resnet_type` 正交；用法注释含整组替换/改键/仅 mid 三种 |

依赖（仅 WSL/Linux）：`mamba-ssm==2.2.5`、`causal-conv1d==1.7.0`（官方预编译 wheel cu12torch2.8cxx11abiTRUE-cp311）、`transformers<5`（runtime 依赖）。安装脚本 `PythonProject9/wsl_setup/install_mamba_wheels.sh`，验证脚本 `test_mamba_block.py` / `test_mamba_gpu.py`。

正确性验证：15 项集成测试全过（单元/初始化保护/整 Decoder 前向反向/默认路径回归/hydra 组合/CFM 真实路径 loss+backward），详见 `docs/2026-09-08_summary.md`。

## 十、Sway Sampling 支持（2026-09-12，推理期）

| 文件 | 类型 | 说明 |
|---|---|---|
| `matcha/models/components/flow_matching.py` | 修改 | `BASECFM.__init__` 读取 `sway_sampling_coef`（缺失默认 `None`）；`forward` 对 `t_span` 做 sway 重参数化 `t + a·(cos(πt/2) − 1 + t)`，端点不变；`None` 时与改动前行为完全一致 |
| `configs/model/cfm/default.yaml` | 修改 | 新增 `sway_sampling_coef: null` |
| `scripts/evaluate.py` | 修改 | 新增 `--sway_sampling_coef`（运行时覆盖 ckpt 配置）与 `--seed`（每条语句固定噪声种子，供逐句配对比较） |
| `wsl_env/sway_grid.sh`、`sway_compare.py`、`sway_multi.sh`、`sway_multi_compare.py`、`inspect_ckpts.py` | 新增 | A/B 网格、配对检验、多模型通用性验证、checkpoint 训练量检查脚本 |

来源：F5-TTS 的 Sway Sampling（arXiv 2410.06885）。属**推理期技巧**，不改变模型参数与训练目标，**不需要重训**；旧 checkpoint 可直接使用（配置缺键 → 默认关闭）。实测结论见 `docs/2026-09-12_summary.md`。
