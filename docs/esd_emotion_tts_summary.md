# ESD 情感可控 Matcha-TTS：项目总结（2026-09-14 ~ 09-17）

> 目标：在 Matcha-TTS（CFM 流匹配）上加 emotion2vec 情感条件，实现情感可控合成。
> 数据：ESD 英文子集（10 说话人 × 5 情感 × 350 句 = 17,500 条）+ RAVDESS（备用）。
> 结果：**MCD 61.41 → 56.30，WER 45.7% → 13.0%**（GT 上限 7.9%），情感可控性待 v4 增强。

**状态声明（先说结论）**：本次实验的目的只是验证**能不能做出带情感的 TTS**——这个任务本身完成了：全链路（emotion2vec 特征 → 条件注入 → 训练 → 合成）跑通，可懂度达标（WER 13.0%）。但**实际听感效果很差**，原因有二：① **架构只是最简单的拼接**——emotion2vec 句级向量直接拼进 decoder 输入通道，情感表达强度先天不足；② **学习率判断失误**——因对该领域的知识储备尚不充分，v1 训练做了学习率衰减，到后期已衰至 6.25e-6，**远小于同类生成式语音模型的常用水平（约 1e-4）**，模型被"冻"进低步长平台期；续训时配置的 3e-5 又一度被断点恢复覆盖而未生效，最终 800 轮收益甚微（MCD 仅 −0.55dB、WER 持平）。后针对学习率做了专项调研（[《300M 量级语音模型学习率速查》](https://captwd.github.io/2026/09/17/300m-speech-models-learning-rate-reference/)：生成式语音模型 LR 集中在 7.5e-5~2e-4，微调 1e-5）。本文如实记录全过程与数据，改进方向见第五、八节。

## 一、训练时间线

| 阶段 | 环境 | 轮数 | LR | 说明 |
|---|---|---|---|---|
| v0 本地验证 | 5060 笔记本 | 127 | 1e-4 恒定 | 跑通全链路（emotion2vec 特征 → 条件注入 → 辅助分类器） |
| v1 服务器长跑 | 4090 | ep130 → **1199** | 1e-4 恒定 600 轮后阶梯衰减至 6.25e-6 | 协议指标单调改善，"过拟合曲线"后被证伪（见第四节） |
| v2 监控修正 | 4090 | ep130 → 263（EarlyStopping） | 1e-4 恒定 | 盯 val_mcd + EarlyStopping；教训：2 句样本 val_mcd 是噪声 |
| **v3 回温续训** | 4090 | ep1199 → **2000** | **3e-5 恒定**（回温重启） | 最终版本；+20.8 万步，MCD −0.55（显著）、WER 持平 |

## 二、最终评估总表（统一协议：875 句 · HiFi-GAN T2 · 10 步 ODE · 情感匹配条件 · seed 1234）

| 指标 | ep126 | ep183 | ep1199 | **ep2000（最终）** | GT 上限 |
|---|---|---|---|---|---|
| MCD ↓ | 61.41 | 60.52 | 56.85 | **56.30** | — |
| WER ↓ | 45.7% | 33.5% | 13.0% | **13.0%** | 7.9% |
| CER ↓ | 26.4% | 18.0% | 4.9% | **4.9%** | — |
| ECA ↓（emotion2vec 探针） | 26.9% | 29.3% | 27.4% | **28.8%** | 68.6%（真实录音） |

ep1199 → ep2000 的配对 Wilcoxon：MCD p=2.8e-06（−0.55dB 显著但幅度小）、WER p=0.944（完全持平）→ **模型已收敛**。

## 三、ep2000 细分拆解

**按说话人（MCD）**：最好 spk11=49.2，最差 spk13=63.1（差 14dB——多说话人方差是主要难度来源）。
**按情感（MCD）**：neutral=49.1 ≈ LJSpeech 系统水平（51）；angry=60.4 / sad=61.5（情感实现方差 +11dB）。
**按说话人（WER）**：最好 spk18=10.2% / spk19=10.7%，最差 spk11=17.0%。

**关键结论**：剥掉情感与多说话人因素后，模型在"中性单风格"子任务上已追平 LJSpeech 140 轮的自己（49.1 vs 51）——高出的部分是情感 TTS 的固有题目难度（MCD 下限 9.94 vs 3.6），不是训练不足。

## 四、重要修正：一条"过拟合曲线"的证伪

v1 训练时的 2 句样本 val_mcd 曲线曾显示"ep349 触底 54.39 后 +7.6dB 过拟合"，据此差点放弃 1200 轮成果。三方协议评估证伪了这个判断：**协议指标从 ep126 到 ep1199 单调改善，1200 轮没有过拟合**。

根因：2 句样本的 MCD 均值，抽样误差 ±10dB 起步（句间标准差 ±13~15），54.4→62 的"趋势"完全在噪声带内。
**教训**：小样本验证指标既不能诊断过拟合，也不能做 checkpoint 监控（v2 的 EarlyStopping 因此提前 800+ 轮停在了次优处）。已改为加大样本量 + 定期协议评估。

## 五、情感平淡问题（未解决，v4 方向）

emotion2vec 探针评估（ECA）显示：合成音频的情感可识别度 ~28%，且**不随训练变化**（ep126~2000 全部 27~29%），远低于真实录音的 68.6%；域内探针（合成音频自训自测）= 20.2%，即随机水平。人耳听感佐证：情感差异微弱。

**诊断**：训练期的辅助分类器（99% 准确）读的是模型自己输出的 mel 微差——闭环自嗨，不等于人耳可闻的情感。

**v4 方向（未启动）**：
1. `emo_aux_loss` 0.2 → 0.5，辅助分类器 3 层 → 5 层；
2. 辅助分类器先用真实 ESD mel 预训练（冻结数轮再联合训练），锚定人耳可闻的情感线索；
3. 每 25 轮自动跑一次 ECA 探针（合成 32 句 → emotion2vec），写进 tensorboard 作为"人耳视角"监控；
4. 数据扩容（全量 ESD 含中文 / RAVDESS）是根本解。

## 六、工程教训（运维）

1. **Lightning 2.6 checkpoint**：`save_top_k=0` + `save_last=true` 不落盘；`last.ckpt` 只在当轮存过编号 ckpt 时写入——续训保护必须 top_k≥1；
2. **checkpoint monitor 别用 epoch**：本次最优轮（ep349/183）都因此没被存下；
3. **hydra 覆盖值含 `=`**（如文件名 `checkpoint_epoch=1199.ckpt`）必须加引号，否则 grammar 报错，且时灵时不灵极具迷惑性；
4. **被 kill 的 pip 会留下混装的包树**：跨重启后 numpy/scipy ABI 崩溃；修复 = 彻底删除重装，且 numpy/scipy 副本放 /root/pylibs（PYTHONPATH 优先）对抗平台开机回滚；
5. **服务器重启后端口会变**，清理命令里 `pkill -f` 的模式会自匹配杀掉自己的 SSH 会话（两次）；
6. **断点续传按大小判断会跳过"新文件更小"的更新**——小配置文件上传前先删远端旧文件。

## 七、文件清单

**最终权重**（`D:\PycharmProjects\PythonProject9\final_checkpoints\`）：
| 文件 | 轮数 | 说明 |
|---|---|---|
| `ep2000_v3_final.ckpt` | 1999 | **推荐使用**（val_mcd 与 1199 持平、步数最多） |
| `ep1199_v1_final.ckpt` | 1199 | 与 ep2000 质量等效，备用 |
| `ep183_v2_best_valmcd.ckpt` | 183 | 历史存档 |
| `superseded/` | — | ep1707 中间态 × 2（可删） |

**评估结果**（`Matcha-TTS/results/`）：
- 协议评估：`eval_ep126_local / eval_ep183_local / eval_ep1199_local / eval_v3_ep2000_local`（各含 results.csv + 875 wav）
- ECA：`eca_gt_ceiling.json / eca_ep126.json / eca_ep183.json / eca_ep1199.json / eca_v3_ep2000.json`
- 曲线：`val_mcd_curve.png/csv`（v1）、`val_mcd_v2.csv`
- 试听：`synth_v3_ep2000_demo / synth_ep1199_demo / synth_ep183_demo`（3 句 × 2 强度 × 多版本对照）

**工具脚本**（`Matcha-TTS/scripts/`，已入库）：`evaluate.py`（协议评估，支持情感模型）、`synthesize_emo.py`（情感合成）、`eval_eca.py`（ECA 探针）、`indomain_eca.py`（域内判决实验）、`extract_emotion2vec_funasr.py`（特征抽取）、`build_emo_labels.py`、`prepare_esd.py / prepare_ravdess.py`。

项目根目录（`D:\PycharmProjects\PythonProject9\`）的一次性诊断/修复脚本（`diag*_remote.sh`、`fix*_remote.sh`、`server_ssh.py`、`traj*_remote.sh`、`verify_remote.sh`、`patch_eca.py`、`remote_v3b_check.sh` 等）为会话工具，可留可删。

## 八、下一步（按优先级）

1. **v4 情感增强训练**（第五节方案，~11 小时 GPU）；
2. **ECA 纳入常规评估**（对标文献 94%）+ 小规模 MOS 听测（对标 nMOS 3.88）；
3. **数据扩容**：全量 ESD（+中文 10 说话人，注意中文 cleaner）或 RAVDESS 混训；
4. **博客**：《给 Matcha-TTS 加情感》已按协议评估修正，确认后 `hexo deploy`。
