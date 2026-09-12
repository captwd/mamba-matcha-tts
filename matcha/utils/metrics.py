# -*- coding: utf-8 -*-
"""
评价指标库（Evaluation Metrics）
=================================
TTS 复现/消融实验常用的客观指标，纯 numpy 实现，不引入新依赖：

1. MCD（Mel Cepstral Distortion，梅尔倒谱失真）
   - 合成 mel vs 真实 mel 的倒谱距离，单位 dB，越低越好
   - 支持 DTW（动态时间规整）对齐帧：合成语音与真实语音时长/语速不同时更公平
     （TTS 论文里常用 DTW-MCD）；也支持 min_len 直接截齐
   - 参考公式（Kubichek 1993 / 语音转换文献通用）：
         MCD = (10 * sqrt(2) / ln10) * (1/T) * Σ_t || c_t^(gen) - c_t^(ref) ||_2
     其中 c_t 为逐帧 mel-cepstral 系数（对 mel 带能量做 DCT-II），默认去掉 c0（能量项），
     因为能量差异主要反映录音增益而非音色/清晰度。

   【中文说明】★ 关于绝对数值的标定（重要）★
   本实现的提取器是"log-mel(80) + DCT-II"，与 SPTK/merlin 的 mcep 提取器数值标定不同，
   绝对值普遍比论文里 3~8dB 的 MCD 大得多，**只应在同一提取器下做相对比较**。
   本仓库实测标定参考（LJSpeech + BigVGAN，22050/80/256 配置）：
       声码器往返下限（GT mel -> BigVGAN -> 重提 mel）   ≈ 3.6 dB
       同文本合成 vs GT（早期模型，可懂但偏糊）          ≈ 40~90 dB
       跨语句自然语音 vs 自然语音（内容不匹配基线）      ≈ 55~90 dB
   解读建议：
       1) 看趋势不看绝对值：同一评估设置下，MCD 下降 = 细节保真度提升
       2) 合成 vs GT 应低于"跨语句基线"才说明模型细节质量过关
          （过早期的模型可能暂时高于基线——输出"可懂但偏糊"，属正常现象）
       3) 可用 scripts/evaluate.py --calibrate 获得当前环境下的声码器往返下限

2. WER / CER（词错误率 / 字符错误率）
   - 用 ASR 模型转写合成音频，与参考文本对齐，算编辑距离比例，越低越好
   - Levenshtein 编辑距离纯 python DP 实现（文本都很短，性能足够）
   - 归一化：小写 + 去标点 + 压缩空白（英文）——与 ASR 输出粒度对齐

典型用法见 scripts/evaluate.py（批量评估）和 baselightningmodule（tensorboard 里的 val/MCD）。
"""
import numpy as np
import re
import string
from scipy.fftpack import dct

# MCD 公式里的尺度系数：10*sqrt(2)/ln(10) ≈ 12.2326（把倒谱欧氏距离换算成 dB 量纲）
_MCD_SCALE = 10.0 * np.sqrt(2.0) / np.log(10.0)


def _dtw_align(cost):
    """【中文说明】经典 DTW（动态时间规整）。

    Args:
        cost (np.ndarray): (T1, T2) 的逐帧代价矩阵（如两串倒谱向量的欧氏距离平方）

    Returns:
        list[(i, j)]: 从 (0, 0) 到 (T1-1, T2-1) 的最优规整路径（允许 1步/斜步/停顿）
    """
    t1, t2 = cost.shape
    inf = np.inf
    dp = np.full((t1 + 1, t2 + 1), inf, dtype=np.float64)
    dp[0, 0] = 0.0
    # 【中文说明】标准递推：dp[i,j] = cost[i-1,j-1] + min(上, 左, 左上)
    #   用向量化写法逐行填表：T~千帧时纯 numpy 也只要几十毫秒
    for i in range(1, t1 + 1):
        row = cost[i - 1]
        dp[i, 1:] = row + np.minimum(np.minimum(dp[i - 1, 1:], dp[i - 1, :-1]), dp[i, :-1])

    # 【中文说明】从终点回溯路径
    path = []
    i, j = t1, t2
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = (
            (dp[i - 1, j - 1], (i - 1, j - 1)),  # 斜步（两帧配对）
            (dp[i - 1, j], (i - 1, j)),          # 垂直（gen 停顿/多帧）
            (dp[i, j - 1], (i, j - 1)),          # 水平（ref 停顿/多帧）
        )
        _, (i, j) = min(candidates)
    path.reverse()
    return path


def mel_cepstral_distortion(
    gen_mel,
    ref_mel,
    n_mfcc=13,
    skip_c0=True,
    align="dtw",
):
    """【中文说明】计算一条 utterance 的 MCD。

    Args:
        gen_mel (np.ndarray | torch.Tensor): 合成梅尔 (n_mels, T_gen)，
            对数域（ln 压缩后的，即 matcha.utils.audio.mel_spectrogram 的输出，或模型 denormalize 后的输出）
        ref_mel (np.ndarray | torch.Tensor): 真实梅尔 (n_mels, T_ref)，同样对数域、同样的 mel 提取参数
        n_mfcc (int): 取前 n_mfcc 个 DCT 系数（默认 13，论文常用值）
        skip_c0 (bool): True 时去掉第 0 个系数（能量项），聚焦音色/清晰度
        align (str): "dtw" 帧对齐（推荐，容忍语速差异）或 "min_len" 截齐到较短长度

    Returns:
        float: MCD（dB）。绝对量级取决于提取器（见模块头部的标定说明），
               只用于同一提取器下的相对比较；越低越好
    """
    if hasattr(gen_mel, "detach"):  # torch.Tensor -> numpy
        gen_mel = gen_mel.detach().cpu().numpy()
    if hasattr(ref_mel, "detach"):
        ref_mel = ref_mel.detach().cpu().numpy()

    gen_mel = np.asarray(gen_mel, dtype=np.float64)
    ref_mel = np.asarray(ref_mel, dtype=np.float64)
    assert gen_mel.ndim == 2 and ref_mel.ndim == 2, "mel 形状应为 (n_mels, T)"

    # ---------- Step 1: mel -> mel-cepstrum（DCT-II 沿 mel 带方向，保持正交归一） ----------
    gen_mc = dct(gen_mel, type=2, axis=0, norm="ortho")[:n_mfcc, :]
    ref_mc = dct(ref_mel, type=2, axis=0, norm="ortho")[:n_mfcc, :]
    if skip_c0:
        gen_mc = gen_mc[1:, :]
        ref_mc = ref_mc[1:, :]

    # ---------- Step 2: 帧对齐 ----------
    if align == "dtw":
        # 【中文说明】帧间欧氏距离平方作为 DTW 代价；路径给出 gen 帧 -> ref 帧 的多对一映射。
        #   用 ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b> 展开，只需 (T1, T2) 的代价矩阵，省内存
        g, r = gen_mc.T, ref_mc.T  # (T, D)
        sq = (g**2).sum(-1)[:, None] + (r**2).sum(-1)[None, :] - 2.0 * (g @ r.T)
        np.maximum(sq, 0.0, out=sq)  # 【中文说明】浮点误差可能出小负数，裁掉
        path = _dtw_align(sq)
        diffs = np.array([np.linalg.norm(gen_mc[:, i] - ref_mc[:, j]) for i, j in path])
    elif align == "min_len":
        # 【中文说明】简单做法：两串都截到较短的长度，逐帧硬对齐（时长差大时会引入偏置）
        t = min(gen_mc.shape[1], ref_mc.shape[1])
        diffs = np.linalg.norm(gen_mc[:, :t] - ref_mc[:, :t], axis=0)
    else:
        raise ValueError(f"Unknown align: {align} (use 'dtw' or 'min_len')")

    # ---------- Step 3: 汇总 ----------
    return float(_MCD_SCALE * diffs.mean())


# ------------------------- WER / CER -------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text_for_eval(text):
    """【中文说明】评估用文本归一化：小写、去标点、压缩空白（与 ASR 转写粒度对齐）"""
    text = text.lower().translate(_PUNCT_TABLE)
    return " ".join(text.split())


def _levenshtein(a, b):
    """【中文说明】Levenshtein 编辑距离（插入/删除/替换各计 1），经典 DP，O(len(a)*len(b))"""
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            # 【中文说明】替换 / 删除 / 插入 三种操作取最小
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(reference, hypothesis):
    """【中文说明】词错误率 = 编辑距离 / 参考词数。两边先做 normalize_text_for_eval"""
    ref_tokens = normalize_text_for_eval(reference).split()
    hyp_tokens = normalize_text_for_eval(hypothesis).split()
    if len(ref_tokens) == 0:
        return float("nan") if len(hyp_tokens) else 0.0
    return _levenshtein(ref_tokens, hyp_tokens) / len(ref_tokens)


def cer(reference, hypothesis):
    """【中文说明】字符错误率 = 编辑距离 / 参考字符数（按空格剔除后逐字符比较）"""
    ref_chars = normalize_text_for_eval(reference).replace(" ", "")
    hyp_chars = normalize_text_for_eval(hypothesis).replace(" ", "")
    if len(ref_chars) == 0:
        return float("nan") if len(hyp_chars) else 0.0
    return _levenshtein(ref_chars, hyp_chars) / len(ref_chars)
