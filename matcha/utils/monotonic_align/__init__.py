# -*- coding: utf-8 -*-
"""
单调对齐搜索（Monotonic Alignment Search，MAS，源自 Glow-TTS）
================================================================
训练时根据注意力概率寻找"音素-音频帧"的最优单调对齐路径，从而把音素特征
扩展到帧级别再交给 CFM 解码器。

- 优先使用 Cython 编译的加速版本（core.pyx，安装时由 setup.py 编译）
- 若机器没有 C++ 编译器导致编译失败，自动回退到本文件中的纯 Python/Numpy 实现
  （算法逻辑与 Cython 版逐行一致，仅速度稍慢，对合成结果无任何影响）
"""
import numpy as np
import torch

try:
    # 编译过的 Cython 加速版本（首选）
    from matcha.utils.monotonic_align.core import maximum_path_c
except (ImportError, ModuleNotFoundError):
    maximum_path_c = None  # 未编译成功时为 None，运行时使用下面的回退实现


def _maximum_path_each(path, value, t_x, t_y, max_neg_val=-1e9):
    """单个样本的动态规划，逐行复刻 core.pyx 中 maximum_path_each 的逻辑。

    path:  int32  二维数组 [t_x, t_y]，输出 0/1 对齐路径
    value: float32 二维数组 [t_x, t_y]，输入累计收益（会被原地修改）
    t_x:   有效音素数；t_y: 有效帧数
    """
    # 前向 DP：value[x, y] 变为"走到 (x, y) 的最大累计收益"
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[x, y - 1]
            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg_val
            else:
                v_prev = value[x - 1, y - 1]
            value[x, y] = max(v_cur, v_prev) + value[x, y]

    # 反向回溯：从右下角沿最优路径标记 1
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        # 注意 y == 0 时 Cython 版会读越界内存，但其结果不影响最终路径，这里安全跳过
        if index != 0 and y > 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index -= 1


def _maximum_path_c_fallback(paths, values, t_xs, t_ys):
    """纯 Numpy 批量版本，与 Cython 版 maximum_path_c 接口一致。"""
    for b in range(values.shape[0]):
        _maximum_path_each(paths[b], values[b], int(t_xs[b]), int(t_ys[b]))


def maximum_path(value, mask):
    """Cython optimised version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    value = value * mask
    device = value.device
    dtype = value.dtype
    value = value.data.cpu().numpy().astype(np.float32)
    path = np.zeros_like(value).astype(np.int32)
    mask = mask.data.cpu().numpy()

    t_x_max = mask.sum(1)[:, 0].astype(np.int32)
    t_y_max = mask.sum(2)[:, 0].astype(np.int32)

    if maximum_path_c is not None:
        maximum_path_c(path, value, t_x_max, t_y_max)
    else:
        _maximum_path_c_fallback(path, value, t_x_max, t_y_max)

    return torch.from_numpy(path).to(device=device, dtype=dtype)
