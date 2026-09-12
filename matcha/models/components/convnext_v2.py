# -*- coding: utf-8 -*-
"""
ConvNeXt V2 1D 组件（供 Decoder 的局部算子替换用）
====================================================
【中文说明】设计定位（重要，避免概念混淆）：
ConvNeXt V2 不是"卷积的替代品"，而是一种"特征提取单元"的设计方案：
    深度卷积 k=7 → LayerNorm → 1×1升维(×4) → GELU → GRN → 1×1降维
本文件把这套设计移植到 1D 时序（mel 序列），作为 Decoder 中
Block1D(k3 密集卷积+GN+Mish) 的替代品。**只替换局部算子**，
U-Net 骨架（down/mid/up、skip、掩码纪律）、Transformer 注意力块、
时间条件注入路径全部保持不变——控制变量只留一个。

文件内结构：
    GlobalResponseNorm1D    GRN（ConvNeXt V2 的标志组件，5 行核心公式）
    ConvNeXtV2Unit1D        特征提取单元（替代 Block1D 的位置）
    ConvNeXtV2ResBlock1D    残差时间块（替代 ResnetBlock1D，外壳与原版逐行对齐）

参考：Woo et al., "ConvNeXt V2: Co-designing and Co-training with
      Masked Autoencoders" (CVPR 2023)。FCMAE 掩码预训练不适用于本任务，
      这里只复用其块设计（深度卷积 + GRN）。
"""
import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import


class GlobalResponseNorm1D(nn.Module):
    """【中文说明】全局响应归一化（GRN，ConvNeXt V2 的核心增量）。

    公式（逐通道）：
        x_hat = x / ||x||_{2,序列维}
        y = gamma * x_hat + beta
    作用：让每个通道对"整条序列的全局能量"敏感，补偿深度卷积缺乏的
    跨通道/跨位置竞争机制（V1→V2 论文的核心动机）。
    gamma/beta 零初始化：训练开始时 y = 0，单元从"接近恒等"慢慢学起。
    输入约定为 channels-last (B, T, C)。
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        gx = x.norm(p=2, dim=1, keepdim=True)
        return self.beta + self.gamma * x / (gx + self.eps)


class ConvNeXtV2Unit1D(nn.Module):
    """【中文说明】ConvNeXt V2 特征提取单元（1D 版），Block1D 的替代品。

    结构（与官方 ConvNeXt V2 块对齐，**内部残差 + LayerScale 必须保留**）：
        xin = stem(x*mask)                      # 仅 dim≠dim_out 时是 1×1 投影，否则恒等
        h   = LN( dwconv(xin) ) → 1×1升维 → GELU → GRN → 1×1降维
        out = (xin + layer_scale * h) * mask    # ★ 内部残差：GRN 零初始化时此路是"活"的

    ★ 为什么必须带内部残差：GRN 的 gamma/beta 零初始化会让 GRN 分支初始输出恒为 0；
      若单元没有自己的残差通路（早期实现踩过的坑），整个单元初始时变成
      "输入无关的常数输出"，夹在两级之间的时间条件注入会被彻底抹掉。
    掩码纪律与原 Block1D 相同：进卷积前乘 mask、出单元后乘 mask。
    forward 签名与 Block1D 一致：forward(x, mask)，可直接互换。
    """

    def __init__(self, dim, dim_out, mlp_ratio=4, layer_scale_init=1e-6):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        # 【中文说明】通道数变化用 1×1 卷积承担（深度卷积本身不改通道数），同时充当内部残差投影
        self.stem = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.dwconv = nn.Conv1d(dim_out, dim_out, 7, padding=3, groups=dim_out)  # 深度卷积
        self.norm = nn.LayerNorm(dim_out)          # channels-last 上的 LayerNorm
        hidden = int(dim_out * mlp_ratio)
        self.pw1 = nn.Conv1d(dim_out, hidden, 1)   # 1×1 升维（等价 Linear）
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm1D(hidden)    # ★ V2 标志
        self.pw2 = nn.Conv1d(hidden, dim_out, 1)   # 1×1 降维
        # 【中文说明】LayerScale（官方 ConvNeXt 设计，1e-6 起步）：残差分支"温和接入"
        self.layer_scale = nn.Parameter(layer_scale_init * torch.ones(1, dim_out, 1))

    def forward(self, x, mask):
        """【中文说明】x: (B, dim, T), mask: (B, 1, T) -> (B, dim_out, T)"""
        xin = self.stem(x * mask)
        h = self.dwconv(xin) * mask
        h = self.norm(h.transpose(1, 2))
        h = self.pw1(h.transpose(1, 2))  # pw 用 1×1 卷积，直接在 (B,C,T) 上做
        h = self.act(h)
        h = self.grn(h.transpose(1, 2)).transpose(1, 2)
        h = self.pw2(h)
        return (xin + self.layer_scale * h) * mask


class ConvNeXtV2ResBlock1D(nn.Module):
    """【中文说明】ConvNeXt V2 版残差时间块——ResnetBlock1D 的逐行对齐替代品。

    与原版 ResnetBlock1D 的对应关系（外壳一致，只换内部算子）：
        原：Block1D(Conv3+GN+Mish) → +time_emb → Block1D(Conv3+GN+Mish) → +1×1残差
        新：ConvNeXtV2Unit1D       → +time_emb → ConvNeXtV2Unit1D       → +1×1残差
    注：ConvNeXtV2Unit1D 内部还带自己的残差+LayerScale（官方设计，保证 GRN 零初始化时
    信息通路不被堵死），因此整体残差比原版多一层——这是有意的稳定性设计。
    时间条件注入点（mlp(time_emb) 广播相加）与残差捷径（res_conv）原样保留——
    这正是"只换 CNN、不改框架"的落点。
    构造签名与 forward 签名和 ResnetBlock1D 完全相同，可在 Decoder 里一键互换。
    """

    def __init__(self, dim, dim_out, time_emb_dim, groups=8):  # groups 仅为签名兼容，未用
        super().__init__()
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = ConvNeXtV2Unit1D(dim, dim_out)
        self.block2 = ConvNeXtV2Unit1D(dim_out, dim_out)

        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        """【中文说明】x: (B, dim, T), mask: (B, 1, T), time_emb: (B, time_emb_dim)"""
        h = self.block1(x, mask)
        # 【中文说明】时间条件注入：与原 ResnetBlock1D 相同的位置、相同的方式
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output
