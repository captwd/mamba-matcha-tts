"""【中文说明】Mamba 全局建模子块（BasicTransformerBlock 的对位替换）。

定位：U-Net 每级 down/mid/up 里的"全局混合器"，与 BasicTransformerBlock（自注意力）/
ConformerWrapper 同级互换，签名完全对齐：
    forward(hidden_states (B,T,C), attention_mask (B,T), timestep) -> (B,T,C)

设计要点（对应 docs/2026-09-06_summary.md 的工程教训）：
1. 双向化：Mamba2 是因果单向序列模型，mel 生成需要全局上下文 ⇒
   正/反序两个独立实例，各自扫描后按原位求和（Vim 式双向，不共享权重）。
2. LayerScale 零初始化：输出分支乘 gamma=1e-6，训练初期块行为≈恒等映射，
   避免随机初始化的全局混合器在残差主干上注入大噪声（GRN/LayerScale 教训的复用）。
3. mask 纪律：与现状对齐——BasicTransformerBlock 本身不消费 attention_mask，
   padding 位置的影响由 decoder 末端 masking 兜底；本块同样不在内部处理 mask。
4. 时间条件：timestep 收下不用（与普通 BasicTransformerBlock 分支一致，
   时间信息已由每级 ResnetBlock1D 注入）。

依赖：mamba-ssm（CUDA-only，仅 Linux/WSL；Windows 无官方 wheel）。
为不破坏 Windows 侧的导入链，mamba_ssm 在 __init__ 内惰性导入；
Decoder.get_block 也只在 block_type=="mamba" 分支才导入本模块。
"""

from typing import Optional

import torch
from torch import nn


class MambaBlock1D(nn.Module):
    """【中文说明】双向 Mamba2 子块；(B,T,C) 进出，残差 + LayerScale 外壳。

    参数：
        dim            通道数（与所在级 channels[i] 一致，默认配置下为 256）
        dropout        分支输出 dropout（与原 Transformer 子块的 dropout 语义对齐）
        d_state        SSM 状态维度（Mamba2 默认 128；64 为小模型常用值）
        d_conv         局部深度卷积核宽（Mamba 系列默认 4）
        expand         通道扩张率（d_inner = expand * dim）
        bidirectional  True=双向（默认）；False=仅正向（消融用）
        layerscale     输出分支初始缩放（1e-6 ≈ 零初始化）
    """

    def __init__(
        self,
        dim,
        dropout=0.0,
        d_state=64,
        d_conv=4,
        expand=2,
        bidirectional=True,
        layerscale=1e-6,
    ):
        super().__init__()
        # 【中文说明】惰性导入：Windows 环境没有 mamba-ssm，只有实际构建本块时才要求已安装
        from mamba_ssm import Mamba2

        self.norm = nn.LayerNorm(dim)
        self.mamba_fwd = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = (
            Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand) if bidirectional else None
        )
        self.dropout = nn.Dropout(dropout)
        # 【中文说明】LayerScale：逐通道缩放向量，初始 1e-6 ⇒ 块输出初始近似为零，
        #   残差主干在训练起点保持恒等，梯度通过 gamma 逐步"打开"该分支
        self.gamma = nn.Parameter(layerscale * torch.ones(dim))

    def forward(  # pylint: disable=unused-argument
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Optional[dict] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ):
        """【中文说明】decoder 调用时只传 (hidden_states, attention_mask, timestep)；
        其余参数仅为与 BasicTransformerBlock 签名对齐的占位，一律不使用。"""
        x = self.norm(hidden_states)
        y = self.mamba_fwd(x)
        if self.mamba_bwd is not None:
            # 【中文说明】反序扫描再翻回原位求和；.contiguous() 保证 CUDA kernel 输入连续
            y = y + self.mamba_bwd(x.flip(1).contiguous()).flip(1).contiguous()
        y = self.dropout(y)
        return hidden_states + self.gamma * y
