# -*- coding: utf-8 -*-
"""
文本编码器（Text Encoder，源自 Glow-TTS）
============================================
把音素 ID 序列编码为声学特征向量序列，同时内置一个时长预测器(Duration Predictor)：
1. Encoder：多头注意力 + 卷积堆叠（ConvReluNorm），输出音素的条件向量 mu_x
2. DurationPredictor：预测每个音素应持续的帧数（用于把音素特征对齐到音频帧级别）
3. 多说话人时会把说话人嵌入(speaker embedding)拼接到输入中

【中文说明】文件内结构（自下而上组装）：
    LayerNorm                 通道维 LayerNorm（对 (B, C, T) 在 C 上归一化，非标准 LN 的写法）
    ConvReluNorm              残差卷积栈（prenet 用，平滑音素嵌入）
    DurationPredictor         两层卷积 + 投影到 1 通道，输出每音素的对数时长 logw
    RotaryPositionalEmbeddings  RoPE 旋转位置编码（只作用于注意力 q/k 的一半维度）
    MultiHeadAttention        带卷积投影 + RoPE 的多头注意力
    FFN                       卷积前馈网络（1D 卷积版 MLP）
    Encoder                   标准编码器栈：交替 [自注意力 + FFN]（各带残差与 LayerNorm）
    TextEncoder               ★ 总装：Embedding -> (prenet) -> (+spk) -> Encoder -> 投影出 mu/logw
"""
""" from https://github.com/jaywalnut310/glow-tts """

import math

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
from einops import rearrange

import matcha.utils as utils  # pylint: disable=consider-using-from-import
from matcha.utils.model import sequence_mask

log = utils.get_pylogger(__name__)


class LayerNorm(nn.Module):
    """【中文说明】手写 LayerNorm：在"通道维"上归一化（输入约定为 (B, C, T)，对 C 求 mean/var）。

    与 nn.LayerNorm 的区别：标准 LN 对最后一维归一化，而这里序列模型习惯 (B,C,T) 布局，
    通道在中间，所以自己写。gamma/beta 是逐通道的可学习仿射参数。
    """

    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        """【中文说明】(B, C, T) -> (B, C, T)；逐样本、逐时间步、按通道统计"""
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        # 【中文说明】把 gamma/beta 重塑成可广播形状（适配 2D/3D 输入）
        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ConvReluNorm(nn.Module):
    """【中文说明】残差卷积栈（Glow-TTS 的 prenet）：n 层 [Conv1d + LayerNorm + ReLU + Dropout]。

    特点：末端投影层 proj 零初始化（weight/bias 置零），训练开始时该模块等价于恒等映射，
    避免随机卷积破坏音素嵌入；随训练逐渐"接入"非线性变换。
    """

    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        # 【中文说明】第一层 in->hidden，其余层 hidden->hidden；每层后接自写的 LayerNorm
        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(torch.nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        # 【中文说明】零初始化：初始时残差分支输出为 0，x 原样通过
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        """【中文说明】(B, C, T) -> (B, out_channels, T)；每层先乘掩码再卷积"""
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DurationPredictor(nn.Module):
    """【中文说明】时长预测器：预测每个音素持续多少帧（输出对数时长 logw）。

    结构：[Conv1d + LN + ReLU + Drop] ×2 -> 1x1 卷积投影到 1 通道。
    训练目标是 MAS 得到的"真实"对数时长（见 matcha_tts.forward Step 5）。
    """

    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        """【中文说明】(B, C, T) -> (B, 1, T)；每维输出 = 该音素的对数持续帧数"""
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class RotaryPositionalEmbeddings(nn.Module):
    """
    ## RoPE module

    Rotary encoding transforms pairs of features by rotating in the 2D plane.
    That is, it organizes the $d$ features as $\frac{d}{2}$ pairs.
    Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it
    by an angle depending on the position of the token.

    【中文说明】旋转位置编码（RoPE）：把位置信息"旋转"进 q/k 向量，使注意力分数天然只依赖
    两个 token 的相对距离。相比加性位置编码，外推到更长序列时更稳。
    本实现：只旋转每个注意力头通道数的一半（k_channels * 0.5），其余通道直接透传。
    """

    def __init__(self, d: int, base: int = 10_000):
        r"""
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = int(d)
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        r"""
        Cache $\cos$ and $\sin$ values

        【中文说明】预计算旋转表并缓存：只要请求的序列长度不超过已缓存的，就直接复用。
        """
        # Return if cache is already built
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        # Get sequence length
        seq_len = x.shape[0]

        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        # 【中文说明】几何递减的旋转频率组
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)

        # Calculate the product of position index and $\theta_i$
        # 【中文说明】每个位置与每个频率两两相乘：角度表 (seq_len, d/2)
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        # 【中文说明】复制拼接成 d 维（后半与前半相同），与"旋转对"的排布方式对应
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache them
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        """【中文说明】构造旋转所需的"对偶"排列：后半取负放前，前半放后"""
        # $\frac{d}{2}$
        d_2 = self.d // 2

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`

        【中文说明】对前一半特征做旋转（乘 cos + 乘 sin 对偶项），后一半原样透传。
        """
        # Cache $\cos$ and $\sin$ values
        x = rearrange(x, "b h t d -> t b h d")

        self._build_cache(x)

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = self._neg_half(x_rope)

        x_rope = (x_rope * self.cos_cached[: x.shape[0]]) + (neg_half_x * self.sin_cached[: x.shape[0]])

        return rearrange(torch.cat((x_rope, x_pass), dim=-1), "t b h d -> b h t d")


class MultiHeadAttention(nn.Module):
    """【中文说明】多头注意力（VITS/Glow-TTS 风格）。

    与标准 Transformer 注意力的区别：
    1. q/k/v 用 1x1 卷积投影（对 (B,C,T) 直接做，等价于逐点线性层）
    2. q/k 各自过 RoPE 旋转位置编码（相对位置信息）
    3. 可选 proximal bias（近邻偏置，本模型未用）与 proximal_init（k 初始 = q，本模型未用）
    """

    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        heads_share=True,
        p_dropout=0.0,
        proximal_bias=False,
        proximal_init=False,
    ):
        super().__init__()
        # 【中文说明】通道数必须能被头数整除（每头 k_channels = channels/n_heads）
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.heads_share = heads_share
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None  # 【中文说明】推理后保存注意力权重（调试/可视化用）

        self.k_channels = channels // n_heads
        # 【中文说明】q/k/v 的 1x1 卷积投影
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)

        # from https://nn.labml.ai/transformers/rope/index.html
        # 【中文说明】q、k 各一个 RoPE（旋转 k_channels*0.5 个通道，奇数头数/通道会报错，见 assert）
        self.query_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)
        self.key_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)

        # 【中文说明】输出投影：多头拼接后压到 out_channels
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        # 【中文说明】投影层 Xavier 初始化；proximal_init 时 k 直接复制 q（有助于训练初期注意力接近均匀）
        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        """【中文说明】x: query 源 (B,C,T)；c: key/value 源 (B,C,T)。自注意力时 x == c"""
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        """【中文说明】核心注意力计算：分头 -> RoPE -> 缩放点积 -> (bias/mask) -> softmax -> 加权求和"""
        b, d, t_s, t_t = (*key.size(), query.size(2))
        # 【中文说明】(B, C, T) -> (B, heads, T, C/heads)
        query = rearrange(query, "b (h c) t-> b h t c", h=self.n_heads)
        key = rearrange(key, "b (h c) t-> b h t c", h=self.n_heads)
        value = rearrange(value, "b (h c) t-> b h t c", h=self.n_heads)

        # 【中文说明】RoPE：q/k 旋转，v 不动
        query = self.query_rotary_pe(query)
        key = self.key_rotary_pe(key)

        # 【中文说明】缩放点积注意力（按每头通道数缩放）
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)

        if self.proximal_bias:
            # 【中文说明】近邻偏置：距离越近分数越高（仅自注意力可用），本模型未启用
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
        if mask is not None:
            # 【中文说明】掩码位置分数置 -1e4，softmax 后概率趋近 0
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        # 【中文说明】(B, h, T, c) -> (B, C, T)：分头结果拼接还原
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    @staticmethod
    def _attention_bias_proximal(length):
        """【中文说明】生成 (1,1,L,L) 的近邻偏置矩阵：bias = -ln(1+|i-j|)"""
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    """【中文说明】前馈网络（1D 卷积版 MLP）：Conv(k) -> ReLU -> Drop -> Conv(k)。

    用带 kernel 的卷积而不是 1x1，使每个位置能"看到"邻域（kernel_size 由配置决定）。
    """

    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = torch.nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        """【中文说明】(B, C, T) -> (B, out_channels, T)"""
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class Encoder(nn.Module):
    """【中文说明】Glow-TTS 风格编码器栈：n_layers 层 [自注意力 + FFN]，各带残差和 LayerNorm。

    与标准 Transformer 的差异：残差是"post-LN"变体（x + y 后过 LN），注意力带掩码。
    """

    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        # 【中文说明】逐层构建注意力/FFN 及各自的后置 LayerNorm
        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        """【中文说明】(B, C, T) -> (B, C, T)。

        每层流程：x *= mask -> 自注意力(y) -> 残差+LN1 -> FFN(y) -> 残差+LN2 -> 乘 mask。
        attn_mask 是 (B, T, T) 的二维联合掩码，屏蔽 padding 对 padding 的注意力。
        """
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class TextEncoder(nn.Module):
    """【中文说明】文本编码器总装（matcha_tts.encoder）。

    数据流（单/多说话人通用）：
        音素 ID (B, T) -> Embedding(*sqrt(C)) -> (ConvReluNorm prenet)
        -> [多说话人: 拼接说话人嵌入] -> Encoder 栈 -> 两路输出：
            proj_m: 条件向量 mu (B, 80, T)   —— 给 CFM 当条件
            proj_w: 对数时长 logw (B, 1, T)  —— 给对齐模块
    """

    def __init__(
        self,
        encoder_type,               # 【中文说明】编码器类型名（配置里固定为 "roentgen"）
        encoder_params,             # 【中文说明】Encoder 栈结构参数
        duration_predictor_params,  # 【中文说明】时长预测器结构参数
        n_vocab,                    # 【中文说明】音素符号表大小
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.n_vocab = n_vocab
        self.n_feats = encoder_params.n_feats        # 【中文说明】输出条件向量维度（=80，与梅尔同维）
        self.n_channels = encoder_params.n_channels  # 【中文说明】内部隐藏通道（LJSpeech 配置 512）
        self.spk_emb_dim = spk_emb_dim
        self.n_spks = n_spks

        # ---------- 音素嵌入 ----------
        # 【中文说明】初始化方差取 1/sqrt(C)：配合 forward 里乘 sqrt(C)，期望上保持单位方差
        self.emb = torch.nn.Embedding(n_vocab, self.n_channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, self.n_channels**-0.5)

        # ---------- 可选 prenet ----------
        # 【中文说明】配置 pirenz=True 时启用残差卷积栈平滑嵌入；否则用恒等 lambda 占位
        if encoder_params.prenet:
            self.prenet = ConvReluNorm(
                self.n_channels,
                self.n_channels,
                self.n_channels,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.5,
            )
        else:
            self.prenet = lambda x, x_mask: x

        # ---------- 编码器栈（多说话人时输入通道扩一个 spk_emb_dim） ----------
        self.encoder = Encoder(
            encoder_params.n_channels + (spk_emb_dim if n_spks > 1 else 0),
            encoder_params.filter_channels,
            encoder_params.n_heads,
            encoder_params.n_layers,
            encoder_params.kernel_size,
            encoder_params.p_dropout,
        )

        # ---------- 双输出头 ----------
        # 【中文说明】proj_m：Encoder 特征 -> 80 维条件向量（mu）
        self.proj_m = torch.nn.Conv1d(self.n_channels + (spk_emb_dim if n_spks > 1 else 0), self.n_feats, 1)
        # 【中文说明】proj_w：DurationPredictor，同一特征 -> 对数时长（logw）
        self.proj_w = DurationPredictor(
            self.n_channels + (spk_emb_dim if n_spks > 1 else 0),
            duration_predictor_params.filter_channels_dp,
            duration_predictor_params.kernel_size,
            duration_predictor_params.p_dropout,
        )

    def forward(self, x, x_lengths, spks=None):
        """Run forward pass to the transformer based encoder and duration predictor

        Args:
            x (torch.Tensor): text input
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): text input lengths
                shape: (batch_size,)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size,)

        Returns:
            mu (torch.Tensor): average output of the encoder
                shape: (batch_size, n_feats, max_text_length)
            logw (torch.Tensor): log duration predicted by the duration predictor
                shape: (batch_size, 1, max_text_length)
            x_mask (torch.Tensor): mask for the text input
                shape: (batch_size, 1, max_text_length)
        """
        # ---------- Step 1: 嵌入 + 掩码 ----------
        # 【中文说明】ID (B,T) -> (B,C,T)；乘 sqrt(C) 是 Transformer 经典做法（补偿嵌入缩放）
        x = self.emb(x) * math.sqrt(self.n_channels)
        x = torch.transpose(x, 1, -1)
        # 【中文说明】由真实长度生成 (B,1,T) 掩码
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        # ---------- Step 2: prenet（可选平滑） ----------
        x = self.prenet(x, x_mask)
        # ---------- Step 3: 多说话人条件拼接 ----------
        # 【中文说明】说话人向量 (B,spk,1) 沿时间复制 T 份 -> 拼到通道维
        if self.n_spks > 1:
            x = torch.cat([x, spks.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)
        # ---------- Step 4: 编码器栈 ----------
        x = self.encoder(x, x_mask)
        # ---------- Step 5: 条件向量输出 mu ----------
        mu = self.proj_m(x) * x_mask

        # ---------- Step 6: 时长预测 ----------
        # 【中文说明】★ x_dp = detach：时长预测器不接收来自 mu/CFM 的梯度——
        #   两路输出解耦训练，避免时长损失干扰声学特征学习（Glow-TTS 传统做法）
        x_dp = torch.detach(x)
        logw = self.proj_w(x_dp, x_mask)

        return mu, logw, x_mask
