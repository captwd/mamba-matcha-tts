# -*- coding: utf-8 -*-
"""
CFM 解码器骨干网络（Decoder Backbone）
=======================================
流匹配 estimator 所使用的神经网络，负责预测速度场 v_theta。
结构：多个 1D 卷积块 + Conformer/Transformer 块交替堆叠（U-Net 风格，带时间步嵌入 SinusoidalPosEmb），
输入为 (带噪梅尔频谱 + 编码器输出 + 时间步嵌入 + 说话人嵌入)，输出为预测的速度场。

【中文说明】★ 这是将来"U-Net 换 Mamba"的替换目标文件 ★
接口契约（来自 flow_matching.CFM，见 flow_matching.py 末尾注释）：
    forward(x, mask, mu, t, spks=None, cond=None) -> (B, 80, T)
    x: (B, 80, T) 当前状态   |  mask: (B, 1, T) 掩码   |  mu: (B, 80, T) 编码器条件
    t: (B,) 时间步          |  spks: (B, spk_dim) 说话人嵌入（可选）
注意 mu 不是事先拼好的——本类 forward 内部用 einops.pack 把 [x, mu(, spks)] 沿通道拼接。

【中文说明】文件内结构（自上而下）：
    SinusoidalPosEmb   正弦位置编码：标量时间步 t -> (B, in_channels) 连续向量
    Block1D            基础卷积块：Conv1d + GroupNorm + Mish（含掩码相乘）
    ResnetBlock1D      残差卷积块：两个 Block1D + 注入时间嵌入 + 1x1 残差捷径
    Downsample1D       下采样：步长 2 的卷积（时间维减半）
    TimestepEmbedding  时间嵌入 MLP：Linear-SiLU-Linear，把正弦编码升维
    Upsample1D         上采样：转置卷积 ×2（时间维加倍）
    ConformerWrapper   Conformer 块的薄封装（可选 block 类型）
    Decoder            ★ 主干：组装 down/mid/up 三段 + final 输出头（见类内注释）
形状流（默认配置 channels=(256,256)，注意：只有第一级是真下采样，最后一级的
"下采样"被替换成普通 3x1 卷积，长度不变——因此实际最低分辨率为 T/2 而非 T/4）：
    输入拼接后 (B, 160+spk, T)
    -> down[0]: Resnet+Transformer -> (B, 256, T) → 下采样 → (B, 256, T/2)
    -> down[1]: Resnet+Transformer -> (B, 256, T/2)（末级用普通卷积，不再降长度）
    -> mid  ×2: (B, 256, T/2)      ← 最粗分辨率处
    -> up[0]: skip 拼接 (512→256) @T/2 → 上采样 → (B, 256, T)
    -> up[1]: skip 拼接 @T → 末级普通卷积 → (B, 256, T)
    -> final: (B, 256, T) -> proj 1x1 -> (B, 80, T)
（长度约定：输入须为 4 的倍数，由 fix_len_compatibility 保证，见 utils/model.py）
"""
import math
from typing import Optional

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
import torch.nn.functional as F
from conformer import ConformerBlock
from diffusers.models.activations import get_activation
from einops import pack, rearrange, repeat

from matcha.models.components.transformer import BasicTransformerBlock


class SinusoidalPosEmb(torch.nn.Module):
    """【中文说明】正弦时间步嵌入（与 Transformer 位置编码同型，但输入是"时间步 t"）。

    作用：把标量 t ∈ [0,1] 映射为高维连续向量，让网络能区分"当前处于流的哪一步"。
    公式：emb(t) = [sin(t/10000^(2i/d)), cos(t/10000^(2i/d))] 拼接，i = 0..d/2-1
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        """【中文说明】x: (B,) 或标量时间步 -> (B, dim)。scale 放大 t 的数值范围以利用 sin 周期"""
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        # 【中文说明】频率几何递减：emb 系数 = exp(-ln(10000) * i / (half_dim-1))
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        # 【中文说明】scale(=1000) * t * 频率 -> 各维相位；一半 sin、一半 cos 拼接
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block1D(torch.nn.Module):
    """【中文说明】最基础的 1D 卷积块：Conv1d(k=3) + GroupNorm(8) + Mish。

    所有 conv 前后都乘 mask：把 padding 区域置零，避免 padding 内容渗入卷积统计。
    输入输出形状不变：(B, dim, T) -> (B, dim_out, T)
    """

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    """【中文说明】注入时间条件的残差卷积块（U-Net 的标准构件，思想源自扩散模型 UNet）。

    结构：
        x -> Block1D -> (+ time_emb 投影) -> Block1D -> (+ 1x1 残差捷径) -> 输出
    time_emb 由 SinusoidalPosEmb+MLP 得到 (B, time_emb_dim)，
    经 Mish+Linear 投到当前通道数后 unsqueeze 成 (B, C, 1) 按时间广播相加。
    这就是"同一个网络在不同 ODE 步表现为不同函数"的实现机制。
    """

    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        # 【中文说明】时间嵌入的通道投影：Mish -> Linear(time_emb_dim -> dim_out)
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)

        # 【中文说明】残差捷径：通道数不一致时用 1x1 卷积对齐；一致时它也照常存在（可学习缩放）
        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        # 【中文说明】时间条件注入点：(B, dim_out) -> (B, dim_out, 1) 沿时间广播
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class Downsample1D(nn.Module):
    """【中文说明】下采样：kernel=3, stride=2, padding=1 的 1D 卷积，时间维 T -> ceil(T/2)"""

    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class TimestepEmbedding(nn.Module):
    """【中文说明】时间嵌入 MLP（源自 diffusers）。

    正弦编码 (B, in_channels) -> Linear -> SiLU -> Linear -> (B, time_embed_dim)。
    升维 4 倍（time_embed_dim = channels[0]*4），给 ResnetBlock1D 提供条件向量。
    cond_proj / post_act 本项目未使用，保留是为兼容 diffusers 的实现。
    """

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        # 【中文说明】可选的额外条件投影（如外部条件向量加到 sample 上）；本项目恒为 None
        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        # 【中文说明】输出维度默认等于 time_embed_dim
        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        """【中文说明】sample: 正弦时间编码 (B, in_channels) -> 条件向量 (B, time_embed_dim)"""
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=True, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        # 【中文说明】两种上采样实现二选一：
        #   转置卷积（默认，本项目用这个）：kernel=4, stride=2, padding=1，严格 2 倍放大
        #   最近邻插值(+可选卷积)：F.interpolate 简单复制，不引入棋盘伪影但有台阶感
        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        """【中文说明】(B, channels, T) -> (B, out_channels, 2T)"""
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs


class ConformerWrapper(ConformerBlock):
    """【中文说明】Conformer 块的适配封装（可选的 block 类型，默认配置未启用）。

    继承 conformer 包的 ConformerBlock；forward 只透传 (hidden_states, attention_mask)，
    其余参数（encoder_hidden_states/timestep）收下但不用——为了与 BasicTransformerBlock
    保持相同的调用签名，方便 Decoder 里统一调度。
    """

    def __init__(  # pylint: disable=useless-super-delegation
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0,
        ff_dropout=0,
        conv_dropout=0,
        conv_causal=False,
    ):
        super().__init__(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
            conv_causal=conv_causal,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
    ):
        """【中文说明】签名对齐 BasicTransformerBlock；实际只用 hidden_states + mask"""
        return super().forward(x=hidden_states, mask=attention_mask.bool())


class Decoder(nn.Module):
    """【中文说明】U-Net 风格的流匹配 estimator（CFM 的速度场网络）。

    三段式结构（默认配置 channels=(256,256)、各段 1/2/1 个 transformer 块）：
        down_blocks：逐级"卷积感知局部 + 注意力建模全局"，并下采样时间维
        mid_blocks ：最粗分辨率处堆叠（感受野最大，负责整体韵律结构）
        up_blocks  ：逐级上采样，并与 down 的隐层做 skip 连接（拼接后过 Resnet）
        final      ：一层卷积块 + 1x1 投影回 80 通道，输出速度场
    时间条件：t -> 正弦编码 -> MLP，在 每个 ResnetBlock1D 内注入。

    【中文说明】局部算子开关 resnet_type（"只换 CNN、不改框架"的落点）：
        "conv"        原版 Block1D（Conv3+GN+Mish），默认，老 checkpoint 兼容
        "convnext_v2" ConvNeXt V2 单元（深度卷积 k7 + LN + GRN），见 convnext_v2.py
        两种选择的 ResBlock 外壳（残差、时间注入、掩码纪律）完全一致；
        down/mid/up/skip/Transformer 块/采样层一律不动。老 checkpoint 不带此参数，自动回退 "conv"。
    """

    def __init__(
        self,
        in_channels,        # 【中文说明】输入通道：2*80（x_t + mu）+ spk_emb_dim(多说话人时)
        out_channels,       # 【中文说明】输出通道：80（速度场）
        channels=(256, 256),  # 【中文说明】每级下采样的通道数，长度=下采样级数
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,         # 【中文说明】每个 down/up 块里 transformer 子块数量
        num_mid_blocks=2,   # 【中文说明】中间段块数量
        num_heads=4,
        act_fn="snakebeta",  # 【中文说明】transformer 前馈层的激活；与 hydra 默认(configs/model/decoder/default.yaml)对齐。
                             # 注意：上游原本写 "snake" 但 FeedForward 无此分支（潜在 bug），此处修正为 snakebeta
        down_block_type="transformer",  # 【中文说明】可选 "transformer" / "conformer" / "mamba"（mamba 需 Linux/WSL + mamba-ssm）
        mid_block_type="transformer",
        up_block_type="transformer",
        resnet_type="conv",  # 【中文说明】局部算子开关："conv"=原版 Block1D | "convnext_v2"=ConvNeXt V2 单元
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ---------- 局部算子选择 ----------
        # 【中文说明】按 resnet_type 选 ResBlock/末级块的实现；外壳（残差+时间注入+掩码）两者一致
        if resnet_type == "convnext_v2":
            from matcha.models.components.convnext_v2 import ConvNeXtV2ResBlock1D, ConvNeXtV2Unit1D

            resnet_cls = ConvNeXtV2ResBlock1D
            final_block_cls = ConvNeXtV2Unit1D
        elif resnet_type == "conv":
            resnet_cls = ResnetBlock1D
            final_block_cls = Block1D
        else:
            raise ValueError(f"Unknown resnet_type: {resnet_type} (use 'conv' or 'convnext_v2')")
        self.resnet_type = resnet_type

        # ---------- 时间嵌入流水线 ----------
        # 【中文说明】标量 t (B,) -> 正弦编码 (B, in_channels) -> MLP (B, channels[0]*4)
        #   time_embed_dim 取首级通道的 4 倍（256*4=1024），是扩散类模型的经验配置
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        # ---------- 三个阶段的容器 ----------
        # 【中文说明】每级 down 块 = [ResnetBlock1D, n×Transformer块, 下采样层]（ModuleList 嵌套）
        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])     # 【中文说明】每块 = [ResnetBlock1D, n×Transformer块]
        self.up_blocks = nn.ModuleList([])      # 【中文说明】每级 = [ResnetBlock1D, n×Transformer块, 上采样层]

        # ---------- 构建下采样段 ----------
        # 【中文说明】逐级：通道 in->channels[i]，除最后一级外都接 Downsample1D；
        #   最后一级用普通 3x1 卷积代替下采样（保持长度不变，只做特征变换）
        output_channel = in_channels
        for i in range(len(channels)):  # pylint: disable=consider-using-enumerate
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = resnet_cls(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        down_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        # ---------- 构建中间段 ----------
        # 【中文说明】最低分辨率处（T/4），通道恒为 channels[-1]；无下/上采样
        for i in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]

            resnet = resnet_cls(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        mid_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        # ---------- 构建上采样段 ----------
        # 【中文说明】通道序列 = 反转的 channels 再补一个 channels[0]：如 (256,256) -> (256,256,256)
        #   每级 Resnet 的输入通道是 2*input_channel——多出来的部分就是 skip 连接拼进来的 down 段隐层
        #   除最后一级外接 Upsample1D（转置卷积 ×2）；最后一级同样换成普通卷积
        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i]
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2

            resnet = resnet_cls(
                dim=2 * input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        up_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        # ---------- 输出头 ----------
        # 【中文说明】final_block：最后一层卷积块（恢复非线性表征），
        #   final_proj：1x1 卷积把 channels[-1] 压回 80 通道 = 速度场
        self.final_block = final_block_cls(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)

        # 【中文说明】自定义初始化（见 initialize_weights）；下一行被注释掉的normal初始化已废弃
        self.initialize_weights()
        # nn.init.normal_(self.final_proj.weight)

    @staticmethod
    def get_block(block_type, dim, attention_head_dim, num_heads, dropout, act_fn):
        """【中文说明】按类型名构建一个"全局建模"子块（工厂函数）。

        transformer：diffusers 的 BasicTransformerBlock（自注意力 + GEGLU/snake 前馈），默认
        conformer  ：卷积+注意力混合块（语音界常用），可选
        mamba      ：双向 Mamba2 序列混合器（线性复杂度全局建模），见 mamba_block.py；
                     CUDA-only（Windows 不可构建，分支内才导入，不影响其它路径）
        三者签名一致（都吃 hidden_states + attention_mask + timestep），可互换。
        """
        if block_type == "conformer":
            block = ConformerWrapper(
                dim=dim,
                dim_head=attention_head_dim,
                heads=num_heads,
                ff_mult=1,
                conv_expansion_factor=2,
                ff_dropout=dropout,
                attn_dropout=dropout,
                conv_dropout=dropout,
                conv_kernel_size=31,
            )
        elif block_type == "transformer":
            block = BasicTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=act_fn,
            )
        elif block_type == "mamba":
            # 【中文说明】惰性导入：Windows 无 mamba-ssm，只有选了 mamba 才要求已安装
            from matcha.models.components.mamba_block import MambaBlock1D

            block = MambaBlock1D(dim=dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown block type {block_type}")

        return block

    def initialize_weights(self):
        """【中文说明】自定义权重初始化（只作用于 Conv1d / GroupNorm / Linear）。

        卷积与线性层：Kaiming 正态（针对 ReLU 系激活，这里 Mish 也适用）+ 零偏置；
        GroupNorm：gamma=1, beta=0（恒等启动）。
        注意 Transformer/Conformer/Mamba 子块内部有自己的初始化，不受这里影响：
        Mamba2 的 A_log / D / dt_proj 等是精心设计的特殊初始化，被 Kaiming 覆盖会破坏
        SSM 步长语义，故整个 MambaBlock1D 子树跳过。
        """
        # 【中文说明】收集所有 MambaBlock1D 子树成员的 id；Windows 上导入失败 = 不存在 mamba 块 = 空集合
        try:
            from matcha.models.components.mamba_block import MambaBlock1D

            mamba_ids = {
                id(sub)
                for m in self.modules()
                if isinstance(m, MambaBlock1D)
                for sub in m.modules()
            }
        except ImportError:
            mamba_ids = set()

        for m in self.modules():
            if id(m) in mamba_ids:
                continue

            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # ---------- Step 1: 时间条件 ----------
        # 【中文说明】标量/向量 t -> 正弦编码 -> MLP：得到 (B, time_embed_dim) 条件向量，
        #   供所有 ResnetBlock1D 注入；Transformer 块收到它但默认不使用（普通 LayerNorm）
        t = self.time_embeddings(t)
        t = self.time_mlp(t)

        # ---------- Step 2: 沿通道拼接输入 ----------
        # 【中文说明】pack 把 [x_t(80), mu(80)] 拼成 (B, 160, T)；
        #   多说话人时再把 spks (B, spk_dim) 沿时间复制 T 份后拼成 (B, 160+spk_dim, T)。
        #   ⇒ 换骨干（如 Mamba）时照抄这一段即可，之后的 down/mid/up 可整体替换。
        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]

        # ---------- Step 3: 下采样段 ----------
        # 【中文说明】masks 列表保存各级分辨率的掩码，up 段按相反顺序弹出使用（对称结构）。
        #   每级流程：Resnet(+时间嵌入) -> 转置成 (B,T,C) 过 Transformer 块（mask 同步转置）
        #   -> 存隐层供 skip -> (x*mask) 过下采样层 -> 掩码也按 ::2 抽稀对齐新长度
        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c")
            mask_down = rearrange(mask_down, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_down,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_down = rearrange(mask_down, "b t -> b 1 t")
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        # ---------- Step 4: 中间段 ----------
        # 【中文说明】最低分辨率处逐块处理；masks[:-1] 是去掉刚才多加的最后一个抽稀掩码，
        #   mask_mid = 最末一级（T/4 长度）的掩码
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c")
            mask_mid = rearrange(mask_mid, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_mid,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_mid = rearrange(mask_mid, "b t -> b 1 t")

        # ---------- Step 5: 上采样段（含 skip 连接） ----------
        # 【中文说明】每级：弹出对应分辨率的掩码和 down 段隐层 -> pack 拼接 (2C, T) 过 Resnet
        #   -> Transformer 块 -> (x*mask) 上采样 ×2。up 结束后时间维恢复到 T、通道为 channels[0]
        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            x = resnet(pack([x, hiddens.pop()], "b * t")[0], mask_up, t)
            x = rearrange(x, "b c t -> b t c")
            mask_up = rearrange(mask_up, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_up,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_up = rearrange(mask_up, "b t -> b 1 t")
            x = upsample(x * mask_up)

        # ---------- Step 6: 输出头 ----------
        # 【中文说明】final 卷积块 -> 1x1 投影到 80 通道 -> 乘掩码清零 padding 区域。
        #   返回值即速度场预测 v_theta (B, 80, T)，交给 CFM 做 Euler 积分
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)

        return output * mask
