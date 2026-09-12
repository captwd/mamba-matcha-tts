# -*- coding: utf-8 -*-
"""
条件流匹配模块（Conditional Flow Matching, CFM）
==================================================
Matcha-TTS 的核心解码器：学习从简单噪声分布到梅尔频谱分布的"流"(flow)。
训练时：把真实梅尔频谱和噪声做插值，让 estimator 网络预测速度场(v_theta)，计算流匹配损失。
推理时：用 ODE 求解器（Euler / Midpoint / RK4）从噪声出发，沿学习到的流走 n_timesteps 步，还原出梅尔频谱。
ODE 步数越少速度越快（默认 10 步即可），这是 Matcha-TTS 合成快的关键。

【中文说明】CFM 数学原理（rectified-flow 风格的线性插值路径）：
    设 x1 = 真实梅尔（归一化后），x0 ~ N(0, T) 为噪声（temperature 缩放方差），
    取时间 t ∈ [0, 1]，插值样本与该路径上恒定的"真实速度"分别为：
        y_t = (1 - (1 - sigma_min) * t) * x0 + t * x1   # t=0 时是纯噪声，t=1 时是真实梅尔
        u   = x1 - (1 - sigma_min) * x0
    网络 estimator(y_t, mask, mu, t, spks) 拟合 u（MSE 损失）。
    学到 v_theta 后，推理即解 ODE：dx/dt = v_theta(x, t)，从 t=0（噪声）积分到 t=1（梅尔）。

【中文说明】文件内结构：
    BASECFM  抽象基类：持有 estimator、实现推理 ODE forward()/solve_euler() 和训练损失 compute_loss()
    CFM      具体子类：只负责"按配置把 estimator 实例化出来"（全项目唯一实例化点，见文件末尾）
"""
from abc import ABC

import torch
import torch.nn.functional as F

from matcha.models.components.decoder import Decoder
from matcha.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class BASECFM(torch.nn.Module, ABC):
    """【中文说明】CFM 抽象基类：定义数据流和损失，不关心 estimator 具体是什么网络。

    它与子类的关系：本类实现"流"的全部数学，子类只负责实例化 estimator。
    """

    def __init__(
        self,
        n_feats,        # 【中文说明】梅尔通道数（=80，同时是 estimator 的输入/输出通道基准）
        cfm_params,     # 【中文说明】hydra 配置对象：solver、n_timesteps、sigma_min
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver          # 【中文说明】ODE 求解器名（配置默认 Euler，本实现实际只用 Euler）
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min  # 【中文说明】插值路径收缩系数（默认 1e-4，路径近似匀速直线）
        else:
            self.sigma_min = 1e-4

        # 【中文说明】estimator 由子类 CFM.__init__ 实例化；基类先置 None 占位
        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        # ---------- Step 1: 采起点噪声 ----------
        # 【中文说明】x(0) ~ N(0, T)：temperature 越小起点方差越低，合成更稳但多样性下降
        z = torch.randn_like(mu) * temperature
        # ---------- Step 2: 等分时间轴 ----------
        # 【中文说明】t_span: [0, 1/n, 2/n, ..., 1]，共 n_timesteps+1 个点
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        # ---------- Step 3: 沿流积分到 t=1 ----------
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        # 【中文说明】初始化：t=0，步长 dt = 1/n_timesteps（等分时间轴）
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        # 【中文说明】sol 保存每一步积分后的 x_t；想可视化 ODE 轨迹可在此循环里收集/断点
        sol = []

        # ---------- Euler 积分主循环 ----------
        # 【中文说明】每步：向 estimator 问当前时刻速度 -> x += dt * v -> t 前进一格。
        #   共 n_timesteps 次 estimator 前向，是推理时的主要计算量；
        #   Matcha 的卖点：10 步即可出高质量结果（对比扩散模型动辄几十上百步）
        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # ---------- Step 1: 随机采时间步 ----------
        # random timestep
        # 【中文说明】t ~ U(0,1)，形状 (B,1,1) 以便与 (B,C,T) 张量广播相乘
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # ---------- Step 2: 采噪声 p(x_0) ----------
        # sample noise p(x_0)
        # 【中文说明】x0 ~ N(0, I)，与真实梅尔 x1 同形状
        z = torch.randn_like(x1)

        # ---------- Step 3: 构造线性插值样本与目标速度 ----------
        # 【中文说明】y_t：t=0 时退化为噪声 z、t=1 时为真实梅尔 x1 的"直线运输"路径上的中间点；
        #   u：该路径的解析速度（全程恒定，因为路径对 t 是线性的）。训练目标就是让网络输出 ≈ u
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # ---------- Step 4: 流匹配 MSE 损失 ----------
        # 【中文说明】estimator 输入：(插值样本 y, 掩码 mask, 条件 mu, 时间步 t, 说话人 spks)；
        #   reduction="sum" 后按"有效帧数 × 通道数"归一化，使损失量纲与序列长度、batch 大小无关；
        #   返回 (loss, y)：y 是插值样本，供上层调试/可视化用（通常忽略）
        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), spks), u, reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )
        return loss, y


class CFM(BASECFM):
    """【中文说明】具体 CFM：继承基类的全部数学，只补一件事——按配置实例化 estimator。

    ★ 这是全项目 estimator 的唯一实例化点：换 U-Net 为 Mamba 等新骨干时，
      只需要改这里（或改成按 decoder_params.name 从注册表选择）。
    """

    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        # 【中文说明】多说话人时，输入通道在 (2*80) 基础上再拼一个说话人嵌入维度：
        #   单说话人: in_channels = 160  →  [x_t(80) | mu_y(80)]
        #   多说话人: in_channels = 160 + spk_emb_dim → [x_t | mu_y | spk（时间维复制后拼接）]
        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        # Just change the architecture of the estimator here
        # 【中文说明】★ 唯一实例化点。estimator 的接口契约（换骨干为 Mamba 等时必须保持一致）：
        #   调用签名  estimator(x, mask, mu, t, spks, cond)
        #     x:    (B, 80, T)   当前状态（带噪梅尔或插值样本 y_t）
        #     mask: (B, 1, T)    有效帧掩码
        #     mu:   (B, 80, T)   编码器条件向量（Decoder.forward 内部会把 x 和 mu 沿通道 pack 拼接，
        #                        因此这里的 in_channels 才是 2*80(+spk) 而不是 80）
        #     t:    (B,)         当前时间步
        #     spks: (B, spk_emb_dim) 或 None
        #   输出      (B, 80, T)   预测的速度场，形状与输入 x 一致
        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
