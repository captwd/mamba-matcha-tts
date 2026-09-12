# -*- coding: utf-8 -*-
"""
Matcha-TTS 核心模型定义（基于条件流匹配 Conditional Flow Matching 的 TTS 模型）
================================================================================
模型整体结构（前向合成流程）：
    文本音素序列 x
      -> TextEncoder（文本编码器，提取音素特征 + 预测音素时长）
      -> 时长对齐（单调对齐搜索，把音素特征扩展到帧级别）
      -> CFM 解码器（条件流匹配，生成梅尔频谱）
      -> （推理时再由 hifigan/ 中的声码器把梅尔频谱转成波形音频）

训练时的损失 = 先验损失(Flow Matching 损失) + 时长预测损失

【中文说明】文件内结构速查（本文件 = "装配车间"，只做组合与调度，网络细节在 components/）：
    class MatchaTTS(BaseLightningClass)
        __init__     把 TextEncoder + CFM 两大部分拼起来，保存超参和数据统计量
        synthesise   推理入口：文本 -> 梅尔频谱（内部调用 encoder -> 对齐 -> CFM ODE）
        forward      训练入口：一个 batch 前向，返回 3 项损失（时长/先验/流匹配）
        关键成员：
            self.encoder = TextEncoder(...)   # matcha/models/components/text_encoder.py
            self.decoder = CFM(...)           # matcha/models/components/flow_matching.py
                                              #   内部持有 estimator（U-Net 骨干，components/decoder.py）
        数据统计量 mel_mean / mel_std 用于把 CFM 的输出反归一化成真实梅尔频谱。
"""
import datetime as dt
import math
import random

import torch

import matcha.utils.monotonic_align as monotonic_align  # pylint: disable=consider-using-from-import
from matcha import utils
from matcha.models.baselightningmodule import BaseLightningClass
from matcha.models.components.flow_matching import CFM
from matcha.models.components.text_encoder import TextEncoder
from matcha.utils.model import (
    denormalize,          # 【中文说明】(归一化梅尔 - mean) / std 还原为真实梅尔
    duration_loss,        # 【中文说明】时长预测的 L2 损失
    fix_len_compatibility,# 【中文说明】把长度调整为 U-Net 下采样步数的整数倍（避免形状报错）
    generate_path,        # 【中文说明】按时长把音素序列"铺"到帧级别，得到对齐矩阵 attn
    sequence_mask,        # 【中文说明】由长度张量生成 0/1 掩码
)

log = utils.get_pylogger(__name__)


class MatchaTTS(BaseLightningClass):  # 🍵
    """【中文说明】Matcha-TTS 主模型（PyTorch Lightning Module）。

    两大组件：
        self.encoder: TextEncoder —— 文本侧（音素嵌入 -> 注意力编码 -> mu_x 和时长 logw）
        self.decoder: CFM         —— 声学侧（条件流匹配，内部 estimator 即 U-Net 骨干）
    训练走 forward()，推理走 synthesise()，两者的中间量同名对应（mu_x / attn / mu_y）。
    """

    def __init__(
        self,
        n_vocab,            # 【中文说明】音素符号表大小（由 symbols 表决定，LJSpeech 系约 178+）
        n_spks,             # 【中文说明】说话人数量；=1 单说话人（LJSpeech），>1 多说话人（VCTK）
        spk_emb_dim,        # 【中文说明】说话人嵌入维度（多说话人才用）
        n_feats,            # 【中文说明】梅尔频谱通道数（=80）
        encoder,            # 【中文说明】TextEncoder 的结构配置（嵌套 omegaconf 对象）
        decoder,            # 【中文说明】U-Net 骨干的结构配置（channels/n_blocks 等）
        cfm,                # 【中文说明】CFM 的配置（solver、ODE 步数、sigma_min 等）
        data_statistics,    # 【中文说明】训练集的 mel_mean/mel_std（归一化常数，随数据集变化）
        out_size,           # 【中文说明】训练时从整条梅尔上随机截取的段长（显存技巧，见 forward）
        optimizer=None,     # 【中文说明】优化器构造函数（由 hydra 配置实例化，见 baselightningmodule）
        scheduler=None,     # 【中文说明】学习率调度器配置
        prior_loss=True,    # 【中文说明】是否计算先验损失（论文中的 L_prior，通常开启）
        use_precomputed_durations=False,  # 【中文说明】True 时跳过 MAS，直接用外部提供的时长（FastSpeech 风格）
    ):
        super().__init__()

        # 【中文说明】把全部构造参数存入 hparams：
        # 1) Lightning 会把它们写进 checkpoint，load_from_checkpoint 时能原样还原模型
        # 2) logger=False：不往 tensorboard 里重复记录
        self.save_hyperparameters(logger=False)

        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.out_size = out_size
        self.prior_loss = prior_loss
        self.use_precomputed_durations = use_precomputed_durations

        # ---------- 说话人嵌入（仅多说话人模型） ----------
        # 【中文说明】说话人 ID -> 向量 (spk_emb_dim)。单说话人模型没有这一层，
        # 因此 checkpoint 里也没有 spk_emb 权重（LJSpeech 版本体积更小）。
        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)

        # ---------- 文本编码器 ----------
        # 【中文说明】输入音素 ID 序列 (B, T_text)，输出：
        #   mu_x (B, 80, T_text)  —— 每个音素的条件向量（CFM 的条件项）
        #   logw (B, 1, T_text)   —— 每个音素的对数时长预测
        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            n_spks,
            spk_emb_dim,
        )

        # ---------- CFM 解码器 ----------
        # 【中文说明】in_channels = 2*80：U-Net 输入是 [带噪梅尔 x_t, 条件 mu_y] 沿通道拼接；
        # 多说话人时 CFM 内部还会再拼一个 spk_emb_dim 维（在 flow_matching.CFM.__init__ 里处理）。
        # 注意：变量名叫 decoder，但它是"流匹配模块"，不是传统的自回归/自编码解码器。
        self.decoder = CFM(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        # ---------- 数据统计量 ----------
        # 【中文说明】把 mel_mean/mel_std 注册为 buffer：
        #   1) 会随 checkpoint 保存/加载（推理时不需要再传统计量）
        #   2) 不参与梯度更新
        #   3) .to(device) 时自动跟着模型走
        self.update_data_statistics(data_statistics)

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, n_timesteps, temperature=1.0, spks=None, length_scale=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            spks (bool, optional): speaker ids.
                shape: (batch_size,)
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.

        Returns:
            dict: {
                "encoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Average mel spectrogram generated by the encoder
                "decoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Refined mel spectrogram improved by the CFM
                "attn": torch.Tensor, shape: (batch_size, max_text_length, max_mel_length),
                # Alignment map between text and mel spectrogram
                "mel": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Denormalized mel spectrogram
                "mel_lengths": torch.Tensor, shape: (batch_size,),
                # Lengths of mel spectrograms
                "rtf": float,
                # Real-time factor
            }
        """
        # 【中文说明】===== 推理计时起点：RTF = 合成耗时 / 音频时长，越小说明越接近实时 =====
        t = dt.datetime.now()

        # ---------- Step 1: 说话人嵌入 ----------
        # 【中文说明】多说话人模型：把说话人 ID (B,) 变成向量 (B, spk_emb_dim)，供编码器/解码器拼接。
        if self.n_spks > 1:
            # Get speaker embedding
            spks = self.spk_emb(spks.long())

        # ---------- Step 2: 文本编码 ----------
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        # 【中文说明】mu_x: (B, 80, T_text) 每音素条件向量；logw: (B,1,T_text) 对数时长预测；
        #             x_mask: (B,1,T_text) 文本掩码（padding 位置为 0）
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)

        # ---------- Step 3: 由时长预测决定目标梅尔长度 ----------
        # 【中文说明】把对数时长还原成帧数 w (B,1,T_text)；
        #   ceil 取整（每音素至少 1 帧）* length_scale 控制语速（>1 变慢，<1 变快）；
        #   y_lengths: 每条语句对应的总帧数 (B,)
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        # 【中文说明】batch 内统一到最长的长度，再对齐到下采样步数的整数倍，保证 U-Net 形状合法
        y_max_length_ = fix_len_compatibility(y_max_length)

        # ---------- Step 4: 构造对齐路径 attn ----------
        # Using obtained durations `w` construct alignment map `attn`
        # 【中文说明】attn: (B, 1, T_text, T_mel)，attn[b,0,i,j]=1 表示第 i 个音素占据第 j 帧。
        #   由时长矩阵 w_ceil 逐行"铺"出来；attn_mask 把 padding 区域置零。
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # ---------- Step 5: 音素条件向量对齐到帧级别 ----------
        # Align encoded text and get mu_y
        # 【中文说明】按对齐矩阵把 mu_x 沿时间维复制展开：
        #   mu_y: (B, 80, T_mel)，与梅尔频谱同长度，作为 CFM 的条件项。
        #   这一步等价于"把每个音素的特征向量重复其时长那么多帧"。
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # ---------- Step 6: CFM 流匹配解码（ODE 采样，模型里最耗时的部分） ----------
        # Generate sample tracing the probability flow
        # 【中文说明】从纯噪声出发，沿学到的速度场积分 n_timesteps 步，得到归一化梅尔频谱。
        #   内部即 flow_matching.solve_euler()；默认 10 步即可出声，这是 Matcha 快的核心。
        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, temperature, spks)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        # ---------- Step 7: 统计 RTF ----------
        # 【中文说明】合成时间 * 采样率 / 音频帧数*hop(256)：RTF<1 表示快于实时
        t = (dt.datetime.now() - t).total_seconds()
        rtf = t * 22050 / (decoder_outputs.shape[-1] * 256)

        # 【中文说明】返回字典逐项含义（都是裁到真实长度 y_max_length 的）：
        #   encoder_outputs —— 编码器条件向量（未过 CFM），用于对比"只有编码器"的合成效果
        #   decoder_outputs —— CFM 输出的"归一化"梅尔（均值方差仍处于训练用的标准化空间）
        #   attn            —— 音素-帧对齐矩阵（可视化对齐质量/强制对齐分析用）
        #   mel             —— 反归一化后的真实梅尔频谱（送声码器的就是它）
        #   mel_lengths     —— 每条样本的真实帧数
        #   rtf             —— 本次合成的实时率
        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": attn[:, :, :y_max_length],
            "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std),
            "mel_lengths": y_lengths,
            "rtf": rtf,
        }

    def forward(self, x, x_lengths, y, y_lengths, spks=None, out_size=None, cond=None, durations=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotonic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. flow matching loss: loss between mel-spectrogram and decoder outputs.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            y (torch.Tensor): batch of corresponding mel-spectrograms.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size,)
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
            spks (torch.Tensor, optional): speaker ids.
                shape: (batch_size,)
        """
        # ---------- Step 1: 说话人嵌入（同 synthesise） ----------
        if self.n_spks > 1:
            # Get speaker embedding
            spks = self.spk_emb(spks)

        # ---------- Step 2: 文本编码 ----------
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)
        y_max_length = y.shape[-1]

        # ---------- Step 3: 构造注意力掩码 ----------
        # 【中文说明】attn_mask: (B, T_text, T_mel)，双维度联合掩码，禁止文本 padding 与梅尔 padding 参与对齐
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # ---------- Step 4: 求对齐矩阵 attn（两条路径） ----------
        if self.use_precomputed_durations:
            # 【中文说明】路径 A：外部给定每音素时长（如用 FastSpeech2 教师蒸馏、或数据集自带强制对齐），
            # 直接按时长铺路径，省掉 MAS 的计算
            attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
        else:
            # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
            # 【中文说明】路径 B：单调对齐搜索（Monotonic Alignment Search, MAS，源自 Glow-TTS）。
            #   在"文本必须单调、不回跳"的约束下，穷举所有可行对齐并选"先验概率最高"的一条。
            #   实现上利用高斯先验的对数似然可分解性：log N(y | mu_x, 1) 沿对齐路径求和即可，
            #   因此先构造每帧的逐点对数似然 log_prior (B, T_text, T_mel)：
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats          # 【中文说明】高斯对数似然常数项
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y**2)        # 【中文说明】展开 (y - mu)^2 的 y^2 项
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)  # 【中文说明】交叉项 -2*y*mu
                mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)   # 【中文说明】mu^2 项（逐音素）
                # 【中文说明】log_prior[b,i,j] = -0.5*||y[b,:,j] - mu_x[b,:,i]||^2 + 常数
                log_prior = y_square - y_mu_double + mu_square + const

                # 【中文说明】DP 求最大似然单调路径（内部是 CUDA/numba 加速的 C 扩展），
                # attn: (B, T_text, T_mel) 0/1 矩阵；detach：对齐结果不回传梯度
                attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()  # b, t_text, T_mel

        # ---------- Step 5: 时长预测损失 ----------
        # Compute loss between predicted log-scaled durations and those obtained from MAS
        # referred to as prior loss in the paper
        # 【中文说明】MAS 得到的对齐里，每音素占的帧数求和再取对数 = "教师"对数时长 logw_；
        #   与编码器预测的 logw 做 L2（duration_loss 内部还会除以有效音素数取平均）。
        #   注意：这一项训练的是"时长预测器"，是模型能自动决定语速/停顿的关键。
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # ---------- Step 6（显存技巧）：从整条梅尔上随机截一小段来训解码器 ----------
        # Cut a small segment of mel-spectrogram in order to increase batch size
        #   - "Hack" taken from Grad-TTS, in case of Grad-TTS, we cannot train batch size 32 on a 24GB GPU without it
        #   - Do not need this hack for Matcha-TTS, but it works with it as well
        # 【中文说明】out_size 来自配置（LJSpeech min 显存版是 172）。做法：
        #   对每条样本随机选一个起点，只取 [起点, 起点+out_size) 的梅尔片段和对齐片段；
        #   较短的样本从头取满 y_lengths。等效于对帧数做数据增强 + 显存减半。
        #   注意 out_size 必须能被 U-Net 下采样倍数整除（channels 数决定，本项目为 2 层下采样）。
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)   # 【中文说明】每条样本允许的最大起点
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            # 【中文说明】逐样本随机抽起点（CPU 上抽完再搬回 GPU，避免 device 端随机数不一致）
            out_offset = torch.LongTensor(
                [torch.tensor(random.choice(range(start, end)) if end > start else 0) for start, end in offset_ranges]
            ).to(y_lengths)
            # 【中文说明】预分配截取后的张量（形状统一为 out_size，便于组 batch）
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)

            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                # 【中文说明】样本不足 out_size 帧时，y_cut_length = 原长度（clamp 上界为 0 增量）
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            # 【中文说明】用截取版替换原变量，后续步骤只处理这一小段
            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # ---------- Step 7: 条件向量对齐到（截取后的）帧级别 ----------
        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # ---------- Step 8: 流匹配损失（模型的主损失） ----------
        # Compute loss of the decoder
        # 【中文说明】进入 CFM：内部随机采时间步 t、构造插值样本 y_t，让 estimator 预测速度场 u，
        #   以 MSE 求差。详见 flow_matching.compute_loss()；返回的第二个值是插值样本（调试用，这里丢弃）
        diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, spks=spks, cond=cond)

        # ---------- Step 9: 先验损失（可选） ----------
        # 【中文说明】鼓励 CFM 的条件 mu_y 本身就接近真实梅尔（高斯负对数似然），
        #   为 ODE 起点提供更可靠的"均值场"，帮助少步数推理时更稳。
        #   归一化：除以有效帧数 * 80 通道，量纲与帧数无关。
        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0

        # 【中文说明】三项损失由 baselightningmodule.get_losses() 求和反传；attn 供可视化
        return dur_loss, prior_loss, diff_loss, attn
