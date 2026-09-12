# -*- coding: utf-8 -*-
"""
HiFi-GAN 声码器适配器（原版随 Matcha-TTS 提供）
================================================
两种型号：
    hifigan_T2_v1   —— LJSpeech 专用（跟官方 matcha_ljspeech 配套）
    hifigan_univ_v1 —— 通用型（VCTK 多说话人训练，自定义模型建议用它）

权重由 matcha/cli.py 的 VOCODER_URLS 负责下载到本地，本模块只负责加载。
输出规格：mel [B, 80, T] -> 波形 [B, 1, T*256]（22050Hz）。
HiFi-GAN 存在生成器偏置伪影，故同时返回 Denoiser（mode="zeros"）。
"""
import torch

from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.vocoders import HIFIGAN_T2_V1, HIFIGAN_UNIV_V1, register_vocoder


def _load_hifigan_generator(checkpoint_path, device):
    """按官方 config v1 构建生成器并恢复权重（去除 weight norm 后仅供推理）"""
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan


def _load_hifigan_with_denoiser(checkpoint_path, device):
    vocoder = _load_hifigan_generator(checkpoint_path, device)
    denoiser = Denoiser(vocoder, mode="zeros")
    return vocoder, denoiser


@register_vocoder(HIFIGAN_T2_V1)
def load_hifigan_t2_v1(checkpoint_path, device):
    return _load_hifigan_with_denoiser(checkpoint_path, device)


@register_vocoder(HIFIGAN_UNIV_V1)
def load_hifigan_univ_v1(checkpoint_path, device):
    return _load_hifigan_with_denoiser(checkpoint_path, device)
