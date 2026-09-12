# -*- coding: utf-8 -*-
"""
声码器适配层（Vocoder Adapters）
====================================
目的：把不同厂商的声码器收敛到统一接口，供 matcha/cli.py 调用，
新增声码器时只加文件、不改推理框架。

注册表机制：
1. 每个声码器文件用 @register_vocoder("名字") 注册一个加载函数
2. 加载函数签名统一为 loader(checkpoint_path, device) -> (vocoder, denoiser)
   - vocoder：可调用对象，mel [B, 80, T] -> 波形 [B, 1, T*hop]（或等价形状）
   - denoiser：无偏置伪影的声码器直接返回 None（如 BigVGAN）
3. 本包末尾统一 import 各适配器模块以触发注册，
   以后加声码器只需要：新写一个文件 + 在下方补一行 import
"""
VOCODER_REGISTRY = {}

HIFIGAN_T2_V1 = "hifigan_T2_v1"          # LJSpeech 专用 HiFi-GAN
HIFIGAN_UNIV_V1 = "hifigan_univ_v1"      # 通用 HiFi-GAN（VCTK 训练）
BIGVGAN_BASE_22KHZ_80BAND = "bigvgan_base_22khz_80band"


def register_vocoder(name):
    """把加载函数注册进 VOCODER_REGISTRY 的装饰器"""

    def decorator(loader):
        if name in VOCODER_REGISTRY:
            raise KeyError(f"Vocoder '{name}' is already registered!")
        VOCODER_REGISTRY[name] = loader
        return loader

    return decorator


def load_vocoder(name, checkpoint_path, device):
    """注册表统一入口：按名字分发到对应加载函数

    Args:
        name (str): 声码器名（须已注册）
        checkpoint_path (str | None): 本地权重路径；不需要本地权重的声码器（如 BigVGAN）可忽略
        device (torch.device): 加载目标设备

    Returns:
        (vocoder, denoiser)：denoiser 可能为 None
    """
    if name not in VOCODER_REGISTRY:
        raise NotImplementedError(
            f"Vocoder '{name}' not implemented! Registered: {sorted(VOCODER_REGISTRY)}. "
            f"To add one, create matcha/vocoders/<name>.py and register it with @register_vocoder."
        )
    return VOCODER_REGISTRY[name](checkpoint_path, device)


# 触发各适配器注册（新声码器在这里补一行）
from matcha.vocoders import hifigan as _hifigan  # noqa: E402,F401  pylint: disable=wrong-import-position
from matcha.vocoders import bigvgan as _bigvgan  # noqa: E402,F401  pylint: disable=wrong-import-position
