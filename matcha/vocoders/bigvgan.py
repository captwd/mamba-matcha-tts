# -*- coding: utf-8 -*-
"""
BigVGAN 声码器适配器（NVIDIA，MIT 协议）
==========================================
选用型号：nvidia/bigvgan_base_22khz_80band（14M 参数）
与 Matcha-TTS 的 mel 规格完全对齐：
    22050Hz / 80 mel / hop 256 / fmax 8000 / librosa 滤波 + ln 动态压缩
因此**无需重训声学模型**即可即插即用。

权重来源：HuggingFace 仓库（自动下载并缓存在本地，~55MB）
依赖：pip install bigvgan

实现说明：不使用 bigvgan 包自带的 from_pretrained（其依赖的 huggingface_hub
PyTorchModelHubMixin 签名与新版 hub 1.x 不兼容），改为手动三步加载：
    config.json -> hparams | 构建网络 | 加载 state_dict
这样与 hub 版本解耦，最稳。

用法：
    from matcha.vocoders import load_vocoder
    vocoder, denoiser = load_vocoder("bigvgan_base_22khz_80band", None, device)
    waveform = vocoder(mel)             # mel: [B, 80, T] -> 波形 [B, 1, T*256]
"""
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from matcha.hifigan.denoiser import Denoiser
from matcha.vocoders import BIGVGAN_BASE_22KHZ_80BAND, register_vocoder

BIGVGAN_MODEL_ID = "nvidia/bigvgan_base_22khz_80band"


class BigVGANWrapper(torch.nn.Module):
    """把 bigvgan 包的模型包一层，调用方式与 HiFi-GAN Generator 保持一致"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def forward(self, mel):
        """mel: [B, 80, T] -> 波形 [B, 1, T*256]"""
        audio = self.model(mel)
        if audio.dim() == 2:  # [B, T] -> [B, 1, T]
            audio = audio.unsqueeze(1)
        return audio

    def remove_weight_norm(self):
        # bigvgan 包的模型在加载时已做 remove_weight_norm，这里仅为保持接口一致
        pass


def _resolve_snapshot(model_id: str) -> Path:
    """定位模型快照目录。

    优先使用本地缓存（离线可用）；缓存不全时才联网按需下载，
    且只取推理所需的两个文件（config.json + bigvgan_generator.pt），
    避免拖下 ~140MB 的判别器训练残留。
    """
    need = ("config.json", "bigvgan_generator.pt")

    # 1) 本地缓存回退：直接找 snapshots/*/<文件>
    hf_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ) / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    for candidate in hf_home.glob("*"):
        if all((candidate / f).exists() for f in need):
            return candidate

    # 2) 联网按需下载（只取推理所需文件）
    return Path(snapshot_download(model_id, allow_patterns=list(need)))


def load_bigvgan(device, model_id=BIGVGAN_MODEL_ID):
    """手动加载 BigVGAN：config.json 构建网络 + bigvgan_generator.pt 恢复权重"""
    from bigvgan import BigVGAN as _BigVGAN
    from bigvgan.env import AttrDict

    print(f"[!] Loading {model_id}!")
    snapshot = _resolve_snapshot(model_id)

    # 1) config.json -> hparams（AttrDict 支持属性访问）
    with open(snapshot / "config.json", encoding="utf-8") as f:
        h = AttrDict(json.load(f))

    # 2) 按超参构建网络
    model = _BigVGAN(h, use_cuda_kernel=False)

    # 3) 加载官方生成器权重（文件格式为 {"generator": state_dict}，与 HiFi-GAN 习惯一致）
    state_dict = torch.load(snapshot / "bigvgan_generator.pt", map_location="cpu")
    if "generator" in state_dict:
        state_dict = state_dict["generator"]
    model.load_state_dict(state_dict, strict=True)
    model.remove_weight_norm()
    model = model.eval().to(device)

    print(f"[+] BigVGAN loaded!")
    return BigVGANWrapper(model).to(device)


@register_vocoder(BIGVGAN_BASE_22KHZ_80BAND)
def load_bigvgan_adapter(checkpoint_path, device):
    """注册表入口：BigVGAN 权重走 HuggingFace 缓存，忽略 checkpoint_path。

    【公平对比修正】早期版本不挂 Denoiser，导致 A/B 对比时 HiFi-GAN 有去噪而
    BigVGAN 裸奔（电音偏重的嫌疑之一）。WaveGlow 式去偏置原理对任何声码器通用
    （喂全零测出固有偏置谱再反相扣除），这里同样挂上，保证对比公平。
    """
    vocoder = load_bigvgan(device)
    denoiser = Denoiser(vocoder, mode="zeros")
    return vocoder, denoiser
