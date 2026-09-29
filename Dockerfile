# Matcha-TTS(ESD 情感微调版)自建镜像 v0.1
# 底座:公司官方 PyTorch 运行时(非个人镜像),torch 2.5.1 + CUDA 12.4 + cuDNN9,已内置 python3
FROM docker.v2.aispeech.com/aispeech/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# apt 换阿里云源(抄自公司基础镜像的标准写法)+ 系统依赖
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list \
    && apt-get clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        espeak-ng \
        build-essential \
        git \
        sudo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 先装 Python 依赖(独立一层:改代码时此层走缓存)
# pip 走阿里云源;底座已带 torch 2.5.1,requirements 的 torch>=2.0 直接满足,不会重下
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple \
    -r requirements.txt

# 再装 matcha 本体(可编辑模式,方便改代码)
COPY . /workspace/matcha-tts
RUN pip install --no-cache-dir -e /workspace/matcha-tts \
    -i https://mirrors.aliyun.com/pypi/simple
