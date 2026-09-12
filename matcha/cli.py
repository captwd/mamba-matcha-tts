# -*- coding: utf-8 -*-
"""
命令行推理脚本（Command Line Inference）
=====================================
在终端中直接用文本合成语音，是 app.py 网页版背后的核心推理逻辑来源。

主要功能：
1. 解析命令行参数（模型、声码器、说话人 ID、ODE 步数、语速、温度等）
2. 自动检查/下载预训练模型 checkpoint 到用户数据目录
3. 文本 -> 音素序列 -> 梅尔频谱 -> 波形音频，支持单条/批量推理
4. 保存合成结果（.wav 音频、.png 频谱图、.npy 梅尔数据）并统计 RTF（实时率）

使用方法：
    matcha "要合成的英文文本" --spk 0 --temperature 0.667 --steps 10
    matcha --file text.txt --batched --vocoder hifigan_univ_v1
    matcha "text" --checkpoint_path my.ckpt --vocoder bigvgan_base_22khz_80band --cleaner english_cleaners2

【中文说明】声码器通过注册表（matcha/vocoders/）分发，--vocoder 选项与 --cleaner
必须分别匹配声码器训练规格和声学模型训练时的文本前端。
"""
import argparse
import datetime as dt
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import assert_model_downloaded, get_user_data_dir, intersperse
from matcha import vocoders as vocoder_registry

MATCHA_URLS = {
    "matcha_ljspeech": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_ljspeech.ckpt",
    "matcha_vctk": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_vctk.ckpt",
}

VOCODER_URLS = {
    "hifigan_T2_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1",  # Old url: https://drive.google.com/file/d/14NENd4equCBLyyCSke114Mv6YR_j_uFs/view?usp=drive_link
    "hifigan_univ_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000",  # Old url: https://drive.google.com/file/d/1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW/view?usp=drive_link
}

MULTISPEAKER_MODEL = {
    "matcha_vctk": {"vocoder": "hifigan_univ_v1", "speaking_rate": 0.85, "spk": 0, "spk_range": (0, 107)}
}

SINGLESPEAKER_MODEL = {"matcha_ljspeech": {"vocoder": "hifigan_T2_v1", "speaking_rate": 0.95, "spk": None}}

# 【中文说明】BigVGAN 声码器（matcha/vocoders/bigvgan.py 适配器）
# 权重从 HuggingFace 自动下载并缓存，无需 GitHub 直链，故不出现在 VOCODER_URLS 中
BIGVGAN_VOCODER_NAME = vocoder_registry.BIGVGAN_BASE_22KHZ_80BAND


def plot_spectrogram_to_numpy(spectrogram, filename):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.title("Synthesised Mel-Spectrogram")
    fig.canvas.draw()
    plt.savefig(filename)


def process_text(i: int, text: str, device: torch.device, cleaner_names=("english_cleaners2",)):
    """文本 -> 音素张量。

    【中文说明】cleaner 必须与训练时数据配置（configs/data/*.yaml 的 cleaners 字段）一致，
    否则音素分布错位、合成质量下降。默认 english_cleaners2 对应 LJSpeech 系配置。
    """
    print(f"[{i}] - Input text: {text}")
    x = torch.tensor(
        intersperse(text_to_sequence(text, list(cleaner_names))[0], 0),
        dtype=torch.long,
        device=device,
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    print(f"[{i}] - Phonetised text: {x_phones[1::2]}")

    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}


def get_texts(args):
    if args.text:
        texts = [args.text]
    else:
        with open(args.file, encoding="utf-8") as f:
            texts = f.readlines()
    return texts


def assert_required_models_available(args):
    save_dir = get_user_data_dir()
    # 【中文说明】修复官方逻辑 bug：原代码无论是否传了 --checkpoint_path，
    # 都会强制下载官方预训练模型（白白浪费流量）。现在指定了自定义 checkpoint 就直接用它。
    if args.checkpoint_path is not None:
        model_path = Path(args.checkpoint_path)
    else:
        model_path = save_dir / f"{args.model}.ckpt"
        assert_model_downloaded(model_path, MATCHA_URLS[args.model])
    # 【中文说明】BigVGAN 的权重在 HuggingFace 缓存里（load_bigvgan 自行处理），跳过本地文件检查
    if args.vocoder == BIGVGAN_VOCODER_NAME:
        vocoder_path = None
    else:
        vocoder_path = save_dir / f"{args.vocoder}"
        assert_model_downloaded(vocoder_path, VOCODER_URLS[args.vocoder])
    return {"matcha": model_path, "vocoder": vocoder_path}


def load_vocoder(vocoder_name, checkpoint_path, device):
    """【中文说明】声码器加载统一入口：按名字从注册表分发（matcha/vocoders/）。

    各声码器的加载逻辑在 matcha/vocoders/<name>.py 中，用 @register_vocoder 注册，
    这里不再维护 if/else 分支；新增声码器只需加适配器文件并在注册表包中 import。
    """
    print(f"[!] Loading {vocoder_name}!")
    vocoder, denoiser = vocoder_registry.load_vocoder(vocoder_name, checkpoint_path, device)
    print(f"[+] {vocoder_name} loaded!")
    return vocoder, denoiser


def load_matcha(model_name, checkpoint_path, device):
    print(f"[!] Loading {model_name}!")
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    _ = model.eval()

    print(f"[+] {model_name} loaded!")
    return model


def to_waveform(mel, vocoder, denoiser=None, denoiser_strength=0.00025):
    audio = vocoder(mel).clamp(-1, 1)
    if denoiser is not None:
        audio = denoiser(audio.squeeze(), strength=denoiser_strength).cpu().squeeze()

    return audio.cpu().squeeze()


def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    plot_spectrogram_to_numpy(np.array(output["mel"].squeeze().float().cpu()), f"{filename}.png")
    np.save(folder / f"{filename}", output["mel"].cpu().numpy())
    sf.write(folder / f"{filename}.wav", output["waveform"], 22050, "PCM_24")
    return folder.resolve() / f"{filename}.wav"


def validate_args(args):
    assert (
        args.text or args.file
    ), "Either text or file must be provided Matcha-T(ea)TTS need sometext to whisk the waveforms."
    assert args.temperature >= 0, "Sampling temperature cannot be negative"
    assert args.steps > 0, "Number of ODE steps must be greater than 0"

    if args.checkpoint_path is None:
        # When using pretrained models
        if args.model in SINGLESPEAKER_MODEL:
            args = validate_args_for_single_speaker_model(args)

        if args.model in MULTISPEAKER_MODEL:
            args = validate_args_for_multispeaker_model(args)
    else:
        # When using a custom model
        # 【中文说明】自定义模型未指定声码器时默认 hifigan_T2_v1（LJ Speech 专用），避免后续 KeyError
        if args.vocoder is None:
            args.vocoder = "hifigan_T2_v1"
        # 【中文说明】BigVGAN 是通用声码器，不会有 LJ Speech 专属警告
        if args.vocoder not in ("hifigan_univ_v1", BIGVGAN_VOCODER_NAME):
            warn_ = "[-] Using custom model checkpoint! I would suggest passing --vocoder hifigan_univ_v1, unless the custom model is trained on LJ Speech."
            warnings.warn(warn_, UserWarning)
        if args.speaking_rate is None:
            args.speaking_rate = 1.0

    if args.batched:
        assert args.batch_size > 0, "Batch size must be greater than 0"
    assert args.speaking_rate > 0, "Speaking rate must be greater than 0"

    return args


def validate_args_for_multispeaker_model(args):
    if args.vocoder is not None:
        if args.vocoder != MULTISPEAKER_MODEL[args.model]["vocoder"]:
            warn_ = f"[-] Using {args.model} model! I would suggest passing --vocoder {MULTISPEAKER_MODEL[args.model]['vocoder']}"
            warnings.warn(warn_, UserWarning)
    else:
        args.vocoder = MULTISPEAKER_MODEL[args.model]["vocoder"]

    if args.speaking_rate is None:
        args.speaking_rate = MULTISPEAKER_MODEL[args.model]["speaking_rate"]

    spk_range = MULTISPEAKER_MODEL[args.model]["spk_range"]
    if args.spk is not None:
        assert (
            args.spk >= spk_range[0] and args.spk <= spk_range[-1]
        ), f"Speaker ID must be between {spk_range} for this model."
    else:
        available_spk_id = MULTISPEAKER_MODEL[args.model]["spk"]
        warn_ = f"[!] Speaker ID not provided! Using speaker ID {available_spk_id}"
        warnings.warn(warn_, UserWarning)
        args.spk = available_spk_id

    return args


def validate_args_for_single_speaker_model(args):
    if args.vocoder is not None:
        if args.vocoder != SINGLESPEAKER_MODEL[args.model]["vocoder"]:
            warn_ = f"[-] Using {args.model} model! I would suggest passing --vocoder {SINGLESPEAKER_MODEL[args.model]['vocoder']}"
            warnings.warn(warn_, UserWarning)
    else:
        args.vocoder = SINGLESPEAKER_MODEL[args.model]["vocoder"]

    if args.speaking_rate is None:
        args.speaking_rate = SINGLESPEAKER_MODEL[args.model]["speaking_rate"]

    if args.spk != SINGLESPEAKER_MODEL[args.model]["spk"]:
        warn_ = f"[-] Ignoring speaker id {args.spk} for {args.model}"
        warnings.warn(warn_, UserWarning)
        args.spk = SINGLESPEAKER_MODEL[args.model]["spk"]

    return args


@torch.inference_mode()
def cli():
    # 【中文说明】中文 Windows 控制台默认 GBK 编码，打印 emoji（🍵）会抛 UnicodeEncodeError；
    # 改为"替换容错"：不可编码字符降级为 ?，中文正常输出
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(
        description=" 🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="matcha_ljspeech",
        help="Model to use",
        choices=MATCHA_URLS.keys(),
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the custom model checkpoint",
    )

    parser.add_argument(
        "--vocoder",
        type=str,
        default=None,
        help="Vocoder to use (default: will use the one suggested with the pretrained model))",
        choices=sorted(vocoder_registry.VOCODER_REGISTRY.keys()),  # 【中文说明】选项自动来自注册表
    )
    parser.add_argument(
        "--cleaner",
        type=str,
        default="english_cleaners2",
        # 【中文说明】必须与训练数据配置里的 cleaners 一致（如 piper 系模型用 english_cleaners_piper）
        help="Text cleaner used for phonemisation; MUST match the one used at training (default: english_cleaners2)",
    )
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--file", type=str, default=None, help="Text file to synthesize")
    parser.add_argument("--spk", type=int, default=None, help="Speaker ID")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.667,
        help="Variance of the x0 noise (default: 0.667)",
    )
    parser.add_argument(
        "--speaking_rate",
        type=float,
        default=None,
        help="change the speaking rate, a higher value means slower speaking rate (default: 1.0)",
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of ODE steps  (default: 10)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for inference (default: use GPU if available)")
    parser.add_argument(
        "--denoiser_strength",
        type=float,
        default=0.00025,
        help="Strength of the vocoder bias denoiser (default: 0.00025)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=os.getcwd(),
        help="Output folder to save results (default: current dir)",
    )
    parser.add_argument("--batched", action="store_true", help="Batched inference (default: False)")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size only useful when --batched (default: 32)"
    )

    args = parser.parse_args()

    args = validate_args(args)
    device = get_device(args)
    print_config(args)
    paths = assert_required_models_available(args)

    if args.checkpoint_path is not None:
        print(f"[🍵] Loading custom model from {args.checkpoint_path}")
        paths["matcha"] = args.checkpoint_path
        args.model = "custom_model"

    model = load_matcha(args.model, paths["matcha"], device)
    vocoder, denoiser = load_vocoder(args.vocoder, paths["vocoder"], device)

    texts = get_texts(args)

    spk = torch.tensor([args.spk], device=device, dtype=torch.long) if args.spk is not None else None
    if len(texts) == 1 or not args.batched:
        unbatched_synthesis(args, device, model, vocoder, denoiser, texts, spk)
    else:
        batched_synthesis(args, device, model, vocoder, denoiser, texts, spk)


class BatchedSynthesisDataset(torch.utils.data.Dataset):
    def __init__(self, processed_texts):
        self.processed_texts = processed_texts

    def __len__(self):
        return len(self.processed_texts)

    def __getitem__(self, idx):
        return self.processed_texts[idx]


def batched_collate_fn(batch):
    x = []
    x_lengths = []

    for b in batch:
        x.append(b["x"].squeeze(0))
        x_lengths.append(b["x_lengths"])

    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    x_lengths = torch.concat(x_lengths, dim=0)
    return {"x": x, "x_lengths": x_lengths}


def batched_synthesis(args, device, model, vocoder, denoiser, texts, spk):
    total_rtf = []
    total_rtf_w = []
    processed_text = [process_text(i, text, "cpu", (args.cleaner,)) for i, text in enumerate(texts)]
    dataloader = torch.utils.data.DataLoader(
        BatchedSynthesisDataset(processed_text),
        batch_size=args.batch_size,
        collate_fn=batched_collate_fn,
        num_workers=8,
    )
    for i, batch in enumerate(dataloader):
        i = i + 1
        start_t = dt.datetime.now()
        b = batch["x"].shape[0]
        output = model.synthesise(
            batch["x"].to(device),
            batch["x_lengths"].to(device),
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=spk.expand(b) if spk is not None else spk,
            length_scale=args.speaking_rate,
        )

        output["waveform"] = to_waveform(output["mel"], vocoder, denoiser, args.denoiser_strength)
        t = (dt.datetime.now() - start_t).total_seconds()
        rtf_w = t * 22050 / (output["waveform"].shape[-1])
        print(f"[🍵-Batch: {i}] Matcha-TTS RTF: {output['rtf']:.4f}")
        print(f"[🍵-Batch: {i}] Matcha-TTS + VOCODER RTF: {rtf_w:.4f}")
        total_rtf.append(output["rtf"])
        total_rtf_w.append(rtf_w)
        for j in range(output["mel"].shape[0]):
            base_name = f"utterance_{j:03d}_speaker_{args.spk:03d}" if args.spk is not None else f"utterance_{j:03d}"
            length = output["mel_lengths"][j]
            new_dict = {"mel": output["mel"][j][:, :length], "waveform": output["waveform"][j][: length * 256]}
            location = save_to_folder(base_name, new_dict, args.output_folder)
            print(f"[🍵-{j}] Waveform saved: {location}")

    print("".join(["="] * 100))
    print(f"[🍵] Average Matcha-TTS RTF: {np.mean(total_rtf):.4f} ± {np.std(total_rtf)}")
    print(f"[🍵] Average Matcha-TTS + VOCODER RTF: {np.mean(total_rtf_w):.4f} ± {np.std(total_rtf_w)}")
    print("[🍵] Enjoy the freshly whisked 🍵 Matcha-TTS!")


def unbatched_synthesis(args, device, model, vocoder, denoiser, texts, spk):
    total_rtf = []
    total_rtf_w = []
    for i, text in enumerate(texts):
        i = i + 1
        base_name = f"utterance_{i:03d}_speaker_{args.spk:03d}" if args.spk is not None else f"utterance_{i:03d}"

        print("".join(["="] * 100))
        text = text.strip()
        text_processed = process_text(i, text, device, (args.cleaner,))

        print(f"[🍵] Whisking Matcha-T(ea)TS for: {i}")
        start_t = dt.datetime.now()
        output = model.synthesise(
            text_processed["x"],
            text_processed["x_lengths"],
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=spk,
            length_scale=args.speaking_rate,
        )
        output["waveform"] = to_waveform(output["mel"], vocoder, denoiser, args.denoiser_strength)
        # RTF with HiFiGAN
        t = (dt.datetime.now() - start_t).total_seconds()
        rtf_w = t * 22050 / (output["waveform"].shape[-1])
        print(f"[🍵-{i}] Matcha-TTS RTF: {output['rtf']:.4f}")
        print(f"[🍵-{i}] Matcha-TTS + VOCODER RTF: {rtf_w:.4f}")
        total_rtf.append(output["rtf"])
        total_rtf_w.append(rtf_w)

        location = save_to_folder(base_name, output, args.output_folder)
        print(f"[+] Waveform saved: {location}")

    print("".join(["="] * 100))
    print(f"[🍵] Average Matcha-TTS RTF: {np.mean(total_rtf):.4f} ± {np.std(total_rtf)}")
    print(f"[🍵] Average Matcha-TTS + VOCODER RTF: {np.mean(total_rtf_w):.4f} ± {np.std(total_rtf_w)}")
    print("[🍵] Enjoy the freshly whisked 🍵 Matcha-TTS!")


def print_config(args):
    print("[!] Configurations: ")
    print(f"\t- Model: {args.model}")
    print(f"\t- Vocoder: {args.vocoder}")
    print(f"\t- Temperature: {args.temperature}")
    print(f"\t- Speaking rate: {args.speaking_rate}")
    print(f"\t- Number of ODE steps: {args.steps}")
    print(f"\t- Speaker: {args.spk}")


def get_device(args):
    if torch.cuda.is_available() and not args.cpu:
        print("[+] GPU Available! Using GPU")
        device = torch.device("cuda")
    else:
        print("[-] GPU not available or forced CPU run! Using CPU")
        device = torch.device("cpu")
    return device


if __name__ == "__main__":
    cli()
