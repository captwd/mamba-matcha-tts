# -*- coding: utf-8 -*-
"""
演示页音频准备脚本（Prepare Demo Audio）
==========================================
为 demo/index.html 生成试听音频：4 个系统 × 5 条句子，统一输出 PCM_16 / 22050Hz。

系统分组：
    gt              LJSpeech 真实录音（上限参照）
    ours_bigvgan    本复现模型(140ep) + BigVGAN
    ours_hifigan    本复现模型(140ep) + HiFi-GAN T2
    official_hifigan 官方预训练 matcha_ljspeech + HiFi-GAN T2
    convnext140_*   ConvNeXt V2 局部算子(140ep)
    mamba139_*      双向 Mamba2 全局混合器(140ep, 衰减 LR)

用法：python scripts/prepare_demo_audio.py
换模型/声码器后重跑本脚本即可刷新 demo 音频。
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf
import torch

from matcha.cli import load_matcha, load_vocoder, process_text

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo" / "audio"
VAL_FILELIST = ROOT / "data" / "LJSpeech-1.1" / "val.txt"
# 【中文说明】"ours" = U-Net 基线 ep140（demo 里的原始两组，保持不动）
OURS_CKPT = ROOT / "logs/train/ljspeech_min/runs/2026-09-05_12-02-02/checkpoints/checkpoint_epoch=139.ckpt"
OFFICIAL_CKPT = ROOT / "logs/train/ljspeech_min/runs/matcha_ljspeech.ckpt"
# 【中文说明】ConvNeXt V2 ep140（2026-09-06 追加的两组）
CONVNEXT_CKPT = ROOT / "logs/train/ljspeech_min/runs/2026-09-06_19-53-10/checkpoints/checkpoint_epoch=139.ckpt"
# 【中文说明】双向 Mamba2 ep139（2026-09-08 run，衰减 LR）
MAMBA_CKPT = ROOT / "logs/train/ljspeech_min/runs/2026-09-08_13-30-03/checkpoints/checkpoint_epoch=139.ckpt"

# 【中文说明】演示用的 5 条句子（来自 val 集，WER 与 GT-WER 均为 0，长度错开）
SELECTED_UTTS = ["LJ050-0184", "LJ009-0077", "LJ016-0002", "LJ010-0097", "LJ003-0136"]


def load_texts():
    """【中文说明】从 val.txt 取所选句子的文本，返回 {utt_id: (wav相对路径, text)}"""
    out = {}
    with open(VAL_FILELIST, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            utt_id = Path(parts[0]).stem
            if utt_id in SELECTED_UTTS:
                text = parts[2] if len(parts) >= 3 else parts[1]
                out[utt_id] = (parts[0], text)
    assert len(out) == len(SELECTED_UTTS), f"val.txt 里没找齐句子：{SELECTED_UTTS} vs {list(out)}"
    return {u: out[u] for u in SELECTED_UTTS}


def save_wav(wav, path):
    """【中文说明】统一存成 PCM_16 单声道 22050（demo 体积友好）"""
    sf.write(path, wav.squeeze().cpu().numpy(), 22050, "PCM_16")


def synth_group(name, ckpt, vocoder_name, items, device):
    """【中文说明】合成一组（5 句）试听音频"""
    out_dir = DEMO / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_matcha(name, str(ckpt), device)
    from matcha.utils.utils import get_user_data_dir

    voc_path = None if vocoder_name.startswith("bigvgan") else get_user_data_dir() / vocoder_name
    vocoder, denoiser = load_vocoder(vocoder_name, voc_path, device)
    for i, (utt_id, (_, text)) in enumerate(items.items(), start=1):
        p = process_text(i, text, device)
        out = model.synthesise(p["x"], p["x_lengths"], n_timesteps=10, temperature=0.667, length_scale=1.0)
        wav = vocoder(out["mel"]).clamp(-1, 1)
        if denoiser is not None:
            wav = denoiser(wav.squeeze(), strength=0.00025).unsqueeze(0)
        save_wav(wav, out_dir / f"sentence_{i}.wav")
        print(f"[+] {name} sentence_{i} ({utt_id})")
    del model, vocoder
    torch.cuda.empty_cache()


def synth_gt(items):
    """【中文说明】GT 真实录音组"""
    gt_dir = DEMO / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for i, (utt_id, (wav_rel, _)) in enumerate(items.items(), start=1):
        wav, sr = sf.read(ROOT / wav_rel, dtype="float32", always_2d=True)
        if sr != 22050:
            import torchaudio

            wav = torchaudio.functional.resample(torch.from_numpy(wav[:, 0]), sr, 22050).unsqueeze(1).numpy()
        save_wav(torch.from_numpy(wav[:, 0]), gt_dir / f"sentence_{i}.wav")
    print("[+] gt done")


@torch.inference_mode()
def main():
    import argparse

    parser = argparse.ArgumentParser(description="生成/增补 demo 试听音频")
    parser.add_argument(
        "--systems",
        type=str,
        default=None,
        help="只重生成指定组（逗号分隔）：gt,ours_bigvgan,ours_hifigan,official_hifigan,"
        "convnext140_bigvgan,convnext140_hifigan,mamba139_bigvgan,mamba139_hifigan；默认全部",
    )
    args = parser.parse_args()
    only = set(args.systems.split(",")) if args.systems else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = load_texts()

    # ---------- 系统注册表：名字 -> (是否合成, ckpt, 声码器) ----------
    groups = {
        "gt": ("special",),
        "ours_bigvgan": (OURS_CKPT, "bigvgan_base_22khz_80band"),
        "ours_hifigan": (OURS_CKPT, "hifigan_T2_v1"),
        "official_hifigan": (OFFICIAL_CKPT, "hifigan_T2_v1"),
        "convnext140_bigvgan": (CONVNEXT_CKPT, "bigvgan_base_22khz_80band"),
        "convnext140_hifigan": (CONVNEXT_CKPT, "hifigan_T2_v1"),
        "mamba139_bigvgan": (MAMBA_CKPT, "bigvgan_base_22khz_80band"),
        "mamba139_hifigan": (MAMBA_CKPT, "hifigan_T2_v1"),
    }

    if only is None or "gt" in only:
        synth_gt(items)

    for name, spec in groups.items():
        if name == "gt" or (only is not None and name not in only):
            continue
        if only is not None and name not in only:
            continue
        ckpt, vocoder_name = spec
        synth_group(name, ckpt, vocoder_name, items, device)

    # ---------- 汇总检查 ----------
    for sub in sorted(p.name for p in DEMO.iterdir() if p.is_dir()):
        n = len(list((DEMO / sub).glob("*.wav")))
        print(f"    {sub}: {n} wavs")
    print("[✓] All demo audio ready ->", DEMO)


if __name__ == "__main__":
    main()
