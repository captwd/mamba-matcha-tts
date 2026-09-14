# -*- coding: utf-8 -*-
"""
情感可控合成（推理端情感注入入口）
====================================
给 文本 + 情感条件 -> 合成带情感的语音。模型侧 cond/emo_scale 接口
（MatchaTTS.synthesise）已就绪，本脚本是第一个把线接上的调用入口。

情感条件三选一：
  1. --emo_npy xx.npy    已抽好的 emotion2vec 句级向量 (768,)，如 data/ESD/emo_feat/0011_000351.npy
  2. --emo_wav ref.wav   任意参考音频（现场用 FunASR emotion2vec 抽取，需 --emo_model 指模型目录）
  3. --emotion angry     类别中心向量（先用 --build_centroids 从训练特征生成 centroids.json）

--emo_scale：classifier-free guidance 强度。
  1.0 = 不引导（纯条件生成，默认）；>1 放大情感。建议从 {1.0, 1.5, 2.0, 3.0} 逐档试听。
  注意：当前实现是对整条 ODE 的最终解做线性外推（非逐步速度场引导），scale 过大
  可能越出训练分布出现伪影/过饱和，试听为准。

用法示例（在仓库根目录）：
  # 0) 一次性：从训练特征构建 5 类情感中心
  python scripts/synthesize_emo.py --checkpoint_path <ckpt> --build_centroids \
      --emo_feat_dir data/ESD/emo_feat --emo_labels data/ESD/emo_labels.json \
      --centroids_out data/ESD/emo_centroids.json

  # 1) 类别中心合成（最省事）
  python scripts/synthesize_emo.py --checkpoint_path <ckpt> \
      --text "I can't believe it!" --emotion surprise --emo_scale 2.0 \
      --spk 0 --output_folder results/synth_emo

  # 2) 参考音频合成（语气随参考走，最灵活；需 pip install funasr）
  python scripts/synthesize_emo.py --checkpoint_path <ckpt> \
      --text "..." --emo_wav ref_angry.wav \
      --emo_model /path/to/emo2vec_base_funasr --emo_scale 1.5

  # 3) 现成 npy 向量合成
  python scripts/synthesize_emo.py --checkpoint_path <ckpt> \
      --text "..." --emo_npy data/ESD/emo_feat/0011_000401.npy
"""
import argparse
import sys
from pathlib import Path

# 【中文说明】让脚本无论从哪个目录启动都能 import matcha 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch

from matcha.cli import BIGVGAN_VOCODER_NAME, VOCODER_URLS, load_matcha, load_vocoder, process_text, to_waveform
from matcha.utils.utils import assert_model_downloaded, get_user_data_dir


def resolve_vocoder_path(vocoder_name):
    """【中文说明】HiFi-GAN 系声码器权重按名下载到用户数据目录；BigVGAN 走 HF 缓存不需要路径"""
    if vocoder_name == BIGVGAN_VOCODER_NAME:
        return None
    vocoder_path = get_user_data_dir() / vocoder_name
    assert_model_downloaded(vocoder_path, VOCODER_URLS[vocoder_name])
    return vocoder_path

# ESD 情感类别（与 scripts/build_emo_labels.py 的索引顺序一致）
ESD_EMOTIONS = ["angry", "happy", "neutral", "sad", "surprise"]
TARGET_SR = 16000  # emotion2vec 输入采样率


def build_centroids(args):
    """【中文说明】按类别对训练特征取平均 -> {类别名: 768维向量}，存 json。
    推理时只需类别名即可拿到一个"典型情感向量"，不必找参考音频。"""
    import json

    emo_feat_dir = Path(args.emo_feat_dir)
    with open(args.emo_labels, encoding="utf-8") as f:
        labels = json.load(f)

    sums, counts = {}, {}
    for utt_id, idx in labels.items():
        feat_path = emo_feat_dir / f"{utt_id}.npy"
        if not feat_path.exists():
            continue
        vec = np.load(feat_path).astype(np.float32).reshape(-1)
        sums[idx] = sums.get(idx, 0.0) + vec
        counts[idx] = counts.get(idx, 0) + 1

    missing = [ESD_EMOTIONS[i] for i in range(len(ESD_EMOTIONS)) if i not in sums]
    if missing:
        raise SystemExit(f"[!] 以下类别没有任何特征，检查 emo_feat_dir/emo_labels: {missing}")

    centroids = {ESD_EMOTIONS[i]: (sums[i] / counts[i]).tolist() for i in sorted(sums)}
    out = Path(args.centroids_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(centroids, ensure_ascii=False), encoding="utf-8")
    print(f"[+] 类别中心 {len(centroids)} 个 -> {out}")
    for name, i in [(ESD_EMOTIONS[i], i) for i in sorted(sums)]:
        print(f"    {name:<10s} {counts[i]} 条")


def extract_emo_from_wav(wav_path, emo_model_dir, device):
    """【中文说明】任意参考音频 -> 16k 重采样 -> FunASR emotion2vec 句级向量 (768,)"""
    try:
        from funasr import AutoModel
    except ImportError as e:
        raise SystemExit("[!] --emo_wav 需要 funasr：pip install funasr（或改用 --emo_npy/--emotion）") from e

    import torchaudio

    wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # 立体声取均值
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, TARGET_SR).numpy()
    tmp = Path("_emo_ref_16k_tmp.wav")
    sf.write(tmp, wav, TARGET_SR)
    try:
        model = AutoModel(model=emo_model_dir, device=str(device), disable_update=True)
        res = model.generate(input=str(tmp), granularity="utterance", extract_embedding=True)
        return np.asarray(res[0]["feats"], dtype=np.float32).reshape(-1)
    finally:
        tmp.unlink(missing_ok=True)


def load_condition(args, model, device):
    """【中文说明】按三种模式加载情感条件，返回 (768,) float32 向量 + 条件名（用于命名输出）"""
    emo_dim = getattr(model.hparams, "emo_dim", 0)
    if emo_dim <= 0:
        raise SystemExit("[!] 该 checkpoint 不带情感条件（emo_dim=0），请用带情感训练的 ckpt")

    if args.emo_npy:
        cond = np.load(args.emo_npy).astype(np.float32).reshape(-1)
        name = Path(args.emo_npy).stem
    elif args.emo_wav:
        if not args.emo_model:
            raise SystemExit("[!] --emo_wav 需同时给 --emo_model（FunASR emotion2vec 模型目录）")
        cond = extract_emo_from_wav(args.emo_wav, args.emo_model, device)
        name = Path(args.emo_wav).stem
    elif args.emotion:
        import json

        if not Path(args.centroids).exists():
            raise SystemExit(f"[!] 找不到 {args.centroids}，先跑 --build_centroids 生成")
        with open(args.centroids, encoding="utf-8") as f:
            centroids = json.load(f)
        if args.emotion not in centroids:
            raise SystemExit(f"[!] 未知情感 '{args.emotion}'，可选: {list(centroids)}")
        cond = np.asarray(centroids[args.emotion], dtype=np.float32)
        name = args.emotion
    else:
        raise SystemExit("[!] 必须指定情感条件之一：--emo_npy / --emo_wav / --emotion")

    if cond.shape[0] != emo_dim:
        raise SystemExit(f"[!] 情感向量维度 {cond.shape[0]} 与模型 emo_dim={emo_dim} 不一致")
    return cond, name


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Matcha-TTS 情感可控合成")
    parser.add_argument("--checkpoint_path", required=True, help="带情感训练的 checkpoint (.ckpt)")
    parser.add_argument("--vocoder", default="hifigan_T2_v1", help="声码器名（与评估协议一致默认 HiFi-GAN T2）")
    parser.add_argument("--cleaner", default="english_cleaners2", help="与训练一致的 cleaner")
    parser.add_argument("--text", action="append", default=None, help="待合成文本，可重复传多次")
    parser.add_argument("--file", default=None, help="批量文本文件（每行一句）")
    # 情感条件三选一
    emo_group = parser.add_argument_group("emotion condition（三选一）")
    emo_group.add_argument("--emo_npy", default=None, help="现成 emotion2vec 句级向量 .npy")
    emo_group.add_argument("--emo_wav", default=None, help="参考音频（现场抽取特征）")
    emo_group.add_argument("--emo_model", default=None, help="FunASR emotion2vec 模型目录（--emo_wav 时必填）")
    emo_group.add_argument("--emotion", default=None, help="类别名（如 angry），需 --centroids")
    emo_group.add_argument("--centroids", default="data/ESD/emo_centroids.json", help="类别中心 json 路径")
    emo_group.add_argument("--build_centroids", action="store_true", help="只构建类别中心后退出")
    emo_group.add_argument("--emo_feat_dir", default="data/ESD/emo_feat", help="构建中心用的特征目录")
    emo_group.add_argument("--emo_labels", default="data/ESD/emo_labels.json", help="构建中心用的标签 json")
    emo_group.add_argument("--centroids_out", default="data/ESD/emo_centroids.json", help="中心输出路径")
    # 合成参数
    parser.add_argument("--emo_scale", type=float, default=1.0,
                        help="CFG 强度：1.0=不引导；>1 放大情感（建议试 1.5/2.0/3.0）")
    parser.add_argument("--spk", type=int, default=0, help="说话人 id（多说话人 ckpt 时用，ESD 0~9）")
    parser.add_argument("--steps", type=int, default=10, help="ODE 步数")
    parser.add_argument("--temperature", type=float, default=0.667)
    parser.add_argument("--speaking_rate", type=float, default=1.0, help="<1 更慢，>1 更快")
    parser.add_argument("--sway_sampling_coef", type=float, default=None,
                        help="Sway 系数（推理期技巧，免重训）；不给=关闭")
    parser.add_argument("--no_denoiser", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_folder", default="results/synth_emo")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # ---------- 只构建类别中心 ----------
    if args.build_centroids:
        build_centroids(args)
        return

    # ---------- 收集文本 ----------
    texts = list(args.text or [])
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts += [ln.strip() for ln in f if ln.strip()]
    if not texts:
        raise SystemExit("[!] 请用 --text 或 --file 给至少一句文本")

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
    print(f"[!] Device: {device} | emo_scale={args.emo_scale} | steps={args.steps} | seed={args.seed}")
    torch.manual_seed(args.seed)

    # ---------- 模型 / 声码器 / 情感条件 ----------
    model = load_matcha("custom_model", args.checkpoint_path, device)
    if args.sway_sampling_coef is not None:
        model.decoder.sway_sampling_coef = args.sway_sampling_coef
    vocoder, denoiser = load_vocoder(args.vocoder, resolve_vocoder_path(args.vocoder), device)
    if args.no_denoiser:
        denoiser = None

    cond, cond_name = load_condition(args, model, device)
    cond = torch.from_numpy(cond)[None].to(device)  # (1, emo_dim)
    spks = None
    if getattr(model.hparams, "n_spks", 1) > 1:
        spks = torch.tensor([args.spk], dtype=torch.long, device=device)
        print(f"[!] 多说话人模型：spk={args.spk}")

    # ---------- 逐句合成 ----------
    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(texts):
        processed = process_text(i, text, device, (args.cleaner,))
        output = model.synthesise(
            processed["x"],
            processed["x_lengths"],
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=spks,
            length_scale=args.speaking_rate,
            cond=cond,
            emo_scale=args.emo_scale,
        )
        mel = output["mel"]
        waveform = to_waveform(mel, vocoder, denoiser).cpu()
        stem = f"{i:02d}_{cond_name}_s{args.emo_scale:g}"  # 带 scale，方便强度扫描不互相覆盖
        sf.write(out_dir / f"{stem}.wav", waveform.numpy(), 22050, "PCM_24")
        np.save(out_dir / stem, mel[0][:, : output["mel_lengths"][0]].cpu().numpy())
        print(f"[+] {stem}.wav  ({mel.shape[-1]} frames)  <- {text[:40]}")

    print(f"[🍵] 完成 -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
