# -*- coding: utf-8 -*-
"""
用 FunASR 批量抽取 emotion2vec 情感特征（句级/帧级）
======================================================
无需 fairseq：FunASR 自带 emotion2vec 重实现，可加载原始 fairseq 权重。

模型目录准备（一次性）:
    <model_dir>/
        config.yaml            # 从 ModelScope iic/emotion2vec_base 下载（仅几 KB）
        configuration.json     # 同上
        emotion2vec_base.pt    # 原始 fairseq 权重（会按 configuration.json 的 init_param 名字加载）

用法（在 Matcha-TTS 根目录运行）:
    python scripts/extract_emotion2vec_funasr.py \
        --wav_dir "D:/.../Emotion Speech Dataset" \
        --out_dir data/ESD/emo_feat \
        --model   "D:/.../emo2vec_base_funasr" \
        --speakers 0011-0020 \
        --granularity utterance \
        --batch_size 32

输出: out_dir/{utt_id}.npy
    utterance -> (768,)   整句向量（做条件用这个）
    frame     -> (T, 768) 帧级序列
"""
import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from funasr import AutoModel

TARGET_SR = 16000


def is_native_16k(path):
    info = sf.info(str(path))
    return info.samplerate == TARGET_SR and info.channels == 1


def resample_to_file(src, dst):
    wav, sr = sf.read(str(src), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torchaudio.functional.resample(
        torch.from_numpy(wav).unsqueeze(0), sr, TARGET_SR
    ).squeeze(0).numpy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), wav, TARGET_SR)


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
    parser = argparse.ArgumentParser(description="FunASR 批量抽取 emotion2vec 特征")
    parser.add_argument("--wav_dir", required=True, help="待处理音频目录（递归）")
    parser.add_argument("--out_dir", required=True, help="输出 npy 目录")
    parser.add_argument("--model", required=True,
                        help="模型目录（含 config.yaml/configuration.json/*.pt）或 ModelScope id")
    parser.add_argument("--speakers", default=None, help="可选说话人范围，如 0011-0020（按文件名前缀过滤）")
    parser.add_argument("--granularity", default="utterance", choices=["utterance", "frame"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefixes = None
    if args.speakers:
        lo, hi = args.speakers.split("-")
        prefixes = tuple(f"{i:04d}_" for i in range(int(lo), int(hi) + 1))

    wav_files = sorted(Path(args.wav_dir).rglob("*.wav"))
    if prefixes is not None:
        wav_files = [w for w in wav_files if w.stem.startswith(prefixes)]
    todo = [w for w in wav_files if not (out_dir / f"{w.stem}.npy").exists()]
    print(f"音频总数 {len(wav_files)}，待处理 {len(todo)}，模型={args.model}，设备={args.device}")

    if not todo:
        print("全部已存在，无需处理")
        return

    model = AutoModel(model=args.model, device=args.device, disable_update=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="emo2vec_16k_"))
    try:
        # 统一成 16k 路径列表；原生 16k 直接用，否则写临时文件
        items = []  # (stem, path)
        for wav in todo:
            if is_native_16k(wav):
                items.append((wav.stem, str(wav)))
            else:
                tmp = tmp_dir / f"{wav.stem}.wav"
                resample_to_file(wav, tmp)
                items.append((wav.stem, str(tmp)))

        done = 0
        for chunk in chunks(items, args.batch_size):
            stems = [s for s, _ in chunk]
            paths = [p for _, p in chunk]
            res = model.generate(
                input=paths,
                granularity=args.granularity,
                extract_embedding=True,
                batch_size=args.batch_size,
            )
            by_key = {}
            for r in res:
                k = str(r["key"])
                k = Path(k).stem if ("/" in k or "\\" in k) else k
                by_key[k] = r
            for stem, _ in chunk:
                feats = np.asarray(by_key[stem]["feats"], dtype=np.float32)
                np.save(out_dir / f"{stem}.npy", feats)
            done += len(chunk)
            if done % 1000 < args.batch_size:
                print(f"  已处理 {done}/{len(todo)}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"完成 -> {out_dir}")


if __name__ == "__main__":
    main()
