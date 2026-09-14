# -*- coding: utf-8 -*-
"""
ESD (Emotional Speech Dataset) 预处理脚本
==========================================
ESD 目录结构:
    Emotion Speech Dataset/
      0001..0010/            # 中文说话人
      0011..0020/            # 英文说话人
        Angry/ Happy/ Neutral/ Sad/ Surprise/   # 5 情感
          0011_000351.wav ...
        0011.txt             # 转写: utt_id \t 文本 \t 情感 (1750 行)
  原始音频为 16kHz 单声道。

本脚本:
  1. 只取英文说话人（默认 0011-0020），重采样 16kHz -> 22050Hz
  2. 从 {speaker}.txt 读取文本
  3. 生成 Matcha 文件列表: 路径|说话人ID(0-indexed)|文本

用法（在 Matcha-TTS 根目录运行）:
    python scripts/prepare_esd.py \
        --src "D:/PycharmProjects/PythonProject7/emotion2vec/Emotion Speech Dataset" \
        --out "data/ESD"
"""
import argparse
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TARGET_SR = 22050
VAL_RATIO = 0.05
SEED = 1234


def read_transcripts(txt_path):
    """读取 {speaker}.txt -> {utt_id: text}；兼容 utf-8 / gbk 编码"""
    for enc in ("utf-8", "gbk"):
        try:
            lines = txt_path.read_text(encoding=enc).splitlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError(f"无法解码转写文件: {txt_path}")

    mapping = {}
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            mapping[parts[0]] = parts[1]
    return mapping


def resample(src, dst, sr=TARGET_SR):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ar", str(sr), "-ac", "1", str(dst)],
        check=True,
    )
    return dst


def main():
    parser = argparse.ArgumentParser(description="ESD 重采样 + 生成文件列表")
    parser.add_argument("--src", required=True, help="ESD 根目录")
    parser.add_argument("--out", default="data/ESD", help="输出目录")
    parser.add_argument("--speakers", default="0011-0020",
                        help="要使用的说话人范围，如 0011-0020（英文）")
    parser.add_argument("--workers", type=int, default=12, help="并行线程数")
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    wav_out = out_dir / "wav"
    wav_out.mkdir(parents=True, exist_ok=True)

    lo, hi = args.speakers.split("-")
    speakers = [f"{i:04d}" for i in range(int(lo), int(hi) + 1)]

    entries, tasks = [], []
    for spk_idx, spk in enumerate(speakers):
        spk_dir = src_dir / spk
        txt_path = spk_dir / f"{spk}.txt"
        if not txt_path.exists():
            print(f"[跳过] 缺少转写文件: {txt_path}")
            continue
        transcripts = read_transcripts(txt_path)

        wavs = sorted(spk_dir.glob("*/*.wav"))
        for wav in wavs:
            utt_id = wav.stem
            text = transcripts.get(utt_id)
            if text is None:
                continue
            dst = wav_out / f"{utt_id}.wav"
            tasks.append((wav, dst))
            entries.append(f"{dst.as_posix()}|{spk_idx}|{text}")

    print(f"说话人 {speakers[0]}-{speakers[-1]}，共 {len(tasks)} 个音频待重采样 -> {TARGET_SR}Hz")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda t: resample(t[0], t[1]), tasks))

    random.seed(SEED)
    random.shuffle(entries)
    n_val = max(1, int(len(entries) * VAL_RATIO))
    val, train = entries[:n_val], entries[n_val:]

    (out_dir / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")

    print(f"训练集 {len(train)} 条 -> {out_dir / 'train.txt'}")
    print(f"验证集 {len(val)} 条 -> {out_dir / 'val.txt'}")
    print(f"重采样音频 -> {wav_out}")


if __name__ == "__main__":
    main()
