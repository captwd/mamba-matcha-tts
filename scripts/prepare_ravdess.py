# -*- coding: utf-8 -*-
"""
RAVDESS 预处理脚本（重采样 48kHz -> 22050Hz + 生成 Matcha 文件列表）
=====================================================================
RAVDESS 原始文件为 48kHz 单声道 wav，文件名格式：
    03-01-06-01-02-01-12.wav
    |  |  |  |  |  |  └ Actor(01-24, 奇男偶女)
    |  |  |  |  |  └ Repetition
    |  |  |  |  └ Statement(01="Kids are talking by the door", 02="Dogs are sitting by the door")
    |  |  |  └ Intensity(01=normal, 02=strong)
    |  |  └ Emotion(01中性 02平静 03开心 04悲伤 05愤怒 06恐惧 07厌恶 08惊讶)
    |  └ Vocal channel(01=speech)
    └ Modality(03=audio-only)

用法（在 Matcha-TTS 根目录运行）:
    python scripts/prepare_ravdess.py \
        --src "D:/PycharmProjects/PythonProject7/emotion2vec/Radvess" \
        --out "data/RAVDESS"

输出:
    data/RAVDESS/wav/{utt_id}.wav      # 22050Hz 单声道
    data/RAVDESS/train.txt             # 格式: 路径|说话人ID|文本 (n_spks>1)
    data/RAVDESS/val.txt
"""
import argparse
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EMOTIONS = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fearful", 7: "disgust", 8: "surprised",
}
STATEMENTS = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}

TARGET_SR = 22050
VAL_RATIO = 0.1
SEED = 1234


def parse_filename(stem):
    """从文件名解析 (emotion, statement, actor)，非法返回 None"""
    parts = stem.split("-")
    if len(parts) != 7 or parts[0] != "03":
        return None
    emotion = int(parts[2])
    statement = parts[4]
    actor = int(parts[6])
    return emotion, statement, actor


def resample(src, dst, sr=TARGET_SR):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ar", str(sr), "-ac", "1", str(dst)],
        check=True,
    )
    return dst


def main():
    parser = argparse.ArgumentParser(description="RAVDESS 重采样 + 生成文件列表")
    parser.add_argument("--src", required=True, help="RAVDESS 原始目录（含 Actor_01..24）")
    parser.add_argument("--out", default="data/RAVDESS", help="输出目录")
    parser.add_argument("--workers", type=int, default=8, help="并行线程数")
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    wav_out = out_dir / "wav"
    wav_out.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(src_dir.glob("Actor_*/*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"未在 {src_dir} 找到 Actor_*/*.wav")

    entries = []
    tasks = []
    for wav in wav_files:
        parsed = parse_filename(wav.stem)
        if parsed is None:
            print(f"[跳过] 文件名不符合规范: {wav.name}")
            continue
        emotion, statement, actor = parsed
        dst = wav_out / f"{wav.stem}.wav"
        tasks.append((wav, dst))
        # 说话人 ID 需 0-indexed 以匹配 nn.Embedding(n_spks)
        text = STATEMENTS[statement]
        entries.append(f"{dst.as_posix()}|{actor - 1}|{text}")

    print(f"共 {len(tasks)} 个音频待重采样 -> {TARGET_SR}Hz")
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
