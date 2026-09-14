# -*- coding: utf-8 -*-
"""
生成情感类别标签 {utt_id: 类别索引}（供情感辅助分类损失使用）
================================================================
用法:
    # ESD（按情感子目录名）
    python scripts/build_emo_labels.py --type esd \
        --src "D:/PycharmProjects/PythonProject7/emotion2vec/Emotion Speech Dataset" \
        --speakers 0011-0020 --out data/ESD/emo_labels.json

    # RAVDESS（按文件名第 3 段）
    python scripts/build_emo_labels.py --type ravdess \
        --src "D:/PycharmProjects/PythonProject7/emotion2vec/Radvess" \
        --out data/RAVDESS/emo_labels.json
"""
import argparse
import json
from pathlib import Path

ESD_EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
RAVDESS_EMOTIONS = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fearful", 7: "disgust", 8: "surprised",
}


def build_esd(src, speakers):
    prefixes = None
    if speakers:
        lo, hi = speakers.split("-")
        prefixes = tuple(f"{i:04d}_" for i in range(int(lo), int(hi) + 1))

    labels = {}
    for spk_dir in sorted(Path(src).glob("0*")):
        if not spk_dir.is_dir():
            continue
        for emo_dir in sorted(spk_dir.iterdir()):
            if not emo_dir.is_dir() or emo_dir.name not in ESD_EMOTIONS:
                continue
            idx = ESD_EMOTIONS.index(emo_dir.name)
            for wav in emo_dir.glob("*.wav"):
                if prefixes is not None and not wav.stem.startswith(prefixes):
                    continue
                labels[wav.stem] = idx
    return labels, len(ESD_EMOTIONS)


def build_ravdess(src):
    labels = {}
    for wav in Path(src).glob("Actor_*/*.wav"):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            continue
        emo = int(parts[2])
        labels[wav.stem] = emo - 1  # 0-indexed: neutral=0 ... surprised=7
    return labels, len(RAVDESS_EMOTIONS)


def main():
    parser = argparse.ArgumentParser(description="生成情感类别标签 json")
    parser.add_argument("--type", required=True, choices=["esd", "ravdess"])
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--speakers", default=None, help="ESD 说话人范围，如 0011-0020")
    args = parser.parse_args()

    if args.type == "esd":
        labels, n = build_esd(args.src, args.speakers)
        emotion_names = ESD_EMOTIONS
    else:
        labels, n = build_ravdess(args.src)
        emotion_names = [RAVDESS_EMOTIONS[i] for i in range(1, 9)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    print(f"类别数 {n}: {emotion_names}")
    print(f"标签 {len(labels)} 条 -> {out}")


if __name__ == "__main__":
    main()
