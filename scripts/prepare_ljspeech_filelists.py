# -*- coding: utf-8 -*-
"""
LJSpeech 文件列表划分脚本（生成 train.txt / val.txt）
========================================================
把 metadata.csv（格式: id|原文|规范化文本）转换成 Matcha-TTS 训练需要的文件列表格式:

    data/LJSpeech-1.1/wavs/LJ001-0001.wav|规范化文本

说明:
1. 使用第 3 列（规范化文本），数字/缩写已转写为单词，是 TTS 训练的标准做法
2. 按 13000 训练 / 100 验证 划分（与官方 Matcha-TTS 文件列表一致）
3. 固定随机种子，保证每次划分结果相同
4. 路径为相对项目根目录的路径 -> 必须在 Matcha-TTS 根目录下启动训练:
       cd D:\\PycharmProjects\\PythonProject9\\Matcha-TTS
       python matcha/train.py experiment=matcha_ljspeech
"""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # Matcha-TTS 根目录
DATA_DIR = ROOT / "data" / "LJSpeech-1.1"
METADATA = DATA_DIR / "metadata.csv"
TRAIN_OUT = DATA_DIR / "train.txt"
VAL_OUT = DATA_DIR / "val.txt"

TRAIN_NUM = 13000  # 训练集条数（官方划分）
SEED = 1234


def main():
    entries = []
    with open(METADATA, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            utt_id, _, normalized_text = parts
            wav_rel = f"data/LJSpeech-1.1/wavs/{utt_id}.wav"  # 相对项目根目录
            entries.append(f"{wav_rel}|{normalized_text}")

    print(f"共读取 {len(entries)} 条数据")

    random.seed(SEED)
    random.shuffle(entries)

    train, val = entries[:TRAIN_NUM], entries[TRAIN_NUM:]
    TRAIN_OUT.write_text("\n".join(train) + "\n", encoding="utf-8")
    VAL_OUT.write_text("\n".join(val) + "\n", encoding="utf-8")

    print(f"训练集: {TRAIN_OUT}  ({len(train)} 条)")
    print(f"验证集: {VAL_OUT}  ({len(val)} 条)")


if __name__ == "__main__":
    main()
