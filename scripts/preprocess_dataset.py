# -*- coding: utf-8 -*-
"""
数据集离线预处理脚本（生成训练缓存，大幅加快每个 epoch 的速度）
====================================================================
训练时 dataloader 每轮都要对同样的数据重复做两件 CPU 重活：
  1. 读 wav -> 算梅尔频谱 -> 归一化        （本脚本存为 cache/mel/<样本名>.npy）
  2. 文本清洗 -> espeak 音素化 -> 转 ID    （本脚本存入 cache/phonemes.json）
提前算好存盘后，训练时直接读缓存，每轮大约快 30 秒。

使用方法（在 Matcha-TTS 根目录运行）:
    python scripts/preprocess_dataset.py --data_config configs/data/ljspeech.yaml
可选参数:
    --limit 50    每个文件列表只处理前 50 条（测试用）
    --workers 8   并行进程数（默认 8，内存紧张时调小）

特性:
    - 断点续跑：已处理的样本自动跳过，中断后重跑即可
    - 参数校验：缓存里存了 meta.json，训练配置与缓存参数不一致时自动禁用缓存并警告
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rootutils
import torchaudio as ta
from omegaconf import OmegaConf
from tqdm.auto import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from matcha.text import text_to_sequence
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import normalize
from matcha.utils.utils import intersperse


def process_one(item, params):
    """处理单个样本：梅尔频谱 + 音素化（计算逻辑与 TextMelDataset.get_mel/get_text 逐行一致）"""
    filepath, text = item
    stem = Path(filepath).stem

    # 断点续跑：梅尔谱已存且音素表已有 -> 跳过
    if (params["mel_dir"] / f"{stem}.npy").exists() and stem in params["done"]:
        return stem, None

    # ---- 梅尔频谱（与 TextMelDataset.get_mel 完全一致）----
    audio, sr = ta.load(filepath)
    if sr != params["sample_rate"]:
        raise ValueError(f"{filepath}: 采样率 {sr} != 配置 {params['sample_rate']}，请检查数据")
    mel = mel_spectrogram(
        audio,
        params["n_fft"],
        params["n_feats"],
        params["sample_rate"],
        params["hop_length"],
        params["win_length"],
        params["f_min"],
        params["f_max"],
        center=False,
    ).squeeze()
    mel = normalize(mel, params["mel_mean"], params["mel_std"])
    np.save(params["mel_dir"] / f"{stem}.npy", mel.cpu().numpy().astype(np.float32))

    # ---- 文本 -> 音素 ID 序列（与 TextMelDataset.get_text 完全一致）----
    seq, cleaned = text_to_sequence(text, list(params["cleaners"]))
    if params["add_blank"]:
        seq = intersperse(seq, 0)
    return stem, {"x": [int(i) for i in seq], "text": cleaned}


def parse_filelist(path, limit=None):
    """解析文件列表，兼容单说话人（wav|文本）与多说话人（wav|说话人|文本）格式"""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                items.append((parts[0], "|".join(parts[2:])))
            elif len(parts) == 2:
                items.append((parts[0], parts[1]))
    return items[:limit] if limit else items


def main():
    parser = argparse.ArgumentParser(description="离线预处理：生成梅尔谱与音素缓存")
    parser.add_argument("--data_config", default="configs/data/ljspeech.yaml", help="数据配置文件")
    parser.add_argument("--limit", type=int, default=None, help="每个文件列表最多处理多少条（测试用）")
    parser.add_argument("--workers", type=int, default=8, help="并行进程数")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.data_config)
    params = dict(
        n_fft=cfg.n_fft,
        n_feats=cfg.n_feats,
        sample_rate=cfg.sample_rate,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        mel_mean=float(cfg.data_statistics.mel_mean),
        mel_std=float(cfg.data_statistics.mel_std),
        cleaners=list(cfg.cleaners),
        add_blank=bool(cfg.add_blank),
    )

    # 缓存目录约定：数据集根目录/cache（与 dataloader 的查找逻辑一致）
    filelists = {"train": cfg.train_filelist_path, "val": cfg.valid_filelist_path}
    for name, path in filelists.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} 文件列表不存在: {path}")
    cache_dir = Path(cfg.train_filelist_path).resolve().parent / "cache"
    mel_dir = cache_dir / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)

    # 读取已有的音素缓存（断点续跑用）
    ph_path = cache_dir / "phonemes.json"
    existing = json.loads(ph_path.read_text(encoding="utf-8")) if ph_path.exists() else {}
    params["mel_dir"] = mel_dir
    params["done"] = set(existing.keys())

    # 合并去重两个文件列表的样本
    all_items, seen = [], set()
    for path in filelists.values():
        for it in parse_filelist(path, args.limit):
            stem = Path(it[0]).stem
            if stem not in seen:
                seen.add(stem)
                all_items.append(it)
    todo = [it for it in all_items if Path(it[0]).stem not in params["done"]]
    print(f"样本总数 {len(all_items)}，已缓存 {len(all_items) - len(todo)}，本次待处理 {len(todo)}")
    print(f"缓存目录: {cache_dir}")

    phonemes = dict(existing)
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for stem, result in tqdm(
                ex.map(process_one, todo, [params] * len(todo)),
                total=len(todo),
                desc="预处理进度",
                unit="条",
            ):
                if result is not None:
                    phonemes[stem] = result

    ph_path.write_text(json.dumps(phonemes, ensure_ascii=False), encoding="utf-8")
    meta = {k: v for k, v in params.items() if k not in ("mel_dir", "done")}
    meta["version"] = 1
    (cache_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"完成! 音素缓存共 {len(phonemes)} 条 -> {ph_path}")
    print(f"梅尔谱目录 -> {mel_dir}")
    print(f"参数记录   -> {cache_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
