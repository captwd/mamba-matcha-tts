import os
import pandas as pd
from scipy.stats import wilcoxon

BASE = "results/sway_multi"
MODELS = ["unet139", "convnext139_const", "mamba139", "official"]
EXTRA = {"convnext_decay134": "results/sway_ab"}  # 之前跑过的（带 ASR 的那组）

print(f"{'model':<22}{'steps':>6}{'sway':>9}{'off':>9}{'delta':>9}{'p':>10}")
print("-" * 66)


def row(name, steps, off_dir, on_dir):
    a = pd.read_csv(f"{on_dir}/results.csv").set_index("utt_id")
    b = pd.read_csv(f"{off_dir}/results.csv").set_index("utt_id")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    col = "mcd_dtw"
    d = a[col].astype(float) - b[col].astype(float)
    try:
        p = wilcoxon(d, zero_method="wilcox").pvalue
    except Exception:
        p = float("nan")
    print(f"{name:<22}{steps:>6}{a[col].astype(float).mean():>9.2f}{b[col].astype(float).mean():>9.2f}"
          f"{d.mean():>+9.2f}{p:>10.4f}")


for m in MODELS:
    for steps in (4, 6, 10):
        row(m, steps, f"{BASE}/{m}/s{steps}_off", f"{BASE}/{m}/s{steps}_m1")
    print("-" * 66)

# 之前那组（含 WER/CER）
for steps in (4, 6, 10):
    row("convnext_decay134", steps, f"results/sway_ab/s{steps}_off", f"results/sway_ab/s{steps}_m1")
