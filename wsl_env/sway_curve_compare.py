import pandas as pd
from scipy.stats import wilcoxon

CURVE = "results/sway_curve"
MODELS = ["convnext_decay134", "official"]
# (显示名, 目录A, 目录B)
PAIRS = [
    ("2步 sway vs 2步 关", "s2_m10", "s2_off"),
    ("2步 sway vs 10步 关", "s2_m10", "s10_off"),
    ("1步 vs 10步 关", "s1_off", "s10_off"),
    ("1步 vs 2步 sway", "s1_off", "s2_m10"),
]

for m in MODELS:
    print(f"\n===== {m} =====")
    print(f"{'对比':<22}{'A':>9}{'B':>9}{'Δ(A-B)':>10}{'p':>10}")
    print("-" * 60)
    for label, da, db in PAIRS:
        a = pd.read_csv(f"{CURVE}/{m}/{da}/results.csv").set_index("utt_id")
        b = pd.read_csv(f"{CURVE}/{m}/{db}/results.csv").set_index("utt_id")
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        d = a["mcd_dtw"].astype(float) - b["mcd_dtw"].astype(float)
        try:
            p = wilcoxon(d, zero_method="wilcox").pvalue
        except Exception:
            p = float("nan")
        print(f"{label:<22}{a['mcd_dtw'].astype(float).mean():>9.2f}{b['mcd_dtw'].astype(float).mean():>9.2f}"
              f"{d.mean():>+10.2f}{p:>10.4f}")
