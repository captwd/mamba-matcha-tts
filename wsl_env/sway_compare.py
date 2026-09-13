import pandas as pd
from scipy.stats import wilcoxon

BASE = "results/sway_ab"
print(f"{'steps':<6}{'metric':<9}{'sway(-1.0)':>11}{'off':>10}{'delta':>10}{'p':>10}")
print("-" * 58)
for steps in (4, 6, 10):
    a = pd.read_csv(f"{BASE}/s{steps}_m1/results.csv").set_index("utt_id")
    b = pd.read_csv(f"{BASE}/s{steps}_off/results.csv").set_index("utt_id")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    for col, fmt in (("mcd_dtw", ".2f"), ("wer", ".4f"), ("cer", ".4f")):
        d = a[col].astype(float) - b[col].astype(float)
        try:
            p = wilcoxon(d, zero_method="wilcox").pvalue
        except Exception:
            p = float("nan")
        print(f"{steps:<6}{col:<9}{a[col].astype(float).mean():>11{fmt}}{b[col].astype(float).mean():>10{fmt}}{d.mean():>+10{fmt}}{p:>10.4f}")
    print("-" * 58)
