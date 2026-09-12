#!/bin/bash
~/miniconda3/envs/matcha/bin/python - <<'PYEOF'
import pandas as pd
from scipy.stats import wilcoxon

mamba = pd.read_csv("/home/zkw/projects/Matcha-TTS/results/eval_mamba_ep139_hifigan/results.csv").set_index("utt_id")
W = "/mnt/d/PycharmProjects/PythonProject9/Matcha-TTS/results"
bases = {
    "U-Net ep140  (HiFi+Deno, 恒定LR)": f"{W}/eval_baseline_ep140_hifigan/results.csv",
    "ConvNeXt ep134 (HiFi+Deno, 衰减LR)": "/home/zkw/projects/Matcha-TTS/results/eval_decayconvnext_ep134_hifigan/results.csv",
}
for name, path in bases.items():
    b = pd.read_csv(path).set_index("utt_id")
    common = mamba.index.intersection(b.index)
    a, c = mamba.loc[common], b.loc[common]
    print(f"\n== Mamba vs {name.strip()} (n={len(common)}) ==")
    print(f"{'metric':<12}{'mamba':>10}{'base':>10}{'delta':>10}{'p':>10}")
    for col, dfmt in (("wer", 4), ("cer", 4), ("mcd_dtw", 2)):
        d = a[col].astype(float) - c[col].astype(float)
        try:
            p = wilcoxon(d, zero_method="wilcox").pvalue
        except Exception:
            p = float("nan")
        w = f"p={round(p,4)}"
        print(f"{col:<12}{round(a[col].astype(float).mean(), dfmt):>10}{round(c[col].astype(float).mean(), dfmt):>10}{round(d.mean(), dfmt):>10}{w:>10}")
PYEOF
