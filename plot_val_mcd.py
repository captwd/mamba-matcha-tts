# -*- coding: utf-8 -*-
"""画出 ESD 情感长跑的 val_mcd 曲线（含 LR 衰减里程碑）"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV = "/mnt/d/PycharmProjects/PythonProject9/Matcha-TTS/results/val_mcd_curve.csv"
PNG = "/mnt/d/PycharmProjects/PythonProject9/Matcha-TTS/results/val_mcd_curve.png"

df = pd.read_csv(CSV)
plt.figure(figsize=(11, 5))
plt.plot(df["step"], df["val_mcd"], lw=0.8, color="#2a6f97")
for x, label in [(600, "lr->5e-5"), (900, "lr->2.5e-5"), (1050, "lr->1.25e-5"), (1150, "lr->6.25e-6")]:
    plt.axvline(x, color="gray", ls="--", lw=0.8)
    plt.text(x + 5, df["val_mcd"].min() - 0.8, label, fontsize=8, color="gray")
i = df["val_mcd"].idxmin()
plt.scatter([df["step"][i]], [df["val_mcd"][i]], color="red", zorder=5)
plt.annotate("min %.2f @ep%d" % (df["val_mcd"][i], df["step"][i]),
             (df["step"][i], df["val_mcd"][i]),
             xytext=(df["step"][i] + 30, df["val_mcd"][i] - 1.5), fontsize=9, color="red")
plt.xlabel("epoch")
plt.ylabel("val MCD (dB)")
plt.title("ESD emotion long-run: val MCD per epoch (ep130-1199)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PNG, dpi=130)
print("PNG saved:", PNG)
v = df["val_mcd"].values
s = df["step"].values
for lo, hi, name in [(130, 600, "constant 130-600"), (600, 900, "decay 600-900"), (900, 1200, "decay 900-1199")]:
    m = (s >= lo) & (s < hi)
    print("%s: %.2f +/- %.2f" % (name, v[m].mean(), v[m].std()))
i_min, i_last = v.argmin(), len(v) - 1
print("min %.2f @ep%d | last %.2f @ep%d | tail(20ep) mean %.2f" % (
    v[i_min], s[i_min], v[i_last], s[i_last], v[-20:].mean()))
