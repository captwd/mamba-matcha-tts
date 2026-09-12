# -*- coding: utf-8 -*-
"""
参数漂移归因分析（Analyze Checkpoint Drift）
=============================================
回答"为什么恒定 LR 会导致细节质量震荡"的第一层证据：
把两个 checkpoint 之间的参数位移做分解，看漂移集中在谁身上。

核心思想：
    MCD 震荡 ≈ 生成细节随 checkpoint 摆动
    若震荡由"敏感小参数"（GRN gamma/beta、LayerScale、time_mlp）驱动，
    它们的相对漂移会远大于大权重层；
    若漂移均匀分布在大卷积/注意力权重上，则指向整体权重游走（LR 未衰减的直接后果）。

用法：
    python scripts/analyze_drift.py <ckpt_a> <ckpt_b> [--top 20]
示例（对比震荡波谷与波峰）：
    python scripts/analyze_drift.py ^
      logs/train/ljspeech_min/runs/2026-09-06_17-01-07/checkpoints/checkpoint_epoch=099.ckpt ^
      logs/train/ljspeech_min/runs/2026-09-06_19-53-10/checkpoints/checkpoint_epoch=139.ckpt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

# 【中文说明】模块分组规则：按 key 前缀/子串归类，顺序即优先级（先命中先归组）
GROUP_RULES = [
    ("encoder (文本编码器)", ("encoder.",)),
    ("estimator.time_mlp (时间嵌入)", ("decoder.estimator.time_mlp",)),
    ("estimator GRN gamma/beta", ("grn.gamma", "grn.beta")),
    ("estimator LayerScale", ("layer_scale",)),
    ("estimator Transformer 注意力", ("attn1.", "attn2.", "norm1", "norm3", "ff.net")),
    ("estimator Conv/Linear 大权重", (".dwconv", ".pw1", ".pw2", ".stem", "res_conv", "down_blocks", "mid_blocks", "up_blocks", "final_block", "final_proj")),
    ("其余", (),),
]


def group_of(key):
    for name, subs in GROUP_RULES:
        if any(s in key for s in subs):
            return name
    return "其余"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_a", help="基准 checkpoint（如 ep100）")
    parser.add_argument("ckpt_b", help="对比 checkpoint（如 ep140）")
    parser.add_argument("--top", type=int, default=20, help="打印漂移最大的参数个数")
    args = parser.parse_args()

    print(f"[!] Loading {args.ckpt_a}")
    a = torch.load(args.ckpt_a, map_location="cpu", weights_only=False)["state_dict"]
    print(f"[!] Loading {args.ckpt_b}")
    b = torch.load(args.ckpt_b, map_location="cpu", weights_only=False)["state_dict"]

    common = [k for k in a if k in b and torch.is_tensor(a[k]) and a[k].dtype.is_floating_point]
    print(f"[!] 可比浮点参数张量: {len(common)} / {len(a)}")

    # ---------- 分组漂移统计 ----------
    # 相对漂移 = ||ΔW||_2 / ||W_a||_2（按组聚合分子分母，避免小张量虚高）
    groups = {}
    rows = []
    for k in common:
        da = a[k].float()
        db = b[k].float()
        num = torch.linalg.vector_norm((db - da).flatten()).item()
        den = torch.linalg.vector_norm(da.flatten()).item() + 1e-12
        g = group_of(k)
        groups.setdefault(g, [0.0, 0.0])
        groups[g][0] += num**2
        groups[g][1] += den**2
        rows.append((num / (den + 1e-12), k, num, den))

    print("\n=== 模块组相对漂移（||ΔW|| / ||W||，越大越活跃）===")
    print("%-42s %10s" % ("模块组", "相对漂移"))
    for name, (num2, den2) in sorted(groups.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
        print("%-42s %10.4f" % (name, (num2**0.5) / (den2**0.5)))

    # ---------- 单参数 Top 漂移 ----------
    rows.sort(reverse=True)
    print(f"\n=== 相对漂移 Top{args.top} 单参数 ===")
    for rel, k, num, den in rows[: args.top]:
        print(f"{rel:8.4f}  |ΔW|={num:9.3f}  ||W||={den:9.3f}  {k}")

    # ---------- 总漂移 ----------
    total_num = sum(r[2] ** 2 for r in rows) ** 0.5
    print(f"\n[✓] 全模型总漂移 ||ΔW||_2 = {total_num:.3f}")
    print("解读提示：把 GRN/LayerScale 组与 Conv 大权重组的相对漂移对比——")
    print("  若小参数组显著更活跃 → 细节质量由这些敏感参数的摆动驱动（LR 衰减直接受益对象）")
    print("  若均匀分布 → 整体权重游走，LR 衰减通过收缩总位移起效（两种机制都支持观察到的震荡）")


if __name__ == "__main__":
    main()
