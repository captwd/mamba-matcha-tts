# -*- coding: utf-8 -*-
"""
情感辅助分类器（Auxiliary Emotion Classifier）
================================================
用途：接在 CFM 预测出的"干净梅尔" x1_hat 上做情感分类，用真实情感标签产生交叉熵损失。
            因为 x1_hat 依赖情感条件 cond，该损失会反向逼迫生成器真正使用情感条件，
            避免模型忽略条件（"条件崩溃"）。

结构：3 层 1D 卷积 + (mean, max) 池化 + 线性头。参数量约几百 K，相对 U-Net 很轻。
"""
import torch
import torch.nn as nn


class EmotionClassifier(nn.Module):
    def __init__(self, n_feats=80, hidden=256, n_emotions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_feats, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden * 2, n_emotions)

    def forward(self, mel, mask):
        """mel: (B, n_feats, T) 预测梅尔；mask: (B, 1, T) 有效帧掩码 -> logits (B, n_emotions)"""
        h = self.net(mel * mask)
        if mask.shape[1] != h.shape[1]:
            mask = mask.expand(-1, h.shape[1], -1)
        h = h * mask
        denom = mask.sum(dim=-1).clamp(min=1.0)
        mean = h.sum(dim=-1) / denom
        maxp = h.masked_fill(mask == 0, -1e9).max(dim=-1).values
        return self.head(torch.cat([mean, maxp], dim=-1))
