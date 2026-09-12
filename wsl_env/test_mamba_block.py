"""Mamba block integration test: unit -> full Decoder -> init-skip -> hydra e2e."""
import math
import sys

import torch

sys.path.insert(0, "/home/zkw/projects/Matcha-TTS")

from matcha.models.components.decoder import Decoder
from matcha.models.components.mamba_block import MambaBlock1D

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print("device:", DEV)


def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    assert cond, name


# ---------- 1. unit: MambaBlock1D ----------
blk = MambaBlock1D(dim=256).to(DEV)
x = torch.randn(2, 128, 256, device=DEV, requires_grad=True)

blk.eval()
with torch.no_grad():
    y = blk(hidden_states=x, attention_mask=torch.ones(2, 128, device=DEV), timestep=None)
check("shape (B,T,C)", y.shape == x.shape)
check("identity at init (LayerScale 1e-6)", (y - x).abs().max().item() < 1e-3)

blk.train()
out = blk(hidden_states=x, attention_mask=None, timestep=None)
out.sum().backward()
check("backward grads finite", all(p.grad is None or torch.isfinite(p.grad).all() for p in blk.parameters()))
check("gamma receives grad", blk.gamma.grad is not None and blk.gamma.grad.abs().sum() > 0)

blk_f = MambaBlock1D(dim=256, bidirectional=False).to(DEV)
with torch.no_grad():
    _ = blk_f(hidden_states=x.detach(), timestep=None)
print("PASS forward-only mode runs")

# long odd-length sequence
with torch.no_grad():
    _ = blk(hidden_states=torch.randn(1, 777, 256, device=DEV), timestep=None)
print("PASS odd length T=777")

# ---------- 2. init-skip verification ----------
dec = Decoder(
    in_channels=160, out_channels=80,
    down_block_type="mamba", mid_block_type="mamba", up_block_type="mamba",
).to(DEV)
mamba0 = dec.down_blocks[0][1][0]
conv_bias = mamba0.mamba_fwd.conv1d.bias
check("mamba conv1d.bias untouched (nonzero, not Kaiming-zeroed)", conv_bias is not None and conv_bias.abs().sum().item() > 0)

# ---------- 3. integration: full Decoder with mamba ----------
B, T = 2, 128
xx = torch.randn(B, 80, T, device=DEV)
mu = torch.randn(B, 80, T, device=DEV)
mask = torch.ones(B, 1, T, device=DEV)
mask[:, :, -13:] = 0.0
t = torch.rand(B, device=DEV)

dec.train()
v = dec(x=xx, mask=mask, mu=mu, t=t)
check("decoder out shape (B,80,T)", v.shape == (B, 80, T))
check("decoder out finite", torch.isfinite(v).all().item())
v.sum().backward()
gnorms = [p.grad.abs().sum().item() for p in dec.parameters() if p.grad is not None]
check("decoder grads flow", len(gnorms) > 0 and math.isfinite(sum(gnorms)))

# ---------- 4. regression: default transformer path ----------
dec_def = Decoder(in_channels=160, out_channels=80).to(DEV)
dec_def.eval()
with torch.no_grad():
    v2 = dec_def(x=xx, mask=mask, mu=mu, t=t)
check("default (transformer) decoder still works", v2.shape == (B, 80, T) and torch.isfinite(v2).all().item())

# ---------- 5. param counts ----------
def n_params(m):
    return sum(p.numel() for p in m.parameters())

print(f"params: decoder[mamba]      = {n_params(dec)/1e6:.1f}M")
print(f"params: decoder[transformer]= {n_params(dec_def)/1e6:.1f}M")

# ---------- 6. hydra e2e: compose + instantiate ----------
from hydra import compose, initialize_config_dir
import hydra.utils
from omegaconf import OmegaConf

cfg_dir = "/home/zkw/projects/Matcha-TTS/configs"
with initialize_config_dir(config_dir=cfg_dir, version_base=None):
    cfg = compose(config_name="train", overrides=["model/decoder=mamba"])
print("composed decoder cfg:", OmegaConf.to_yaml(OmegaConf.select(cfg, "model.decoder"))[:300].replace("\n", " | "))
# 【说明】真实路径：MatchaTTS 把 decoder 组配置作为 decoder_params 字典喂给 CFM，
#   CFM 内部 Decoder(**decoder_params)。这里照抄并跑一次完整训练 loss：
from matcha.models.components.flow_matching import CFM

cfm_h = CFM(
    in_channels=160,  # 【说明】2*n_feats，与 matcha_tts.py:111 一致
    out_channel=80,
    cfm_params=OmegaConf.select(cfg, "model.cfm"),
    decoder_params=OmegaConf.select(cfg, "model.decoder"),
).to(DEV)
cfm_h.train()
loss, _y = cfm_h.compute_loss(x1=torch.randn(B, 80, T, device=DEV), mask=mask, mu=torch.randn(B, 80, T, device=DEV))
check("CFM(composed cfg) training loss finite", torch.isfinite(loss).item())
loss.backward()
gnorms_cfm = [p.grad.abs().sum().item() for p in cfm_h.parameters() if p.grad is not None]
check("CFM grads flow", len(gnorms_cfm) > 0 and math.isfinite(sum(gnorms_cfm)))
print(f"params: full CFM[mamba] = {n_params(cfm_h)/1e6:.1f}M")

print("MAMBA_BLOCK_ALL_OK")
