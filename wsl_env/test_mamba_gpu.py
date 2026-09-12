import torch
import causal_conv1d, mamba_ssm
print("causal_conv1d:", causal_conv1d.__version__, "| mamba_ssm:", mamba_ssm.__version__)

from causal_conv1d import causal_conv1d_fn
x = torch.randn(2, 64, 100, device="cuda", dtype=torch.float16, requires_grad=True)
w = torch.randn(64, 4, device="cuda", dtype=torch.float16, requires_grad=True)
y = causal_conv1d_fn(x, w, bias=None, activation="silu")
y.sum().backward()
print("CAUSAL_CONV1D_CUDA_OK", tuple(y.shape))

from mamba_ssm import Mamba, Mamba2
for cls, d_state in ((Mamba, 16), (Mamba2, 64)):
    m = cls(d_model=128, d_state=d_state).cuda().to(torch.float16)
    inp = torch.randn(2, 50, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    out = m(inp)
    out.sum().backward()
    assert inp.grad is not None and torch.isfinite(out).all()
    print(cls.__name__, "FWD_BWD_OK", tuple(out.shape))

print("MAMBA_ALL_OK")
