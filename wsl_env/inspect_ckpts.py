import torch
from matcha.utils.utils import _torch_load_compat  # noqa: F401  （确保兼容补丁生效）

paths = {
    "official": "/mnt/d/PycharmProjects/PythonProject9/Matcha-TTS/logs/train/ljspeech_min/runs/matcha_ljspeech.ckpt",
    "unet139": "/mnt/d/PycharmProjects/PythonProject9/Matcha-TTS/logs/train/ljspeech_min/runs/2026-09-05_12-02-02/checkpoints/checkpoint_epoch=139.ckpt",
    "mamba139": "/home/zkw/projects/Matcha-TTS/logs/train/ljspeech_min/runs/2026-09-08_13-30-03/checkpoints/checkpoint_epoch=139.ckpt",
}

for name, p in paths.items():
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print(f"\n===== {name} =====")
    print("epoch:", ck.get("epoch"), "| global_step:", ck.get("global_step"))
    hp = ck.get("hyper_parameters", {})
    for k in ("n_feats", "out_size", "prior_loss", "use_precomputed_durations"):
        if k in hp:
            print(f"  {k}: {hp[k]}")
    ds = hp.get("data_statistics", None)
    if ds is not None:
        print("  data_statistics:", dict(ds) if hasattr(ds, "items") else ds)
    dec = hp.get("decoder", None)
    if dec is not None:
        try:
            print("  decoder cfm:", dict(dec.get("cfm_params", {})))
        except Exception:
            pass
    # 训练轮数相关
    for k in ("n_epochs", "max_epochs", "num_epochs"):
        if k in hp:
            print(f"  {k}: {hp[k]}")
    if "loops" in ck:
        try:
            ep = ck["loops"]["fit_loop"]["epoch_loop.state_dict"].get("_batches_that_stepped")
            print("  batches_that_stepped:", ep)
        except Exception:
            pass
