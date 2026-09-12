"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.

【中文说明】PyTorch Lightning 模型的基类（BaseLightningClass）
所有模型（如 MatchaTTS）都继承自这个类，它封装了与模型结构无关的通用训练逻辑：
1. configure_optimizers：根据配置创建优化器(Optimizer)和学习率调度器(Scheduler)
2. training_step / validation_step：计算损失、记录指标（loss、grad_norm 等）
3. on_validation_epoch_end：在验证结束后把梅尔频谱图保存为日志图片，方便观察合成效果
"""
import inspect
from abc import ABC
from typing import Any, Dict

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from matcha import utils
from matcha.utils.metrics import mel_cepstral_distortion
from matcha.utils.model import denormalize
from matcha.utils.utils import plot_tensor

log = utils.get_pylogger(__name__)


class BaseLightningClass(LightningModule, ABC):
    """【中文说明】Lightning 训练逻辑基类（与模型结构解耦，可被任意 TTS 模型复用）。

    子类（MatchaTTS）需要提供：
        forward(x, x_lengths, y, y_lengths, spks, out_size, durations) -> 三个损失
        synthesise(...)                                                -> 推理输出
    本类提供：优化器配置、训练/验证步的损失统计、验证后的 tensorboard 可视化、梯度范数监控。
    """

    def update_data_statistics(self, data_statistics):
        """【中文说明】把训练集的 mel_mean/mel_std 注册为 buffer（随 checkpoint 保存、随设备迁移）。

        若未提供统计量则用 mean=0/std=1 的"恒等归一化"兜底。
        MatchaTTS.__init__ 末尾会调用本方法。
        """
        if data_statistics is None:
            data_statistics = {
                "mel_mean": 0.0,
                "mel_std": 1.0,
            }

        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))

    def configure_optimizers(self) -> Any:
        """【中文说明】按 hydra 配置创建优化器 + 学习率调度器（Lightning 标准钩子）。

        调度器配置形如：
            scheduler: {scheduler: {...}, lightning_args: {interval: epoch, frequency: 1}}
        对指数型调度器需要处理 last_epoch（从 checkpoint 恢复时接着训练不重启学习率）。
        """
        # 【中文说明】optimizer 是配置里存的"类 + 参数"，这里传入模型参数实例化
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler not in (None, {}):
            scheduler_args = {}
            # Manage last epoch for exponential schedulers
            # 【中文说明】若调度器签名含 last_epoch（如 ExponentialLR），
            #   则用恢复 checkpoint 时记录的 epoch 续接（on_load_checkpoint 里存的 ckpt_loaded_epoch）
            if "last_epoch" in inspect.signature(self.hparams.scheduler.scheduler).parameters:
                if hasattr(self, "ckpt_loaded_epoch"):
                    current_epoch = self.ckpt_loaded_epoch - 1
                else:
                    current_epoch = -1

            scheduler_args.update({"optimizer": optimizer})
            scheduler = self.hparams.scheduler.scheduler(**scheduler_args)
            scheduler.last_epoch = current_epoch
            # 【中文说明】返回字典形式：interval/frequency 来自配置（按 epoch 或 step 调度）
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler.lightning_args.interval,
                    "frequency": self.hparams.scheduler.lightning_args.frequency,
                    "name": "learning_rate",
                },
            }

        return {"optimizer": optimizer}

    def get_losses(self, batch):
        """【中文说明】从 dataloader 的 batch 取张量 -> 调用模型的 forward -> 收集三个损失。

        batch 字段（由 TextMelDataModule 提供）：
            x/x_lengths: 音素 ID 序列及长度；y/y_lengths: 归一化梅尔及长度
            spks: 说话人 ID（单说话人为 None）；durations: 外部时长（默认路径不用）
        """
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch["spks"]

        dur_loss, prior_loss, diff_loss, *_ = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            spks=spks,
            out_size=self.out_size,
            durations=batch["durations"],
        )
        return {
            "dur_loss": dur_loss,
            "prior_loss": prior_loss,
            "diff_loss": diff_loss,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """【中文说明】恢复 checkpoint 时记下当前 epoch，供 configure_optimizers 续接调度器"""
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

    def training_step(self, batch: Any, batch_idx: int):
        """【中文说明】训练步：算三项损失 -> 分别记录 -> 求和作为反传总损失。

        日志项：step（全局步数）、三项子损失（step+epoch 两个粒度）、总损失 loss/train。
        sync_dist=True：多卡训练时自动做跨进程平均。
        """
        loss_dict = self.get_losses(batch)
        self.log(
            "step",
            float(self.global_step),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "sub_loss/train_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        # 【中文说明】总损失 = 三项直接相加（权重均为 1）；返回给 Lightning 负责反传
        total_loss = sum(loss_dict.values())
        self.log(
            "loss/train",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": total_loss, "log": loss_dict}

    def validation_step(self, batch: Any, batch_idx: int):
        """【中文说明】验证步：与训练步同构，但只记录不反传；返回总损失供 Lightning 汇总。

        注意：验证时模型仍走训练路径 forward()（含 MAS 求对齐），因此 val loss
        的 diff_loss 会比训练时略高/有波动——MAS 每次都基于当前编码器输出重新算。
        """
        loss_dict = self.get_losses(batch)
        self.log(
            "sub_loss/val_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        # 【中文说明】验证集总损失
        total_loss = sum(loss_dict.values())
        self.log(
            "loss/val",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def on_validation_end(self) -> None:
        """【中文说明】验证结束后往 tensorboard 写"看得见"的图（只在主进程做一次）。

        epoch 0：额外记录两条真实梅尔（original/），方便与生成结果对照；
        每个 epoch：取验证集第一条样本，用当前模型 synthesise 10 步 ODE，
        记录 编码器梅尔 / 解码器梅尔 / 对齐矩阵 三张图——
        训练时观察"对齐是否学起来、梅尔是否清晰"主要就看这里。
        """
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"original/{i}",
                        plot_tensor(y.squeeze().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )

            log.debug("Synthesising...")
            # 【中文说明】MCD：合成(反归一化)mel vs 真实(反归一化)mel，DTW 对齐；逐样本记录再求均值
            mcd_values = []
            y_lengths = one_batch["y_lengths"]
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                spks = one_batch["spks"][i].unsqueeze(0).to(self.device) if one_batch["spks"] is not None else None
                # 【中文说明】固定 10 步 ODE，保证各 epoch 之间可视化可比
                output = self.synthesise(x[:, :x_lengths], x_lengths, n_timesteps=10, spks=spks)
                y_enc, y_dec = output["encoder_outputs"], output["decoder_outputs"]
                attn = output["attn"]
                self.logger.experiment.add_image(
                    f"generated_enc/{i}",
                    plot_tensor(y_enc.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"generated_dec/{i}",
                    plot_tensor(y_dec.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"alignment/{i}",
                    plot_tensor(attn.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                # ---------- val/MCD ----------
                # 【中文说明】两边都回到"对数梅尔"域再比较：
                #   合成侧 output["mel"] 已是 denormalize 后的对数梅尔；
                #   真实侧 y 是归一化域，需同样 denormalize；并裁到该样本真实帧数去掉 padding
                gen_mel = output["mel"][0][:, : output["mel_lengths"][0]]
                ref_mel = denormalize(one_batch["y"][i, :, : y_lengths[i]].to(self.device), self.mel_mean, self.mel_std)
                try:
                    mcd_value = mel_cepstral_distortion(gen_mel, ref_mel, align="dtw")
                    mcd_values.append(mcd_value)
                    self.logger.experiment.add_scalar(f"val_mcd/{i}", mcd_value, self.current_epoch)
                except Exception as e:  # pylint: disable=broad-except
                    # 【中文说明】MCD 失败不应中断训练（如某样本过短），记录警告并跳过
                    log.warning(f"MCD computation failed for sample {i}: {e}")
            if mcd_values:
                # 【中文说明】两条样本的均值；tensorboard 里看 val_mcd/mean 曲线，越低越好
                self.logger.experiment.add_scalar(
                    "val_mcd/mean", sum(mcd_values) / len(mcd_values), self.current_epoch
                )

    def on_before_optimizer_step(self, optimizer):
        """【中文说明】优化器步进前记录各参数组的 2-范数梯度——监控训练稳定性（梯度爆炸/消失）。

        注意：放在 optimizer step 之前，避免被梯度裁剪/累积影响测量值。
        """
        self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})
