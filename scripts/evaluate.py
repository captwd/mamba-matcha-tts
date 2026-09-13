# -*- coding: utf-8 -*-
"""
批量评估脚本（Batch Evaluation）
=================================
给定 checkpoint + 声码器 + filelist，对每条样本：
    文本 -> Matcha 合成 mel -> MCD(合成mel, 真实mel)
    合成 mel -> 声码器 -> wav -> （可选）ASR 转写 -> WER / CER（与参考文本比）

输出：
    - 每条样本的合成 wav、mel npy（存到 --output_folder）
    - 每条样本的指标明细 + 汇总均值（results.csv）
    - 控制台汇总表

指标说明：
    MCD      越低越好（DTW 对齐，单位 dB；经验：<3 极好，3~6 可用）
    WER/CER  越低越好（wav2vec2-CTC 转写；同时报告 GT 音频的 WER 作为"ASR 上限"参照）

使用示例（LJSpeech 验证集前 10 条）：
    python scripts/evaluate.py ^
        --checkpoint_path logs\\train\\ljspeech_min\\runs\\...\\checkpoint_epoch=139.ckpt ^
        --vocoder bigvgan_base_22khz_80band ^
        --filelist data\\LJSpeech-1.1\\val.txt --data_root data\\LJSpeech-1.1 ^
        --output_folder results\\eval_lj139 --max_utts 10 --asr wav2vec2

依赖：torchaudio（仅 --asr wav2vec2 时需要；首次运行自动下载 ~360MB 模型）
"""
import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

# 【中文说明】让脚本无论从哪个目录启动都能 import matcha 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch
import torchaudio

from matcha.cli import BIGVGAN_VOCODER_NAME, VOCODER_URLS, get_device, load_matcha, load_vocoder, process_text, to_waveform
from matcha.utils.audio import mel_spectrogram
from matcha.utils.metrics import mel_cepstral_distortion, wer, cer
from matcha.utils.utils import assert_model_downloaded, get_user_data_dir

# 【中文说明】默认 mel 参数 = LJSpeech 系配置（configs/data/ljspeech.yaml），与训练一致
DEFAULT_MEL_PARAMS = dict(n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=8000)


class ASR:
    """【中文说明】torchaudio 预训练 wav2vec2 CTC 的贪心解码封装（懒加载，首次用时才下载权重）"""

    def __init__(self, device):
        self.device = device
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = None

    def _ensure_loaded(self):
        if self.model is None:
            print("[!] Loading wav2vec2 ASR (WAV2VEC2_ASR_BASE_960H)...")
            self.model = self.bundle.get_model().to(self.device).eval()

    @torch.inference_mode()
    def transcribe(self, wav, sample_rate):
        """【中文说明】wav (Tensor, 1D, float, [-1,1]) -> 大写英文字母+空格 的转写字符串"""
        self._ensure_loaded()
        if sample_rate != self.bundle.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.bundle.sample_rate)
        emission, _ = self.model(wav.unsqueeze(0).to(self.device))  # (T, C)
        # 【中文说明】CTC 贪心：逐帧取最大概率 -> 合并重复帧 -> 去掉 blank（index 0）-> "|" 映射成空格
        token_ids = emission[0].argmax(dim=-1).tolist()
        labels = self.bundle.get_labels()
        merged = []
        prev = -1
        for tid in token_ids:
            if tid != prev and tid != 0:
                merged.append(labels[tid])
            prev = tid
        return "".join(merged).replace("|", " ").strip()


def resolve_vocoder_path(vocoder_name):
    """【中文说明】HiFi-GAN 系声码器权重下载到用户数据目录；BigVGAN 走 HF 缓存不需要路径"""
    if vocoder_name == BIGVGAN_VOCODER_NAME:
        return None
    vocoder_path = get_user_data_dir() / vocoder_name
    assert_model_downloaded(vocoder_path, VOCODER_URLS[vocoder_name])
    return vocoder_path


def load_gt_mel(wav_path, mel_params):
    """【中文说明】读真实音频 -> 训练同款 mel_spectrogram -> (80, T) 对数梅尔 numpy"""
    wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav[:, 0])  # 【中文说明】取单声道
    if sr != mel_params["sampling_rate"]:
        wav = torchaudio.functional.resample(wav, sr, mel_params["sampling_rate"])
    mel = mel_spectrogram(
        wav.unsqueeze(0),
        mel_params["n_fft"],
        mel_params["num_mels"],
        mel_params["sampling_rate"],
        mel_params["hop_size"],
        mel_params["win_size"],
        mel_params["fmin"],
        mel_params["fmax"],
        center=False,
    ).squeeze(0)  # (80, T)
    return mel


def parse_filelist_line(line):
    """【中文说明】兼容 "wav|text" 与 "wav|text|normalized_text" 两种行格式（取最后一列文本）"""
    parts = [p.strip() for p in line.strip().split("|")]
    if len(parts) >= 3:
        return parts[0], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"Bad filelist line: {line!r}")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Matcha-TTS 批量评估：MCD + WER/CER")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Matcha-TTS checkpoint (.ckpt)")
    parser.add_argument("--vocoder", type=str, default=BIGVGAN_VOCODER_NAME, help="声码器名（注册表里的名字）")
    parser.add_argument("--cleaner", type=str, default="english_cleaners2", help="与训练一致的 cleaner")
    parser.add_argument("--filelist", type=str, required=True, help='filelist 路径，行格式 "wav|text[|normalized]"')
    parser.add_argument("--data_root", type=str, default=None, help="filelist 里 wav 相对路径的根目录")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--steps", type=int, default=10, help="ODE 步数")
    parser.add_argument("--temperature", type=float, default=0.667)
    parser.add_argument("--speaking_rate", type=float, default=1.0)
    parser.add_argument("--max_utts", type=int, default=None, help="只评估前 N 条（调试用）")
    parser.add_argument("--asr", type=str, default="wav2vec2", choices=["wav2vec2", "none"], help="是否启用 ASR 算 WER/CER")
    parser.add_argument("--mcd_align", type=str, default="dtw", choices=["dtw", "min_len"], help="MCD 帧对齐方式")
    parser.add_argument("--calibrate", action="store_true",
                        help="评估前先做声码器往返校准：GT mel -> 声码器 -> 重提 mel 的 MCD 下限（应接近几 dB）")
    parser.add_argument("--no_denoiser", action="store_true",
                        help="强制不使用去噪器（A/B 对照用；默认按注册表配方自动挂载）")
    parser.add_argument("--sway_sampling_coef", type=float, default=None,
                        help="Sway Sampling 系数（推理期非均匀时间步）：不给=关闭；建议 [-1, 0]，F5-TTS 默认 -1.0")
    parser.add_argument("--seed", type=int, default=1234,
                        help="每条语句的固定随机种子（保证不同配置间可逐句配对；0=不固定）")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
    print(f"[!] Device: {device}")

    # ---------- 加载模型与声码器（复用 cli.py 的逻辑） ----------
    model = load_matcha("custom_model", args.checkpoint_path, device)
    # 【中文说明】覆盖 CFM 的 sway 系数（推理期技巧，不需要重训；None 时保持配置里的值）
    if args.sway_sampling_coef is not None:
        model.decoder.sway_sampling_coef = args.sway_sampling_coef
    print(f"[!] sway_sampling_coef = {getattr(model.decoder, 'sway_sampling_coef', None)}"
          f" | steps = {args.steps} | seed = {args.seed}")
    vocoder_path = resolve_vocoder_path(args.vocoder)
    vocoder, denoiser = load_vocoder(args.vocoder, vocoder_path, device)
    if args.no_denoiser:
        denoiser = None
        print("[!] --no_denoiser：本次评估不使用去噪器")
    asr = ASR(device) if args.asr == "wav2vec2" else None

    # ---------- 读 filelist ----------
    data_root = Path(args.data_root) if args.data_root else Path(".")
    with open(args.filelist, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    if args.max_utts:
        lines = lines[: args.max_utts]
    print(f"[!] Evaluating {len(lines)} utterances")

    # ---------- 可选：声码器往返校准 ----------
    # 【中文说明】GT mel -> 声码器 -> wav -> 重提 mel 的 MCD，是当前提取器+声码器的"下限"。
    #   若这个值不低（比如 >10dB），说明声码器或 mel 参数有问题，后续合成 MCD 都不可信。
    if args.calibrate:
        first_wav = data_root / parse_filelist_line(lines[0])[0]
        ref_cal = load_gt_mel(first_wav, DEFAULT_MEL_PARAMS)
        wav_cal = vocoder(ref_cal.unsqueeze(0).to(device)).clamp(-1, 1)
        from matcha.utils.audio import mel_spectrogram as _ms
        gen_cal = _ms(
            wav_cal.squeeze(0).cpu(), DEFAULT_MEL_PARAMS["n_fft"], DEFAULT_MEL_PARAMS["num_mels"],
            DEFAULT_MEL_PARAMS["sampling_rate"], DEFAULT_MEL_PARAMS["hop_size"], DEFAULT_MEL_PARAMS["win_size"],
            DEFAULT_MEL_PARAMS["fmin"], DEFAULT_MEL_PARAMS["fmax"], center=False,
        ).squeeze(0)
        mcd_floor = mel_cepstral_distortion(gen_cal.numpy(), ref_cal.numpy(), align=args.mcd_align)
        print(f"[!] Vocoder round-trip MCD floor: {mcd_floor:.2f} dB (应远小于合成-vs-GT 的值)")

    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    rows = []

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["utt_id", "mcd_dtw", "wer", "cer", "gt_wer", "gen_len_frames", "ref_len_frames"])

        for idx, line in enumerate(lines):
            t0 = dt.datetime.now()
            wav_rel, text = parse_filelist_line(line)
            wav_path = data_root / wav_rel
            utt_id = Path(wav_rel).stem
            # ---------- 合成 ----------
            # 【中文说明】固定每条语句的噪声种子：同一 utt 在不同配置（步数/sway）下用同一份起点噪声，
            #   这样 MCD/WER 才能做逐句配对比较，而不是被采样随机性淹没。
            if args.seed:
                torch.manual_seed(args.seed + idx)
                torch.cuda.manual_seed_all(args.seed + idx)
            processed = process_text(idx, text, device, (args.cleaner,))
            output = model.synthesise(
                processed["x"],
                processed["x_lengths"],
                n_timesteps=args.steps,
                temperature=args.temperature,
                length_scale=args.speaking_rate,
            )
            gen_mel = output["mel"][0][:, : output["mel_lengths"][0]].cpu()  # (80, T_gen) 对数域
            waveform = to_waveform(output["mel"], vocoder, denoiser).cpu()

            # ---------- 真实音频的 mel ----------
            ref_mel = load_gt_mel(wav_path, DEFAULT_MEL_PARAMS)

            # ---------- MCD ----------
            mcd_value = mel_cepstral_distortion(gen_mel.numpy(), ref_mel.numpy(), align=args.mcd_align)

            # ---------- ASR：合成音频 & 真实音频 都转写 ----------
            if asr is not None:
                hyp = asr.transcribe(waveform, 22050)
                gt_wav, gt_sr = sf.read(wav_path, dtype="float32", always_2d=True)
                gt_hyp = asr.transcribe(torch.from_numpy(gt_wav[:, 0]), gt_sr)
                wer_value, cer_value, gt_wer_value = wer(text, hyp), cer(text, hyp), wer(text, gt_hyp)
            else:
                hyp, gt_hyp = "", ""
                wer_value, cer_value, gt_wer_value = float("nan"), float("nan"), float("nan")

            # ---------- 落盘 ----------
            sf.write(out_dir / f"{utt_id}.wav", waveform.numpy(), 22050, "PCM_24")
            np.save(out_dir / f"{utt_id}", gen_mel.numpy())
            writer.writerow([utt_id, f"{mcd_value:.3f}", f"{wer_value:.4f}", f"{cer_value:.4f}", f"{gt_wer_value:.4f}",
                             gen_mel.shape[1], ref_mel.shape[1]])
            csv_file.flush()
            elapsed = (dt.datetime.now() - t0).total_seconds()
            print(f"[{idx + 1}/{len(lines)}] {utt_id}  MCD={mcd_value:.2f}dB  WER={wer_value:.2f}  "
                  f"CER={cer_value:.2f}  GT-WER={gt_wer_value:.2f}  ({elapsed:.1f}s)")

    # ---------- 汇总 ----------
    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    mcd_all = np.atleast_1d(arr["mcd_dtw"])
    print("\n" + "=" * 60)
    print(f"[🍵] Summary over {len(mcd_all)} utts")
    print(f"    MCD ({args.mcd_align}): {np.nanmean(mcd_all):.2f} dB ± {np.nanstd(mcd_all):.2f}")
    if asr is not None:
        print(f"    WER : {np.nanmean(np.atleast_1d(arr['wer'])) * 100:.1f}%")
        print(f"    CER : {np.nanmean(np.atleast_1d(arr['cer'])) * 100:.1f}%")
        print(f"    GT  WER (ASR 上限参照): {np.nanmean(np.atleast_1d(arr['gt_wer'])) * 100:.1f}%")
    print(f"    详情: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
