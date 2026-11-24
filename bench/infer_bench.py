"""ARTalk 推理时延基准（独立脚本，不依赖 learn-3dmm；只测系数生成，不渲染、不需要 FLAME 文件）。

在仓库根目录运行（需要 conda activate artalk，并已 bash setup/fetch_weights.sh）：
    python bench/infer_bench.py                                   # 默认：bench/audio 下 9 条音频，fp32，GPU
    python bench/infer_bench.py -a bench/audio/to_be_or.wav --repeats 10
    python bench/infer_bench.py --dtype bfloat16                  # 低精度对照
    python bench/infer_bench.py --device cpu --repeats 2          # CPU 对照

每条音频：先预热 --warmup 次，再计时 --repeats 次，报告中位数，JSON 与表情系数 .npz 存到 bench/results/。
ARTalk 推理每一级都取概率最大的比特（argmax），没有随机采样：同样输入每次结果完全相同，所以重复只为取耗时中位数。
分段计时（一次完整推理里）：
    audio_enc_s   wav2vec2 XLS-R 300M 音频编码器，所有 4 秒块的总和（比 DPT 的 HuBERT-base 大得多）
    ar_s          自回归解码（5 级尺度 1/5/25/50/100 的 12 层 Transformer 前向）加 VAE 解码，所有块的总和
    total_s       model.inference 端到端
    rtf           total_s / 音频时长
流式口径（ARTalkStreamer，块长固定 4 秒，不是真正的低延迟，见 docs/realtime.md）：
    chunk_compute_s   处理一个 4 秒整块的耗时，即“攒够一块之后”到“第一批帧出来”的计算延迟
    首帧理论延迟 ≈ 4 秒（等音频）+ chunk_compute_s（计算）+ 0.16 秒（CausalSavgolSmoother）
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math import gcd  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from artalk.models import BitwiseARModel  # noqa: E402
from artalk.streaming import ARTalkStreamer  # noqa: E402


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def gpu_snapshot() -> str:
    """nvidia-smi 当前占用，用来判断机器是否被别人的任务占用（GB10 统一内存，显存字段可能是 N/A）。"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{out} | apps: {apps or 'none'}"
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def load_audio(path: Path) -> torch.Tensor:
    # 不用 torchaudio.load：torchaudio 2.9 起它依赖 torchcodec；soundfile 能读 wav/flac/mp3
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        g = gcd(sr, 16000)
        wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
    return torch.from_numpy(wav)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", "-a", type=Path, nargs="+", default=sorted((ROOT / "bench/audio").glob("*")))
    p.add_argument("--style", default="natural_0", help="assets/style_motion/<名>.pt，none 用空风格")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=None, help="结果 JSON 路径，默认 bench/results/<时间>_<设备>_<dtype>.json")
    args = p.parse_args()

    dev = args.device
    use_amp = args.dtype != "float32" and dev.startswith("cuda")
    dtype = getattr(torch, args.dtype)
    amp = lambda: torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp)  # noqa: E731

    configs = json.loads((ROOT / "assets/config.json").read_text())
    configs["AR_CONFIG"]["AUDIO_ENCODER"] = "wav2vec"
    t0 = time.perf_counter()
    model = BitwiseARModel(configs).eval()
    model.load_state_dict(torch.load(ROOT / "assets/ARTalk_wav2vec.pt", map_location="cpu", weights_only=True),
                          strict=True)
    model.to(dev)
    sync(dev)
    load_s = time.perf_counter() - t0

    style = None
    if args.style != "none":
        style = torch.load(ROOT / f"assets/style_motion/{args.style}.pt", map_location="cpu", weights_only=True)
        style = style[None].to(dev)

    # 分段计时钩子：音频编码器前后同步并累计
    acc = {"audio_enc": 0.0}
    state = {}

    def pre(_m, _i):
        sync(dev)
        state["t"] = time.perf_counter()

    def post(_m, _i, _o):
        sync(dev)
        acc["audio_enc"] += time.perf_counter() - state["t"]

    model.audio_encoder.register_forward_pre_hook(pre)
    model.audio_encoder.register_forward_hook(post)

    result = {
        "time": datetime.now().isoformat(timespec="seconds"), "device": dev, "dtype": args.dtype, "style": args.style,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if dev.startswith("cuda") else None,
        "load_s": round(load_s, 2), "gpu_before": gpu_snapshot(), "runs": [],
    }
    motions = {}

    for audio_path in args.audio:
        audio = load_audio(audio_path)
        dur = len(audio) / 16000
        batch = {"audio": audio[None].to(dev), "style_motion": style}
        totals, encs = [], []
        motion = None
        for it in range(args.warmup + args.repeats):
            acc["audio_enc"] = 0.0
            sync(dev)
            t0 = time.perf_counter()
            with torch.no_grad(), amp():
                motion = model.inference(batch)[0]
            sync(dev)
            total = time.perf_counter() - t0
            if it >= args.warmup:
                totals.append(total)
                encs.append(acc["audio_enc"])
        med, enc = statistics.median(totals), statistics.median(encs)
        windows = -(-int(np.ceil(dur * 25)) // 100)  # 向上取整得到 4 秒块数
        motions[audio_path.stem] = motion.float().cpu().numpy()

        # 流式口径：整块耗时（feed 恰好 1 个 4 秒块；音频不足 4 秒时补零成一整块）
        chunk = audio[: 16000 * 4]
        chunk = torch.nn.functional.pad(chunk, (0, 16000 * 4 - len(chunk)))
        streamer = ARTalkStreamer(model, style_motion=style[0] if style is not None else None)
        chunk_times = []
        for it in range(args.warmup + args.repeats):
            streamer.reset()
            sync(dev)
            t0 = time.perf_counter()
            with torch.no_grad(), amp():
                streamer.feed(chunk.to(dev))
            sync(dev)
            if it >= args.warmup:
                chunk_times.append(time.perf_counter() - t0)

        run = {
            "audio": str(audio_path), "audio_seconds": round(dur, 3), "frames": int(motion.shape[0]),
            "chunks": windows, "total_s": round(med, 4), "total_all_s": [round(x, 4) for x in totals],
            "audio_enc_s": round(enc, 4), "ar_s": round(med - enc, 4), "rtf": round(med / dur, 4),
            "chunk_compute_s": round(statistics.median(chunk_times), 4),
        }
        print(f"[{dev}/{args.dtype}] {audio_path.name:26s} {dur:5.1f}s chunks={windows} total={med:.3f}s "
              f"audio_enc={enc:.3f}s ar={med - enc:.3f}s rtf={run['rtf']:.3f} chunk={run['chunk_compute_s']:.3f}s")
        result["runs"].append(run)

    result["gpu_after"] = gpu_snapshot()
    if dev.startswith("cuda"):
        result["peak_memory_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)

    out_dir = ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or out_dir / f"{stamp}_{dev.replace(':', '')}_{args.dtype}.json"
    np.savez_compressed(out.with_suffix(".npz"), **motions)
    result["motion_npz"] = out.with_suffix(".npz").name
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
