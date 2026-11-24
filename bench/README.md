# bench：ARTalk 时延基准

在仓库根目录、`conda activate artalk` 并运行过 `bash setup/fetch_weights.sh` 之后：

```bash
python bench/infer_bench.py                                    # 9 条音频，fp32，GPU
python bench/infer_bench.py -a bench/audio/to_be_or.wav --repeats 10
python bench/infer_bench.py --dtype bfloat16                   # 低精度对照
python bench/infer_bench.py --device cpu --repeats 2           # CPU 对照
```

`bench/audio/` 里的 9 条音频与 DiffPoseTalk fork 的 `demo/input/audio/` 相同，两个模型的时延可以同口径对比。结果写入 `bench/results/`（不进 git）：JSON（耗时、`nvidia-smi` 前后快照、环境版本）加同名 `.npz`（各音频的 106 维系数）。

## 指标

| 字段 | 含义 |
|---|---|
| `total_s` | `BitwiseARModel.inference` 端到端中位数 |
| `audio_enc_s` | wav2vec2 XLS-R 300M 音频编码器，所有 4 秒块总和（DiffPoseTalk 用的是小得多的 HuBERT-base） |
| `ar_s` | `total_s - audio_enc_s`：自回归解码（5 级尺度、12 层 Transformer）加 VAE 解码 |
| `rtf` | `total_s` / 音频时长，小于 1 表示快于实时 |
| `chunk_compute_s` | 用 `ARTalkStreamer` 处理一个 4 秒整块的耗时 |

ARTalk 的“流式”接口只是把离线推理拆成 `feed / finish`，输出与离线逐位相同，块长固定 4 秒（见 `docs/realtime.md`）。所以首帧延迟 ≈ 4 秒等音频 + `chunk_compute_s` + 0.16 秒平滑器延迟，计算再快也压不到 4 秒以下。

## 与 DiffPoseTalk 对比时注意

- ARTalk 推理没有随机采样，每次结果完全相同；DiffPoseTalk 有扩散采样，且耗时随步数线性增长。
- 两者窗口都是 4 秒 / 100 帧 / 25 fps；ARTalk 输出 106 维（表情 100 + 头部旋转 3 + 下颌 3），DiffPoseTalk 输出 50 维表情加 4 维姿态。
- 参考基线（Mac M5 Pro CPU，约 4 秒音频）：ARTalk 约 1 秒，DiffPoseTalk 500 步约 26 秒。
- spark 可能多人共用：每次结果里有 `gpu_before` / `gpu_after`，别人有任务在跑时数据不可信，重测。
