# 吉他分轨轻量模型实验报告

日期：2026-09-04

## 结论

有更轻量、快很多的候选模型，但目前只能安全地把它们定义成“快速档”，不能替换默认高质量档。

- 木吉他：`mdx_6s_acoustic_guitar_anvuew.onnx`，27.15 MB。
- 电吉他：`mdx_6s_electric_guitar_anvuew.onnx`，27.15 MB。
- Lead/Rhythm：`demucs4_lead_rhythm_guitar_drypaint.ckpt`，109.82 MB。
- 三个候选权重合计 164.12 MB，现有三个权重合计 492.32 MB，缩小 66.66%。
- 在 RTX 3060 Laptop GPU 上，全轻量链的模型推理约 48.13 s；现有 HQ6 链是 858.18 s，提速 17.83 倍。
- 但是它们相对当前 HQ6 的输出变化很大；指定歌曲没有真值 stem，无法证明 SDR 不下降，所以不建议直接替换默认模型。

默认高质量档仍建议采用前一轮的“木/电 BS-RoFormer 共享主干 + 原 Lead Mel-Band RoFormer”，模型推理约 579.93 s。若产品确实需要速度，可以新增独立的“快速”档，而不是静默替换。

## 测试输入与环境

指定输入：

`C:\CloudMusic\新裤子 - 没有理想的人不伤心.mp3`

- SHA-256：`C3CD8B471B422DB2F424AD651246C23AD1827EA28293E0AD3D5ECA6594B34177`
- 它与仓库 `测试用例/新裤子 - 没有理想的人不伤心.mp3` 字节完全一致，已有 HQ6 结果可以直接作为参考。
- 时长：341.053333 s，15,040,452 frames，44.1 kHz，立体声。
- GPU：NVIDIA GeForce RTX 3060 Laptop GPU 6 GiB。
- PyTorch：2.11.0+cu130；CUDA：13.0；ONNX Runtime GPU：1.28.0。

所有新增脚本、模型和输出均放在本目录。没有修改 `python/`、`src/` 或任何依赖库源码。

## 现有三个模型究竟是什么

| 阶段 | 架构 | 参数量 | 权重大小 | 当前 HQ6 推理 |
|---|---|---:|---:|---:|
| 木吉他 | BS-RoFormer | 38,692,876 | 77,624,038 B | 320.65 s |
| 电吉他 | BS-RoFormer | 38,692,876 | 77,624,038 B | 336.37 s |
| Lead | Mel-Band RoFormer | 84,199,748 | 337,073,664 B | 201.16 s |

因此不是三个 BS-RoFormer；只有木吉他和电吉他是 BS-RoFormer，Lead 是 Mel-Band RoFormer。

木/电两个现有 checkpoint 的 `band_split`、全部 Transformer `layers` 和 `final_norm` 逐 tensor 完全一致，只有 `mask_estimators` 不同。完全相同的值占两个 checkpoint 总值数的 68.0443%。它们是“相同主干 + 不同输出头”，但两个完整模型并不完全一致。

前一轮共享主干实验中，电吉他 difference SNR 159.17 dB 表示候选实现与当前实现之间只剩约 `1.43e-9` RMS 的浮点舍入误差；这远低于 float32 音频的实际可闻尺度，可视为数值等价。它不是模型相对真值的 SDR，不能解释成分轨质量 159 dB。

## 轻量候选

两个 MDX-Net ONNX 模型均为 125 个 initializer、6,779,816 个权重/常量值，约为单个现有 BS-RoFormer 参数量的 17.5%。两者结构一致，但按层名归一化后，6,779,816 个值中只有 4 个常量值 bit-identical，不能像现有 BS 模型那样共享主干。

| 阶段 | 候选 | 权重大小 | 整曲推理 | 对当前阶段提速 | 与当前 HQ6 corr | difference SNR |
|---|---|---:|---:|---:|---:|---:|
| 木吉他 | MDX-Net 6s Acoustic | 27,147,460 B | 17.806 s | 18.01x | 0.5202 | 0.714 dB |
| 电吉他 | MDX-Net 6s Electric | 27,147,623 B | 17.570 s | 19.14x | 0.8441 | 4.701 dB |
| Lead，输入为当前 HQ6 Electric | HTDemucs4 Lead/Rhythm，batch=8 | 109,822,623 B | 13.086 s | 15.37x | 0.9360 | 8.284 dB |
| Lead，输入为轻量 MDX Electric | 同上 | 109,822,623 B | 12.750 s | 15.78x | 0.8834 | 6.106 dB |

`corr` 和 difference SNR 只量化候选输出与当前 HQ6 输出的差异，不是真值质量分数。这里的数值足以证明“不是等价替换”，但不能单凭它判断哪一个听感更好。

表中的 15--19 倍是“当前产品 HQ6 六次推理”对“候选公开配置单次推理”的实际档位倍率，并非纯架构倍率。按已有单次推理基线，现有 Electric BS 为 57.42 s，MDX Electric 为 17.57 s，模型/配置本身约快 3.27 倍；现有 Lead MBR 为 34.58 s，HTDemucs batch=8 为 13.09 s，约快 2.64 倍。其余收益来自候选不再执行 HQ6。若也给轻量模型做六次增强，耗时会相应上升，而且仍需要真值测试才能判断质量是否改善。

HTDemucs 原始 `lead + rhythm` 并不能精确重建输入 Electric：在轻量 Electric 上的重建 difference SNR 为 23.88 dB。因此若集成，应继续采用 `rhythm = electric - lead`，保证 BandBuddy 三轨内部严格一致。

## 三种产品档位

| 档位 | 木/电 | Lead | 模型推理 | 相对当前 858.18 s | 质量判断 |
|---|---|---|---:|---:|---|
| 当前实现 | 两个独立 BS，HQ6 | MBR，HQ6 | 858.18 s | 1.00x | 当前基准 |
| 推荐高质量 | 共享主干 BS，HQ6 | MBR，HQ6 | 579.93 s | 1.48x | 木 bit-exact；电 159.17 dB；Lead 不变 |
| 平衡候选 | 共享主干 BS，HQ6 | HTDemucs，单 pass batch=8 | 391.85 s | 2.19x | 木/电等价；Lead 明显变化，需试听/真值验证 |
| 极速候选 | 两个 MDX-Net | HTDemucs，单 pass batch=8 | 48.13 s | 17.83x | 全链均明显变化，只适合独立快速档 |

沿用现有流程中约 6.73 s 的解码、保存等开销估算，极速档进程总耗时约 54.86 s，即这首 5 分 41 秒歌曲可在约 55 秒内完成。

极速档最终输出相对当前 HQ6：

| 输出 | corr | difference SNR |
|---|---:|---:|
| Acoustic | 0.5202 | 0.714 dB |
| Electric（中间结果） | 0.8441 | 4.701 dB |
| Lead | 0.8834 | 6.106 dB |
| Rhythm（残差法） | 0.8136 | 4.034 dB |

## HTDemucs batch 大小

在相同当前 HQ6 Electric 输入上：

- batch=1：18.977 s，峰值 PyTorch CUDA 分配 273,303,552 B。
- batch=8：13.086 s，峰值 PyTorch CUDA 分配 1,332,570,624 B。
- batch=8 相对 batch=1 的 Lead difference SNR 为 75.20 dB、corr 为 0.999999985。

因此 6 GiB 显卡可把 batch=8 作为该轻量 HTDemucs 的速度配置；若显存较小，可回退 batch=1。批处理不改变模型语义，但浮点计算顺序会造成极小数值差异。

## 依赖与安装体积

MDX-Net CUDA 推理需要 `onnxruntime-gpu`。本实验通过普通 pip wheel 安装到隔离目录，没有修改依赖源码：

- `onnxruntime-gpu==1.28.0` 解压后约 306,445,858 B（292.25 MiB）。
- 当前提取运行时中的 CPU ONNX Runtime 约 36,350,768 B（34.67 MiB）。
- 若生产环境用 GPU 包替换 CPU 包，而不是两份并存，则新增运行时净增约 257.58 MiB。

所以“权重缩小 66.66%”不等于安装包也缩小 66.66%。在当前环境尚无 GPU ORT 的前提下，候选权重加 ORT 净增量约 414.10 MiB，较现有三权重 469.51 MiB 只少约 55.4 MiB。其核心收益是速度与显存/模型规模，不是安装包大小。

## 公开质量证据的边界

MVSepLess 的公开模型目录确实列出了两个 Anvuew MDX-Net acoustic/electric 模型和 Dry Paint Dealer Undr 的 Lead/Rhythm HTDemucs，但没有给这三个精确 checkpoint 的真值 SDR：

- https://huggingface.co/noblebarkrr/mvsepless_resources/blob/main/models.json

MVSep 公共接口显示：通用 Guitar 当前默认 BS Roformer SW 为 SDR 9.05；Lead/Rhythm 服务的两阶段方案为 SDR 9.21，一阶段为 9.02。但页面没有把这两个服务选项直接映射到本次三个公开 checkpoint，因此不能把 9.05、9.21 或 9.02 写到这些候选权重名下：

- https://www.mvsep.com/en/full_api
- https://mvsep.com/algorithms/17

MVSep 的 Guitar 验证集有 30 首约一分钟的 mixture，真值部分闭源，只能把结果提交到其服务端评分。用户指定歌曲同样没有木/电/Lead/Rhythm 真值：

- https://www.mvsep.com/quality_checker/custom_leaderboards

因此，在没有真值 SDR 和盲听评分前，结论必须保持为：这些模型快、轻、输出有效，但尚未证明“不降低效果”。

## 推荐决策

1. 默认档：落地前一轮共享 BS 主干方案，保留 Lead MBR；它是目前唯一有严格数值等价证据的方案。
2. 可选平衡档：只把 Lead MBR 换成 HTDemucs，继续使用 `rhythm = electric - lead`。先让用户试听本次整曲输出。
3. 可选极速档：MDX Acoustic + MDX Electric + HTDemucs Lead；UI 必须明确标为快速/预览质量。
4. 在任何候选升级为默认前，建立包含纯木吉他、clean/distorted electric、solo/rhythm 重叠场景的真值集，并记录 SDR、SI-SDR、bleed 与盲听 MOS。
5. 三个候选权重的公开仓库没有清晰 model card/license；若要随安装包再分发，应先核实权重授权。运行时按固定哈希下载可降低再分发风险，但不能替代许可确认。

## 复现与试听

- `benchmark_mdxnet_candidates.py`：两个 MDX-Net 的完整整曲 CUDA 基准。
- `benchmark_htdemucs_lead_candidate.py`：HTDemucs Lead/Rhythm、batch 对照及残差一致性基准。
- `download_verified_ranges.ps1`：断点式分段下载，并按 Hugging Face 官方 LFS SHA-256 验证。
- `outputs/full-341s/mdxnet-benchmark.json`：木/电完整原始指标。
- `outputs/full-341s/htdemucs-lead-benchmark.json`：HTDemucs 使用当前 HQ6 Electric 的指标。
- `outputs/fast-chain-full-341s/htdemucs-lead-benchmark.json`：完整极速链指标。
- `outputs/full-341s/acoustic_guitar_mdxnet.wav`、`electric_guitar_mdxnet.wav`：轻量木/电试听。
- `outputs/fast-chain-full-341s/lead_guitar_htdemucs.wav`、`rhythm_guitar_residual.wav`：完整极速链 Lead/Rhythm 试听。
