# BandBuddy 吉他分轨提速实验报告

日期：2026-09-04

## 结论

最值得落地的方案是把木吉他与电吉他两个 BS-RoFormer 合并成一个“共享主干、两个独立 mask head”的原生双 stem 模型。

- 不改第三方依赖源码，不增加 pip 依赖，不换权重，也不减少 HQ6 的六次推理。
- 341.05 秒整曲上，木吉他 + 电吉他由 **657.02 s 降至 378.77 s**，提速 **1.735x**，节省 **278.25 s**。
- 木吉他输出与现有 HQ6 **逐样本完全一致**；电吉他只有浮点运算顺序造成的 `1.43e-9` RMS 差异，difference SNR 为 **159.17 dB**，相关系数为 **1.0**。
- 整条吉他链的模型推理预计由 **858.18 s 降至 579.93 s**，提速 **1.480x**；按现有 864.91 s 进程总耗时估算，新总耗时约 **586.66 s（9 分 47 秒）**，节省约 **4 分 38 秒**。
- 峰值 CUDA 分配为 **1,190,870,528 bytes**，仅比单个原 BS 模型的 1,140,857,856 bytes 高约 4.4%，适合当前 RTX 3060 Laptop 6 GiB。

Lead 阶段可以再把 chunk batch 从 1 调到 2，但整曲仅由 201.16 s 降到 193.27 s；输出与原 HQ6 的全局 difference SNR 为 63.35 dB，并非严格逐样本相同。收益只有整条链约 1.4%，所以建议作为可选开关，不纳入默认“零质量风险”方案。

## 为什么共享主干成立

两个固定检查点逐 tensor 比较结果：

- 每个检查点 699 个 tensors、38,693,580 个 values；
- 451 个 tensors、26,328,772 个 values 完全相同，占 **68.0443%**；
- `band_split`、全部 Transformer `layers` 和 `final_norm` 完全相同；
- 只有 `mask_estimators` 不同，正好分别代表木吉他和电吉他的输出 head。

上游 BS-RoFormer 本身支持 `num_stems`，并在同一个共享表示上依次执行多个 `mask_estimators`，所以实验只是使用公开模型接口重新组装两个现有 state dict，没有修改 vendored 或 pip 库。上游实现可见 [BS-RoFormer 源码](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/models/bs_roformer/bs_roformer.py)。

组合后的参数量是 51,057,684；同时加载两个独立模型则是 77,385,752。单个 20 秒 chunk 的独立双模型耗时为 8.484 s，组合模型为 4.380 s，提速 1.937x，两个输出在该测试中都是 bit-exact。

## 完整实测

环境：Windows、Python 3.12.10、PyTorch 2.11.0+cu130、CUDA 13.0、RTX 3060 Laptop 6 GiB。整曲输入为 341.053 s 的《新裤子 - 没有理想的人不伤心》。所有对比都保留 20 s BS chunk、3 s Lead chunk、overlap=2、AMP，以及 identity / channel swap / polarity × 两个 half-song shifts 的 HQ6。

| 测试 | 现有耗时 | 候选耗时 | 提速 | 与现有 HQ6 的差异 |
|---|---:|---:|---:|---|
| 15 s，木 + 电 BS 完整 HQ6 | 31.415 s | 17.527 s | 1.792x | 木：bit-exact；电：162.61 dB |
| 341 s，木 + 电 BS 完整 HQ6 | 657.020 s | 378.768 s | 1.735x | 木：bit-exact；电：159.17 dB |
| 341 s，Lead HQ6 batch=2 | 201.160 s | 193.270 s | 1.041x | 63.35 dB；corr 0.99999977 |
| 整条链，推荐默认 | 858.180 s | 579.928 s | 1.480x | BS 等价；Lead 不变 |
| 整条链，再启用 Lead batch=2 | 858.180 s | 572.038 s | 1.500x | Lead 有极小数值差异 |

这里的 difference SNR 是候选输出相对当前 HQ6 输出的误差信噪比，不是相对真值 stem 的 SDR。它只能证明实现等价程度，不能被解释为模型分轨质量分数。

## 其他实验及取舍

| 方向 | 实测结果 | 决策 |
|---|---|---|
| 强制 Flash SDPA | BS 单 chunk 1.244 s，现有/自动约 1.234–1.237 s | 不采用；当前 memory-efficient 路径已经更快 |
| BS chunk batch=2 | 单项吞吐 1.264 s，batch=1 为 1.248 s | 不采用 |
| 木/电两个 CUDA stream 并发 | 8.561 s，串行 8.454 s | 不采用；同一张 3060 争抢算力反而变慢 |
| BS chunk 20 s 改 10 s | 60 s 样本快 1.28x，但相对 20 s 输出仅 16.37 dB | 不默认采用；没有真值集，无法证明质量不降 |
| Lead batch=2 | 整曲快 3.9%，63.35 dB difference SNR | 可选，不作为严格等价默认值 |
| Lead batch=4/8 | 短测吞吐继续提高，但输出相对 HQ6 约 34 dB，显存升至 1.13/1.90 GB | 不采用 |
| 原生 CUDA Graph | `torch.istft` 报 `cudaErrorStreamCaptureUnsupported` | 不采用 |
| `torch.compile` | 当前 Windows 环境缺少 Triton；`reduce-overhead` 又依赖 CUDA Graph，且官方说明不保证所有模型适用 | 不为小概率收益增加环境复杂度 |
| 用专用 HTDemucs 替换 Lead MBR | 单 pass 更快，但输出语义明显不同；两输出对输入有约 -24 dB 重建误差 | 不替换默认高质量路线 |
| 将专用 Lead HTDemucs 直接用于原始 mix | 输出能量异常，模型实际期望 guitar-only 输入 | 不可跳过电吉他隔离阶段 |

PyTorch 官方说明 SDPA 会在 Flash、memory-efficient 和 math 实现之间选择，具体最快实现依赖形状与硬件，因而这里以本机实测为准：[SDPA 性能说明](https://pytorch.org/blog/out-of-the-box-acceleration/)。`torch.compile(mode="reduce-overhead")` 使用 CUDA Graph、可能增加内存且不保证适用：[torch.compile 文档](https://docs.pytorch.org/docs/stable/generated/torch.compile)。

## 为什么不能在完全保质的前提下追平 HT-Demucs

当前吉他链对三个模型各执行 HQ6，共 18 次完整模型推理；15 秒样本的三个模型推理合计 40.86 s，而既有 HTDemucs 六轨基线约 3.96 s。共享主干后的同样 HQ6 约为 26.97 s，差距缩小但仍约为 6.8 倍。

主要原因不是 GPU 没启用，而是高质量策略本身的模型数量、模型规模和重复增强次数。上游 TTA 的确会额外执行 channel reversal 与 polarity inversion 后再平均结果：[MSST `apply_tta`](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/utils/model_utils.py)。要进一步接近 HT-Demucs，必须减少 HQ6 pass、缩短 chunk 或换更快模型，都会改变输出，必须有带真值的吉他数据集和主观试听 AB 才能确认质量代价。

MVSep 当前公开接口也把 Lead/Rhythm 的两阶段模型列为默认、SDR 9.21，一阶段为 9.02；这支持保留“先隔离电吉他，再做 Lead/Rhythm”的高质量路线：[MVSep Full API](https://www.mvsep.com/en/full_api)。其吉他页列出的 BS Roformer SW 为 guitar SDR 9.05：[MVSep Guitar](https://mvsep.com/algorithms/17)。

## 可选速度档位实验

以下只与现有 HQ6 输出比较，没有真值 SDR，因此不能称为等质量：

| 策略 | 15 s 三模型耗时 | 相对现有 HQ6 提速 | Acoustic / Electric / Lead difference SNR |
|---|---:|---:|---|
| HQ6（现有参考） | 40.859 s | 1.00x | 参考 |
| HQ4：去 polarity | 28.627 s | 1.43x | 29.26 / 33.14 / 18.69 dB |
| TTA3：去 half-song shift | 21.705 s | 1.88x | 21.44 / 25.43 / 16.94 dB |
| 仅两个 big shifts | 14.545 s | 2.81x | 24.35 / 29.01 / 10.19 dB |
| 单 pass | 7.550 s | 5.41x | 19.44 / 23.52 / 7.96 dB |

如果以后需要“极速模式”，建议先建立小型真值/人工评分集，再从 HQ4 开始验证；不应把这些配置直接替换默认 HQ6。

## 建议的生产落地方式

本轮没有改动核心代码。后续若决定集成，建议只改 BandBuddy 自己的加载与编排层：

1. 继续下载并校验现有木/电两个固定 checkpoint。
2. 启动时验证两份 config 相同，并验证除 `mask_estimators.0` 外的 state tensors 完全相同；不满足时 fail closed，回退现有两个模型流程。
3. 实例化原生 `BSRoformer(num_stems=2)`，共享 tensor 从任一 checkpoint 装载，两个 mask head 分别从木/电 checkpoint 装载。
4. 一次 HQ6 得到 Acoustic 与 Electric；之后释放共享模型，再按原逻辑运行 Lead MBR，并保持 `rhythm = electric - lead`。
5. manifest 同时保留两份原权重 SHA-256，并记录 `shared_trunk=true`，方便审计与回归。
6. 用当前 15 秒和 341 秒样本设回归门槛：Acoustic bit-exact、Electric difference SNR > 140 dB、三轨长度/采样率/残差一致性不变。

## 复现入口与原始结果

- 共享模型完整 HQ6：[benchmark_shared_bs_pipeline.py](benchmark_shared_bs_pipeline.py)
- checkpoint 相似度检查：[inspect_checkpoint_similarity.py](inspect_checkpoint_similarity.py)
- 单 chunk 共享模型基准：[benchmark_shared_bs.py](benchmark_shared_bs.py)
- HQ 策略比较：[benchmark_hq_policies.py](benchmark_hq_policies.py)
- Lead batch 比较：[benchmark_lead_batches.py](benchmark_lead_batches.py)
- 341 秒共享 BS 结果：[results/shared-bs-pipeline-full-341s.json](results/shared-bs-pipeline-full-341s.json)
- 341 秒 Lead batch=2 结果：[results/lead-batch2-full-341s.json](results/lead-batch2-full-341s.json)
- 全部机器可读结果：[results](results)
