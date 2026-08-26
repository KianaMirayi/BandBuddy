# MSS → MSR MVP

这个目录验证一条与模型、轨数和乐器分类解耦的两阶段链路：

```text
混音 → MSS（输出任意数量、任意名字的 stem）→ 按 stem 选择性 MSR → manifest.json
```

MVP 暂时是独立命令行实验，不接入正式 UI、任务队列或模型下载器。这样可以先验证模型效果和资源开销，再决定产品化范围。

## 文件协议

- MSS 在 `{output}` 根目录写音频文件；文件名（不含扩展名）就是 stem ID。
- 管线动态发现 stem，不内置“6 轨”或固定乐器列表。
- MSR 对选中的每条 stem 单独运行，可使用 `{input}`、`{output}`、`{stem}` 模板变量。
- 未选中的 stem 原样传递；最终目录包含各轨及可复现的 `manifest.json`。
- stage 参数以 argv 数组直接执行，不经过 shell。

`manifest.json` 记录输入/输出 SHA-256、实际命令、耗时、采样率、声道、时长、峰值、RMS、DC offset 和削波比例。没有干净参考轨时，`msrChange` 只能说明信号改变了多少，不能证明音质更好。

## 已有配置

| 配置 | MSS | MSR | 状态 |
| --- | --- | --- | --- |
| `demucs6_passthrough.json` | `htdemucs_6s` | 无 | 已在 4 个用例上实测 |
| `dtt_bsr_8stem_stage1.json` | DTT-BSR 公布的 8 个一阶段模型 | 无 | 已在 4 个用例上实测；不是 DTT-BSR+ |
| `demucs6_xlance_vocals.json` | `htdemucs_6s` | X-Lance vocal dereverb | 适配器已完成；需另行准备官方权重 |

X-Lance 论文系统的顺序是 denoise → MSS，并只对 vocal 做 dereverb；`demucs6_xlance_vocals.json` 是为了验证“现有 MSS 后接 MSR”而设计的实验组合，不等同于论文的完整 X-Lance 系统。

DTT-BSR+ 论文描述的一阶段是 DTT-BSR、二阶段是逐 stem 的 modified Demucs-L。当前官方仓库和公开权重只包含一阶段，因此本目录明确使用 `stage1` 命名，不把它冒充为 DTT-BSR+。

模型来源、commit 和权重摘要见 `model-sources.json`；模型仓库与权重不提交到 BandBuddy。

## 运行

准备统一的短测试片段（NCM 会复用 BandBuddy 现有解码逻辑）：

```powershell
pnpm msr:prepare-cases -- --input 测试用例 --output .codex-tmp/msr-mvp/cases --start 60 --duration 15
```

校验配置：

```powershell
pnpm msr:mvp -- validate --config python/msr_mvp/configs/dtt_bsr_8stem_stage1.json
```

运行 DTT-BSR 一阶段（示意路径需替换为本机实际位置）：

```powershell
pnpm msr:mvp -- run `
  --config python/msr_mvp/configs/dtt_bsr_8stem_stage1.json `
  --input .codex-tmp/msr-mvp/cases/example.wav `
  --output .codex-tmp/msr-mvp/output/example `
  --var model_python=C:/path/to/python.exe `
  --var dtt_repo=C:/path/to/DTT-BSR `
  --var dtt_config=C:/path/to/config.yaml `
  --var dtt_checkpoints=C:/path/to/checkpoints `
  --var dtt_site=C:/path/to/python/site-packages `
  --var device=cuda `
  --var dtt_chunk_seconds=15
```

运行 Demucs → X-Lance vocal dereverb 的变量接口类似，见 `demucs6_xlance_vocals.json`。显存不足时优先减小 `xlance_chunk_seconds`。

失败的运行会保留 `<output>/.work` 方便排查；重试请换一个输出目录，或在确认路径后移走旧的失败目录。

## 测试

```powershell
pnpm test:msr-mvp
```

测试会用伪 MSS/MSR stage 验证动态 stem（包括非预设的 `saxophone`）、选择性恢复、manifest 和缺失 stem 的有界错误。
