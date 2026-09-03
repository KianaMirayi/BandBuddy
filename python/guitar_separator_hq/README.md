# Guitar Separator HQ6

这是 BandBuddy 2.0 使用的固定最高音质吉他分离模块，同时保留一个可审计的独立命令行入口。它没有“快速/标准”等降质选项，一次推理只输出三个独立轨：

1. 原混音 → Acoustic BS-RoFormer HQ6 → `acoustic_guitar.wav`
2. 原混音 → Electric BS-RoFormer HQ6 → 电吉他浮点中间轨
3. 电吉他中间轨 → Lead/Rhythm Mel-Band RoFormer HQ6 → `lead_guitar.wav`
4. `rhythm_guitar = electric_guitar - lead_guitar` → `rhythm_guitar.wav`

每个模型执行 identity、左右声道交换、极性反转三种 TTA，并为每种 TTA 执行两个半曲长 circular big shift，共六次推理后平均。Acoustic/Electric 使用 20 秒块、overlap 2；Lead 使用 3 秒块、overlap 2。宿主音频和 overlap-add 累加器始终为 float32，不做峰值归一化、int16 中间量化或有损中间编码。

产品工作进程会先从同一份解码后的 44.1 kHz stereo float32 混音生成原六轨和上述吉他三轨，再按任务创建时的格式快照一次性编码九轨。去除某条吉他轨由播放器静音、轨道增益或混音导出完成，本模块不会生成 `without_*` 文件。

完整逆向结论、模型版本和整曲基线见 [`docs/experiments/bd-guitar-separator-v14.5-hq6-2026-09-03.md`](../../docs/experiments/bd-guitar-separator-v14.5-hq6-2026-09-03.md)。

## Windows 独立运行

要求 Python 3.12。下面的审计环境使用 PyTorch 2.11.0 CUDA 13.0；BandBuddy 安装器会为 Windows x64 或 macOS Apple Silicon 建立自己的私有运行环境。

```powershell
resources\bin\uv.exe venv .venv-guitar-hq --python 3.12
resources\bin\uv.exe pip install --python .venv-guitar-hq\Scripts\python.exe -r python\guitar_separator_hq\requirements-hq.txt
```

```powershell
.venv-guitar-hq\Scripts\python.exe python\guitar_separator_hq_cli.py `
  "C:\path\song.mp3" `
  --output "C:\path\song-guitar-hq6" `
  --device cuda:0
```

权重从公开的 [ModelScope `Zzzzzzorz/BandBuddy-Models`](https://modelscope.cn/models/Zzzzzzorz/BandBuddy-Models) 固定 `v2.0.0` 分支下载。独立工具校验三份吉他权重；产品安装器还会同时校验原六轨权重，并且只有四份文件的固定字节数和 SHA-256 全部通过后才写入完整标记。下载支持 `.part` 断点续传。

独立输出目录只包含：

- `acoustic_guitar.wav`
- `lead_guitar.wav`
- `rhythm_guitar.wav`
- `separation_manifest.json`：输入/输出哈希、内部推理版本、固定参数、耗时、显存峰值和 Lead/Rhythm 重建误差

`--offline` 禁止网络下载，`--overwrite` 明确允许替换已有同名输出。除此之外没有音质相关参数。

## 独立验收

```powershell
$env:PYTHONPATH = "python"
.venv-guitar-hq\Scripts\python.exe -m guitar_separator_hq.verify `
  "C:\path\song-guitar-hq6" `
  --report "C:\path\song-guitar-hq6\quality_report.json"
```

验收器从磁盘重读三个结果，检查哈希、帧数、44.1 kHz/stereo/float32、NaN/Inf，并报告每 10 秒的 Lead/Rhythm 能量。推理阶段同时验证 `lead + rhythm` 对电吉他中间轨的 float32 误差。该检查不能替代有真值 stems 的 SDR/SI-SDR 测试或人工盲听。

## 固定模型

| 目标 | 架构 | 参数量 | checkpoint SHA-256 |
|---|---:|---:|---|
| 木吉他 | BS-RoFormer | 38,692,876 | `fa386b2e7b1ea4f12b9b5c557444c0dc78648ef4ee299de2759d86457e182b3e` |
| 电吉他 | BS-RoFormer | 38,692,876 | `cd506bfce9474f91a31001967f2c4935ce4e67f643da3df20d04058da927c553` |
| 主音吉他 | Mel-Band RoFormer | 84,199,748 | `b3c47bca33609ca1ba0bb2d2076410bfd1eb941b051b72afc1f3e24d12b17eef` |

精确仓库、分支、路径、大小和配置哈希定义在 [`specs.py`](specs.py)，配置快照位于 [`configs`](configs)。产品四文件清单另固定在 [`../worker/model_download.py`](../worker/model_download.py)。

## 测试

```powershell
$env:PYTHONPATH = "python"
.venv-guitar-hq\Scripts\python.exe -m unittest discover `
  -s python\guitar_separator_hq\tests -t python -p "test_*.py" -v
python -m unittest discover -s python\worker\tests -p "test_*.py" -v
```

测试覆盖配置哈希和目标、HQ6 TTA/overlap-add 等变性、三个 float32 输出、Lead/Rhythm 互补重建、无 `without_*` 产物，以及模型包断点续传、损坏缓存和原子完整标记。

## 来源与许可

模型结构 vendor 了 [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) 固定 commit `0e5f1159fc5ea87fc13b957584e178b4977e5dd3` 所需的 MIT 文件，许可证保留在 [`_vendor/msst/LICENSE`](_vendor/msst/LICENSE)。公开 ModelScope 权重仓库按 GPL-3.0 发布，并包含配置快照、来源说明和第三方声明；BandBuddy 源码本身继续使用 Apache-2.0。
