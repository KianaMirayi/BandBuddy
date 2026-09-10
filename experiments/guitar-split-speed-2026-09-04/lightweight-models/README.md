# Lightweight guitar-model candidates

Isolated follow-up benchmark for lighter replacements of BandBuddy's two
BS-RoFormer guitar classifiers and Lead/Rhythm Mel-Band RoFormer.

The fixed input is:

`C:\CloudMusic\新裤子 - 没有理想的人不伤心.mp3`

Large model files, generated audio, and locally installed benchmark-only
dependencies stay ignored. Production code under `python/` and `src/` remains
read-only.

See [REPORT.md](REPORT.md) for conclusions and
[results/lightweight-summary.json](results/lightweight-summary.json) for the
machine-readable summary.

The tested fast chain completed model inference in 48.13 seconds versus the
current 858.18-second HQ6 chain, but its outputs differ materially from the
current quality baseline. It is suitable only as a separately labeled fast
mode until ground-truth and listening validation are available.
