# Guitar split speed experiments (2026-09-04)

This is an isolated benchmark workspace for BandBuddy's three-stage guitar
separation pipeline. Production sources under `python/` and `src/` were kept
read-only throughout the experiment.

The main result is a lossless shared-trunk BS-RoFormer prototype: the acoustic
and electric checkpoints have an identical transformer trunk, so one native
two-stem BS-RoFormer can execute the trunk once and retain both original mask
heads. On the 341.05 s reference song, acoustic + electric inference fell from
657.02 s to 378.77 s (1.735x), with bit-identical acoustic output and only
float-rounding-level electric error (159.17 dB difference SNR).

See [REPORT.md](REPORT.md) for the conclusion, benchmark matrix, rejected
approaches, and suggested production integration. Raw measurements are under
`results/`; generated audio, local dependencies, and large artifacts are
ignored by Git.
