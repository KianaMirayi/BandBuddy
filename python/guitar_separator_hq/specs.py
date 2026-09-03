from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MODEL_SET_REVISION = "bandbuddy-stems:v2.0.0"
MODELSCOPE_REPOSITORY = "Zzzzzzorz/BandBuddy-Models"
MODELSCOPE_REVISION = "v2.0.0"
MODELSCOPE_RESOLVE_ROOT = (
    f"https://modelscope.cn/models/{MODELSCOPE_REPOSITORY}/resolve/{MODELSCOPE_REVISION}"
)
MSST_SOURCE_COMMIT = "0e5f1159fc5ea87fc13b957584e178b4977e5dd3"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    architecture: str
    target: str
    filename: str
    config_filename: str
    config_sha256: str
    repository: str
    repository_revision: str
    repository_path: str
    sha256: str
    size: int
    tensor_values: int
    trainable_parameters: int

    @property
    def source_url(self) -> str:
        return f"{MODELSCOPE_RESOLVE_ROOT}/{self.repository_path}"


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="acoustic",
        display_name="MVSep Mega 53-stem Acoustic Guitar",
        architecture="bs_roformer",
        target="acoustic-guitar",
        filename="bs_mega_53stem_acoustic-guitar_mvsep.ckpt",
        config_filename="acoustic_guitar.yaml",
        config_sha256="040148363929b68fd4a4e323358a245938568692ca201fc2f703b6449dc8b569",
        repository=MODELSCOPE_REPOSITORY,
        repository_revision=MODELSCOPE_REVISION,
        repository_path="bs_mega_53stem_acoustic-guitar_mvsep.ckpt",
        sha256="fa386b2e7b1ea4f12b9b5c557444c0dc78648ef4ee299de2759d86457e182b3e",
        size=77_624_038,
        tensor_values=38_693_580,
        trainable_parameters=38_692_876,
    ),
    ModelSpec(
        key="electric",
        display_name="MVSep Mega 53-stem Electric Guitar",
        architecture="bs_roformer",
        target="electric-guitar",
        filename="bs_mega_53stem_electric-guitar_mvsep.ckpt",
        config_filename="electric_guitar.yaml",
        config_sha256="e89d0646a99f08e99e56d71de22b5accee2628fd11a91928ad556fc7b8617ecb",
        repository=MODELSCOPE_REPOSITORY,
        repository_revision=MODELSCOPE_REVISION,
        repository_path="bs_mega_53stem_electric-guitar_mvsep.ckpt",
        sha256="cd506bfce9474f91a31001967f2c4935ce4e67f643da3df20d04058da927c553",
        size=77_624_038,
        tensor_values=38_693_580,
        trainable_parameters=38_692_876,
    ),
    ModelSpec(
        key="lead",
        display_name="Mel-Band RoFormer Lead/Rhythm Guitar by listra92",
        architecture="mel_band_roformer",
        target="Lead",
        filename="mbr_lead_rhythm_guitar_listra92.ckpt",
        config_filename="lead_rhythm.yaml",
        config_sha256="a26685bc9ab10aab4dc153b74ec3559f1b4bd2251d129a13b6b22fe3a27d382d",
        repository=MODELSCOPE_REPOSITORY,
        repository_revision=MODELSCOPE_REVISION,
        repository_path="mbr_lead_rhythm_guitar_listra92.ckpt",
        sha256="b3c47bca33609ca1ba0bb2d2076410bfd1eb941b051b72afc1f3e24d12b17eef",
        size=337_073_664,
        tensor_values=84_199_940,
        trainable_parameters=84_199_748,
    ),
)

MODEL_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}


def config_path(spec: ModelSpec) -> Path:
    return Path(__file__).with_name("configs") / spec.config_filename
