from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


GuitarQuality = Literal["fast", "balanced", "high"]
GUITAR_QUALITIES: tuple[GuitarQuality, ...] = ("fast", "balanced", "high")
MODEL_SET_REVISION = "bandbuddy-stems:v2.0.1"
MODELSCOPE_REPOSITORY = "Zzzzzzorz/BandBuddy-Models"
MODELSCOPE_REVISION = "v2.0.1"
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


SHARED_BS_SPEC = ModelSpec(
    key="shared_bs",
    display_name="MVSep Mega 53-stem Acoustic/Electric Shared BS-RoFormer",
    architecture="bs_roformer_shared",
    target="acoustic+electric",
    filename="bs_mega_53stem_acoustic-electric_shared_mvsep.ckpt",
    config_filename="shared_acoustic_electric.yaml",
    config_sha256="aa9118bd30d786f6dfa4bf5344f662e81f5d9e07877c4001bbcfb340e4bf4238",
    repository=MODELSCOPE_REPOSITORY,
    repository_revision=MODELSCOPE_REVISION,
    repository_path="bs_mega_53stem_acoustic-electric_shared_mvsep.ckpt",
    sha256="054c13fc97ff863df55c1e8f0ab620a7697df38a98c564e6d8aa4e314d8fa391",
    size=102_410_137,
    tensor_values=51_058_388,
    trainable_parameters=51_057_684,
)

LEAD_HQ_SPEC = ModelSpec(
    key="lead_hq",
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
)

ACOUSTIC_FAST_SPEC = ModelSpec(
    key="acoustic_fast",
    display_name="MDX-Net 6s Acoustic Guitar by Anvuew",
    architecture="mdx_net",
    target="acoustic_guitar",
    filename="mdx_6s_acoustic_guitar_anvuew.onnx",
    config_filename="fast_acoustic_mdx.yaml",
    config_sha256="0ec8dba21eb5759314bb6a929b73e7011009a454e5da8b554242a18f57b2e780",
    repository=MODELSCOPE_REPOSITORY,
    repository_revision=MODELSCOPE_REVISION,
    repository_path="mdx_6s_acoustic_guitar_anvuew.onnx",
    sha256="2bd8f2af629b279cc1a568f895ee9636f7ce2d76c69aa601e6744eaab8b4916a",
    size=27_147_460,
    tensor_values=6_779_816,
    trainable_parameters=6_779_816,
)

ELECTRIC_FAST_SPEC = ModelSpec(
    key="electric_fast",
    display_name="MDX-Net 6s Electric Guitar by Anvuew",
    architecture="mdx_net",
    target="electric_guitar",
    filename="mdx_6s_electric_guitar_anvuew.onnx",
    config_filename="fast_electric_mdx.yaml",
    config_sha256="e422ffd7cf8a786db51aed36fddb14f0cf8736915aad534ec77fddd9ba838d28",
    repository=MODELSCOPE_REPOSITORY,
    repository_revision=MODELSCOPE_REVISION,
    repository_path="mdx_6s_electric_guitar_anvuew.onnx",
    sha256="bd6fcf40659771568ee180ea69bd4576a9c3d2423ae0f5f6f5afcc8b6a6fd938",
    size=27_147_623,
    tensor_values=6_779_816,
    trainable_parameters=6_779_816,
)

LEAD_FAST_SPEC = ModelSpec(
    key="lead_fast",
    display_name="HTDemucs4 Lead/Rhythm Guitar by Dry Paint Dealer Undr",
    architecture="htdemucs",
    target="lead",
    filename="demucs4_lead_rhythm_guitar_drypaint.ckpt",
    config_filename="fast_lead_rhythm_htdemucs.yaml",
    config_sha256="2c0d2238e5234621f904d25efd661bfb3b96ea22dd9748e4b0ffccc09f566e2e",
    repository=MODELSCOPE_REPOSITORY,
    repository_revision=MODELSCOPE_REVISION,
    repository_path="demucs4_lead_rhythm_guitar_drypaint.ckpt",
    sha256="946ffd50d7f2fd87e447d880525283e88bd9061e1b428d4f1d380e764d54d618",
    size=109_822_623,
    tensor_values=27_405_756,
    trainable_parameters=27_405_756,
)

HQ_MODEL_SPECS: tuple[ModelSpec, ...] = (SHARED_BS_SPEC, LEAD_HQ_SPEC)
FAST_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ACOUSTIC_FAST_SPEC,
    ELECTRIC_FAST_SPEC,
    LEAD_FAST_SPEC,
)
MODEL_SPECS: tuple[ModelSpec, ...] = HQ_MODEL_SPECS + FAST_MODEL_SPECS
MODEL_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}


def normalize_quality(value: str) -> GuitarQuality:
    normalized = value.strip().lower()
    if normalized not in GUITAR_QUALITIES:
        raise ValueError(f"GUITAR_QUALITY_INVALID:{value}")
    return normalized  # type: ignore[return-value]


def model_specs_for_quality(quality: GuitarQuality) -> tuple[ModelSpec, ...]:
    return FAST_MODEL_SPECS if quality == "fast" else HQ_MODEL_SPECS


def config_path(spec: ModelSpec) -> Path:
    return Path(__file__).with_name("configs") / spec.config_filename
