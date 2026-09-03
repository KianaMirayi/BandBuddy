"""BandBuddy's fixed high-quality guitar separation runtime."""

from .separator import (
    GuitarArrayResult,
    SeparationResult,
    load_audio,
    separate_guitar_arrays,
    separate_guitars,
    validate_audio_array,
    write_float_wav,
)
from .specs import MODEL_SET_REVISION

__all__ = [
    "GuitarArrayResult",
    "MODEL_SET_REVISION",
    "SeparationResult",
    "load_audio",
    "separate_guitar_arrays",
    "separate_guitars",
    "validate_audio_array",
    "write_float_wav",
]
__version__ = "2.0.0"
