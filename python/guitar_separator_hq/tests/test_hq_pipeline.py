from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from guitar_separator_hq import inference, separator
from guitar_separator_hq.inference import InferenceReport
from guitar_separator_hq.specs import MODEL_SPECS, config_path


class IdentityStem(torch.nn.Module):
    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        return mix.unsqueeze(1)


class HqPipelineTests(unittest.TestCase):
    def test_configs_are_pinned_and_targets_match(self) -> None:
        for spec in MODEL_SPECS:
            path = config_path(spec)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, spec.config_sha256)
            self.assertEqual(inference.load_config(spec)["training"]["target_instrument"], spec.target)

    def test_hq6_tta_and_overlap_preserve_identity(self) -> None:
        generator = np.random.default_rng(42)
        mix = generator.normal(0, 0.05, size=(2, 4_321)).astype(np.float32)
        estimate = inference._high_quality_predict(
            IdentityStem(),
            mix,
            chunk_size=1_000,
            overlap=2,
            use_amp=False,
            device=torch.device("cpu"),
            progress=None,
        )
        np.testing.assert_allclose(estimate, mix, atol=2e-7, rtol=2e-6)

    def test_three_outputs_are_float_and_lead_rhythm_are_complementary(self) -> None:
        frames = 8_192
        mix = np.linspace(-0.4, 0.4, frames * 2, dtype=np.float32).reshape(2, frames)
        acoustic = mix * np.float32(0.2)
        electric = mix * np.float32(0.7)
        lead = electric * np.float32(0.3)
        estimates = {"acoustic": acoustic, "electric": electric, "lead": lead}
        report = InferenceReport(
            architecture="test",
            target="test",
            chunk_size=1,
            overlap=2,
            big_shifts=2,
            tta_variants=("identity", "channel_swap", "polarity"),
            passes=6,
            amp_enabled=False,
            parameters=1,
            seconds=0.0,
            peak_cuda_bytes=0,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            source.write_bytes(b"synthetic input marker")
            output = root / "out"

            def fake_predict(source_audio, spec, checkpoint, device, progress):
                return estimates[spec.key].copy(), report

            with (
                patch.object(separator, "ensure_models", return_value={spec.key: root for spec in MODEL_SPECS}),
                patch.object(separator, "load_audio", return_value=mix.copy()),
                patch.object(separator, "predict_model", side_effect=fake_predict),
            ):
                result = separator.separate_guitars(source, output, root, device_name="cpu")

            loaded_acoustic, rate = sf.read(result.acoustic_guitar, dtype="float32", always_2d=True)
            loaded_lead, _ = sf.read(result.lead_guitar, dtype="float32", always_2d=True)
            loaded_rhythm, _ = sf.read(result.rhythm_guitar, dtype="float32", always_2d=True)
            self.assertEqual(rate, 44_100)
            self.assertEqual(sf.info(result.lead_guitar).subtype, "FLOAT")
            np.testing.assert_allclose(loaded_acoustic.T, acoustic, atol=3e-8, rtol=1e-7)
            np.testing.assert_allclose(loaded_lead.T + loaded_rhythm.T, electric, atol=3e-8, rtol=1e-7)
            manifest = json.loads(result.manifest.read_text("utf-8"))
            self.assertEqual(manifest["quality_policy"]["only_mode"], "HQ6")
            self.assertLess(manifest["consistency"]["lead_plus_rhythm_minus_electric"]["peak"], 3e-8)
            self.assertEqual(set(manifest["outputs"]), set(separator.OUTPUT_NAMES))
            self.assertEqual(list(output.glob("without_*")), [])

    def test_non_finite_output_is_rejected(self) -> None:
        audio = np.zeros((2, 32), dtype=np.float32)
        audio[0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "AUDIO_NON_FINITE"):
            separator.validate_audio_array(audio, name="bad")


if __name__ == "__main__":
    unittest.main()
