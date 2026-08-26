from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline.py"
FAKE = Path(__file__).with_name("fake_stage.py")


class PipelineTests(unittest.TestCase):
    def synthetic_wav(self, path: Path) -> None:
        with wave.open(str(path), "wb") as destination:
            destination.setnchannels(2)
            destination.setsampwidth(2)
            destination.setframerate(8000)
            destination.writeframes((b"\x10\x00\xf0\xff" * 800))

    def config(self, path: Path) -> None:
        path.write_text(json.dumps({
            "schemaVersion": 1,
            "id": "fake-dynamic-pipeline",
            "mss": {
                "id": "fake-mss",
                "argv": [sys.executable, str(FAKE), "mss", "--input", "{input}", "--output", "{output}"]
            },
            "msr": {
                "id": "fake-msr",
                "selectStems": ["vocals", "saxophone"],
                "argv": [sys.executable, str(FAKE), "msr", "--input", "{input}", "--output", "{output}"]
            }
        }), "utf-8")

    def test_dynamic_stems_and_selected_restoration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            config = root / "config.json"
            output = root / "output"
            self.synthetic_wav(source)
            self.config(config)
            result = subprocess.run(
                [sys.executable, str(PIPELINE), "run", "--config", str(config), "--input", str(source), "--output", str(output)],
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            manifest = json.loads((output / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["stems"], ["drums", "saxophone", "vocals"])
            self.assertEqual(manifest["restoredStems"], ["saxophone", "vocals"])
            self.assertTrue(all((output / f"{stem}.wav").is_file() for stem in manifest["stems"]))
            self.assertFalse((output / ".work").exists())
            saxophone = next(item for item in manifest["files"] if item["stem"] == "saxophone")
            self.assertIn("msrChange", saxophone)
            self.assertIn("qualityNote", manifest)

    def test_missing_selected_stem_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            config = root / "config.json"
            output = root / "output"
            self.synthetic_wav(source)
            self.config(config)
            raw = json.loads(config.read_text("utf-8"))
            raw["msr"]["selectStems"] = ["clarinet"]
            config.write_text(json.dumps(raw), "utf-8")
            result = subprocess.run(
                [sys.executable, str(PIPELINE), "run", "--config", str(config), "--input", str(source), "--output", str(output)],
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("MSR_SELECTED_STEM_MISSING", result.stdout)


if __name__ == "__main__":
    unittest.main()

