from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_download  # noqa: E402


def tiny_file(key: str, filename: str, payload: bytes) -> model_download.BundleFile:
    return model_download.BundleFile(
        key,
        filename,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )


class FakeResponse:
    def __init__(self, payload: bytes, status: int) -> None:
        self.payload = payload
        self.status = status
        self.headers = {"Content-Length": str(len(payload))}
        self.offset = 0

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self, size: int) -> bytes:
        block = self.payload[self.offset : self.offset + size]
        self.offset += len(block)
        return block


class InterruptedResponse(FakeResponse):
    def __init__(self, payload: bytes, first_block: int) -> None:
        super().__init__(payload, 200)
        self.first_block = first_block
        self.failed = False

    def read(self, size: int) -> bytes:
        if self.offset == 0:
            return super().read(min(size, self.first_block))
        if not self.failed:
            self.failed = True
            raise OSError("network interrupted")
        return b""


class ModelBundleTests(unittest.TestCase):
    def test_production_manifest_is_pinned_to_the_public_v201_revision(self) -> None:
        self.assertEqual(model_download.MODEL_REPOSITORY, "Zzzzzzorz/BandBuddy-Models")
        self.assertEqual(model_download.MODEL_REVISION, "v2.0.1")
        self.assertIn("/resolve/v2.0.1", model_download.MODEL_BASE_URL)
        self.assertEqual(
            [(item.size, item.sha256) for item in model_download.BUNDLE_FILES],
            [
                (54_996_327, "34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd"),
                (102_410_137, "183607bffbebdb43dcb3fd583b7cbe3c77fb55aea886cbd1e3aa316d9148698c"),
                (337_073_664, "b3c47bca33609ca1ba0bb2d2076410bfd1eb941b051b72afc1f3e24d12b17eef"),
                (27_147_460, "2bd8f2af629b279cc1a568f895ee9636f7ce2d76c69aa601e6744eaab8b4916a"),
                (27_147_623, "bd6fcf40659771568ee180ea69bd4576a9c3d2423ae0f5f6f5afcc8b6a6fd938"),
                (109_822_623, "946ffd50d7f2fd87e447d880525283e88bd9061e1b428d4f1d380e764d54d618"),
            ],
        )

    def test_range_download_resumes_and_publishes_only_verified_bytes(self) -> None:
        payload = b"verified-model-content"
        spec = tiny_file("test", "test.ckpt", payload)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / spec.filename
            destination.with_name(destination.name + ".part").write_bytes(payload[:8])
            requests = []

            def fake_urlopen(request, **_kwargs):
                requests.append(request)
                return FakeResponse(payload[8:], 206)

            with patch.object(model_download, "urlopen", side_effect=fake_urlopen):
                model_download._download_once(spec, destination, lambda _fraction: None)

            self.assertEqual(requests[0].get_header("Range"), "bytes=8-")
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(destination.with_name(destination.name + ".part").exists())

    def test_complete_verified_partial_is_published_without_another_request(self) -> None:
        payload = b"already-complete-model"
        spec = tiny_file("test", "test.ckpt", payload)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / spec.filename
            destination.with_name(destination.name + ".part").write_bytes(payload)
            with patch.object(model_download, "urlopen") as urlopen:
                model_download._download_once(spec, destination, lambda _fraction: None)
            urlopen.assert_not_called()
            self.assertEqual(destination.read_bytes(), payload)

    def test_server_ignoring_range_restarts_the_partial_file(self) -> None:
        payload = b"complete-response"
        spec = tiny_file("test", "test.ckpt", payload)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / spec.filename
            destination.with_name(destination.name + ".part").write_bytes(b"stale")
            with patch.object(model_download, "urlopen", return_value=FakeResponse(payload, 200)):
                model_download._download_once(spec, destination, lambda _fraction: None)
            self.assertEqual(destination.read_bytes(), payload)

    def test_network_interruption_resumes_from_the_persisted_partial(self) -> None:
        payload = b"resume-after-interruption"
        spec = tiny_file("test", "test.ckpt", payload)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / spec.filename
            requests = []

            def fake_urlopen(request, **_kwargs):
                requests.append(request)
                if len(requests) == 1:
                    return InterruptedResponse(payload, 7)
                self.assertEqual(request.get_header("Range"), "bytes=7-")
                return FakeResponse(payload[7:], 206)

            with (
                patch.object(model_download, "urlopen", side_effect=fake_urlopen),
                patch.object(model_download.time, "sleep"),
            ):
                model_download.download_file(
                    spec, destination, retries=1, progress=lambda _fraction: None
                )
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(len(requests), 2)

    def test_corrupt_complete_cache_is_replaced(self) -> None:
        payload = b"valid-cache"
        spec = tiny_file("test", "test.ckpt", payload)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / spec.filename
            destination.write_bytes(b"x" * len(payload))
            with patch.object(model_download, "urlopen", return_value=FakeResponse(payload, 200)):
                model_download.download_file(
                    spec, destination, retries=0, progress=lambda _fraction: None
                )
            self.assertEqual(destination.read_bytes(), payload)

    def test_missing_pinned_branch_fails_without_publishing_a_file(self) -> None:
        payload = b"missing"
        spec = tiny_file("test", "test.ckpt", payload)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / spec.filename
            missing = HTTPError("https://example.invalid/model", 404, "not found", {}, None)
            with patch.object(model_download, "urlopen", side_effect=missing) as urlopen:
                with self.assertRaisesRegex(RuntimeError, "MODEL_DOWNLOAD_FAILED"):
                    model_download.download_file(
                        spec, destination, retries=8, progress=lambda _fraction: None
                    )
            self.assertEqual(urlopen.call_count, 1)
            self.assertFalse(destination.exists())

    def test_marker_is_atomic_and_corrupt_cache_is_rejected(self) -> None:
        payloads = {"one.ckpt": b"one", "two.ckpt": b"two"}
        specs = tuple(tiny_file(name, name, payload) for name, payload in payloads.items())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def install(spec, destination, **_kwargs):
                destination.write_bytes(payloads[spec.filename])

            with (
                patch.object(model_download, "BUNDLE_FILES", specs),
                patch.object(model_download, "download_file", side_effect=install),
            ):
                bundle = model_download.install_bundle(root)
                self.assertEqual(model_download.verify_bundle(root), bundle)
                (bundle / "two.ckpt").write_bytes(b"bad")
                with self.assertRaisesRegex(RuntimeError, "MODEL_(SIZE|HASH)_MISMATCH"):
                    model_download.verify_bundle(root)

    def test_stage_verification_hashes_only_the_models_needed_by_that_stage(self) -> None:
        payloads = {"six.th": b"six", "guitar.ckpt": b"guitar"}
        specs = tuple(tiny_file(key, filename, payloads[filename]) for key, filename in (
            ("six_stem", "six.th"),
            ("shared_acoustic_electric", "guitar.ckpt"),
        ))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def install(spec, destination, **_kwargs):
                destination.write_bytes(payloads[spec.filename])

            with (
                patch.object(model_download, "BUNDLE_FILES", specs),
                patch.object(model_download, "download_file", side_effect=install),
            ):
                bundle = model_download.install_bundle(root)
                (bundle / "guitar.ckpt").write_bytes(b"broken")
                self.assertEqual(
                    model_download.verify_bundle(root, ("six_stem",)),
                    bundle,
                )
                with self.assertRaisesRegex(RuntimeError, "MODEL_(SIZE|HASH)_MISMATCH"):
                    model_download.verify_bundle(root, ("shared_acoustic_electric",))

    def test_failed_bundle_install_never_writes_the_complete_marker(self) -> None:
        payloads = {"one.ckpt": b"one", "two.ckpt": b"two"}
        specs = tuple(tiny_file(name, name, payload) for name, payload in payloads.items())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def fail_second(spec, destination, **_kwargs):
                if spec.filename == "two.ckpt":
                    raise OSError("network interrupted")
                destination.write_bytes(payloads[spec.filename])

            with (
                patch.object(model_download, "BUNDLE_FILES", specs),
                patch.object(model_download, "download_file", side_effect=fail_second),
            ):
                with self.assertRaisesRegex(OSError, "network interrupted"):
                    model_download.install_bundle(root)
                self.assertFalse(model_download.marker_path(root).exists())


if __name__ == "__main__":
    unittest.main()
