from __future__ import annotations

import hashlib
import os
from pathlib import Path
import ssl
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .specs import MODEL_SPECS, ModelSpec


CHUNK_BYTES = 1024 * 1024
ProgressCallback = Callable[[str, float, str], None]


class ModelStoreError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def model_path(model_root: Path, spec: ModelSpec) -> Path:
    return model_root.resolve() / spec.filename


def verify_model(model_root: Path, spec: ModelSpec) -> Path:
    path = model_path(model_root, spec)
    if not path.is_file():
        raise ModelStoreError(f"MODEL_FILE_MISSING:{spec.key}:{path}")
    if path.stat().st_size != spec.size:
        raise ModelStoreError(
            f"MODEL_SIZE_MISMATCH:{spec.key}:expected={spec.size}:actual={path.stat().st_size}"
        )
    actual = sha256_file(path)
    if actual != spec.sha256:
        raise ModelStoreError(
            f"MODEL_HASH_MISMATCH:{spec.key}:expected={spec.sha256}:actual={actual}"
        )
    return path


def verify_all_models(model_root: Path) -> dict[str, Path]:
    return {spec.key: verify_model(model_root, spec) for spec in MODEL_SPECS}


def _download_once(
    spec: ModelSpec,
    destination: Path,
    callback: ProgressCallback | None,
) -> None:
    partial = destination.with_name(destination.name + ".part")
    if partial.is_file() and partial.stat().st_size == spec.size:
        if sha256_file(partial) == spec.sha256:
            os.replace(partial, destination)
            return
        partial.unlink()
    if partial.is_file() and partial.stat().st_size > spec.size:
        partial.unlink()

    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "BandBuddy/2.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(spec.source_url, headers=headers)
    with urlopen(request, timeout=300, context=ssl.create_default_context()) as response:
        status = getattr(response, "status", response.getcode())
        resumed = offset > 0 and status == 206
        if not resumed:
            offset = 0
        mode = "ab" if resumed else "wb"
        written = offset
        with partial.open(mode) as handle:
            while True:
                block = response.read(CHUNK_BYTES)
                if not block:
                    break
                handle.write(block)
                written += len(block)
                if callback:
                    callback(
                        spec.key,
                        min(0.99, written / spec.size),
                        f"下载 {spec.display_name}",
                    )
            handle.flush()
            os.fsync(handle.fileno())

    if written != spec.size:
        raise OSError(f"MODEL_DOWNLOAD_INCOMPLETE:{spec.key}:{written}/{spec.size}")
    actual = sha256_file(partial)
    if actual != spec.sha256:
        partial.unlink(missing_ok=True)
        raise ModelStoreError(
            f"MODEL_HASH_MISMATCH:{spec.key}:expected={spec.sha256}:actual={actual}"
        )
    os.replace(partial, destination)


def download_model(
    model_root: Path,
    spec: ModelSpec,
    *,
    retries: int = 8,
    callback: ProgressCallback | None = None,
) -> Path:
    destination = model_path(model_root, spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        verified = verify_model(model_root, spec)
        if callback:
            callback(spec.key, 1.0, f"已校验 {spec.display_name}")
        return verified
    except ModelStoreError:
        pass

    last_error: BaseException | None = None
    retry_types = (HTTPError, URLError, TimeoutError, OSError, ssl.SSLError, ModelStoreError)
    for attempt in range(retries + 1):
        try:
            _download_once(spec, destination, callback)
            verified = verify_model(model_root, spec)
            if callback:
                callback(spec.key, 1.0, f"已下载并校验 {spec.display_name}")
            return verified
        except retry_types as error:
            last_error = error
            if attempt >= retries:
                break
            if callback:
                callback(
                    spec.key,
                    0.0,
                    f"网络中断，保留断点并重试 {attempt + 1}/{retries}",
                )
            time.sleep(min(10.0, 1.0 + attempt * 1.5))
    raise ModelStoreError(f"MODEL_DOWNLOAD_FAILED:{spec.key}:{last_error}") from last_error


def ensure_models(
    model_root: Path,
    *,
    download_missing: bool,
    callback: ProgressCallback | None = None,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for spec in MODEL_SPECS:
        try:
            resolved[spec.key] = verify_model(model_root, spec)
        except ModelStoreError:
            if not download_missing:
                raise
            resolved[spec.key] = download_model(model_root, spec, callback=callback)
    return resolved
