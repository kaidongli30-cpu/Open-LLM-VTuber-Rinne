"""Install Rinne V2 weights and verify the public reference WAVs."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

RELEASE_TAG = "rinne-gpt-sovits-v2-20260923"
RELEASE_BASE = (
    "https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne/releases/"
    f"download/{RELEASE_TAG}"
)
INFERENCE_CONFIG_NAME = "rinne_public_v2_tts_infer.yaml"
REFERENCE_DIR = Path(__file__).resolve().parent / "assets" / "rinne-voice-v2"
REFERENCE_HASHES = {
    "default.wav": "a25fa9fa0f873989bf330acbb8b3386ba87657008019d1a6899053ec982b3640",
    "surprise.wav": "0f4911f93b27e055420d5792f7574608b4b92bde87afb92012952591dc4e2dbb",
    "shy_aux.wav": "a03653e31ea4f07bddaade481b52f3ec4196b316ae2f7bb5505005a3454dcb19",
    "angry.wav": "291ef50f9b4e51395b30b474c279f98d83ec41f40594fc695cca77b6dbc98949",
    "spirit.wav": "bf41c4d27e1fdccb6d0806c0703d5e568d0ec7064830845b3e7f502293a86e41",
}
RUNTIME_HASHES = {
    # Hash UTF-8 text after normalizing line endings: the same package was
    # observed with both CRLF and LF copies, while Python executes them alike.
    "api_v2.py": "7a34a5bad6c06fb282b20463987ae5d6a0a65a9cfaa4aadbc520fafb4722f743",
    "GPT_SoVITS/TTS_infer_pack/TTS.py": "83a849f2d0accc8a9d51e7ad1cb7475fa135eb7952de8dfae377ea95e5d85c09",
    "config.py": "be2264473f508ee566f711218ceb203fd50b2844de2878fee3c35766815e7013",
}


@dataclass(frozen=True)
class ModelAsset:
    relative_path: str
    sha256: str


ASSETS = (
    ModelAsset(
        "GPT_weights_v2/rinne_e15.ckpt",
        "503dd54e06ce69b359fe8638d9eefb9367e83b3cc3b9bf1664132a0d8d2938d3",
    ),
    ModelAsset(
        "SoVITS_weights_v2/rinne_e8_s456.pth",
        "4c851f813dbbd80c7d275d20460be7690c5171d4cb443105eb19f26b0132a19a",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(directory: Path, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=".rinne-v2-", suffix=suffix, dir=directory, delete=False
    ) as handle:
        return Path(handle.name)


def install_asset(asset: ModelAsset, gpt_root: Path) -> None:
    target = gpt_root / asset.relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != asset.sha256:
            raise RuntimeError(
                f"Existing model differs from the public V2 release: {target}. "
                "Move it aside manually before retrying."
            )
        print(f"Verified existing V2 model: {target}")
        return

    temporary = _temporary_path(target.parent, ".download")
    url = f"{RELEASE_BASE}/{target.name}"
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "Rinne-public-V2-installer"}
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            with temporary.open("wb") as output:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
        if sha256_file(temporary) != asset.sha256:
            raise RuntimeError(f"SHA-256 verification failed for {url}")
        os.replace(temporary, target)
        print(f"Installed verified V2 model: {target}")
    finally:
        temporary.unlink(missing_ok=True)


def verify_references() -> None:
    for name, expected in REFERENCE_HASHES.items():
        path = REFERENCE_DIR / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Public V2 reference missing or altered: {path}")
    print(f"Verified {len(REFERENCE_HASHES)} public V2 reference WAV files")


def verify_runtime(gpt_root: Path) -> None:
    for relative_path, expected in RUNTIME_HASHES.items():
        path = gpt_root / relative_path
        actual = (
            hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            if path.is_file()
            else None
        )
        if actual != expected:
            raise RuntimeError(
                f"GPT-SoVITS runtime differs from the tested "
                f"GPT-SoVITS-v2pro-20250604 package: {path}"
            )
    print("Verified tested GPT-SoVITS-v2pro-20250604 runtime files")


def write_inference_config(gpt_root: Path) -> Path:
    source = gpt_root / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    if not source.is_file():
        raise FileNotFoundError(f"GPT-SoVITS inference template not found: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if (
        not isinstance(loaded, dict)
        or not isinstance(loaded.get("custom"), dict)
        or not isinstance(loaded.get("v2"), dict)
    ):
        raise ValueError("GPT-SoVITS inference template needs custom and v2 profiles")
    profile = dict(loaded["custom"])
    profile.update(
        {
            "device": "cpu",
            "is_half": False,
            "t2s_weights_path": ASSETS[0].relative_path.replace("\\", "/"),
            "vits_weights_path": ASSETS[1].relative_path.replace("\\", "/"),
            "version": "v2",
        }
    )
    for name in ("bert_base_path", "cnhuhbert_base_path"):
        value = str(profile.get(name, "")).strip()
        if not value or not (gpt_root / value).exists():
            raise FileNotFoundError(f"GPT-SoVITS pretrained model missing: {name}")
    target = gpt_root / INFERENCE_CONFIG_NAME
    loaded["custom"] = profile
    content = yaml.safe_dump(loaded, allow_unicode=True, sort_keys=False)
    if target.exists():
        if target.read_text(encoding="utf-8") != content:
            raise RuntimeError(
                f"Existing inference config differs: {target}. "
                "Move it aside manually before retrying."
            )
    else:
        temporary = _temporary_path(gpt_root, ".yaml")
        try:
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Prepared V2 inference config: {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt-root", type=Path, required=True)
    args = parser.parse_args()
    gpt_root = args.gpt_root.resolve()
    if (
        not (gpt_root / "api_v2.py").is_file()
        or not (gpt_root / "runtime" / "python.exe").is_file()
    ):
        parser.error(
            "--gpt-root must be the extracted Windows GPT-SoVITS package "
            "directory containing api_v2.py and runtime/python.exe"
        )

    if not (gpt_root / "GPT_SoVITS" / "configs" / "tts_infer.yaml").is_file():
        parser.error("GPT-SoVITS inference template is missing")

    verify_runtime(gpt_root)
    verify_references()
    for asset in ASSETS:
        install_asset(asset, gpt_root)
    write_inference_config(gpt_root)
    print("Start the voice service from --gpt-root with:")
    print(f"  runtime\\python.exe api_v2.py -c {INFERENCE_CONFIG_NAME}")
    print(f"Reference WAVs are supplied by this repository: {REFERENCE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
