"""Install user-owned Rinne game assets without redistributing them.

The public project contains setup code and three approved transparent spirit
expression overlays. Original PCK files and generated GPU bundles stay local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from string import hexdigits
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_VERSION = 1
INSTALL_MARKER = ".rinne-local-install.json"
FAMILY_MANIFEST = "first-outfit-manifest.json"
EXPECTED_PORTRAIT_IDS = tuple(range(60101, 60116))
OUTFIT_PROFILES = {
    1: ("mp_summer_uniform", "rinne-legacy-gpu-first-outfit", FAMILY_MANIFEST),
    2: (
        "mp_red_white_ruffled_casual",
        "rinne-legacy-gpu-outfit-family",
        "outfit-manifest.json",
    ),
    3: (
        "mp_red_cardigan_brown_skirt",
        "rinne-legacy-gpu-outfit-family",
        "outfit-manifest.json",
    ),
    4: (
        "mp_dark_navy_winter_uniform",
        "rinne-legacy-gpu-outfit-family",
        "outfit-manifest.json",
    ),
    5: (
        "mp_spirit_dress",
        "rinne-spirit-dress-expression-preview",
        "spirit-expression-preview.json",
    ),
}
EXPECTED_RUNTIME_FILES = frozenset(
    {
        "manifest.json",
        "texture.rgba8",
        "geometry.bin",
        "dynamic-manifest.json",
        "deformation.rgba32f",
        "eye-dynamic-manifest.json",
        "eye-deformation.rgba32f",
        "runtime-control-manifest.json",
        "timeline-manifest.json",
        "amb-channels.f32",
    }
)


class AssetSetupError(RuntimeError):
    """A readable failure that leaves the game source untouched."""


@dataclass(frozen=True)
class BundleValidation:
    root: Path
    profile_id: str
    outfit_number: int
    manifest_sha256: str
    portrait_count: int
    total_bytes: int

    def public_payload(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "portrait_count": self.portrait_count,
            "total_bytes": self.total_bytes,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path(root: str | Path | None = None) -> Path:
    base = project_root() if root is None else Path(root).resolve()
    return base / "local_config" / "rinne_game_assets.json"


def default_asset_root(environ: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environ is None else environ
    explicit = values.get("RINNE_GAME_ASSET_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    local_app_data = values.get("LOCALAPPDATA")
    if local_app_data:
        return (
            Path(local_app_data)
            / "Open-LLM-VTuber-Rinne"
            / "game-assets"
        )
    return Path.home() / ".local" / "share" / "Open-LLM-VTuber-Rinne" / "game-assets"


def desktop_settings_path(environ: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environ is None else environ
    explicit = values.get("RINNE_RENDERER_SETTINGS_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    app_data = values.get("APPDATA")
    if app_data:
        return Path(app_data) / "open-llm-vtuber" / "rinne-legacy-renderer.json"
    return Path.home() / ".config" / "open-llm-vtuber" / "rinne-legacy-renderer.json"


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AssetSetupError(f"{label}不存在：{path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssetSetupError(f"{label}无法读取或不是有效 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise AssetSetupError(f"{label}顶层必须是 JSON 对象：{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise AssetSetupError(f"资源清单包含不安全路径：{relative}")
    resolved_root = root.resolve()
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise AssetSetupError(f"资源清单路径越过资源目录：{relative}") from exc
    return resolved


def validate_first_outfit_bundle(
    bundle_directory: str | Path,
    *,
    outfit_number: int = 1,
    verify_hashes: bool = True,
) -> BundleValidation:
    """Validate one SDK-produced fifteen-portrait local runtime bundle."""

    root = Path(bundle_directory).expanduser().resolve()
    if not root.is_dir():
        raise AssetSetupError(f"运行资源目录不存在：{root}")
    if outfit_number not in OUTFIT_PROFILES:
        raise AssetSetupError(f"不支持的服装编号：{outfit_number}")
    if outfit_number == 5:
        return validate_spirit_bundle(root)
    profile_id, format_name, manifest_name = OUTFIT_PROFILES[outfit_number]
    expected_ids = tuple(range(60000 + outfit_number * 100 + 1, 60000 + outfit_number * 100 + 16))
    manifest_path = root / manifest_name
    manifest = _load_json(manifest_path, label="凛祢运行资源清单")
    if manifest.get("format") != format_name:
        raise AssetSetupError(f"不是受支持的第 {outfit_number} 套服装运行资源格式")
    if outfit_number > 1 and manifest.get("outfit_number") != outfit_number:
        raise AssetSetupError("运行资源清单的服装编号不匹配")
    if manifest.get("version") != 1:
        raise AssetSetupError(f"不支持的运行资源版本：{manifest.get('version')!r}")
    if manifest.get("portrait_count") != len(expected_ids):
        raise AssetSetupError("运行资源必须完整包含 15 个表情肖像")
    source_policy = manifest.get("source_policy")
    if not isinstance(source_policy, dict) or not all(
        source_policy.get(key) is True
        for key in (
            "bundle_is_local_only",
            "game_assets_are_not_stored_in_source_repository",
            "original_pcks_remain_read_only",
        )
    ):
        raise AssetSetupError("运行资源清单缺少本地只读来源声明")

    portraits = manifest.get("portraits")
    if not isinstance(portraits, list):
        raise AssetSetupError("运行资源清单缺少 portraits 列表")
    by_id: dict[int, dict[str, Any]] = {}
    for raw in portraits:
        if not isinstance(raw, dict) or isinstance(raw.get("portrait_id"), bool):
            raise AssetSetupError("运行资源清单包含无效肖像条目")
        portrait_id = raw.get("portrait_id")
        if not isinstance(portrait_id, int) or portrait_id in by_id:
            raise AssetSetupError("运行资源清单包含重复或无效肖像编号")
        by_id[portrait_id] = raw
    if tuple(sorted(by_id)) != expected_ids:
        raise AssetSetupError(f"运行资源的肖像编号不符合第 {outfit_number} 套服装")

    total_bytes = manifest_path.stat().st_size
    for portrait_id in expected_ids:
        entry = by_id[portrait_id]
        expected_directory = f"MP{portrait_id:06d}"
        if entry.get("directory") != expected_directory:
            raise AssetSetupError(f"肖像 {portrait_id} 的目录名不匹配")
        portrait_root = _safe_member(root, expected_directory)
        if not portrait_root.is_dir():
            raise AssetSetupError(f"缺少肖像目录：{expected_directory}")
        files = entry.get("files")
        if not isinstance(files, list):
            raise AssetSetupError(f"肖像 {portrait_id} 缺少文件清单")
        described: set[str] = set()
        for file_entry in files:
            if not isinstance(file_entry, dict):
                raise AssetSetupError(f"肖像 {portrait_id} 包含无效文件记录")
            name = file_entry.get("name")
            byte_length = file_entry.get("byte_length")
            expected_hash = file_entry.get("sha256")
            if not isinstance(name, str) or name in described:
                raise AssetSetupError(f"肖像 {portrait_id} 包含重复或无效文件名")
            if not isinstance(byte_length, int) or isinstance(byte_length, bool):
                raise AssetSetupError(f"肖像 {portrait_id}/{name} 文件长度无效")
            if (
                not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(character not in hexdigits for character in expected_hash)
            ):
                raise AssetSetupError(f"肖像 {portrait_id}/{name} 哈希无效")
            described.add(name)
            file_path = _safe_member(portrait_root, name)
            if not file_path.is_file():
                raise AssetSetupError(f"缺少运行文件：{expected_directory}/{name}")
            actual_length = file_path.stat().st_size
            if actual_length != byte_length:
                raise AssetSetupError(f"运行文件长度不符：{expected_directory}/{name}")
            if verify_hashes and _sha256(file_path) != expected_hash.lower():
                raise AssetSetupError(f"运行文件哈希不符：{expected_directory}/{name}")
            total_bytes += actual_length
        missing = EXPECTED_RUNTIME_FILES - described
        if missing:
            raise AssetSetupError(
                f"肖像 {portrait_id} 缺少必需文件：{', '.join(sorted(missing))}"
            )

    return BundleValidation(
        root=root,
        profile_id=profile_id,
        outfit_number=outfit_number,
        manifest_sha256=_sha256(manifest_path),
        portrait_count=len(expected_ids),
        total_bytes=total_bytes,
    )


def validate_spirit_bundle(bundle_directory: str | Path) -> BundleValidation:
    """Verify the seven native portraits and fifteen spirit expressions."""

    from .sdk.rinne_legacy_runtime.spirit_expression_preview import (
        verify_rinne_spirit_expression_preview_directory,
    )

    root = Path(bundle_directory).expanduser().resolve()
    try:
        manifest = verify_rinne_spirit_expression_preview_directory(root)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise AssetSetupError(f"灵装运行资源校验失败：{exc}") from exc
    if manifest.get("portrait_count") != 7 or manifest.get("expression_count") != 15:
        raise AssetSetupError("灵装必须包含七个原生肖像和十五个表情")
    total_bytes = (root / "spirit-expression-preview.json").stat().st_size
    for entry in (*manifest["portraits"], *manifest["overlays"]):
        total_bytes += sum(
            item["byte_length"] for item in entry.get("files", [entry])
        )
    return BundleValidation(
        root=root,
        profile_id="mp_spirit_dress",
        outfit_number=5,
        manifest_sha256=_sha256(root / "spirit-expression-preview.json"),
        portrait_count=7,
        total_bytes=total_bytes,
    )


def detect_outfit_number(bundle_directory: str | Path) -> int:
    root = Path(bundle_directory).expanduser().resolve()
    if (root / "spirit-expression-preview.json").is_file():
        return 5
    first = root / FAMILY_MANIFEST
    if first.is_file():
        return 1
    manifest = _load_json(root / "outfit-manifest.json", label="服装运行资源清单")
    number = manifest.get("outfit_number")
    if isinstance(number, bool) or not isinstance(number, int) or number not in OUTFIT_PROFILES:
        raise AssetSetupError("运行资源清单包含不支持的服装编号")
    return number


def load_local_asset_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = default_config_path() if path is None else Path(path).resolve()
    config = _load_json(config_path, label="本地游戏资源配置")
    if config.get("version") != CONFIG_VERSION:
        raise AssetSetupError(f"不支持的本地游戏资源配置版本：{config.get('version')!r}")
    assets = config.get("assets")
    if not isinstance(assets, dict):
        raise AssetSetupError("本地游戏资源配置缺少 assets")
    return config


def configured_bundle_path(
    profile_id: str = "mp_summer_uniform",
    *,
    config_path: str | Path | None = None,
) -> Path | None:
    try:
        config = load_local_asset_config(config_path)
    except AssetSetupError:
        return None
    entry = config["assets"].get(profile_id)
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        return None
    return Path(entry["path"]).expanduser().resolve()


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _public_config_payload(
    validation: BundleValidation,
    *,
    managed: bool,
    current: Mapping[str, object] | None = None,
) -> dict[str, object]:
    existing = dict(current or {})
    old_assets = existing.get("assets")
    assets = dict(old_assets) if isinstance(old_assets, dict) else {}
    kind = (
        "legacy_first_outfit" if validation.outfit_number == 1
        else "spirit" if validation.outfit_number == 5
        else "legacy_outfit_family"
    )
    assets[validation.profile_id] = {
        "kind": kind,
        "path": str(validation.root),
        "managed_by_setup": managed,
        **validation.public_payload(),
    }
    return {
        "version": CONFIG_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "assets": assets,
    }


def _desktop_settings_payload(
    validation: BundleValidation,
    current: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload = dict(current or {})
    normalized = validation.root.as_posix()
    first_outfit = (
        normalized if validation.outfit_number == 1
        else payload.get("first_outfit_dir") or normalized
    )
    payload.update(
        {
            "version": 2,
            "profile_id": validation.profile_id,
            "renderer": "rinne",
            "outfit_id": validation.profile_id,
            "outfit_dir": normalized,
            "first_outfit_dir": first_outfit,
            "asset_kind": "spirit" if validation.outfit_number == 5 else "outfit",
        }
    )
    payload.pop("custom_outfit_dir", None)
    return payload


def _existing_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _file_snapshot(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _restore_file_snapshot(path: Path, snapshot: bytes | None) -> None:
    if snapshot is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.restore")
    temporary.write_bytes(snapshot)
    os.replace(temporary, path)


def _write_runtime_pointers(
    validation: BundleValidation,
    *,
    managed: bool,
    config_path: Path,
    settings_path: Path,
) -> None:
    config_payload = _public_config_payload(
        validation,
        managed=managed,
        current=_existing_json_or_empty(config_path),
    )
    _write_json_atomic(
        config_path,
        config_payload,
    )
    current_settings = _existing_json_or_empty(settings_path)
    assets = config_payload.get("assets")
    first_entry = assets.get("mp_summer_uniform") if isinstance(assets, dict) else None
    if isinstance(first_entry, dict) and isinstance(first_entry.get("path"), str):
        current_settings["first_outfit_dir"] = (
            Path(first_entry["path"]).expanduser().resolve().as_posix()
        )
    _write_json_atomic(
        settings_path,
        _desktop_settings_payload(
            validation,
            current=current_settings,
        ),
    )


def install_prepared_bundle(
    bundle_directory: str | Path,
    *,
    destination: str | Path | None = None,
    use_in_place: bool = False,
    replace: bool = False,
    outfit_number: int | None = None,
    verify_hashes: bool = True,
    config_path: str | Path | None = None,
    settings_path: str | Path | None = None,
) -> BundleValidation:
    """Install or point at a complete SDK-produced runtime bundle atomically."""

    selected_number = outfit_number or detect_outfit_number(bundle_directory)
    source = validate_first_outfit_bundle(
        bundle_directory,
        outfit_number=selected_number,
        verify_hashes=verify_hashes,
    )
    local_config = default_config_path() if config_path is None else Path(config_path).resolve()
    desktop_config = (
        desktop_settings_path() if settings_path is None else Path(settings_path).resolve()
    )
    if use_in_place:
        _write_runtime_pointers(
            source,
            managed=False,
            config_path=local_config,
            settings_path=desktop_config,
        )
        return source

    final = (
        default_asset_root() / source.profile_id
        if destination is None
        else Path(destination).expanduser().resolve()
    )
    if final == source.root:
        raise AssetSetupError("源目录与安装目录相同；请使用 --use-in-place")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.install-{uuid.uuid4().hex}"
    backup = final.parent / f".{final.name}.backup-{uuid.uuid4().hex}"
    if final.exists() and not replace:
        raise AssetSetupError(f"安装目录已存在；请先移除或明确使用 --replace：{final}")
    old_moved = False
    final_installed = False
    config_snapshot = _file_snapshot(local_config)
    settings_snapshot = _file_snapshot(desktop_config)
    try:
        shutil.copytree(source.root, staging, copy_function=shutil.copy2)
        installed = validate_first_outfit_bundle(
            staging,
            outfit_number=selected_number,
            verify_hashes=verify_hashes,
        )
        marker = {
            "version": 1,
            "managed_by": "rinne_game_asset_setup",
            "profile_id": installed.profile_id,
            "install_path": str(final),
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": installed.manifest_sha256,
        }
        _write_json_atomic(staging / INSTALL_MARKER, marker)
        if final.exists():
            os.replace(final, backup)
            old_moved = True
        os.replace(staging, final)
        final_installed = True
        final_validation = BundleValidation(
            root=final,
            profile_id=installed.profile_id,
            outfit_number=installed.outfit_number,
            manifest_sha256=installed.manifest_sha256,
            portrait_count=installed.portrait_count,
            total_bytes=installed.total_bytes,
        )
        _write_runtime_pointers(
            final_validation,
            managed=True,
            config_path=local_config,
            settings_path=desktop_config,
        )
        if old_moved:
            shutil.rmtree(backup, ignore_errors=True)
        return final_validation
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if final_installed and final.exists():
            shutil.rmtree(final, ignore_errors=True)
        if old_moved and backup.exists():
            os.replace(backup, final)
        _restore_file_snapshot(local_config, config_snapshot)
        _restore_file_snapshot(desktop_config, settings_snapshot)
        raise


def build_from_game_source(
    game_directory: str | Path,
    sdk_directory: str | Path | None = None,
    *,
    output_directory: str | Path,
    outfit_number: int = 1,
    python_executable: str | Path | None = None,
) -> Path:
    """Run the bundled open SDK against the user's read-only PCKs."""

    source = Path(game_directory).expanduser().resolve()
    sdk_root = (
        Path(sdk_directory).expanduser().resolve()
        if sdk_directory is not None
        else Path(__file__).resolve().parent / "sdk"
    )
    output = Path(output_directory).expanduser().resolve()
    if outfit_number not in OUTFIT_PROFILES:
        raise AssetSetupError(f"不支持的服装编号：{outfit_number}")
    expected_ids = (
        range(160101, 160108)
        if outfit_number == 5
        else range(60000 + outfit_number * 100 + 1, 60000 + outfit_number * 100 + 16)
    )
    expected = tuple(source / f"MP{portrait_id:06d}.pck" for portrait_id in expected_ids)
    missing = [item.name for item in expected if not item.is_file()]
    if missing:
        raise AssetSetupError(
            f"所选目录缺少第 {outfit_number} 套服装的 PCK："
            + ", ".join(missing[:5])
        )
    exporter = sdk_root / "tools" / (
        "export_rinne_spirit_dress_bundle.py"
        if outfit_number == 5
        else "export_rinne_gpu_first_outfit_bundle.py"
    )
    package = sdk_root / "rinne_legacy_runtime" / "__init__.py"
    if not exporter.is_file() or not package.is_file():
        raise AssetSetupError("SDK 目录缺少导出器或 rinne_legacy_runtime 包")
    if output.exists():
        raise AssetSetupError(f"SDK 输出目录必须尚不存在：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    executable = str(python_executable or sys.executable)
    command = [executable, str(exporter), str(source), str(output)]
    if outfit_number == 5:
        command.extend(
            ["--overlay-directory", str(project_root() / "assets" / "rinne-spirit-overlays")]
        )
    else:
        command.extend(["--outfit-number", str(outfit_number)])
    try:
        subprocess.run(command, cwd=sdk_root, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)
        raise AssetSetupError(f"本地 SDK 转换失败：{type(exc).__name__}") from exc
    validate_first_outfit_bundle(output, outfit_number=outfit_number)
    return output


def remove_installed_bundle(
    *,
    profile_id: str = "mp_summer_uniform",
    config_path: str | Path | None = None,
    settings_path: str | Path | None = None,
    remove_data: bool = False,
) -> Path | None:
    """Remove pointers, and optionally setup-owned data, without touching sources."""

    local_config = default_config_path() if config_path is None else Path(config_path).resolve()
    desktop_config = (
        desktop_settings_path() if settings_path is None else Path(settings_path).resolve()
    )
    configured: Path | None = None
    managed = False
    remaining_assets: dict[str, Any] = {}
    if local_config.is_file():
        config = load_local_asset_config(local_config)
        remaining_assets = dict(config["assets"])
        entry = remaining_assets.pop(profile_id, None)
        if isinstance(entry, dict) and isinstance(entry.get("path"), str):
            configured = Path(entry["path"]).expanduser().resolve()
            managed = entry.get("managed_by_setup") is True
        if remove_data and configured is not None:
            marker = configured / INSTALL_MARKER
            if not managed or not marker.is_file():
                raise AssetSetupError("拒绝删除未由本工具管理的目录；配置尚未更改")
            marker_payload = _load_json(marker, label="本地安装标记")
            if (
                marker_payload.get("managed_by") != "rinne_game_asset_setup"
                or marker_payload.get("profile_id") != profile_id
                or marker_payload.get("install_path") != str(configured)
                or configured == Path(configured.anchor).resolve()
                or configured == Path.home().resolve()
                or configured == project_root().resolve()
            ):
                raise AssetSetupError("安装标记或目标路径不匹配；配置尚未更改")
        if remaining_assets:
            _write_json_atomic(
                local_config,
                {
                    "version": CONFIG_VERSION,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "assets": remaining_assets,
                },
            )
        else:
            local_config.unlink()
    current = _existing_json_or_empty(desktop_config)
    if current.get("renderer") == "rinne" and current.get("profile_id") == profile_id:
        for key in (
            "outfit_id",
            "outfit_dir",
            "first_outfit_dir",
            "asset_kind",
            "custom_outfit_dir",
        ):
            current.pop(key, None)
        replacement = next(
            (
                (key, value)
                for key, value in remaining_assets.items()
                if key in {entry[0] for entry in OUTFIT_PROFILES.values()}
                and isinstance(value, dict)
                and isinstance(value.get("path"), str)
            ),
            None,
        )
        if replacement is None:
            current.update({"version": 2, "profile_id": "live2d", "renderer": "live2d"})
        else:
            next_id, next_entry = replacement
            normalized = Path(next_entry["path"]).expanduser().resolve().as_posix()
            current.update(
                {
                    "version": 2,
                    "profile_id": next_id,
                    "renderer": "rinne",
                    "outfit_id": next_id,
                    "outfit_dir": normalized,
                    "first_outfit_dir": normalized,
                    "asset_kind": "spirit" if next_id == "mp_spirit_dress" else "outfit",
                }
            )
        _write_json_atomic(desktop_config, current)
    if remove_data and configured is not None:
        shutil.rmtree(configured)
    return configured


def _select_directory(title: str) -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise AssetSetupError("当前 Python 不包含 tkinter，请改用命令行参数") from exc
    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askdirectory(title=title, mustexist=True)
    finally:
        root.destroy()
    return Path(selected).resolve() if selected else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从用户自己的游戏文件配置本地凛祢运行资源（不会上传或修改源文件）"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    install = subparsers.add_parser("install", help="导入已由 SDK 生成的运行资源")
    install.add_argument("bundle_directory", nargs="?", type=Path)
    install.add_argument("--destination", type=Path)
    install.add_argument("--use-in-place", action="store_true")
    install.add_argument("--replace", action="store_true")
    install.add_argument("--fast", action="store_true", help="跳过大文件 SHA-256 复核")
    install.add_argument("--gui", action="store_true", help="用文件夹选择窗口选择资源")

    build = subparsers.add_parser("build", help="调用本机 SDK 从原版 PCK 生成并安装")
    build.add_argument("game_directory", nargs="?", type=Path)
    build.add_argument("--sdk-directory", type=Path)
    build.add_argument("--outfit-number", type=int, choices=(1, 2, 3, 4, 5), default=1)
    build.add_argument("--destination", type=Path)
    build.add_argument("--replace", action="store_true")
    build.add_argument("--gui", action="store_true")

    status = subparsers.add_parser("status", help="检查当前配置和资源完整性")
    status.add_argument("--fast", action="store_true")
    status.add_argument("--profile", choices=[item[0] for item in OUTFIT_PROFILES.values()])

    remove = subparsers.add_parser("remove", help="移除本地配置或工具生成的数据")
    remove.add_argument("--profile", choices=[item[0] for item in OUTFIT_PROFILES.values()], default="mp_summer_uniform")
    remove.add_argument("--delete-generated-data", action="store_true")
    remove.add_argument("--yes", action="store_true", help="确认删除工具生成的数据")
    return parser


def _status(*, verify_hashes: bool, profile_id: str | None = None) -> int:
    selected = (
        [profile_id]
        if profile_id is not None
        else [profile[0] for profile in OUTFIT_PROFILES.values()]
    )
    count = 0
    for candidate in selected:
        configured = configured_bundle_path(candidate)
        if configured is None:
            continue
        outfit_number = next(
            number
            for number, values in OUTFIT_PROFILES.items()
            if values[0] == candidate
        )
        result = validate_first_outfit_bundle(
            configured,
            outfit_number=outfit_number,
            verify_hashes=verify_hashes,
        )
        print(f"{candidate}: {result.root} ({result.portrait_count} 肖像)")
        print(f"清单 SHA-256：{result.manifest_sha256}")
        count += 1
    if count == 0:
        print("未配置本地游戏凛祢资源。")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "install":
            source = args.bundle_directory
            if args.gui:
                source = _select_directory("选择已由 Rinne SDK 生成的运行资源目录")
            if source is None:
                raise AssetSetupError("没有选择运行资源目录")
            result = install_prepared_bundle(
                source,
                destination=args.destination,
                use_in_place=args.use_in_place,
                replace=args.replace,
                verify_hashes=not args.fast,
            )
            print(f"安装完成：{result.root}")
            print("请完全退出并重新启动凛祢桌面客户端。")
            return 0
        if args.command == "build":
            game_directory = args.game_directory
            sdk_directory = args.sdk_directory
            if args.gui:
                game_directory = _select_directory("选择包含该套服装 PCK 的目录")
            if game_directory is None:
                raise AssetSetupError("必须选择游戏 PCK 目录")
            build_parent = default_asset_root().parent
            build_parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="rinne-game-build-",
                dir=build_parent,
            ) as temporary:
                generated = Path(temporary) / "generated"
                build_from_game_source(
                    game_directory,
                    sdk_directory,
                    output_directory=generated,
                    outfit_number=args.outfit_number,
                )
                result = install_prepared_bundle(
                    generated,
                    destination=args.destination,
                    replace=args.replace,
                    outfit_number=args.outfit_number,
                    verify_hashes=True,
                )
            print(f"转换并安装完成：{result.root}")
            print("请完全退出并重新启动凛祢桌面客户端。")
            return 0
        if args.command == "status":
            return _status(verify_hashes=not args.fast, profile_id=args.profile)
        if args.command == "remove":
            if args.delete_generated_data and not args.yes:
                raise AssetSetupError("删除生成数据必须同时使用 --yes")
            removed = remove_installed_bundle(
                profile_id=args.profile,
                remove_data=args.delete_generated_data,
            )
            if args.delete_generated_data and removed is not None:
                print(f"已删除本工具生成的运行资源：{removed}")
            else:
                print("已移除本地资源指针；用户源文件和已有运行资源未被删除。")
            return 0
    except AssetSetupError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
