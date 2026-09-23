"""Update an existing Rinne conf.yaml in place without moving personal data."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "config_templates" / "conf.rinne.public.yaml"
CONFIG = ROOT / "conf.yaml"

# These values belong to each installation. An update must never publish, print,
# replace, or move them, even when all other runtime settings are upgraded.
PRIVATE_FIELD_NAMES = {
    "llm_api_key",
    "api_key",
    "api_key_file",
    "bocha_api_key",
    "proxy_url",
    "conf_uid",
    "human_name",
    "persona_prompt",
    "model_cache_dir",
    "terminology_path",
    "ref_audio_path",
    "aux_ref_audio_paths",
    "prompt_text",
    "prompt_lang",
    "avatar",
    "live2d_model_name",
    "api_url",
}
PRIVATE_PATH_PREFIXES = {
    ("character_config", "tts_preprocessor_config", "translator_config", "ollama_local", "glossary_path"),
}
LOCAL_SYSTEM_FIELDS = {"host", "port", "config_alts_dir"}


def _preserve(path: tuple[str, ...], old_value: Any) -> bool:
    if old_value is None or old_value == "":
        return False
    if len(path) == 2 and path[0] == "system_config":
        return path[1] in LOCAL_SYSTEM_FIELDS
    return path[-1] in PRIVATE_FIELD_NAMES or any(
        path[: len(prefix)] == prefix for prefix in PRIVATE_PATH_PREFIXES
    )


def _upgrade(existing: Any, desired: Any, path: tuple[str, ...] = ()) -> list[str]:
    """Overlay current public behavior, leaving private and unknown keys in place."""

    changed: list[str] = []
    for key, new_value in desired.items():
        field_path = (*path, str(key))
        if key not in existing:
            existing[key] = new_value
            changed.append(".".join(field_path))
        elif isinstance(new_value, dict) and isinstance(existing[key], dict):
            changed.extend(_upgrade(existing[key], new_value, field_path))
        elif not _preserve(field_path, existing[key]) and existing[key] != new_value:
            existing[key] = new_value
            changed.append(".".join(field_path))
    return changed


def update_config(config_path: Path = CONFIG, *, apply: bool = False) -> tuple[list[str], Path | None]:
    config_path = config_path.resolve(strict=True)
    if config_path.name.lower() != "conf.yaml":
        raise ValueError("只接受现有的 conf.yaml")
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    with config_path.open("r", encoding="utf-8") as stream:
        existing = yaml.load(stream)
    with TEMPLATE.open("r", encoding="utf-8") as stream:
        desired = yaml.load(stream)
    if not isinstance(existing, dict) or not isinstance(desired, dict):
        raise ValueError("配置文件不是 YAML 对象")
    character = existing.get("character_config")
    if not isinstance(character, dict) or character.get("conf_name") != "rinne":
        raise ValueError("此工具只更新已有的凛祢配置；未修改文件")

    changed = _upgrade(existing, desired)
    if not apply or not changed:
        return changed, None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = config_path.with_name(f"conf.yaml.backup-{stamp}")
    if backup.exists():
        raise FileExistsError(f"备份已存在：{backup}")
    shutil.copy2(config_path, backup)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=config_path.parent,
            prefix="conf.yaml.updating-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            yaml.dump(existing, stream)
            stream.flush()
            os.fsync(stream.fileno())
        with temporary.open("r", encoding="utf-8") as stream:
            if not isinstance(yaml.load(stream), dict):
                raise ValueError("更新后的配置未通过 YAML 解析")
        os.replace(temporary, config_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return changed, backup


def main() -> None:
    parser = argparse.ArgumentParser(description="在原有 conf.yaml 上更新凛祢设置")
    parser.add_argument("--apply", action="store_true", help="先备份再写入；省略时只预览")
    args = parser.parse_args()
    changed, backup = update_config(apply=args.apply)
    if not changed:
        print("配置已经是当前版本；未修改文件。")
        return
    print("将更新的配置项：" if not args.apply else "已更新的配置项：")
    for field in changed:
        print(f"  {field}")
    if backup:
        print(f"原配置备份：{backup}")
        print("聊天、日记与资料库文件未移动。")
    else:
        print("预览完成；未修改文件。确认后加 --apply 执行。")


if __name__ == "__main__":
    main()
