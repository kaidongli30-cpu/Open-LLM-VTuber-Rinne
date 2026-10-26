"""Optional local background selection and compatibility aliases."""

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass(frozen=True)
class BackgroundCatalog:
    files: tuple[str, ...]
    aliases: dict[str, str]


def load_background_catalog(directory: str | Path) -> BackgroundCatalog | None:
    """Use only existing local images; ordinary installations need no catalog."""
    root = Path(directory)
    manifest = root / "catalog.json"
    if not manifest.is_file():
        return None

    def image_name(value):
        return (
            isinstance(value, str)
            and bool(value)
            and "/" not in value
            and "\\" not in value
            and Path(value).name == value
            and Path(value).suffix.lower() in (".jpg", ".jpeg", ".png", ".gif")
        )

    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        files, aliases = data["files"], data.get("aliases", {})
        if not isinstance(files, list) or not files or not all(map(image_name, files)):
            raise ValueError("invalid image list")
        if len(set(files)) != len(files):
            raise ValueError("duplicate image names")
        if not isinstance(aliases, dict) or not all(
            image_name(key) and value in files for key, value in aliases.items()
        ):
            raise ValueError("invalid image aliases")
        if any(key in files for key in aliases):
            raise ValueError("an image cannot also be an alias")
        resolved_root = root.resolve()
        if not all(
            (root / name).is_file() and (root / name).resolve().parent == resolved_root
            for name in files
        ):
            raise ValueError("image missing or outside background directory")
        return BackgroundCatalog(tuple(files), aliases)
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("背景清单不可用，继续使用原有背景列表。")
        return None
