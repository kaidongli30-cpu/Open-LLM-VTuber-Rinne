"""MCP bridge for local documents, images, and cached video observations."""

from __future__ import annotations

import base64
import logging
import os
import sys
from pathlib import Path

# The MCP stdio protocol owns stdout.  Keep diagnostics on stderr.
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("rinne-library")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.server.fastmcp import Image  # noqa: E402

from src.open_llm_vtuber.file_library import (  # noqa: E402
    find_library_files,
    list_library_folders,
    list_library_files,
    read_document,
    view_image,
)
from src.open_llm_vtuber.library_filename_search import (  # noqa: E402
    default_filename_search,
)
from src.open_llm_vtuber.video_analysis import (  # noqa: E402
    read_cached_video_analysis,
)
LIBRARY_ROOT = os.environ.get("RINNE_LIBRARY_ROOT")
FILENAME_SEARCH = default_filename_search(LIBRARY_ROOT, repo_root=REPO_ROOT)
mcp = FastMCP("rinne-library")


@mcp.tool()
def library_list_files(
    kind: str = "all",
    folder: str = "",
    name_contains: str = "",
) -> dict:
    """列出凛祢本地资料库中的文件。

    资料库只按文件名、文件夹和文件类型筛选，不搜索文件正文。
    kind 可选 all、document、image、video；folder 使用 documents/、images/ 或
    videos/ 下的相对路径。
    """

    return {
        "library": "rinne_01",
        "files": list_library_files(
            LIBRARY_ROOT,
            kind=kind,
            folder=folder,
            name_contains=name_contains,
        ),
    }


@mcp.tool()
def library_list_folders(
    kind: str = "all",
    folder: str = "",
    recursive: bool = False,
) -> dict:
    """浏览凛祢资料库中的真实文件夹，不读取文件正文。

    folder 是资料库相对路径；不带 documents/、images/ 或 videos/ 前缀时，
    kind=all 会同时浏览三个类型根目录。默认只返回直接子目录，需要展开整棵
    树时传 recursive=true。
    """

    return {
        "library": "rinne_01",
        "folders": list_library_folders(
            LIBRARY_ROOT,
            kind=kind,
            folder=folder,
            recursive=recursive,
        ),
    }


@mcp.tool()
def library_find_by_filename(
    name: str = "",
    kind: str = "all",
    query: str = "",
    limit: int = 8,
    semantic: bool = True,
    min_score: float = 0.45,
) -> dict:
    """按文件名查找资料库文件，可用本地 BGE 补充语义相近的文件名。

    ``query`` is kept as a compatibility alias because some OpenAI-compatible
    models use that natural parameter name after seeing an earlier tool result.
    搜索范围始终只有文件名，不读取文档、图片或视频内容。
    """

    resolved_name = str(name or query).strip()
    exact = [
        {**item, "match_type": "literal_filename", "semantic_score": 1.0}
        for item in find_library_files(resolved_name, LIBRARY_ROOT, kind=kind)
    ]
    files = list(exact[:limit])
    semantic_status = "disabled"
    if semantic and len(files) < limit:
        try:
            semantic_matches = FILENAME_SEARCH.search(
                resolved_name,
                kind=kind,
                limit=limit,
                min_score=min_score,
            )
            known_paths = {item["relative_path"] for item in files}
            for item in semantic_matches:
                if item["relative_path"] in known_paths:
                    continue
                files.append(item)
                known_paths.add(item["relative_path"])
                if len(files) >= limit:
                    break
            semantic_status = "used"
        except Exception as exc:  # exact filename lookup remains available
            logger.warning("Filename semantic search unavailable: %s", type(exc).__name__)
            semantic_status = "unavailable"
    return {
        "query": resolved_name,
        "search_scope": "filenames_only",
        "semantic_status": semantic_status,
        "files": files,
    }


@mcp.tool()
def library_read_document(
    file_id_or_path: str = "",
    start: int = 0,
    max_chars: int = 12_000,
    file_id: str = "",
    path: str = "",
    file_path: str = "",
) -> dict:
    """读取 txt、md、pdf 或 docx 文件的一段正文。需要继续阅读时增加 start。

    ``file_id``, ``path`` and ``file_path`` are compatibility aliases for
    ``file_id_or_path``.
    """

    reference = file_id_or_path or file_id or path or file_path
    return read_document(
        reference,
        start=start,
        max_chars=max_chars,
        root=LIBRARY_ROOT,
    )


@mcp.tool()
def library_view_image(
    file_id_or_path: str = "",
    file_id: str = "",
    path: str = "",
    file_path: str = "",
) -> Image:
    """查看资料库中的图片原图；传入文件 ID、相对路径或兼容别名。"""

    reference = file_id_or_path or file_id or path or file_path
    record = view_image(reference, root=LIBRARY_ROOT)
    image_format = record["mime_type"].removeprefix("image/")
    return Image(
        data=base64.b64decode(record["data"]),
        format=image_format,
    )


@mcp.tool()
def library_read_video_analysis(
    file_id_or_path: str = "",
    file_id: str = "",
    path: str = "",
    file_path: str = "",
) -> dict:
    """读取已完成并缓存的视频观察；不能根据文件名猜测视频内容。"""

    reference = file_id_or_path or file_id or path or file_path
    return read_cached_video_analysis(reference, root=LIBRARY_ROOT)


if __name__ == "__main__":
    logger.info("Starting Rinne library MCP server at %s", LIBRARY_ROOT or "rinne_library/rinne_01")
    mcp.run()
