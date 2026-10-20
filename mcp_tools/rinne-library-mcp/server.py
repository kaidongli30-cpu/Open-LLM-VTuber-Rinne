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
    list_library_folders,
    list_library_files,
    read_document,
    view_image,
)
from src.open_llm_vtuber.library_filename_search import (  # noqa: E402
    default_filename_search,
    find_filename_keywords,
    browse_filename_records,
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
    树时传 recursive=true。一次返回所选目录全部后代文件的名字和路径，
    files 可直接交给读取工具；无需为了看内容逐层再次列举。没有打开文件内容。
    """

    return {
        "library": "rinne_01",
        "folders": list_library_folders(
            LIBRARY_ROOT,
            kind=kind,
            folder=folder,
            recursive=recursive,
        ),
        "files": browse_filename_records(LIBRARY_ROOT, kind=kind, folder=folder),
        "files_scope": "all_descendants_filenames_only",
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
    """递归搜索 documents、images、videos 下的文件名，不搜索文件内容。

    多个关键词用空格分隔：全部命中优先，部分命中随后。目录层级不限，
    kind=all 包含任意文件格式。关键词已完整命中时直接返回；否则可用
    本地 BGE 补充意思相近的文件名。文件夹名不参与匹配。

    ``query`` is kept as a compatibility alias because some OpenAI-compatible
    models use that natural parameter name after seeing an earlier tool result.
    搜索范围始终只有文件名，不读取文档、图片或视频内容。
    """

    resolved_name = str(name or query).strip()
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    if not -1.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between -1 and 1")
    keywords = find_filename_keywords(resolved_name, LIBRARY_ROOT, kind=kind)
    files = list(keywords[:limit])
    semantic_status = "disabled"
    complete_keyword_match = any(
        item["keyword_match_count"] == item["keyword_count"] for item in keywords
    )
    if semantic and complete_keyword_match:
        semantic_status = "not_needed"
    elif semantic:
        try:
            search = getattr(FILENAME_SEARCH, "search_bounded", FILENAME_SEARCH.search)
            semantic_matches = search(
                resolved_name,
                kind=kind,
                limit=limit,
                min_score=min_score,
            )
            by_path = {item["relative_path"]: dict(item) for item in keywords}
            for item in semantic_matches:
                existing = by_path.get(item["relative_path"])
                if existing is None:
                    by_path[item["relative_path"]] = dict(item)
                else:
                    existing["semantic_score"] = item["semantic_score"]
            files = sorted(
                by_path.values(),
                key=lambda item: (
                    -item.get("keyword_match_count", 0),
                    -item.get("semantic_score", 0.0),
                    item["relative_path"].casefold(),
                ),
            )[:limit]
            semantic_status = "used"
        except Exception as exc:  # exact filename lookup remains available
            logger.warning(
                "Filename semantic search unavailable: %s", type(exc).__name__
            )
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
    """读取同一视频先前完成的观察，不能根据文件名猜测视频内容。

    这不是针对当前提问重新分析的结果；请保留原观察的范围和不确定性。
    视频文件发生变化时拒绝读取旧观察。
    """

    reference = file_id_or_path or file_id or path or file_path
    return read_cached_video_analysis(
        reference, root=LIBRARY_ROOT, require_matching_focus=False
    )


if __name__ == "__main__":
    try:
        FILENAME_SEARCH.prepare_dependencies()
    except Exception as exc:
        logger.warning(
            "Filename semantic dependencies unavailable: %s; keyword search remains available",
            type(exc).__name__,
        )
    logger.info(
        "Starting Rinne library MCP server at %s",
        LIBRARY_ROOT or "rinne_library/rinne_01",
    )
    mcp.run()
