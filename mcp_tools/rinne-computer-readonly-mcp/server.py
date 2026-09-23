"""Strictly read-only MCP bridge for explicitly allowlisted computer folders."""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict
from pathlib import Path

# The MCP stdio protocol owns stdout. Keep diagnostics on stderr.
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("rinne-computer-readonly")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from src.open_llm_vtuber.agent_runtime.read_only_config import (  # noqa: E402
    ReadOnlyComputerConfig,
)


_config = ReadOnlyComputerConfig.from_environment()
_files = _config.create_service()
mcp = FastMCP("rinne-computer-readonly")


@mcp.tool()
def computer_file_roots() -> dict:
    """查看凛祢当前获准读取的文件夹根目录；不会访问目录内容。"""

    return {"allowed_roots": list(_files.allowed_roots)}


@mcp.tool()
def computer_file_stat(path: str) -> dict:
    """只读查看一个已授权文件或文件夹的类型、大小和修改时间。"""

    return {"entry": asdict(_files.stat(path))}


@mcp.tool()
def computer_list_directory(path: str, limit: int = 50) -> dict:
    """只读列出已授权目录的直接子项；不会递归，也不会修改任何文件。"""

    entries = _files.list_directory(path, limit=limit)
    return {"entries": [asdict(entry) for entry in entries]}


@mcp.tool()
def computer_find_by_name(
    query: str,
    within: str,
    limit: int = 50,
) -> dict:
    """只按名称在已授权目录中查找文件或文件夹；不搜索正文。"""

    entries = _files.find_by_name(query, within=within, limit=limit)
    return {
        "query": query,
        "entries": [asdict(entry) for entry in entries],
    }


@mcp.tool()
def computer_read_text(
    path: str,
    start: int = 0,
    max_chars: int = 12_000,
) -> dict:
    """分段读取已授权的 UTF-8 或带 BOM UTF-16 文本；不支持二进制文件。"""

    if start < 0:
        raise ValueError("start must not be negative")
    if max_chars < 1 or max_chars > 20_000:
        raise ValueError("max_chars must be between 1 and 20000")

    entry = _files.stat(path)
    text = _files.read_text(path)
    chunk = text[start : start + max_chars]
    next_start = start + len(chunk)
    return {
        "path": entry.path,
        "text": chunk,
        "start": start,
        "next_start": next_start if next_start < len(text) else None,
        "total_chars": len(text),
    }


if __name__ == "__main__":
    logger.info(
        "Starting read-only computer MCP server with %d allowed root(s)",
        len(_files.allowed_roots),
    )
    mcp.run()
