"""Request-scoped library evidence; history text is never a new file observation."""

import json
import re
from pathlib import PurePosixPath
from typing import Any

LIBRARY_NAVIGATION_TOOLS = frozenset(
    {
        "library_find_by_filename",
        "library_list_files",
        "library_list_folders",
    }
)


def navigation_key(name: str, arguments: dict[str, Any]) -> str | None:
    if name not in LIBRARY_NAVIGATION_TOOLS:
        return None
    values = dict(arguments)
    if name == "library_find_by_filename":
        query = (
            str(values.pop("name", "") or values.pop("query", "")).strip().casefold()
        )
        values.pop("query", None)
        values["name"] = re.sub(r"(?<=[a-z0-9])[-_](?=[a-z0-9])", "", query)
        defaults = {"kind": "all", "limit": 8, "semantic": True, "min_score": 0.45}
    elif name == "library_list_files":
        defaults = {"kind": "all", "folder": "", "name_contains": ""}
    else:
        defaults = {"kind": "all", "folder": "", "recursive": False}
    for key, value in defaults.items():
        values.setdefault(key, value)
    if "folder" in values:
        values["folder"] = str(values["folder"]).replace("\\", "/").strip("/")
    return json.dumps([name, values], ensure_ascii=False, sort_keys=True)


class LibraryTurnEvidence:
    def __init__(self, *, current_image: bool = False):
        self.current_image = current_image
        self.calls: dict[str, tuple[str, dict[str, Any]]] = {}
        self.failed_search = False
        self.empty_search = False
        self.successful_reads: set[str] = set()
        self.failed_reads: set[str] = set()
        self.image_requested = False
        self.navigation_results: dict[str, tuple[Any, bool]] = {}
        self.navigation_exhausted = False
        self.error_ids: set[str] = set()

    def cached_navigation(self, tool_id: str):
        name, arguments = self.calls[str(tool_id)]
        return self.navigation_results.get(navigation_key(name, arguments))

    def register(self, tool_id: str, name: str, arguments: dict[str, Any]):
        if name.startswith("library_"):
            self.calls[str(tool_id)] = (name, arguments)
            self.image_requested |= (
                name == "library_view_image" or arguments.get("kind") == "image"
            )

    @staticmethod
    def _reference(arguments: dict[str, Any]) -> str:
        return str(
            next(
                (
                    arguments[key]
                    for key in ("file_id_or_path", "file_id", "path", "file_path")
                    if arguments.get(key)
                ),
                "文件",
            )
        )

    def observe_status(self, update: dict[str, Any]):
        call = self.calls.get(str(update.get("tool_id", "")))
        if call is None or update.get("status") not in {"error", "completed"}:
            return
        name, arguments = call
        failed = update["status"] == "error"
        if failed:
            self.error_ids.add(str(update.get("tool_id")))
        else:
            self.error_ids.discard(str(update.get("tool_id")))
        if name in {
            "library_find_by_filename",
            "library_list_files",
            "library_list_folders",
        }:
            if failed:
                self.failed_search = True
        elif name in {
            "library_view_image",
            "library_read_document",
            "library_read_video_analysis",
        }:
            reference = self._reference(arguments)
            if failed:
                self.failed_reads.add(reference)
            else:
                self.failed_reads.discard(reference)
                self.successful_reads.add(reference)

    def observe_results(self, results: list[dict[str, Any]]):
        for result in results:
            tool_id = str(
                result.get("tool_call_id")
                or result.get("tool_use_id")
                or result.get("tool_id")
                or ""
            )
            call = self.calls.get(tool_id)
            if call is None or call[0] not in LIBRARY_NAVIGATION_TOOLS:
                continue
            content = result.get("content")
            key = navigation_key(*call)
            cached_error = bool(
                result.get("is_error")
                or tool_id in self.error_ids
                or re.search(
                    r"^(?:Error|Unexpected error|工具.*(?:未执行|超时|不可用|未完成))",
                    str(content),
                )
            )
            self.navigation_results[key] = (content, cached_error)
            if isinstance(content, list):
                content = "\n".join(
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict)
                )
            try:
                payload = json.loads(content)
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict) and isinstance(payload.get("files"), list):
                if (
                    call[0] == "library_list_folders"
                    and not payload["files"]
                    and payload.get("folders")
                ):
                    # Discovering an empty directory does not recover an earlier
                    # failed filename lookup or supply file-content evidence.
                    continue
                self.empty_search = not payload["files"]
                self.failed_search = cached_error or (
                    self.empty_search
                    and payload.get("semantic_status") == "unavailable"
                )

    def instructions(self) -> str:
        if not self.calls:
            return ""
        return (
            "【本轮资料库证据】\n"
            f"检索失败={self.failed_search}；没有匹配候选={self.empty_search}；"
            f"实际读取成功={sorted(self.successful_reads)}；读取失败={sorted(self.failed_reads)}。\n"
            "只有本轮实际读取成功的文件或本轮新图片可作为内容证据。"
            "搜索候选仅证明文件名匹配。历史图片描述不能作为本轮已找到或已看到的证据。"
            "失败须如实说明；后续改写也必须保持这些事实，不得把失败改写成成功。"
            "没找到时用你平时和用户说话的语气，可以亲昵、难为情或轻轻道歉，"
            "不要向他念检索状态，不要断言文件不存在。"
            + (
                "相同的目录或查询结果已经返回，不再重复列举或搜索；请用现有结果回答，必要时只打开候选文件。"
                if self.navigation_exhausted
                else ""
            )
        )

    def guard_final(self, reply: str) -> str:
        if not self.calls:
            return reply
        failure = self.failed_search or self.empty_search or bool(self.failed_reads)
        if not failure:
            return reply
        if not self.successful_reads and not self.current_image:
            reports_failure = re.search(
                r"没(?:有)?(?:能)?(?:找到|找着|搜到|翻到|翻出|打开|看清|看见|看到)|找不到|打不开|暂时没|还没",
                reply,
            )
            claims_success = re.search(
                r"找到了|翻出来了|(?<!没)看到了|现在能看|已经看到|照片里|图片里|这张就是",
                reply,
            )
            claims_absence = re.search(
                r"不存在|没(?:有)?存|手机里没有|相册里没有", reply
            )
            technical_fallback = any(
                word in reply for word in ("资料库", "返回匹配", "检索", "搜索结果")
            )
            if reports_failure and not (
                claims_success or claims_absence or technical_fallback
            ):
                return reply
            label = "那张照片" if self.image_requested else "那份文件"
            if self.failed_search or self.failed_reads:
                return f"[embarrassed]（有些不好意思地靠近你）用户，你说的{label}，这次还没能打开确认……抱歉呀，本来想和你一起再看看的。"
            return f"[embarrassed]（轻轻捏了捏你的手）用户，我刚刚翻了一圈，还没翻到你说的{label}……有点不好意思，本来想拿出来和你一起看的。"
        # A successful subsequent read recovers a failed search, but not a
        # separate unreadable file in a multiple-file request.
        if self.failed_reads:
            names = [
                PurePosixPath(ref.replace("\\", "/")).name
                for ref in sorted(self.failed_reads)
            ]
            notice = "；".join(names)
            if not all(
                name in reply and ("失败" in reply or "没能" in reply) for name in names
            ):
                return f"{reply}\n另外，{notice}读取失败，我还没能确认它的内容。"
        return reply
