"""Runtime-only selection for Rinne's renderer and emotion profile.

The selected profile is applied in memory. ``conf.yaml`` and ``model_dict.json``
remain untouched during backend startup.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping


MP_EXPRESSION_MAP: dict[str, int] = {
    "focused": 9,
    "neutral": -1,
    "confused": 3,
    "gentle": 10,
    "awkward": 2,
    "happy": 4,
    "worried": 8,
    "surprise": 7,
    "surprised": 7,
    "dissatisfaction": 11,
    "flustered": 12,
    "sad": 5,
    "embarrassed": 13,
    "shy": 6,
    "uneasy": 14,
    "angry": 1,
}

LIVE2D_EXPRESSION_MAP: dict[str, int] = {
    "neutral": -1,
    "ambitious": 0,
    "happy": 4,
    "shy": 6,
    "surprise": 7,
    "surprised": 7,
    "sad": 5,
    "worried": 8,
    "angry": 1,
    "awkward": 2,
    "confused": 3,
}

_MP_GUIDANCE_START = "但是在回复中，你【必须】使用表情标签来表达情绪。"
_CORE_PRINCIPLE = "【核心原则】"
_MP_LATER_RULE = (
    "你【必须】按照上文的表情说明，在对应句子或语义段落开头使用表情标签。"
    "只有主要情绪发生变化时才切换标签，一段完整回复中使用的表情标签不得超过三个。"
)
_LIVE2D_GUIDANCE = (
    "但是在回复中，你【必须】使用Live2D表情标签来表达情绪。可用的情绪有："
    "[happy][shy][surprise][angry][sad][confused][ambitious][awkward][worried]。"
    "这能让你的Live2D形象根据情绪做出对应的表情。请根据当前的语境和你的真实感受，"
    "自然地选择合适的标签。"
)
_LIVE2D_LATER_RULE = (
    "你【必须】在每一句回复的开头加上提示当前表情的词语，包括："
    "[happy][shy][surprise][angry][sad][confused][ambitious][awkward][worried]。"
)
_SPIRIT_TTS_REFERENCE_CONFIG_KEY = "spirit_form_reference"
_PROFILE_ID_ALIASES = {
    "custom_navy_blazer_jk_skirt": "custom_navy_sailor_jk_skirt",
}


def _canonical_profile_id(profile_id: str) -> str:
    """Resolve renamed renderer profiles without breaking saved startup choices."""

    return _PROFILE_ID_ALIASES.get(profile_id, profile_id)


def _load_local_asset_entries(project_root: Path) -> dict[str, dict[str, object]]:
    config_path = project_root / "local_config" / "rinne_game_assets.json"
    try:
        parsed = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict) or parsed.get("version") != 1:
        return {}
    assets = parsed.get("assets")
    if not isinstance(assets, dict):
        return {}
    return {
        key: value
        for key, value in assets.items()
        if isinstance(key, str) and isinstance(value, dict)
    }


def _configured_path(
    entries: Mapping[str, Mapping[str, object]],
    profile_id: str,
    field: str,
) -> Path | None:
    value = entries.get(profile_id, {}).get(field)
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser().resolve()


def _profile_assets_available(profile: "RinneRendererProfile", root: Path) -> bool:
    if profile.renderer == "live2d":
        return (root / "live2d-models" / "rinne" / "rinne.model3.json").is_file()
    if profile.outfit_dir is None or not profile.outfit_dir.is_dir():
        return False
    manifest_names = (
        ("spirit-expression-preview.json",)
        if profile.asset_kind == "spirit"
        else ("outfit-manifest.json", "first-outfit-manifest.json")
    )
    if not any((profile.outfit_dir / name).is_file() for name in manifest_names):
        return False
    if profile.asset_kind != "custom_outfit":
        return True
    return bool(
        profile.custom_outfit_dir is not None
        and (profile.custom_outfit_dir / "custom-outfit-preview.json").is_file()
    )


@dataclass(frozen=True)
class RinneRendererProfile:
    profile_id: str
    menu_label: str
    renderer: str
    expression_profile: str
    outfit_dir: Path | None
    asset_kind: str = "outfit"
    tts_reference_config_key: str | None = None
    custom_outfit_dir: Path | None = None
    enabled: bool = True


@dataclass(frozen=True)
class RinneRendererStartupOutcome:
    profile: RinneRendererProfile
    settings_path: Path


class RinneRuntimeOutfitController:
    """Switch Rinne's outfit and matching default voice without a restart."""

    _TTS_REFERENCE_FIELDS = (
        "ref_audio_path",
        "prompt_lang",
        "prompt_text",
        "emotion_references",
    )

    def __init__(
        self,
        project_root: str | Path,
        source_config_data: Mapping[str, object],
        startup: RinneRendererStartupOutcome,
    ) -> None:
        if startup.profile.renderer != "rinne":
            raise ValueError("原 Live2D 模型不支持运行中换装")
        self._project_root = Path(project_root).resolve()
        self._source_config_data = copy.deepcopy(dict(source_config_data))
        self._settings_path = startup.settings_path
        self._profiles = {
            profile.profile_id: profile
            for profile in rinne_outfit_profiles(self._project_root)
        }
        self._current_profile = startup.profile

    @property
    def current_profile(self) -> RinneRendererProfile:
        return self._current_profile

    def catalog_payload(self) -> dict[str, object]:
        return {
            "available": True,
            "current_profile_id": self._current_profile.profile_id,
            "profiles": [
                {
                    "profile_id": profile.profile_id,
                    "menu_label": profile.menu_label,
                    "group": (
                        "custom"
                        if profile.asset_kind == "custom_outfit"
                        else "game"
                    ),
                    "asset_kind": profile.asset_kind,
                }
                for profile in self._profiles.values()
            ],
        }

    def switch(
        self,
        profile_id: str,
        *,
        tts_engines: Iterable[object | None],
    ) -> RinneRendererProfile:
        canonical_id = _canonical_profile_id(profile_id.strip())
        target = self._profiles.get(canonical_id)
        if target is None or not target.enabled:
            raise ValueError(f"不可用的凛祢服装：{profile_id}")
        if target.profile_id == self._current_profile.profile_id:
            return target

        _validate_profile_assets(target)
        reference = self._tts_reference_for(target)
        engines = self._unique_compatible_tts_engines(tts_engines)
        if not engines:
            raise RuntimeError("当前 TTS 不支持在运行中同步凛祢形态参考音")

        previous = self._current_profile
        snapshots = [
            {
                field: copy.deepcopy(getattr(engine, field))
                for field in self._TTS_REFERENCE_FIELDS
            }
            for engine in engines
        ]
        try:
            write_renderer_settings(target, self._settings_path)
            for engine in engines:
                for field in self._TTS_REFERENCE_FIELDS:
                    setattr(engine, field, copy.deepcopy(reference[field]))
        except Exception:
            for engine, snapshot in zip(engines, snapshots):
                for field, value in snapshot.items():
                    setattr(engine, field, value)
            write_renderer_settings(previous, self._settings_path)
            raise

        self._current_profile = target
        return target

    def _tts_reference_for(self, profile: RinneRendererProfile) -> dict[str, Any]:
        effective = apply_rinne_renderer_profile_to_config(
            self._source_config_data,
            profile,
        )
        character = effective.get("character_config")
        tts_config = character.get("tts_config") if isinstance(character, dict) else None
        gpt_sovits = (
            tts_config.get("gpt_sovits_tts")
            if isinstance(tts_config, dict)
            else None
        )
        if not isinstance(gpt_sovits, dict):
            raise RuntimeError("conf.yaml 缺少 gpt_sovits_tts，无法同步形态参考音")
        reference = {
            field: copy.deepcopy(gpt_sovits.get(field))
            for field in self._TTS_REFERENCE_FIELDS
        }
        if not all(reference[field] for field in self._TTS_REFERENCE_FIELDS[:3]):
            raise ValueError("形态默认参考音配置不完整")
        if not isinstance(reference["emotion_references"], dict):
            reference["emotion_references"] = {}
        return reference

    def _unique_compatible_tts_engines(
        self,
        engines: Iterable[object | None],
    ) -> list[object]:
        unique: list[object] = []
        seen: set[int] = set()
        for engine in engines:
            if engine is None or id(engine) in seen:
                continue
            if not all(hasattr(engine, field) for field in self._TTS_REFERENCE_FIELDS):
                continue
            seen.add(id(engine))
            unique.append(engine)
        return unique


def rinne_renderer_profiles(
    project_root: str | Path,
) -> tuple[RinneRendererProfile, ...]:
    root = Path(project_root).resolve()
    model_root = root / "Rinne_model"
    entries = _load_local_asset_entries(root)
    first_outfit = _configured_path(entries, "mp_summer_uniform", "path")
    if first_outfit is None:
        first_outfit = model_root / "rinne_legacy_runtime_bundle"

    def game_path(profile_id: str, fallback_name: str) -> Path:
        return _configured_path(entries, profile_id, "path") or model_root / fallback_name

    def custom_path(profile_id: str, fallback_name: str) -> Path:
        configured = _configured_path(entries, profile_id, "custom_outfit_path")
        public = root / "assets" / "rinne-original-outfits" / profile_id
        if configured is not None:
            return configured
        if public.is_dir():
            return public
        return model_root / fallback_name

    candidates = (
        RinneRendererProfile(
            profile_id="mp_summer_uniform",
            menu_label="游戏原画：夏季校服",
            renderer="rinne",
            expression_profile="mp",
            outfit_dir=first_outfit,
        ),
        RinneRendererProfile(
            profile_id="mp_red_white_ruffled_casual",
            menu_label="游戏原画：红白荷叶边便装",
            renderer="rinne",
            expression_profile="mp",
            outfit_dir=game_path(
                "mp_red_white_ruffled_casual",
                "rinne_legacy_runtime_bundle_red_white_ruffled_casual",
            ),
        ),
        RinneRendererProfile(
            profile_id="mp_red_cardigan_brown_skirt",
            menu_label="游戏原画：红色开衫与棕色半身裙便装",
            renderer="rinne",
            expression_profile="mp",
            outfit_dir=game_path(
                "mp_red_cardigan_brown_skirt",
                "rinne_legacy_runtime_bundle_red_cardigan_brown_skirt",
            ),
        ),
        RinneRendererProfile(
            profile_id="mp_dark_navy_winter_uniform",
            menu_label="游戏原画：深蓝色冬季校服",
            renderer="rinne",
            expression_profile="mp",
            outfit_dir=game_path(
                "mp_dark_navy_winter_uniform",
                "rinne_legacy_runtime_bundle_dark_navy_winter_uniform",
            ),
        ),
        RinneRendererProfile(
            profile_id="live2d",
            menu_label="原 Live2D 模型",
            renderer="live2d",
            expression_profile="live2d",
            outfit_dir=None,
            asset_kind="live2d",
        ),
        RinneRendererProfile(
            profile_id="mp_spirit_dress",
            menu_label="游戏原画：灵装凛祢",
            renderer="rinne",
            expression_profile="mp",
            outfit_dir=game_path(
                "mp_spirit_dress",
                "rinne_spirit_dress_runtime_bundle",
            ),
            asset_kind="spirit",
            tts_reference_config_key=_SPIRIT_TTS_REFERENCE_CONFIG_KEY,
        ),
        RinneRendererProfile(
            profile_id="custom_pink_long_sleeve_pajamas",
            menu_label="作者自制：粉色长袖睡衣",
            renderer="rinne",
            expression_profile="mp",
            outfit_dir=first_outfit,
            asset_kind="custom_outfit",
            custom_outfit_dir=custom_path(
                "custom_pink_long_sleeve_pajamas",
                "rinne_custom_outfit_pink_long_sleeve_pajamas",
            ),
        ),
        RinneRendererProfile(
            profile_id="custom_navy_sailor_jk_skirt",
            menu_label="作者自制：藏青水手服JK短裙装",
            renderer="rinne",
            expression_profile="mp",
            outfit_dir=first_outfit,
            asset_kind="custom_outfit",
            custom_outfit_dir=custom_path(
                "custom_navy_sailor_jk_skirt",
                "rinne_custom_outfit_navy_sailor_jk_skirt",
            ),
        ),
        RinneRendererProfile(
            profile_id="custom_ivory_duffle_winter_coat",
            menu_label="作者自制：象牙白牛角扣连帽棉服",
            renderer="rinne",
            expression_profile="mp",
            outfit_dir=first_outfit,
            asset_kind="custom_outfit",
            custom_outfit_dir=custom_path(
                "custom_ivory_duffle_winter_coat",
                "rinne_custom_outfit_ivory_duffle_winter_coat",
            ),
        ),
        RinneRendererProfile(
            profile_id="custom_pink_ruffle_swimdress",
            menu_label="作者自制：粉色蝴蝶结荷叶边裙式泳装",
            renderer="rinne",
            expression_profile="mp",
            outfit_dir=first_outfit,
            asset_kind="custom_outfit",
            custom_outfit_dir=custom_path(
                "custom_pink_ruffle_swimdress",
                "rinne_custom_outfit_pink_ruffle_swimdress",
            ),
        ),
    )
    return tuple(
        RinneRendererProfile(
            profile_id=profile.profile_id,
            menu_label=profile.menu_label,
            renderer=profile.renderer,
            expression_profile=profile.expression_profile,
            outfit_dir=profile.outfit_dir,
            asset_kind=profile.asset_kind,
            tts_reference_config_key=profile.tts_reference_config_key,
            custom_outfit_dir=profile.custom_outfit_dir,
            enabled=_profile_assets_available(profile, root),
        )
        for profile in candidates
    )


def rinne_outfit_profiles(
    project_root: str | Path,
) -> tuple[RinneRendererProfile, ...]:
    """Return the nine MP outfits exposed by the in-app clothes picker."""

    return tuple(
        profile
        for profile in rinne_renderer_profiles(project_root)
        if profile.renderer == "rinne" and profile.enabled
    )


def renderer_settings_path(environ: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environ is None else environ
    explicit = values.get("RINNE_RENDERER_SETTINGS_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    roaming = values.get("APPDATA")
    if roaming:
        return Path(roaming) / "open-llm-vtuber" / "rinne-legacy-renderer.json"
    return (
        Path.home()
        / "AppData"
        / "Roaming"
        / "open-llm-vtuber"
        / "rinne-legacy-renderer.json"
    )


def _load_settings(path: Path) -> dict:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _path_key(value: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(value)))


def _profile_from_saved_settings(
    profiles: tuple[RinneRendererProfile, ...], settings: Mapping[str, object]
) -> RinneRendererProfile | None:
    saved_id = settings.get("profile_id")
    if isinstance(saved_id, str):
        canonical_id = _canonical_profile_id(saved_id)
        matched = next(
            (item for item in profiles if item.profile_id == canonical_id), None
        )
        if matched is not None and matched.enabled:
            return matched
    if str(settings.get("renderer", "")).lower() == "live2d":
        return next(
            (
                item
                for item in profiles
                if item.profile_id == "live2d" and item.enabled
            ),
            None,
        )
    saved_dir = settings.get("outfit_dir") or settings.get("first_outfit_dir")
    if isinstance(saved_dir, str):
        saved_key = _path_key(saved_dir)
        for profile in profiles:
            if (
                profile.enabled
                and profile.outfit_dir is not None
                and _path_key(profile.outfit_dir) == saved_key
            ):
                return profile
    return None


def _validate_profile_assets(profile: RinneRendererProfile) -> None:
    if profile.renderer != "rinne":
        return
    if profile.outfit_dir is None or not profile.outfit_dir.is_dir():
        raise FileNotFoundError(
            f"所选凛祢服装资源目录不存在：{profile.outfit_dir or '未配置'}"
        )
    manifest_names = (
        ("spirit-expression-preview.json",)
        if profile.asset_kind == "spirit"
        else ("outfit-manifest.json", "first-outfit-manifest.json")
    )
    if not any((profile.outfit_dir / name).is_file() for name in manifest_names):
        raise FileNotFoundError(f"所选凛祢服装缺少资源清单：{profile.outfit_dir}")
    if profile.asset_kind == "custom_outfit":
        custom_root = profile.custom_outfit_dir
        if custom_root is None or not custom_root.is_dir():
            raise FileNotFoundError(
                f"所选凛祢自制服装目录不存在：{custom_root or '未配置'}"
            )
        if not (custom_root / "custom-outfit-preview.json").is_file():
            raise FileNotFoundError(f"所选凛祢自制服装缺少资源清单：{custom_root}")


def choose_rinne_renderer_profile(
    project_root: str | Path,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    environ: MutableMapping[str, str] | None = None,
    settings_path: str | Path | None = None,
    interactive: bool | None = None,
) -> RinneRendererProfile:
    values = os.environ if environ is None else environ
    profiles = rinne_renderer_profiles(project_root)
    destination = (
        renderer_settings_path(values) if settings_path is None else Path(settings_path)
    )
    saved = _profile_from_saved_settings(profiles, _load_settings(destination))
    forced_id = values.get("RINNE_RENDERER_PROFILE", "").strip()
    if forced_id:
        canonical_id = _canonical_profile_id(forced_id)
        selected = next(
            (profile for profile in profiles if profile.profile_id == canonical_id), None
        )
        if selected is None or not selected.enabled:
            raise ValueError(f"RINNE_RENDERER_PROFILE 不可用：{forced_id}")
        _validate_profile_assets(selected)
        return selected

    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        selected = saved or next((profile for profile in profiles if profile.enabled), None)
        if selected is None:
            raise FileNotFoundError(
                "没有可用的凛祢形象。请先运行 setup_rinne_game_assets.py，"
                "或按 public_docs/GAME_ASSET_SETUP.md 配置本地资源。"
            )
        _validate_profile_assets(selected)
        return selected

    output_fn("")
    output_fn("请选择本次启动时凛祢使用的形象：")
    for index, profile in enumerate(profiles, start=1):
        suffix = "" if profile.enabled else "（未安装）"
        output_fn(f"  [{index}] {profile.menu_label}{suffix}")
    while True:
        try:
            raw = input_fn(f"请输入 1-{len(profiles)} 后按回车：").strip()
        except EOFError:
            selected = saved or profiles[0]
            output_fn(f"未读取到输入，沿用：{selected.menu_label}")
            _validate_profile_assets(selected)
            return selected
        if not raw.isdigit() or not 1 <= int(raw) <= len(profiles):
            choices = "、".join(str(index) for index in range(1, len(profiles) + 1))
            output_fn(f"输入无效，请输入 {choices}。")
            continue
        selected = profiles[int(raw) - 1]
        if not selected.enabled:
            output_fn("这套衣服当前不可用，请选择其他选项。")
            continue
        _validate_profile_assets(selected)
        return selected


def write_renderer_settings(profile: RinneRendererProfile, destination: Path) -> None:
    settings = _load_settings(destination)
    settings.update(
        {
            "version": 2,
            "profile_id": profile.profile_id,
            "renderer": profile.renderer,
        }
    )
    if profile.outfit_dir is not None:
        normalized = profile.outfit_dir.as_posix()
        settings["outfit_id"] = profile.profile_id
        settings["outfit_dir"] = normalized
        settings["first_outfit_dir"] = normalized
    settings["asset_kind"] = profile.asset_kind
    if profile.custom_outfit_dir is not None:
        settings["custom_outfit_dir"] = profile.custom_outfit_dir.as_posix()
    else:
        settings.pop("custom_outfit_dir", None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def prepare_rinne_renderer_startup(
    project_root: str | Path,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    environ: MutableMapping[str, str] | None = None,
    settings_path: str | Path | None = None,
    interactive: bool | None = None,
) -> RinneRendererStartupOutcome:
    values = os.environ if environ is None else environ
    destination = (
        renderer_settings_path(values) if settings_path is None else Path(settings_path)
    )
    profile = choose_rinne_renderer_profile(
        project_root,
        input_fn=input_fn,
        output_fn=output_fn,
        environ=values,
        settings_path=destination,
        interactive=interactive,
    )
    write_renderer_settings(profile, destination)
    values["RINNE_RENDERER_PROFILE"] = profile.profile_id
    values["RINNE_EXPRESSION_PROFILE"] = profile.expression_profile
    output_fn(f"本次已选择：{profile.menu_label}")
    return RinneRendererStartupOutcome(profile=profile, settings_path=destination)


def active_rinne_emotion_map(
    live2d_model_name: str, environ: Mapping[str, str] | None = None
) -> dict[str, int] | None:
    if live2d_model_name.lower() != "rinne":
        return None
    values = os.environ if environ is None else environ
    profile = values.get("RINNE_EXPRESSION_PROFILE", "mp").strip().lower()
    if profile == "live2d":
        return dict(LIVE2D_EXPRESSION_MAP)
    return dict(MP_EXPRESSION_MAP)


def apply_active_rinne_emotion_map(
    model_dict: list[dict], environ: Mapping[str, str] | None = None
) -> list[dict]:
    result = copy.deepcopy(model_dict)
    for model in result:
        if isinstance(model, dict) and str(model.get("name", "")).lower() == "rinne":
            model["emotionMap"] = active_rinne_emotion_map("rinne", environ)
    return result


def apply_rinne_renderer_profile_to_config(
    config_data: Mapping[str, object], profile: RinneRendererProfile
) -> dict:
    result = copy.deepcopy(dict(config_data))
    character = result.get("character_config")
    if profile.expression_profile == "live2d":
        if not isinstance(character, dict):
            raise ValueError(
                "conf.yaml 缺少 character_config，无法切换 Live2D 表情提示词"
            )
        persona = character.get("persona_prompt")
        if not isinstance(persona, str):
            raise ValueError(
                "conf.yaml 缺少 persona_prompt，无法切换 Live2D 表情提示词"
            )
        start = persona.find(_MP_GUIDANCE_START)
        end = persona.find(_CORE_PRINCIPLE, start)
        if start < 0 or end < 0:
            raise ValueError("未找到 MP 表情提示词边界；为避免损坏个性设定，已停止启动")
        persona = persona[:start] + _LIVE2D_GUIDANCE + "\n\n" + persona[end:]
        if _MP_LATER_RULE not in persona:
            raise ValueError("未找到 MP 表情使用规则；为避免提示词混用，已停止启动")
        character["persona_prompt"] = persona.replace(
            _MP_LATER_RULE, _LIVE2D_LATER_RULE, 1
        )

    tts_config = character.get("tts_config") if isinstance(character, dict) else None
    gpt_sovits = (
        tts_config.get("gpt_sovits_tts") if isinstance(tts_config, dict) else None
    )
    spirit_reference = (
        gpt_sovits.pop(_SPIRIT_TTS_REFERENCE_CONFIG_KEY, None)
        if isinstance(gpt_sovits, dict)
        else None
    )

    if profile.tts_reference_config_key is not None:
        if not isinstance(character, dict):
            raise ValueError("conf.yaml 缺少 character_config，无法切换形态默认参考音")
        if not isinstance(tts_config, dict):
            raise ValueError("conf.yaml 缺少 tts_config，无法切换形态默认参考音")
        if not isinstance(gpt_sovits, dict):
            raise ValueError("conf.yaml 缺少 gpt_sovits_tts，无法切换形态默认参考音")
        if profile.tts_reference_config_key != _SPIRIT_TTS_REFERENCE_CONFIG_KEY:
            raise ValueError(
                f"不支持的凛祢形态语音配置：{profile.tts_reference_config_key}"
            )
        if not isinstance(spirit_reference, dict):
            raise ValueError(
                "conf.yaml 的 gpt_sovits_tts 缺少 spirit_form_reference 配置"
            )

        ref_audio_path = str(spirit_reference.get("ref_audio_path", "")).strip()
        prompt_lang = str(spirit_reference.get("prompt_lang", "")).strip()
        prompt_text = str(spirit_reference.get("prompt_text", "")).strip()
        use_for_all_emotions = spirit_reference.get("use_for_all_emotions", False)
        if not ref_audio_path or not prompt_lang or not prompt_text:
            raise ValueError("spirit_form_reference 的参考音路径或提示文本不完整")
        if not isinstance(use_for_all_emotions, bool):
            raise ValueError(
                "spirit_form_reference.use_for_all_emotions 必须是 true 或 false"
            )

        reference_path = Path(ref_audio_path).expanduser()
        if not reference_path.is_absolute():
            if profile.outfit_dir is None:
                raise ValueError("无法确定灵装参考音相对路径的项目根目录")
            reference_path = profile.outfit_dir.parent.parent / reference_path
        reference_path = reference_path.resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(f"灵装凛祢默认语音参考音不存在：{reference_path}")

        gpt_sovits["ref_audio_path"] = reference_path.as_posix()
        gpt_sovits["prompt_lang"] = prompt_lang
        gpt_sovits["prompt_text"] = prompt_text
        if use_for_all_emotions:
            gpt_sovits["emotion_references"] = {}
    return result
