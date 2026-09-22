from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .amb_v4 import AmbV4Animation, AmbV4Config
from .checked_binary import BinaryBoundsError
from .gpu_bundle import (
    RINNE_GPU_GEOMETRY_FILENAME,
    RINNE_GPU_MANIFEST_FILENAME,
    RINNE_GPU_TEXTURE_FILENAME,
)
from .gpu_eye_dynamics import (
    RINNE_GPU_EYE_DEFORMATION_FILENAME,
    RINNE_GPU_EYE_MANIFEST_FILENAME,
)
from .gpu_linear_dynamics import (
    RINNE_GPU_DEFORMATION_FILENAME,
    RINNE_GPU_DYNAMIC_MANIFEST_FILENAME,
)
from .gpu_runtime_controls import (
    RINNE_GPU_RUNTIME_CONTROL_MANIFEST_FILENAME,
    RinneGpuRuntimeControlBundle,
    verify_rinne_gpu_runtime_control_bundle,
)
from .io_safety import write_new_bytes

RINNE_GPU_TIMELINE_FORMAT = "rinne-legacy-gpu-resident-timeline"
RINNE_GPU_TIMELINE_VERSION = 1
RINNE_GPU_TIMELINE_MANIFEST_FILENAME = "timeline-manifest.json"
RINNE_GPU_TIMELINE_DATA_FILENAME = "amb-channels.f32"
RINNE_GPU_TIMELINE_MANIFEST_MAX_BYTES = 4 * 1024 * 1024
RINNE_GPU_TIMELINE_DATA_MAX_BYTES = 64 * 1024 * 1024

_CAPABILITIES = [
    "amb_playback",
    "automatic_blink",
    "automatic_breath",
    "baseline_expression",
    "live_mouth",
]
_COMPOSITION_ORDER = [
    "automatic_breath",
    "amb_baseline_composition",
    "automatic_blink",
    "live_mouth",
    "gpu_coefficient_easing",
]
_SPEECH_INDICES = [5, 6, 7]
_ACCEPTANCE_TIMES = [0, 33, 80, 160, 333, 1000, 2000, 3007, 3087, 3167]
_ACCEPTANCE_RANDOM_VALUES = [0, 0, 0, 32767, 1, 5]


def _acceptance_controls(expression_count: int) -> dict[str, object]:
    expressions = [0.0] * expression_count
    for index, value in ((0, 0.2), (1, 0.3), (8, 0.15)):
        if index < expression_count:
            expressions[index] = value
    return {
        "expression_weights": expressions,
        "mouth_scale": 0.75,
        "mouth_gains": [0.1, 0.2, 0.3],
        "right_eye_close": 0.1,
        "left_eye_close": 0.25,
        "neck_rotation": [1.0, -0.5, 0.25],
        "neck_translation": [0.01, -0.005, 0.0025],
        "right_pupil_position": [-0.2, 0.1],
        "left_pupil_position": [0.15, -0.05],
        "type2_intensity": 0.8,
    }


@dataclass(frozen=True)
class RinneGpuTimelineBundle:
    runtime_bundle: RinneGpuRuntimeControlBundle
    timeline_manifest_json: bytes
    amb_channels_f32: bytes

    def __post_init__(self) -> None:
        verify_rinne_gpu_timeline_bundle(self)

    def timeline_manifest(self) -> dict[str, Any]:
        decoded = json.loads(self.timeline_manifest_json)
        if not isinstance(decoded, dict):
            raise BinaryBoundsError("GPU timeline manifest root must be an object")
        return decoded


def _channel_binary(animation: AmbV4Animation) -> bytes:
    output = bytearray(animation.channel_count * animation.frame_count * 4)
    cursor = 0
    for channel in animation.channels:
        for frame_index in range(animation.frame_count):
            struct.pack_into("<f", output, cursor, channel.at(frame_index))
            cursor += 4
    return bytes(output)


def _config_manifest(config: AmbV4Config) -> dict[str, object]:
    if not isinstance(config, AmbV4Config):
        raise TypeError("GPU timeline requires the face.uca.bin config")
    return {
        "source_member": "face.uca.bin",
        "blink": {
            "enabled": config.blink_enabled,
            "duration_factor": config.blink_duration_factor,
            "frequencies": list(config.blink_frequencies),
            "random_contract": "inclusive_integer_0_to_32767_three_draw_cycle",
        },
        "breath": {
            "enabled": config.breath_enabled,
            "duration_factor": config.breath_duration_factor,
            "expression_index": 4,
            "initial_phase_degrees": -90.0,
            "phase_step_degrees": 3.5,
            "clock_divisor_milliseconds": 30.0,
        },
    }


def build_rinne_gpu_timeline_bundle(
    runtime_bundle: RinneGpuRuntimeControlBundle,
    animation: AmbV4Animation,
    auto_animation_config: AmbV4Config,
) -> RinneGpuTimelineBundle:
    runtime_manifest = verify_rinne_gpu_runtime_control_bundle(runtime_bundle)
    if not isinstance(animation, AmbV4Animation):
        raise TypeError("GPU timeline requires an AMB v4 animation")
    if not isinstance(auto_animation_config, AmbV4Config):
        raise TypeError("GPU timeline requires the face.uca.bin config")
    if animation.expression_channel_count != len(
        runtime_bundle.eye_bundle.linear_bundle.base_bundle.manifest()["reference"][
            "expression_weights"
        ]
    ):
        raise BinaryBoundsError(
            "GPU timeline expression count does not match the linear bundle"
        )
    if animation.auxiliary_channel_count != runtime_manifest["opacity"]["record_count"]:
        raise BinaryBoundsError(
            "GPU timeline auxiliary count does not match opacity records"
        )
    if any(
        any(animation.selector_words(frame_index))
        for frame_index in range(animation.frame_count)
    ):
        raise BinaryBoundsError(
            "GPU timeline contains a nonzero render resource selector"
        )

    channel_bytes = _channel_binary(animation)
    groups = animation.channel_groups
    manifest = {
        "format": RINNE_GPU_TIMELINE_FORMAT,
        "version": RINNE_GPU_TIMELINE_VERSION,
        "capabilities": _CAPABILITIES,
        "runtime_bundle": {
            "manifest_sha256": hashlib.sha256(
                runtime_bundle.runtime_manifest_json
            ).hexdigest(),
            "base_topology_sha256": runtime_manifest["eye_bundle"][
                "base_topology_sha256"
            ],
        },
        "clock": {
            "unit": "uint32_milliseconds",
            "amb_divisor": 1000.0,
            "caller_controls_loop": True,
        },
        "amb": {
            "file": RINNE_GPU_TIMELINE_DATA_FILENAME,
            "format": "channel_major_float32_little_endian",
            "byte_length": len(channel_bytes),
            "sha256": hashlib.sha256(channel_bytes).hexdigest(),
            "rate": animation.rate,
            "channel_count": animation.channel_count,
            "expression_channel_count": animation.expression_channel_count,
            "auxiliary_channel_count": animation.auxiliary_channel_count,
            "frame_count": animation.frame_count,
            "runtime_loop_start": animation.runtime_loop_start,
            "runtime_loop_end": animation.runtime_loop_end,
            "groups": {
                "core": [groups.core.start, groups.core.stop],
                "expression": [groups.expression.start, groups.expression.stop],
                "auxiliary": [groups.auxiliary.start, groups.auxiliary.stop],
                "trailing_triplet": [
                    groups.trailing_triplet.start,
                    groups.trailing_triplet.stop,
                ],
                "extended": [groups.extended.start, groups.extended.stop],
            },
            "discrete_selector_count": 10,
            "selectors_all_zero": True,
            "continuous_interpolation": "legacy_float32_linear",
            "discrete_selection": "following_frame_when_fraction_at_least_half",
        },
        "automatic": _config_manifest(auto_animation_config),
        "composition": {
            "order": _COMPOSITION_ORDER,
            "speech_expression_indices": _SPEECH_INDICES,
            "pupil_x_scale": 0.1,
            "pupil_y_scale": 0.05,
            "extended_baseline_gain_indices": {
                "pupil": 0,
                "eye": 1,
                "pose": 2,
                "expression": 3,
            },
        },
        "acceptance_sequence": {
            "start_time": 0,
            "loop": True,
            "blink_floor": 0.0,
            "times": _ACCEPTANCE_TIMES,
            "random_values": _ACCEPTANCE_RANDOM_VALUES,
            "base_controls": _acceptance_controls(animation.expression_channel_count),
            "final_time": _ACCEPTANCE_TIMES[-1],
        },
    }
    manifest_json = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return RinneGpuTimelineBundle(
        runtime_bundle=runtime_bundle,
        timeline_manifest_json=manifest_json,
        amb_channels_f32=channel_bytes,
    )


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BinaryBoundsError(f"GPU timeline {label} must be an object")
    return value


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def verify_rinne_gpu_timeline_bundle(
    bundle: RinneGpuTimelineBundle,
) -> dict[str, Any]:
    if not isinstance(bundle, RinneGpuTimelineBundle):
        raise TypeError("GPU timeline verification requires a timeline bundle")
    runtime = verify_rinne_gpu_runtime_control_bundle(bundle.runtime_bundle)
    if (
        not 0
        < len(bundle.timeline_manifest_json)
        <= (RINNE_GPU_TIMELINE_MANIFEST_MAX_BYTES)
    ):
        raise BinaryBoundsError("GPU timeline manifest size exceeds safety limit")
    if not 0 < len(bundle.amb_channels_f32) <= RINNE_GPU_TIMELINE_DATA_MAX_BYTES:
        raise BinaryBoundsError("GPU timeline AMB data size exceeds safety limit")
    try:
        manifest = json.loads(bundle.timeline_manifest_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinaryBoundsError("GPU timeline manifest is invalid JSON") from exc
    root = _require_object(manifest, "manifest")
    if (
        root.get("format") != RINNE_GPU_TIMELINE_FORMAT
        or root.get("version") != RINNE_GPU_TIMELINE_VERSION
        or root.get("capabilities") != _CAPABILITIES
    ):
        raise BinaryBoundsError("GPU timeline format or capabilities are invalid")
    if root.get("runtime_bundle") != {
        "manifest_sha256": hashlib.sha256(
            bundle.runtime_bundle.runtime_manifest_json
        ).hexdigest(),
        "base_topology_sha256": runtime["eye_bundle"]["base_topology_sha256"],
    }:
        raise BinaryBoundsError("GPU timeline runtime-bundle identity is invalid")

    clock = _require_object(root.get("clock"), "clock")
    if clock != {
        "unit": "uint32_milliseconds",
        "amb_divisor": 1000.0,
        "caller_controls_loop": True,
    }:
        raise BinaryBoundsError("GPU timeline clock contract is invalid")

    amb = _require_object(root.get("amb"), "AMB")
    counts = [
        amb.get("channel_count"),
        amb.get("expression_channel_count"),
        amb.get("auxiliary_channel_count"),
        amb.get("frame_count"),
    ]
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in counts
    ):
        raise BinaryBoundsError("GPU timeline AMB counts are invalid")
    channel_count, expression_count, auxiliary_count, frame_count = counts
    if (
        amb.get("file") != RINNE_GPU_TIMELINE_DATA_FILENAME
        or amb.get("format") != "channel_major_float32_little_endian"
        or amb.get("byte_length") != len(bundle.amb_channels_f32)
        or len(bundle.amb_channels_f32) != channel_count * frame_count * 4
        or amb.get("sha256") != hashlib.sha256(bundle.amb_channels_f32).hexdigest()
        or not _finite_number(amb.get("rate"))
        or float(amb["rate"]) <= 0.0
        or expression_count
        != len(
            bundle.runtime_bundle.eye_bundle.linear_bundle.base_bundle.manifest()[
                "reference"
            ]["expression_weights"]
        )
        or auxiliary_count != runtime["opacity"]["record_count"]
        or channel_count != expression_count + auxiliary_count + 29
        or not isinstance(amb.get("runtime_loop_start"), int)
        or not isinstance(amb.get("runtime_loop_end"), int)
        or amb["runtime_loop_start"] < 0
        or amb["runtime_loop_end"] < amb["runtime_loop_start"]
        or amb["runtime_loop_end"] >= frame_count
        or amb.get("discrete_selector_count") != 10
        or amb.get("selectors_all_zero") is not True
        or amb.get("continuous_interpolation") != "legacy_float32_linear"
        or amb.get("discrete_selection")
        != "following_frame_when_fraction_at_least_half"
    ):
        raise BinaryBoundsError("GPU timeline AMB contract is invalid")
    groups = amb.get("groups")
    extended_start = 15 + expression_count + auxiliary_count
    expected_groups = {
        "core": [0, 12],
        "expression": [12, 12 + expression_count],
        "auxiliary": [
            12 + expression_count,
            12 + expression_count + auxiliary_count,
        ],
        "trailing_triplet": [
            12 + expression_count + auxiliary_count,
            15 + expression_count + auxiliary_count,
        ],
        "extended": [extended_start, channel_count],
    }
    if groups != expected_groups:
        raise BinaryBoundsError("GPU timeline channel groups are invalid")
    floats = struct.iter_unpack("<f", bundle.amb_channels_f32)
    if any(not math.isfinite(value[0]) for value in floats):
        raise BinaryBoundsError("GPU timeline AMB data contains non-finite values")
    extended_start = expected_groups["extended"][0]
    for selector_index in range(10):
        channel = extended_start + selector_index
        offset = channel * frame_count * 4
        if any(
            struct.unpack_from("<I", bundle.amb_channels_f32, offset + frame * 4)[0]
            != 0
            for frame in range(frame_count)
        ):
            raise BinaryBoundsError("GPU timeline selector channel is nonzero")

    automatic = _require_object(root.get("automatic"), "automatic config")
    blink = _require_object(automatic.get("blink"), "blink config")
    breath = _require_object(automatic.get("breath"), "breath config")
    frequencies = blink.get("frequencies")
    if (
        automatic.get("source_member") != "face.uca.bin"
        or not isinstance(blink.get("enabled"), int)
        or isinstance(blink.get("enabled"), bool)
        or not _finite_number(blink.get("duration_factor"))
        or float(blink["duration_factor"]) <= 0
        or not isinstance(frequencies, list)
        or len(frequencies) != 3
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in frequencies
        )
        or sum(frequencies) <= 0
        or blink.get("random_contract")
        != "inclusive_integer_0_to_32767_three_draw_cycle"
        or not isinstance(breath.get("enabled"), int)
        or isinstance(breath.get("enabled"), bool)
        or not _finite_number(breath.get("duration_factor"))
        or float(breath["duration_factor"]) <= 0
        or breath.get("expression_index") != 4
        or breath.get("initial_phase_degrees") != -90.0
        or breath.get("phase_step_degrees") != 3.5
        or breath.get("clock_divisor_milliseconds") != 30.0
    ):
        raise BinaryBoundsError("GPU timeline automatic config is invalid")
    composition = _require_object(root.get("composition"), "composition")
    if composition != {
        "order": _COMPOSITION_ORDER,
        "speech_expression_indices": _SPEECH_INDICES,
        "pupil_x_scale": 0.1,
        "pupil_y_scale": 0.05,
        "extended_baseline_gain_indices": {
            "pupil": 0,
            "eye": 1,
            "pose": 2,
            "expression": 3,
        },
    }:
        raise BinaryBoundsError("GPU timeline composition contract is invalid")
    acceptance = _require_object(root.get("acceptance_sequence"), "acceptance")
    if acceptance != {
        "start_time": 0,
        "loop": True,
        "blink_floor": 0.0,
        "times": _ACCEPTANCE_TIMES,
        "random_values": _ACCEPTANCE_RANDOM_VALUES,
        "base_controls": _acceptance_controls(expression_count),
        "final_time": _ACCEPTANCE_TIMES[-1],
    }:
        raise BinaryBoundsError("GPU timeline acceptance sequence is invalid")
    return root


def write_rinne_gpu_timeline_bundle(
    output_directory: Path | str,
    bundle: RinneGpuTimelineBundle,
) -> None:
    verify_rinne_gpu_timeline_bundle(bundle)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    runtime = bundle.runtime_bundle
    eye = runtime.eye_bundle
    linear = eye.linear_bundle
    base = linear.base_bundle
    for filename, payload in (
        (RINNE_GPU_MANIFEST_FILENAME, base.manifest_json),
        (RINNE_GPU_TEXTURE_FILENAME, base.texture_rgba8),
        (RINNE_GPU_GEOMETRY_FILENAME, base.geometry_binary),
        (RINNE_GPU_DYNAMIC_MANIFEST_FILENAME, linear.dynamic_manifest_json),
        (RINNE_GPU_DEFORMATION_FILENAME, linear.deformation_rgba32f),
        (RINNE_GPU_EYE_MANIFEST_FILENAME, eye.eye_manifest_json),
        (RINNE_GPU_EYE_DEFORMATION_FILENAME, eye.eye_deformation_rgba32f),
        (RINNE_GPU_RUNTIME_CONTROL_MANIFEST_FILENAME, runtime.runtime_manifest_json),
        (RINNE_GPU_TIMELINE_MANIFEST_FILENAME, bundle.timeline_manifest_json),
        (RINNE_GPU_TIMELINE_DATA_FILENAME, bundle.amb_channels_f32),
    ):
        write_new_bytes(destination / filename, payload)


__all__ = [
    "RINNE_GPU_TIMELINE_DATA_FILENAME",
    "RINNE_GPU_TIMELINE_FORMAT",
    "RINNE_GPU_TIMELINE_MANIFEST_FILENAME",
    "RINNE_GPU_TIMELINE_VERSION",
    "RinneGpuTimelineBundle",
    "build_rinne_gpu_timeline_bundle",
    "verify_rinne_gpu_timeline_bundle",
    "write_rinne_gpu_timeline_bundle",
]
