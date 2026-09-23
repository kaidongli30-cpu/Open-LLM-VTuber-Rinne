from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rinne_legacy_runtime import (  # noqa: E402
    RINNE_SPIRIT_DRESS_PCK_SPECS,
    BinaryBoundsError,
    RinneGpuSourceEvidence,
    RinneLegacyFrameControls,
    RinneLegacyFrameRenderer,
    acceptance_rinne_gpu_eye_controls,
    build_rinne_gpu_bundle,
    build_rinne_gpu_eye_dynamic_bundle,
    build_rinne_gpu_linear_control_samples,
    build_rinne_gpu_linear_dynamic_bundle,
    build_rinne_gpu_runtime_control_bundle,
    build_rinne_gpu_timeline_bundle,
    compare_rinne_gpu_eye_interpolation_matrix,
    compare_rinne_gpu_eye_reconstruction,
    compare_rinne_gpu_linear_reconstruction,
    resolve_rinne_gpu_linear_coefficients,
    verify_rinne_pck_against_spec,
    write_new_bytes,
    write_rinne_gpu_timeline_bundle,
)


NATIVE_FAMILY_MANIFEST_FILENAME = "spirit-dress-native-family.json"
_COMPATIBILITY_CANDIDATES = {
    160_101: ("neutral", "visual_candidate"),
    160_102: ("worried", "pending_user_review_against_160104"),
    160_103: ("angry", "script_context_candidate"),
    160_104: ("uneasy", "pending_user_review_against_160102"),
    160_105: ("gentle", "visual_candidate"),
    160_106: ("sad", "script_context_candidate"),
    160_107: ("happy", "script_context_candidate"),
}


def _checked_dimension(value: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 2048:
        raise BinaryBoundsError(f"spirit-dress {label} must be within 1..2048")
    return value


def verify_native_family(
    pck_directory: Path,
    *,
    reference_width: int = 128,
    reference_height: int = 128,
    output_directory: Path | None = None,
) -> dict[str, object]:
    width = _checked_dimension(reference_width, "reference width")
    height = _checked_dimension(reference_height, "reference height")
    source = pck_directory.resolve(strict=True)
    if not source.is_dir():
        raise BinaryBoundsError("spirit-dress PCK root is not a directory")
    destination = None
    if output_directory is not None:
        destination = output_directory.resolve()
        destination.mkdir(parents=True, exist_ok=False)

    results: list[dict[str, object]] = []
    for position, spec in enumerate(RINNE_SPIRIT_DRESS_PCK_SPECS, start=1):
        print(
            f"[{position:02d}/07] verifying {spec.filename}",
            file=sys.stderr,
            flush=True,
        )
        verification = verify_rinne_pck_against_spec(
            source / spec.filename,
            spec,
        )
        renderer = RinneLegacyFrameRenderer.from_pck(verification.path)
        evidence = RinneGpuSourceEvidence.from_verified_assets(
            verification,
            renderer.assets,
        )
        reference = renderer.build_draw_plan(width=width, height=height)
        base = build_rinne_gpu_bundle(reference, evidence)
        linear_samples = build_rinne_gpu_linear_control_samples(
            renderer,
            reference,
            width=width,
            height=height,
        )
        linear = build_rinne_gpu_linear_dynamic_bundle(
            base,
            reference,
            linear_samples,
        )
        linear_acceptance = linear.dynamic_manifest()["acceptance_sample"]
        linear_controls = RinneLegacyFrameControls(
            expression_weights=tuple(
                linear_acceptance["raw_expression_weights"]
            ),
            right_pupil_position=linear_acceptance["right_pupil_position"],
            left_pupil_position=linear_acceptance["left_pupil_position"],
        )
        coefficients = resolve_rinne_gpu_linear_coefficients(
            linear_controls.resolve_expression_weights(
                renderer.expression_weight_count
            ),
            renderer.assets.renderer.expression_ease_flags,
            right_pupil_position=linear_controls.right_pupil_position,
            left_pupil_position=linear_controls.left_pupil_position,
        )
        linear_actual = renderer.build_draw_plan(
            linear_controls,
            width=width,
            height=height,
        )
        linear_comparison = compare_rinne_gpu_linear_reconstruction(
            linear,
            reference,
            linear_actual,
            coefficients,
        )
        if not linear_comparison.accepted:
            raise BinaryBoundsError(
                f"{spec.filename} linear reconstruction failed"
            )

        eye = build_rinne_gpu_eye_dynamic_bundle(linear, renderer, reference)
        eye_controls = acceptance_rinne_gpu_eye_controls(
            linear_acceptance["raw_expression_weights"],
            right_pupil_position=linear_acceptance["right_pupil_position"],
            left_pupil_position=linear_acceptance["left_pupil_position"],
        )
        eye_actual = renderer.build_draw_plan(
            eye_controls,
            width=width,
            height=height,
        )
        eye_comparison = compare_rinne_gpu_eye_reconstruction(
            eye,
            reference,
            eye_actual,
            coefficients,
            blink_deformation=eye_controls.blink_geometry.deformation,
            right_eye_close=eye_controls.right_eye_close or 0.0,
            left_eye_close=eye_controls.left_eye_close or 0.0,
        )
        eye_interpolation = compare_rinne_gpu_eye_interpolation_matrix(
            eye,
            renderer,
            eye_controls,
            coefficients,
        )
        if not eye_comparison.accepted or not eye_interpolation.accepted:
            raise BinaryBoundsError(
                f"{spec.filename} nonlinear eye reconstruction failed"
            )

        runtime = build_rinne_gpu_runtime_control_bundle(
            eye,
            renderer,
            reference,
        )
        timeline = build_rinne_gpu_timeline_bundle(
            runtime,
            renderer.assets.animation,
            renderer.assets.auto_animation_config,
        )
        timeline_manifest = timeline.timeline_manifest()
        triggers = renderer.assets.atlas.compositor_triggers
        if destination is not None:
            write_rinne_gpu_timeline_bundle(
                destination / verification.path.stem,
                timeline,
            )
        compatibility_label, label_status = _COMPATIBILITY_CANDIDATES[
            spec.portrait_id
        ]
        results.append(
            {
                "portrait_id": spec.portrait_id,
                "directory": verification.path.stem,
                "compatibility_label_candidate": compatibility_label,
                "compatibility_label_status": label_status,
                "filename": spec.filename,
                "pck_sha256": verification.sha256,
                "texture": {
                    "width": renderer.assets.texture.width,
                    "height": renderer.assets.texture.height,
                },
                "compositor": {
                    "trigger_a": triggers.a_type_id,
                    "trigger_b": triggers.b_type_id,
                    "selected_trigger": renderer.assets.compositor_trigger_type,
                },
                "linear_maximum_error": (
                    linear_comparison.maximum_absolute_error
                ),
                "eye_maximum_error": eye_comparison.maximum_absolute_error,
                "eye_midpoint_maximum_error": (
                    eye_interpolation.maximum_absolute_error
                ),
                "timeline": {
                    "frame_count": timeline_manifest["amb"]["frame_count"],
                    "expression_channel_count": timeline_manifest["amb"][
                        "expression_channel_count"
                    ],
                    "auxiliary_channel_count": timeline_manifest["amb"][
                        "auxiliary_channel_count"
                    ],
                    "blink_enabled": timeline_manifest["automatic"]["blink"][
                        "enabled"
                    ],
                    "breath_enabled": timeline_manifest["automatic"]["breath"][
                        "enabled"
                    ],
                },
            }
        )
        del (
            timeline,
            runtime,
            eye,
            linear,
            base,
            renderer,
            reference,
            linear_samples,
            linear_actual,
            eye_actual,
        )
        gc.collect()

    report = {
        "format": "rinne-spirit-dress-native-family-verification",
        "portrait_count": len(results),
        "reference_dimensions": [width, height],
        "source_remained_read_only": True,
        "runtime_payloads_written": destination is not None,
        "portraits": results,
    }
    if destination is not None:
        family_manifest = {
            "format": "rinne-spirit-dress-native-source-family",
            "version": 1,
            "portrait_count": len(results),
            "default_portrait_id": 160_101,
            "labels_are_compatibility_candidates": True,
            "not_a_fifteen_label_desktop_runtime": True,
            "portraits": [
                {
                    "portrait_id": entry["portrait_id"],
                    "directory": entry["directory"],
                    "compatibility_label_candidate": entry[
                        "compatibility_label_candidate"
                    ],
                    "compatibility_label_status": entry[
                        "compatibility_label_status"
                    ],
                }
                for entry in results
            ],
        }
        write_new_bytes(
            destination / NATIVE_FAMILY_MANIFEST_FILENAME,
            (
                json.dumps(
                    family_manifest,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        report["output_directory"] = str(destination)
        report["manifest_written_last"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify all seven original Rinne spirit-dress PCKs through the "
            "static, expression, eye, runtime-control, and AMB timeline gates "
            "without writing extracted game assets."
        )
    )
    parser.add_argument("pck_directory", type=Path)
    parser.add_argument("--reference-width", type=int, default=128)
    parser.add_argument("--reference-height", type=int, default=128)
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "optionally write seven verified GPU timeline bundles and a "
            "native-source manifest; the original PCKs remain read-only"
        ),
    )
    args = parser.parse_args()
    try:
        report = verify_native_family(
            args.pck_directory,
            reference_width=args.reference_width,
            reference_height=args.reference_height,
            output_directory=args.output_directory,
        )
    except (OSError, TypeError, ValueError, KeyError, IndexError) as exc:
        raise SystemExit(
            f"spirit-dress native-family verification failed: {exc}"
        ) from exc
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
