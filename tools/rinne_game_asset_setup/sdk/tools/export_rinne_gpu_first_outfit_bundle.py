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
    RINNE_GPU_FIRST_OUTFIT_MANIFEST_FILENAME,
    RINNE_GPU_OUTFIT_FAMILY_MANIFEST_FILENAME,
    BinaryBoundsError,
    RinneAssetMismatchError,
    RinneGpuFirstOutfitArtifact,
    RinneGpuSourceEvidence,
    RinneLegacyFrameControls,
    RinneLegacyFrameRenderer,
    acceptance_rinne_gpu_eye_controls,
    build_rinne_gpu_bundle,
    build_rinne_gpu_eye_dynamic_bundle,
    build_rinne_gpu_first_outfit_manifest_from_entries,
    build_rinne_gpu_outfit_family_manifest_from_entries,
    build_rinne_gpu_linear_control_samples,
    build_rinne_gpu_linear_dynamic_bundle,
    build_rinne_gpu_runtime_control_bundle,
    build_rinne_gpu_timeline_bundle,
    compare_rinne_gpu_eye_interpolation_matrix,
    compare_rinne_gpu_eye_reconstruction,
    compare_rinne_gpu_linear_reconstruction,
    describe_rinne_gpu_first_outfit_artifact,
    get_rinne_outfit_portrait_profiles,
    resolve_rinne_gpu_linear_coefficients,
    verify_rinne_gpu_first_outfit_directory,
    verify_rinne_gpu_outfit_family_directory,
    verify_rinne_pck_family_directory,
    write_new_bytes,
    write_rinne_gpu_timeline_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify one MP060x01..MP060x15 family and export one local-only, "
            "selectable outfit GPU timeline directory. Original PCKs remain read-only."
        )
    )
    parser.add_argument("pck_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--outfit-number", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--reference-width", type=int, default=128)
    parser.add_argument("--reference-height", type=int, default=128)
    return parser


def _checked_dimension(value: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 2048:
        raise BinaryBoundsError(f"first-outfit {label} must be within 1..2048")
    return value


def main() -> None:
    args = _parser().parse_args()
    output_created = False
    try:
        width = _checked_dimension(args.reference_width, "reference width")
        height = _checked_dimension(args.reference_height, "reference height")
        outfit_number = args.outfit_number
        profiles = get_rinne_outfit_portrait_profiles(outfit_number)
        verifications = verify_rinne_pck_family_directory(
            args.pck_directory,
            outfit_number,
        )
        by_id = {item.spec.portrait_id: item for item in verifications}
        expected_ids = tuple(profile.portrait_id for profile in profiles)
        if tuple(sorted(by_id)) != expected_ids:
            raise BinaryBoundsError(
                "outfit source directory does not contain all fifteen PCKs"
            )
        destination = args.output_directory
        destination.mkdir(parents=True, exist_ok=False)
        output_created = True
        entries: list[dict[str, object]] = []
        reconstruction_summary: list[dict[str, object]] = []

        for position, profile in enumerate(profiles, start=1):
            verification = by_id[profile.portrait_id]
            print(
                f"[{position:02d}/15] building {verification.spec.filename} "
                f"({profile.compatibility_label})",
                flush=True,
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
                expression_weights=tuple(linear_acceptance["raw_expression_weights"]),
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
                    f"{profile.filename} linear reconstruction failed"
                )
            eye = build_rinne_gpu_eye_dynamic_bundle(
                linear,
                renderer,
                reference,
            )
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
            interpolation = compare_rinne_gpu_eye_interpolation_matrix(
                eye,
                renderer,
                eye_controls,
                coefficients,
            )
            if not eye_comparison.accepted or not interpolation.accepted:
                raise BinaryBoundsError(
                    f"{profile.filename} nonlinear eye reconstruction failed"
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
            artifact = RinneGpuFirstOutfitArtifact(
                profile=profile,
                verification=verification,
                timeline_bundle=timeline,
            )
            entries.append(describe_rinne_gpu_first_outfit_artifact(artifact))
            write_rinne_gpu_timeline_bundle(
                destination / artifact.directory_name,
                timeline,
            )
            reconstruction_summary.append(
                {
                    "portrait_id": profile.portrait_id,
                    "linear_maximum_error": (linear_comparison.maximum_absolute_error),
                    "eye_maximum_error": eye_comparison.maximum_absolute_error,
                    "eye_midpoint_maximum_error": (
                        interpolation.maximum_absolute_error
                    ),
                }
            )
            print(
                f"[{position:02d}/15] wrote {artifact.directory_name}",
                flush=True,
            )
            del (
                artifact,
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

        if outfit_number == 1:
            manifest_bytes = build_rinne_gpu_first_outfit_manifest_from_entries(entries)
            manifest_filename = RINNE_GPU_FIRST_OUTFIT_MANIFEST_FILENAME
            verify_directory = verify_rinne_gpu_first_outfit_directory
        else:
            manifest_bytes = build_rinne_gpu_outfit_family_manifest_from_entries(
                outfit_number,
                entries,
            )
            manifest_filename = RINNE_GPU_OUTFIT_FAMILY_MANIFEST_FILENAME
            verify_directory = verify_rinne_gpu_outfit_family_directory
        write_new_bytes(
            destination / manifest_filename,
            manifest_bytes,
        )
        manifest = verify_directory(destination)
        print(
            json.dumps(
                {
                    "output_directory": str(destination),
                    "portrait_count": manifest["portrait_count"],
                    "default_portrait_id": manifest["default_portrait_id"],
                    "shared_topology": manifest["topology"][
                        "shared_across_all_portraits"
                    ],
                    "distinct_topology_count": len(
                        manifest["topology"]["distinct_hashes"]
                    ),
                    "reconstruction": reconstruction_summary,
                    "source_remained_read_only": True,
                    "manifest_written_last": True,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        BinaryBoundsError,
        RinneAssetMismatchError,
    ) as exc:
        partial = (
            f" Partial output may remain at {args.output_directory}."
            if output_created
            else ""
        )
        raise SystemExit(f"outfit GPU export failed: {exc}.{partial}") from exc


if __name__ == "__main__":
    main()
