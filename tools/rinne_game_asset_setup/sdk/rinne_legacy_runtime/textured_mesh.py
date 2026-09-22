from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .checked_binary import BinaryBoundsError
from .rgba_png import PNG_MAX_DIMENSION, PNG_MAX_PIXELS, encode_rgba_png
from .texture import MotionPortraitTexture


TEXTURED_MESH_MAX_VERTICES = 1_000_000
TEXTURED_MESH_COORDINATE_LIMIT = 16.0
TEXTURED_MESH_MAX_PIXEL_VISITS = 32_000_000
TextureFilter = Literal["nearest", "linear"]


@dataclass(frozen=True)
class TexturedDepthMesh:
    positions_xyz: Sequence[tuple[float, float, float]]
    uvs: Sequence[tuple[float, float]]
    triangle_indices: Sequence[int]
    opacity: float = 1.0
    depth_write: bool = True
    vertex_opacities: Sequence[float] | None = None
    texture_filter: TextureFilter | None = None


def _validate_pair_sequence(
    values: Sequence[tuple[float, float]], label: str
) -> None:
    if len(values) > TEXTURED_MESH_MAX_VERTICES:
        raise BinaryBoundsError(
            f"{label} count {len(values)} exceeds {TEXTURED_MESH_MAX_VERTICES}"
        )
    for index, pair in enumerate(values):
        if len(pair) != 2 or not all(math.isfinite(value) for value in pair):
            raise BinaryBoundsError(f"{label} contains an invalid pair at {index}")
        if label == "position" and any(
            abs(value) > TEXTURED_MESH_COORDINATE_LIMIT for value in pair
        ):
            raise BinaryBoundsError(
                f"position {index} exceeds coordinate safety limit "
                f"{TEXTURED_MESH_COORDINATE_LIMIT}"
            )


def _validate_depth_mesh(mesh: TexturedDepthMesh) -> None:
    if not math.isfinite(mesh.opacity) or mesh.opacity < 0.0 or mesh.opacity > 1.0:
        raise BinaryBoundsError("textured depth mesh opacity must be within 0..1")
    if mesh.texture_filter not in (None, "nearest", "linear"):
        raise ValueError("texture filter must be nearest or linear")
    if len(mesh.positions_xyz) > TEXTURED_MESH_MAX_VERTICES:
        raise BinaryBoundsError("textured depth mesh vertex count exceeds safety limit")
    for index, position in enumerate(mesh.positions_xyz):
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            raise BinaryBoundsError(
                f"textured depth mesh contains an invalid position at {index}"
            )
        if abs(position[0]) > TEXTURED_MESH_COORDINATE_LIMIT or abs(
            position[1]
        ) > TEXTURED_MESH_COORDINATE_LIMIT:
            raise BinaryBoundsError(
                f"textured depth mesh position {index} exceeds coordinate safety limit"
            )
    _validate_pair_sequence(mesh.uvs, "UV")
    if len(mesh.positions_xyz) != len(mesh.uvs):
        raise ValueError("textured depth mesh position and UV counts must match")
    if mesh.vertex_opacities is not None:
        if len(mesh.vertex_opacities) != len(mesh.positions_xyz):
            raise ValueError(
                "textured depth mesh position and vertex-opacity counts must match"
            )
        for index, opacity in enumerate(mesh.vertex_opacities):
            if not math.isfinite(opacity) or opacity < 0.0 or opacity > 1.0:
                raise BinaryBoundsError(
                    "textured depth mesh contains an invalid vertex opacity "
                    f"at {index}"
                )
    if len(mesh.triangle_indices) % 3:
        raise ValueError(
            "textured depth mesh triangle index count must be divisible by three"
        )
    if len(mesh.triangle_indices) > TEXTURED_MESH_MAX_VERTICES * 6:
        raise BinaryBoundsError(
            "textured depth mesh triangle index count exceeds safety limit"
        )
    for offset, index in enumerate(mesh.triangle_indices):
        if index < 0 or index >= len(mesh.positions_xyz):
            raise BinaryBoundsError(
                f"textured depth mesh triangle index {index} at {offset} exceeds "
                f"vertex count {len(mesh.positions_xyz)}"
            )


def _source_over(destination: memoryview, offset: int, source: tuple[int, ...]) -> None:
    source_alpha = source[3]
    if source_alpha == 0:
        return
    destination_alpha = destination[offset + 3]
    inverse_source_alpha = 255 - source_alpha
    output_alpha_scaled = source_alpha * 255 + destination_alpha * inverse_source_alpha
    if output_alpha_scaled == 0:
        return
    for channel in range(3):
        numerator = (
            source[channel] * source_alpha * 255
            + destination[offset + channel]
            * destination_alpha
            * inverse_source_alpha
        )
        destination[offset + channel] = (
            numerator + output_alpha_scaled // 2
        ) // output_alpha_scaled
    destination[offset + 3] = (output_alpha_scaled + 127) // 255


def _legacy_mode0_blend(
    destination: memoryview, offset: int, source: tuple[int, ...]
) -> None:
    source_alpha = source[3]
    if source_alpha == 0:
        return
    inverse_source_alpha = 255 - source_alpha
    for channel in range(3):
        destination[offset + channel] = (
            source[channel] * source_alpha
            + destination[offset + channel] * inverse_source_alpha
            + 127
        ) // 255
    destination[offset + 3] = min(
        255, source_alpha + destination[offset + 3]
    )


def _sample_texture_rgba(
    texture_view: memoryview,
    *,
    texture_width: int,
    texture_height: int,
    u: float,
    v: float,
    texture_filter: TextureFilter,
) -> tuple[float, float, float, float]:
    if texture_filter == "nearest":
        texture_x = min(
            texture_width - 1,
            int(u * (texture_width - 1) + 0.5),
        )
        texture_y = min(
            texture_height - 1,
            int(v * (texture_height - 1) + 0.5),
        )
        offset = (texture_y * texture_width + texture_x) * 4
        return tuple(float(value) for value in texture_view[offset : offset + 4])

    sample_x = u * texture_width - 0.5
    sample_y = v * texture_height - 0.5
    x0_unclamped = math.floor(sample_x)
    y0_unclamped = math.floor(sample_y)
    fraction_x = sample_x - x0_unclamped
    fraction_y = sample_y - y0_unclamped
    x0 = min(texture_width - 1, max(0, x0_unclamped))
    x1 = min(texture_width - 1, max(0, x0_unclamped + 1))
    y0 = min(texture_height - 1, max(0, y0_unclamped))
    y1 = min(texture_height - 1, max(0, y0_unclamped + 1))
    weights = (
        ((1.0 - fraction_x) * (1.0 - fraction_y), x0, y0),
        (fraction_x * (1.0 - fraction_y), x1, y0),
        ((1.0 - fraction_x) * fraction_y, x0, y1),
        (fraction_x * fraction_y, x1, y1),
    )
    output = [0.0, 0.0, 0.0, 0.0]
    for weight, texture_x, texture_y in weights:
        offset = (texture_y * texture_width + texture_x) * 4
        for channel in range(4):
            output[channel] += texture_view[offset + channel] * weight
    return output[0], output[1], output[2], output[3]


def rasterize_textured_mesh_rgba(
    texture: MotionPortraitTexture,
    positions: Sequence[tuple[float, float]],
    uvs: Sequence[tuple[float, float]],
    triangle_indices: Sequence[int],
    *,
    width: int,
    height: int,
    opacity: float = 1.0,
    vertex_opacities: Sequence[float] | None = None,
    texture_filter: TextureFilter = "nearest",
) -> bytes:
    """Rasterize one diagnostic mesh with explicit clamp-to-edge sampling.

    Position Y is normalized bottom-to-top and converted to PNG top-to-bottom
    rows. Texture V indexes the supplied RGBA rows directly. MotionPortrait
    callers supply the runtime atlas coordinates after the group-B loader's
    ``(first_y, second_y) -> (1-second_y, 1-first_y)`` rewrite. Static analysis
    proves that the game uploads decoded TEX rows without reordering them.
    """

    if not math.isfinite(opacity) or opacity < 0.0 or opacity > 1.0:
        raise BinaryBoundsError("textured mesh opacity must be within 0..1")
    if texture_filter not in ("nearest", "linear"):
        raise ValueError("texture filter must be nearest or linear")
    if (
        width <= 0
        or height <= 0
        or width > PNG_MAX_DIMENSION
        or height > PNG_MAX_DIMENSION
        or width * height > PNG_MAX_PIXELS
    ):
        raise BinaryBoundsError("textured mesh output dimensions exceed safety limit")
    if (
        texture.width <= 0
        or texture.height <= 0
        or texture.width > PNG_MAX_DIMENSION
        or texture.height > PNG_MAX_DIMENSION
        or texture.width * texture.height > PNG_MAX_PIXELS
    ):
        raise BinaryBoundsError("texture dimensions exceed safety limit")
    expected_texture_bytes = texture.width * texture.height * 4
    if len(texture.rgba) != expected_texture_bytes:
        raise BinaryBoundsError(
            f"texture RGBA length {len(texture.rgba)} != {expected_texture_bytes}"
        )
    _validate_pair_sequence(positions, "position")
    _validate_pair_sequence(uvs, "UV")
    if len(positions) != len(uvs):
        raise ValueError("position and UV counts must match")
    if vertex_opacities is not None:
        if len(vertex_opacities) != len(positions):
            raise ValueError("position and vertex-opacity counts must match")
        for index, vertex_opacity in enumerate(vertex_opacities):
            if (
                not math.isfinite(vertex_opacity)
                or vertex_opacity < 0.0
                or vertex_opacity > 1.0
            ):
                raise BinaryBoundsError(
                    f"textured mesh contains an invalid vertex opacity at {index}"
                )
    if len(triangle_indices) % 3:
        raise ValueError("triangle index count must be divisible by three")
    if len(triangle_indices) > TEXTURED_MESH_MAX_VERTICES * 6:
        raise BinaryBoundsError("triangle index count exceeds safety limit")
    for offset, index in enumerate(triangle_indices):
        if index < 0 or index >= len(positions):
            raise BinaryBoundsError(
                f"triangle index {index} at {offset} exceeds vertex count "
                f"{len(positions)}"
            )

    pixels = bytearray(width * height * 4)
    pixel_view = memoryview(pixels)
    texture_view = memoryview(texture.rgba).cast("B")

    def edge(
        first: tuple[float, float],
        second: tuple[float, float],
        point: tuple[float, float],
    ) -> float:
        return (point[0] - first[0]) * (second[1] - first[1]) - (
            point[1] - first[1]
        ) * (second[0] - first[0])

    screen_positions = tuple((x * width, (1.0 - y) * height) for x, y in positions)
    pixel_visits = 0
    for triangle_offset in range(0, len(triangle_indices), 3):
        i0, i1, i2 = triangle_indices[triangle_offset : triangle_offset + 3]
        p0, p1, p2 = (
            screen_positions[i0],
            screen_positions[i1],
            screen_positions[i2],
        )
        area = edge(p1, p2, p0)
        if abs(area) < 1e-12:
            continue
        if area < 0.0:
            p1, p2 = p2, p1
            uv1, uv2 = uvs[i2], uvs[i1]
            opacity1, opacity2 = (
                (1.0, 1.0)
                if vertex_opacities is None
                else (vertex_opacities[i2], vertex_opacities[i1])
            )
            area = -area
        else:
            uv1, uv2 = uvs[i1], uvs[i2]
            opacity1, opacity2 = (
                (1.0, 1.0)
                if vertex_opacities is None
                else (vertex_opacities[i1], vertex_opacities[i2])
            )
        minimum_x = max(0, math.floor(min(p0[0], p1[0], p2[0])))
        maximum_x = min(width - 1, math.ceil(max(p0[0], p1[0], p2[0])))
        minimum_y = max(0, math.floor(min(p0[1], p1[1], p2[1])))
        maximum_y = min(height - 1, math.ceil(max(p0[1], p1[1], p2[1])))
        if minimum_x > maximum_x or minimum_y > maximum_y:
            continue
        pixel_visits += (maximum_x - minimum_x + 1) * (
            maximum_y - minimum_y + 1
        )
        if pixel_visits > TEXTURED_MESH_MAX_PIXEL_VISITS:
            raise BinaryBoundsError(
                "textured mesh raster work exceeds pixel-visit safety limit"
            )

        uv0 = uvs[i0]
        opacity0 = 1.0 if vertex_opacities is None else vertex_opacities[i0]
        edge_epsilon = 1e-12 * max(1.0, area)

        def owns_edge(first: tuple[float, float], second: tuple[float, float]) -> bool:
            delta_x = second[0] - first[0]
            delta_y = second[1] - first[1]
            return delta_y < 0.0 or (delta_y == 0.0 and delta_x > 0.0)

        edge0_owned = owns_edge(p1, p2)
        edge1_owned = owns_edge(p2, p0)
        edge2_owned = owns_edge(p0, p1)
        for pixel_y in range(minimum_y, maximum_y + 1):
            for pixel_x in range(minimum_x, maximum_x + 1):
                point = (pixel_x + 0.5, pixel_y + 0.5)
                edge0 = edge(p1, p2, point)
                edge1 = edge(p2, p0, point)
                edge2 = edge(p0, p1, point)
                if (
                    edge0 < -edge_epsilon
                    or edge1 < -edge_epsilon
                    or edge2 < -edge_epsilon
                    or (abs(edge0) <= edge_epsilon and not edge0_owned)
                    or (abs(edge1) <= edge_epsilon and not edge1_owned)
                    or (abs(edge2) <= edge_epsilon and not edge2_owned)
                ):
                    continue
                weight0 = edge0 / area
                weight1 = edge1 / area
                weight2 = edge2 / area
                u = min(1.0, max(0.0, weight0 * uv0[0] + weight1 * uv1[0] + weight2 * uv2[0]))
                v = min(1.0, max(0.0, weight0 * uv0[1] + weight1 * uv1[1] + weight2 * uv2[1]))
                sampled = _sample_texture_rgba(
                    texture_view,
                    texture_width=texture.width,
                    texture_height=texture.height,
                    u=u,
                    v=v,
                    texture_filter=texture_filter,
                )
                vertex_opacity = min(
                    1.0,
                    max(
                        0.0,
                        weight0 * opacity0
                        + weight1 * opacity1
                        + weight2 * opacity2,
                    ),
                )
                source = tuple(int(channel + 0.5) for channel in sampled[:3]) + (
                    int(sampled[3] * opacity * vertex_opacity + 0.5),
                )
                _source_over(pixel_view, (pixel_y * width + pixel_x) * 4, source)

    return bytes(pixels)


def rasterize_textured_depth_meshes_rgba(
    texture: MotionPortraitTexture,
    meshes: Sequence[TexturedDepthMesh],
    *,
    width: int,
    height: int,
    depth_compare: Literal["disabled", "less_equal", "greater_equal"],
    color_blend: Literal["straight_alpha", "legacy_mode0"] = "straight_alpha",
    texture_filter: TextureFilter = "nearest",
) -> bytes:
    """Rasterize ordered atlas draws with an explicit interpolated depth test.

    Transparent samples neither blend nor write depth. Equal depths pass so
    file draw order remains the tie breaker. The comparison direction stays an
    explicit argument until the old renderer's visibility state is
    independently fixed. ``disabled`` preserves submission order and still
    carries the recovered Z values in each input mesh.
    """

    if depth_compare not in ("disabled", "less_equal", "greater_equal"):
        raise ValueError(
            "depth comparison must be disabled, less_equal or greater_equal"
        )
    if color_blend not in ("straight_alpha", "legacy_mode0"):
        raise ValueError(
            "color blend must be straight_alpha or legacy_mode0"
        )
    if texture_filter not in ("nearest", "linear"):
        raise ValueError("texture filter must be nearest or linear")
    blend_pixel = (
        _legacy_mode0_blend if color_blend == "legacy_mode0" else _source_over
    )
    if (
        width <= 0
        or height <= 0
        or width > PNG_MAX_DIMENSION
        or height > PNG_MAX_DIMENSION
        or width * height > PNG_MAX_PIXELS
    ):
        raise BinaryBoundsError("textured depth output dimensions exceed safety limit")
    if (
        texture.width <= 0
        or texture.height <= 0
        or texture.width > PNG_MAX_DIMENSION
        or texture.height > PNG_MAX_DIMENSION
        or texture.width * texture.height > PNG_MAX_PIXELS
    ):
        raise BinaryBoundsError("texture dimensions exceed safety limit")
    expected_texture_bytes = texture.width * texture.height * 4
    if len(texture.rgba) != expected_texture_bytes:
        raise BinaryBoundsError(
            f"texture RGBA length {len(texture.rgba)} != {expected_texture_bytes}"
        )
    if len(meshes) > 256:
        raise BinaryBoundsError("textured depth draw count exceeds safety limit")
    total_vertices = 0
    total_indices = 0
    for mesh in meshes:
        _validate_depth_mesh(mesh)
        total_vertices += len(mesh.positions_xyz)
        total_indices += len(mesh.triangle_indices)
        if total_vertices > TEXTURED_MESH_MAX_VERTICES:
            raise BinaryBoundsError(
                "combined textured depth vertex count exceeds safety limit"
            )
        if total_indices > TEXTURED_MESH_MAX_VERTICES * 6:
            raise BinaryBoundsError(
                "combined textured depth index count exceeds safety limit"
            )

    pixels = bytearray(width * height * 4)
    pixel_view = memoryview(pixels)
    texture_view = memoryview(texture.rgba).cast("B")
    initial_depth = math.inf if depth_compare == "less_equal" else -math.inf
    depth_buffer = [initial_depth] * (width * height)

    def edge(
        first: tuple[float, float],
        second: tuple[float, float],
        point: tuple[float, float],
    ) -> float:
        return (point[0] - first[0]) * (second[1] - first[1]) - (
            point[1] - first[1]
        ) * (second[0] - first[0])

    def owns_edge(first: tuple[float, float], second: tuple[float, float]) -> bool:
        delta_x = second[0] - first[0]
        delta_y = second[1] - first[1]
        return delta_y < 0.0 or (delta_y == 0.0 and delta_x > 0.0)

    pixel_visits = 0
    for mesh in meshes:
        mesh_texture_filter = mesh.texture_filter or texture_filter
        screen_positions = tuple(
            (x * width, (1.0 - y) * height)
            for x, y, _z in mesh.positions_xyz
        )
        for triangle_offset in range(0, len(mesh.triangle_indices), 3):
            i0, i1, i2 = mesh.triangle_indices[
                triangle_offset : triangle_offset + 3
            ]
            p0, p1, p2 = (
                screen_positions[i0],
                screen_positions[i1],
                screen_positions[i2],
            )
            area = edge(p1, p2, p0)
            if abs(area) < 1e-12:
                continue
            if area < 0.0:
                p1, p2 = p2, p1
                uv1, uv2 = mesh.uvs[i2], mesh.uvs[i1]
                z1, z2 = mesh.positions_xyz[i2][2], mesh.positions_xyz[i1][2]
                opacity1, opacity2 = (
                    (1.0, 1.0)
                    if mesh.vertex_opacities is None
                    else (mesh.vertex_opacities[i2], mesh.vertex_opacities[i1])
                )
                area = -area
            else:
                uv1, uv2 = mesh.uvs[i1], mesh.uvs[i2]
                z1, z2 = mesh.positions_xyz[i1][2], mesh.positions_xyz[i2][2]
                opacity1, opacity2 = (
                    (1.0, 1.0)
                    if mesh.vertex_opacities is None
                    else (mesh.vertex_opacities[i1], mesh.vertex_opacities[i2])
                )
            minimum_x = max(0, math.floor(min(p0[0], p1[0], p2[0])))
            maximum_x = min(width - 1, math.ceil(max(p0[0], p1[0], p2[0])))
            minimum_y = max(0, math.floor(min(p0[1], p1[1], p2[1])))
            maximum_y = min(height - 1, math.ceil(max(p0[1], p1[1], p2[1])))
            if minimum_x > maximum_x or minimum_y > maximum_y:
                continue
            pixel_visits += (maximum_x - minimum_x + 1) * (
                maximum_y - minimum_y + 1
            )
            if pixel_visits > TEXTURED_MESH_MAX_PIXEL_VISITS:
                raise BinaryBoundsError(
                    "textured depth raster work exceeds pixel-visit safety limit"
                )

            uv0 = mesh.uvs[i0]
            z0 = mesh.positions_xyz[i0][2]
            opacity0 = (
                1.0
                if mesh.vertex_opacities is None
                else mesh.vertex_opacities[i0]
            )
            edge_epsilon = 1e-12 * max(1.0, area)
            edge0_owned = owns_edge(p1, p2)
            edge1_owned = owns_edge(p2, p0)
            edge2_owned = owns_edge(p0, p1)
            for pixel_y in range(minimum_y, maximum_y + 1):
                for pixel_x in range(minimum_x, maximum_x + 1):
                    point = (pixel_x + 0.5, pixel_y + 0.5)
                    edge0 = edge(p1, p2, point)
                    edge1 = edge(p2, p0, point)
                    edge2 = edge(p0, p1, point)
                    if (
                        edge0 < -edge_epsilon
                        or edge1 < -edge_epsilon
                        or edge2 < -edge_epsilon
                        or (abs(edge0) <= edge_epsilon and not edge0_owned)
                        or (abs(edge1) <= edge_epsilon and not edge1_owned)
                        or (abs(edge2) <= edge_epsilon and not edge2_owned)
                    ):
                        continue
                    weight0 = edge0 / area
                    weight1 = edge1 / area
                    weight2 = edge2 / area
                    depth = weight0 * z0 + weight1 * z1 + weight2 * z2
                    pixel_index = pixel_y * width + pixel_x
                    current_depth = depth_buffer[pixel_index]
                    passes = depth_compare == "disabled" or (
                        depth <= current_depth
                        if depth_compare == "less_equal"
                        else depth >= current_depth
                    )
                    if not passes:
                        continue

                    u = min(
                        1.0,
                        max(
                            0.0,
                            weight0 * uv0[0]
                            + weight1 * uv1[0]
                            + weight2 * uv2[0],
                        ),
                    )
                    v = min(
                        1.0,
                        max(
                            0.0,
                            weight0 * uv0[1]
                            + weight1 * uv1[1]
                            + weight2 * uv2[1],
                        ),
                    )
                    sampled = _sample_texture_rgba(
                        texture_view,
                        texture_width=texture.width,
                        texture_height=texture.height,
                        u=u,
                        v=v,
                        texture_filter=mesh_texture_filter,
                    )
                    vertex_opacity = min(
                        1.0,
                        max(
                            0.0,
                            weight0 * opacity0
                            + weight1 * opacity1
                            + weight2 * opacity2,
                        ),
                    )
                    source_alpha = int(
                        sampled[3] * mesh.opacity * vertex_opacity + 0.5
                    )
                    if source_alpha == 0:
                        continue
                    source = tuple(
                        int(channel + 0.5) for channel in sampled[:3]
                    ) + (source_alpha,)
                    blend_pixel(pixel_view, pixel_index * 4, source)
                    if mesh.depth_write and depth_compare != "disabled":
                        depth_buffer[pixel_index] = depth

    return bytes(pixels)


def render_textured_mesh_preview(
    texture: MotionPortraitTexture,
    positions: Sequence[tuple[float, float]],
    uvs: Sequence[tuple[float, float]],
    triangle_indices: Sequence[int],
    *,
    width: int,
    height: int,
    opacity: float = 1.0,
) -> bytes:
    rgba = rasterize_textured_mesh_rgba(
        texture,
        positions,
        uvs,
        triangle_indices,
        width=width,
        height=height,
        opacity=opacity,
    )
    return encode_rgba_png(width, height, rgba)


__all__ = [
    "TexturedDepthMesh",
    "rasterize_textured_depth_meshes_rgba",
    "rasterize_textured_mesh_rgba",
    "render_textured_mesh_preview",
]
