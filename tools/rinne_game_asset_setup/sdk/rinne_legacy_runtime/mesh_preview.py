from __future__ import annotations

from pathlib import Path

from .mpb_v39_renderer import MpbV39RendererViews
from .io_safety import write_new_bytes
from .rgba_png import encode_rgba_png


_PREVIEW_COLORS = (
    (255, 92, 92, 180),
    (87, 196, 255, 180),
    (255, 209, 84, 200),
    (143, 255, 145, 180),
    (205, 130, 255, 180),
    (255, 143, 216, 180),
    (109, 255, 230, 180),
    (255, 166, 87, 180),
    (214, 226, 255, 200),
)
PREVIEW_MAX_DIMENSION = 8_192
PREVIEW_MAX_PIXELS = 16_777_216


class _RgbaCanvas:
    def __init__(self, width: int, height: int) -> None:
        if (
            width <= 1
            or height <= 1
            or width > PREVIEW_MAX_DIMENSION
            or height > PREVIEW_MAX_DIMENSION
            or width * height > PREVIEW_MAX_PIXELS
        ):
            raise ValueError(
                "preview dimensions exceed the safe output limit "
                f"({PREVIEW_MAX_DIMENSION} per side, {PREVIEW_MAX_PIXELS} pixels)"
            )
        self.width = width
        self.height = height
        self.pixels = bytearray(width * height * 4)

    def blend_pixel(self, x: int, y: int, color: tuple[int, int, int, int]) -> None:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        offset = (y * self.width + x) * 4
        source_alpha = color[3]
        inverse = 255 - source_alpha
        destination_alpha = self.pixels[offset + 3]
        output_alpha = source_alpha + destination_alpha * inverse // 255
        if output_alpha == 0:
            return
        for channel in range(3):
            numerator = (
                color[channel] * source_alpha
                + self.pixels[offset + channel]
                * destination_alpha
                * inverse
                // 255
            )
            self.pixels[offset + channel] = min(255, numerator // output_alpha)
        self.pixels[offset + 3] = output_alpha

    def line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        color: tuple[int, int, int, int],
    ) -> None:
        clipped = self._clip_line(start, end)
        if clipped is None:
            return
        (x0_float, y0_float), (x1_float, y1_float) = clipped
        x0, y0 = round(x0_float), round(y0_float)
        x1, y1 = round(x1_float), round(y1_float)
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            self.blend_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            doubled = error * 2
            if doubled >= dy:
                error += dy
                x0 += sx
            if doubled <= dx:
                error += dx
                y0 += sy

    def _clip_line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Cohen-Sutherland clip before Bresenham bounds its run time."""

        left, right = 1, 2
        top, bottom = 4, 8
        minimum_x, maximum_x = 0.0, float(self.width - 1)
        minimum_y, maximum_y = 0.0, float(self.height - 1)
        x0, y0 = start
        x1, y1 = end

        def outcode(x: float, y: float) -> int:
            code = 0
            if x < minimum_x:
                code |= left
            elif x > maximum_x:
                code |= right
            if y < minimum_y:
                code |= top
            elif y > maximum_y:
                code |= bottom
            return code

        code0 = outcode(x0, y0)
        code1 = outcode(x1, y1)
        for _iteration in range(8):
            if not (code0 | code1):
                return (x0, y0), (x1, y1)
            if code0 & code1:
                return None

            code = code0 if code0 else code1
            if code & top:
                if y1 == y0:
                    return None
                x = x0 + (x1 - x0) * (minimum_y - y0) / (y1 - y0)
                y = minimum_y
            elif code & bottom:
                if y1 == y0:
                    return None
                x = x0 + (x1 - x0) * (maximum_y - y0) / (y1 - y0)
                y = maximum_y
            elif code & right:
                if x1 == x0:
                    return None
                y = y0 + (y1 - y0) * (maximum_x - x0) / (x1 - x0)
                x = maximum_x
            else:
                if x1 == x0:
                    return None
                y = y0 + (y1 - y0) * (minimum_x - x0) / (x1 - x0)
                x = minimum_x

            if code == code0:
                x0, y0 = x, y
                code0 = outcode(x0, y0)
            else:
                x1, y1 = x, y
                code1 = outcode(x1, y1)
        raise AssertionError("line clipping did not converge")

    def png_bytes(self) -> bytes:
        return encode_rgba_png(self.width, self.height, self.pixels)


def render_mesh_topology_preview(
    renderer: MpbV39RendererViews,
    *,
    width: int | None = None,
    height: int | None = None,
) -> bytes:
    """Render a transparent diagnostic PNG from parameter XY/triangle views.

    This intentionally does not claim to render the character artwork. It is a
    parameter-domain topology check used before final positions are connected.
    """

    renderer.validate()
    output_width = width if width is not None else renderer.mask_width
    output_height = height if height is not None else renderer.mask_height
    canvas = _RgbaCanvas(output_width, output_height)

    for record_index, record in enumerate(renderer.mesh_records):
        color = _PREVIEW_COLORS[record_index % len(_PREVIEW_COLORS)]
        positions = [
            (
                max(-2.0, min(3.0, x)) * (output_width - 1),
                max(-2.0, min(3.0, y)) * (output_height - 1),
            )
            for x, y in record.coordinate_source_xy.values()
        ]
        indices = list(record.triangle_indices.values())
        for triangle_offset in range(0, len(indices), 3):
            first, second, third = indices[triangle_offset : triangle_offset + 3]
            canvas.line(positions[first], positions[second], color)
            canvas.line(positions[second], positions[third], color)
            canvas.line(positions[third], positions[first], color)

    return canvas.png_bytes()


def write_mesh_topology_preview(
    renderer: MpbV39RendererViews,
    output: Path,
    *,
    width: int | None = None,
    height: int | None = None,
) -> None:
    write_new_bytes(
        output,
        render_mesh_topology_preview(renderer, width=width, height=height)
    )


__all__ = ["render_mesh_topology_preview", "write_mesh_topology_preview"]
