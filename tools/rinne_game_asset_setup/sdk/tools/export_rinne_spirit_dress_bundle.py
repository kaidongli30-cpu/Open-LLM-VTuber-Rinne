"""Build the local spirit-dress runtime from the user's seven game PCKs."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from verify_rinne_spirit_dress_native_family import verify_native_family

from rinne_legacy_runtime.spirit_expression_preview import (
    write_rinne_spirit_expression_preview_directory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pck_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--overlay-directory", required=True, type=Path)
    args = parser.parse_args()

    output = args.output_directory.resolve()
    if output.exists():
        parser.error(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="rinne-spirit-native-", dir=output.parent
    ) as temporary:
        native = Path(temporary) / "native"
        report = verify_native_family(
            args.pck_directory,
            reference_width=32,
            reference_height=32,
            output_directory=native,
        )
        manifest = write_rinne_spirit_expression_preview_directory(
            output,
            native,
            args.overlay_directory,
        )
    print(
        json.dumps(
            {
                "source_remained_read_only": report["source_remained_read_only"],
                "native_portrait_count": report["portrait_count"],
                "expression_count": manifest["expression_count"],
                "output_directory": str(output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
