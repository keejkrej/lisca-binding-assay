from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from binding.app import app
from binding.services.show import run_show


@app.command()
def show(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Folder containing Pos*/img_channel... TIFFs from convert.",
        ),
    ],
    position: Annotated[int, typer.Option("--position", "-p", help="Position index to show.")] = 0,
    channel: Annotated[int, typer.Option("--channel", "-c", help="Channel index to show.")] = 0,
    time: Annotated[int, typer.Option("--time", "-t", help="Frame to show.")] = 0,
    metadata: Annotated[
        Path | None,
        typer.Option(
            "--metadata",
            exists=True,
            dir_okay=False,
            help="metadata.json containing normalized.pixel_size_um x/y/z values.",
        ),
    ] = None,
) -> None:
    try:
        result = run_show(
            input_dir,
            position=position,
            channel=channel,
            time=time,
            metadata=metadata,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Loaded position={result.position}, channel={result.channel}, time={result.time}: "
        f"shape={result.shape}, dtype={result.dtype}"
    )
    if result.voxel_scale is not None:
        typer.echo(f"Voxel scale (z, y, x): {result.voxel_scale} um")