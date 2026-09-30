from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from binding.app import app
from binding.services.traces import run_traces

ProjectionOption = Literal["mean", "max", "sum"]


@app.command()
def traces(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Folder containing Pos*/img_channel... TIFFs from convert.",
        ),
    ],
    position: Annotated[
        int,
        typer.Option("--position", "-p", help="Position index to sample."),
    ] = 0,
    channel: Annotated[
        int,
        typer.Option("--channel", "-c", help="Channel index to sample."),
    ] = 0,
    sizes: Annotated[
        str,
        typer.Option(
            "--sizes",
            help="Comma-separated square ROI side lengths in pixels, smallest to largest.",
        ),
    ] = "8,16,32,64,128",
    center_y: Annotated[
        int | None,
        typer.Option("--center-y", help="ROI center row in pixels. Defaults to image center."),
    ] = None,
    center_x: Annotated[
        int | None,
        typer.Option("--center-x", help="ROI center column in pixels. Defaults to image center."),
    ] = None,
    z: Annotated[
        int | None,
        typer.Option("--z", help="Use a single z plane instead of projecting the stack."),
    ] = None,
    projection: Annotated[
        ProjectionOption,
        typer.Option("--projection", help="Z projection when --z is not set."),
    ] = "mean",
    time_map: Annotated[
        Path | None,
        typer.Option(
            "--time-map",
            exists=True,
            dir_okay=False,
            help="Optional CSV with t and t_real columns for real time values.",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output CSV file or directory.",
        ),
    ] = Path("."),
) -> None:
    try:
        result = run_traces(
            input_dir,
            position=position,
            channel=channel,
            sizes=sizes,
            center_y=center_y,
            center_x=center_x,
            z=z,
            projection=projection,  # type: ignore[arg-type]
            time_map=time_map,
            output=output,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Saved {result.output_path} with rows={result.row_count}, times={result.time_count}, "
        f"roi_sizes={result.roi_sizes}, center=({result.center_y}, {result.center_x})"
    )