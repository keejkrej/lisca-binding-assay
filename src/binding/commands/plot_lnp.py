from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from binding.app import app
from binding.services.plot_lnp import run_plot_lnp


@app.command(name="plot-lnp")
def plot_lnp(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=True,
            file_okay=False,
            help="Dataset directory with roi/ crops.",
        ),
    ],
    counts_csv: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Spot counts CSV produced by binding spot-counts.",
        ),
    ],
    filtered_dir: Annotated[
        Path,
        typer.Option(
            "--filtered-dir",
            exists=True,
            file_okay=False,
            help="Directory containing filtered spot CSV files.",
        ),
    ] = ...,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output PNG path.",
        ),
    ] = Path("fig5_panels.png"),
    position: Annotated[
        int,
        typer.Option("--position", "-p", help="Position index for ROI loading."),
    ] = 0,
    channel: Annotated[
        int,
        typer.Option("--channel", "-c", help="Fluorescence channel index."),
    ] = 1,
    roi: Annotated[
        int | None,
        typer.Option("--roi", help="ROI index for panels d and e."),
    ] = None,
    time: Annotated[
        int | None,
        typer.Option("--time", "-t", help="Frame for panels d and e."),
    ] = None,
    time_unit: Annotated[
        str,
        typer.Option(
            "--time-unit",
            help="Time axis unit. Counts CSV stores time_real in seconds.",
        ),
    ] = "min",
    movies: Annotated[
        Path | None,
        typer.Option(
            "--movies",
            help="Directory to write per-ROI figB mp4 movies (one per roi). If set, generates movies of B viz for all ROIs after the static figure.",
        ),
    ] = None,
) -> None:
    try:
        result = run_plot_lnp(
            input_dir,
            counts_csv,
            filtered_dir=filtered_dir,
            output=output,
            position=position,
            channel=channel,
            roi=roi,
            time=time,
            time_unit=time_unit,
            movies=movies,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Saved {result.output_path} with roi={result.selected_roi}, time={result.selected_time}, "
        f"spots={result.spot_count}, cells={result.cell_count}"
    )

    for warning in result.movie_warnings:
        typer.echo(warning)