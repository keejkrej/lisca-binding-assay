from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from binding.app import app
from binding.services.picasso import run_picasso_roi, run_picasso_stack

FitMethodOption = Literal["lq", "mle", "mle-integrated"]
GaussModelOption = Literal["spherical", "elliptic", "rotated"]


@app.command()
def picasso(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=True,
            file_okay=False,
            help="Directory containing raw microscopy data.",
        ),
    ],
    position: Annotated[
        int,
        typer.Option("--position", "-p", help="Position index to process."),
    ] = 0,
    channel: Annotated[
        int,
        typer.Option("--channel", "-c", help="Channel index to process."),
    ] = 0,
    time: Annotated[
        int,
        typer.Option("--time", "-t", help="Frame to process."),
    ] = 0,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory where Picasso spot CSV files will be written.",
        ),
    ] = Path("picasso"),
    min_gradient: Annotated[
        float,
        typer.Option("--min-gradient", "-g", help="Minimum radial net gradient threshold."),
    ] = 500.0,
    box_size: Annotated[
        int,
        typer.Option("--box-size", "-b", help="Odd spot extraction box side length in pixels."),
    ] = 7,
    fit_method: Annotated[
        FitMethodOption,
        typer.Option("--fit-method", help="2D Gaussian fitting estimator (lq, mle, mle-integrated)."),
    ] = "lq",
    model: Annotated[
        GaussModelOption,
        typer.Option("--model", "-m", help="2D Gaussian PSF model (spherical, elliptic, rotated)."),
    ] = "elliptic",
    baseline: Annotated[
        float,
        typer.Option("--baseline", help="Camera baseline offset in ADU."),
    ] = 0.0,
    sensitivity: Annotated[
        float,
        typer.Option("--sensitivity", help="Camera sensitivity (electrons per ADU)."),
    ] = 1.0,
    gain: Annotated[
        float,
        typer.Option("--gain", help="Camera EM gain."),
    ] = 1.0,
    gaussian_sigma: Annotated[
        float | None,
        typer.Option("--gaussian-sigma", help="Optional Gaussian pre-filter sigma for candidate identification."),
    ] = None,
    roi: Annotated[
        int | None,
        typer.Option("--roi", help="Single cell ROI index to process."),
    ] = None,
    all_rois: Annotated[
        bool,
        typer.Option("--all-rois", help="Process every cell ROI."),
    ] = False,
    all_times: Annotated[
        bool,
        typer.Option("--all-times", help="Process every available Frame."),
    ] = False,
    use_roi_stacks: Annotated[
        bool,
        typer.Option(
            "--roi-stacks/--stack",
            help="Use roi/Pos*/Roi*.tif crops instead of full-field stacks.",
        ),
    ] = False,
) -> None:
    roi_mode = use_roi_stacks or all_rois or roi is not None

    if roi_mode:
        try:
            result = run_picasso_roi(
                input_dir,
                position=position,
                channel=channel,
                time=time,
                output=output,
                min_gradient=min_gradient,
                box_size=box_size,
                fit_method=fit_method,
                model=model,
                baseline=baseline,
                sensitivity=sensitivity,
                gain=gain,
                gaussian_sigma=gaussian_sigma,
                roi=roi,
                all_rois=all_rois,
                all_times=all_times,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        typer.echo(
            f"Saved Picasso output to {result.output_dir} for rois={result.rois}, times={result.time_count}, "
            f"total_spots={result.total_spots}, fit_method={result.fit_method}, model={result.model}"
        )
        return

    try:
        result = run_picasso_stack(
            input_dir,
            position=position,
            channel=channel,
            time=time,
            output=output,
            min_gradient=min_gradient,
            box_size=box_size,
            fit_method=fit_method,
            model=model,
            baseline=baseline,
            sensitivity=sensitivity,
            gain=gain,
            gaussian_sigma=gaussian_sigma,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Loaded position={result.position}, channel={result.channel}, time={result.time}: "
        f"shape={result.shape}, dtype={result.dtype}, fit_method={result.fit_method}, model={result.model}"
    )
    typer.echo(f"Saved picasso output to {result.output_path}")
