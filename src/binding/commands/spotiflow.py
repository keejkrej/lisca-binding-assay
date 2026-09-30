from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from binding.app import app
from binding.services.spotiflow import run_spotiflow_roi, run_spotiflow_stack


@app.command()
def spotiflow(
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
            help="Directory where Spotiflow CSV files will be written.",
        ),
    ] = Path("spotiflow"),
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Pretrained Spotiflow model name."),
    ] = "general",
    estimate_params: Annotated[
        bool,
        typer.Option("--estimate-params/--no-estimate-params", help="Estimate spot intensity and FWHM."),
    ] = False,
    device: Annotated[
        str,
        typer.Option("--device", "-d", help="Device for Spotiflow prediction."),
    ] = "auto",
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
            result = run_spotiflow_roi(
                input_dir,
                position=position,
                channel=channel,
                time=time,
                output=output,
                model=model,
                estimate_params=estimate_params,
                device=device,
                roi=roi,
                all_rois=all_rois,
                all_times=all_times,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        typer.echo(
            f"Saved Spotiflow output to {result.output_dir} for rois={result.rois}, times={result.time_count}, "
            f"total_spots={result.total_spots}, model={result.model}"
        )
        return

    try:
        result = run_spotiflow_stack(
            input_dir,
            position=position,
            channel=channel,
            time=time,
            output=output,
            model=model,
            estimate_params=estimate_params,
            device=device,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Loaded position={result.position}, channel={result.channel}, time={result.time}: "
        f"shape={result.shape}, dtype={result.dtype}, model={result.model}"
    )
    typer.echo(f"Saved spotiflow output to {result.output_path}")