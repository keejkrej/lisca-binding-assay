from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ShowTracesResult:
    output_path: Path
    roi_sizes: list[int]
    x_axis: str
    row_count: int


def traces_plot_output_path(input_file: Path, output: Path | None) -> Path:
    if output is not None:
        return output
    return input_file.with_name(f"{input_file.stem}_traces.png")


def read_trace_rows(input_file: Path) -> tuple[list[dict[str, str]], bool]:
    with input_file.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"time", "roi_size", "mean_intensity"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Traces CSV must contain time, roi_size, and mean_intensity columns"
            )

        rows = list(reader)
        has_time_real = "time_real" in reader.fieldnames

    if not rows:
        raise ValueError("Traces CSV has no rows")

    return rows, has_time_real


def grouped_traces(
    rows: list[dict[str, str]],
    use_time_real: bool,
) -> dict[int, tuple[list[float], list[float]]]:
    grouped: dict[int, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))

    for row in rows:
        size = int(row["roi_size"])
        x_values, y_values = grouped[size]
        if use_time_real:
            if "time_real" not in row or row["time_real"] == "":
                raise ValueError("Traces CSV is missing time_real values")
            x_values.append(float(row["time_real"]))
        else:
            x_values.append(float(row["time"]))
        y_values.append(float(row["mean_intensity"]))

    for size in grouped:
        x_values, y_values = grouped[size]
        order = sorted(range(len(x_values)), key=lambda index: x_values[index])
        grouped[size] = (
            [x_values[index] for index in order],
            [y_values[index] for index in order],
        )

    return grouped


def run_show_traces(
    input_file: Path,
    *,
    output: Path | None,
    use_time_real: bool,
) -> ShowTracesResult:
    rows, has_time_real = read_trace_rows(input_file)
    grouped = grouped_traces(rows, use_time_real=use_time_real and has_time_real)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_path = traces_plot_output_path(input_file, output)
    x_label = "time_real" if use_time_real and has_time_real else "time"

    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for size in sorted(grouped):
        x_values, y_values = grouped[size]
        axis.plot(x_values, y_values, marker="o", markersize=3, label=f"{size}x{size}")

    axis.set_title("Region intensity traces")
    axis.set_xlabel(x_label)
    axis.set_ylabel("mean_intensity")
    axis.legend(title="roi_size")
    axis.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return ShowTracesResult(
        output_path=output_path,
        roi_sizes=sorted(grouped),
        x_axis=x_label,
        row_count=len(rows),
    )