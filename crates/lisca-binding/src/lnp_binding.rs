//! Studio `lnp-binding` pipeline: Picasso spot detection on each ROI crop,
//! per-ROI counts (`spots.csv`, `binding.csv`, `binding_summary.csv`),
//! sample `.xlsx` workbooks, and per-sample count & intensity trace figures.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use csv::{Reader, WriterBuilder};
use marray::np::s;
use mplot::prelude::{
    AxesStyle, Figure, FigureBuilder, FillBetweenStyle, GridPos, LegendStyle, LineDash, LineStyle,
    SaveOptions, Size, TickFormat,
};
use mplot::Color;
use picasso::prelude::{
    detect_spots_u16_slice, spot_counts, BindingDetectConfig, BindingSpot, CameraInfo, GaussModel,
    SpotFitMethod,
};

use crate::export::write_xlsx_only;
use crate::roi::{load_roi_index, load_roi_stack, roi_dir};
use crate::sample::{sample_pack_dirnames, SampleMapping};

/// Picasso `min_gradient` when `analysis.minGradient` is omitted.
pub const DEFAULT_MIN_GRADIENT: f64 = 50.0;
/// Picasso odd box side length when `analysis.boxSize` is omitted.
pub const DEFAULT_BOX_SIZE: u32 = 7;
/// Distance (px) at which a later-frame spot counts as a new binding event.
const MATCH_DISTANCE_PX: f64 = 5.0;

const FIGURE_SINGLE_WIDTH_IN: f64 = 6.5;
const FIGURE_SINGLE_HEIGHT_IN: f64 = 5.5;
const PANEL_WIDTH_IN: f64 = 5.5;
const PANEL_HEIGHT_IN: f64 = 4.0;
const FIGURE_DPI: u32 = 100;
const LABEL_FONTSIZE: f64 = 18.0;
const TITLE_FONTSIZE: f64 = 18.0;
const TICK_FONTSIZE: f64 = 17.0;
const SAVE_PAD_GRID_INCHES: f64 = 0.28;
const SAVE_PAD_SINGLE_INCHES: f64 = 0.25;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BindingDetectParams {
    pub min_gradient: f64,
    pub box_size: u32,
}

impl Default for BindingDetectParams {
    fn default() -> Self {
        Self {
            min_gradient: DEFAULT_MIN_GRADIENT,
            box_size: DEFAULT_BOX_SIZE,
        }
    }
}

pub fn validate_detect_params(
    min_gradient: Option<f64>,
    box_size: Option<u32>,
) -> Result<BindingDetectParams, String> {
    let min_gradient = min_gradient.unwrap_or(DEFAULT_MIN_GRADIENT);
    if !min_gradient.is_finite() || min_gradient < 0.0 {
        return Err(format!(
            "analysis.minGradient must be >= 0, got {min_gradient}"
        ));
    }
    let box_size = box_size.unwrap_or(DEFAULT_BOX_SIZE);
    if box_size < 3 || box_size % 2 == 0 {
        return Err(format!(
            "analysis.boxSize must be an odd integer >= 3, got {box_size}"
        ));
    }
    Ok(BindingDetectParams {
        min_gradient,
        box_size,
    })
}

pub fn run_pipeline(
    workspace: &Path,
    mapping: &SampleMapping,
    interval_minutes: f64,
    params: BindingDetectParams,
) -> Result<(), String> {
    if interval_minutes <= 0.0 || !interval_minutes.is_finite() {
        return Err(format!(
            "invalid interval.value/unit in assay.json: {interval_minutes}"
        ));
    }
    for position in mapping.positions() {
        run_position(workspace, mapping, position, interval_minutes, params)?;
    }
    run_plot_traces(workspace, mapping, interval_minutes)?;
    write_summary(workspace, mapping)
}

pub fn run_position(
    workspace: &Path,
    mapping: &SampleMapping,
    position: u32,
    interval_minutes: f64,
    params: BindingDetectParams,
) -> Result<(), String> {
    let pos_dir = roi_dir(workspace, position);
    if !pos_dir.is_dir() {
        return Err(format!(
            "No ROI directory found for position {position}: {}",
            pos_dir.display()
        ));
    }
    let index = load_roi_index(workspace, position)?;
    if index.rois.is_empty() {
        return Err(format!(
            "position {position} has no ROI frames to score ({})",
            pos_dir.display()
        ));
    }
    let owners: Vec<_> = mapping
        .iter()
        .filter(|sample| sample.positions.contains(&position))
        .collect();
    let sample = match owners.as_slice() {
        [sample] => *sample,
        [] => return Err(format!("no sample in assay.json lists Pos{position}")),
        _ => return Err(format!("Pos{position} is listed by more than one sample")),
    };
    if interval_minutes <= 0.0 || !interval_minutes.is_finite() {
        return Err(format!("interval must be > 0, got {interval_minutes}"));
    }
    let signal = *sample.signal.first().ok_or_else(|| {
        format!(
            "sample {} has no signal channel for lnp-binding",
            sample.name
        )
    })?;

    let mut spot_rows = Vec::new();
    let mut count_rows = Vec::new();
    for roi_crop in &index.rois {
        let stack = load_roi_stack(workspace, position, roi_crop.roi, signal)?;
        let time_count = stack.shape()[0];
        let height = stack.shape()[1];
        let width = stack.shape()[2];
        let time_indices = index.resolved_time_indices(time_count)?;
        let config = BindingDetectConfig {
            min_gradient: params.min_gradient,
            box_size: params.box_size as usize,
            fit_method: SpotFitMethod::GaussLq,
            gauss_model: GaussModel::Elliptic,
            gaussian_sigma: None,
            camera_info: CameraInfo::default(),
            pos: format!("Pos{position}"),
            roi: roi_crop.roi.to_string(),
            parallel: false,
        };
        let mut roi_spots = Vec::new();
        for (stack_t, &source_t) in time_indices.iter().enumerate() {
            let plane = stack.slice(s![stack_t, .., ..]);
            let pixels: Vec<u16> = plane
                .iter()
                .map(|value| {
                    if !value.is_finite() || *value <= 0.0 {
                        0
                    } else {
                        value.round().clamp(0.0, 65535.0) as u16
                    }
                })
                .collect();
            let detected = detect_frame_spots(
                &pixels,
                height,
                width,
                source_t,
                &config,
                position,
                roi_crop.roi,
            )?;
            roi_spots.extend(detected);
        }
        spot_rows.extend(roi_spots.iter().cloned().map(|spot| SpotRow {
            roi: roi_crop.roi,
            spot,
        }));
        count_rows.extend(count_rows_for_roi(
            roi_crop.roi,
            &roi_spots,
            &time_indices,
            interval_minutes,
        ));
    }

    spot_rows.sort_by(|left, right| {
        left.roi
            .cmp(&right.roi)
            .then(left.spot.time.cmp(&right.spot.time))
    });
    count_rows.sort_by(|left, right| left.roi.cmp(&right.roi).then(left.t.cmp(&right.t)));
    let analysis_dir = workspace.join("analysis").join(format!("Pos{position}"));
    write_spot_csv(&analysis_dir.join("spots.csv"), &spot_rows)?;
    write_count_csv(&analysis_dir.join("binding.csv"), &count_rows)
}

pub fn write_summary(workspace: &Path, mapping: &SampleMapping) -> Result<(), String> {
    let csvs = discover_binding_csvs(&workspace.join("analysis"))?;
    let dirnames = sample_pack_dirnames(mapping);
    let mut traces: BTreeMap<usize, Vec<Vec<String>>> = BTreeMap::new();
    let mut summaries: BTreeMap<usize, Vec<Vec<String>>> = BTreeMap::new();
    let mut spots: BTreeMap<usize, Vec<Vec<String>>> = BTreeMap::new();

    for path in csvs {
        let position = binding_position(&path)?;
        let sample = mapping
            .iter()
            .position(|sample| sample.positions.contains(&position))
            .ok_or_else(|| format!("No sample owns Pos{position} ({})", path.display()))?;
        let rows = read_count_rows(&path, position)?;
        let summary_rows = summarize_rows(&rows);
        let summary_path = path
            .parent()
            .ok_or_else(|| format!("binding csv has no directory: {}", path.display()))?
            .join("binding_summary.csv");
        write_summary_csv(&summary_path, &summary_rows)?;
        for row in &rows {
            traces.entry(sample).or_default().push(count_xlsx_row(row));
        }
        for row in &summary_rows {
            summaries
                .entry(sample)
                .or_default()
                .push(summary_xlsx_row(row));
        }
        let spots_path = path
            .parent()
            .ok_or_else(|| format!("binding csv has no directory: {}", path.display()))?
            .join("spots.csv");
        if spots_path.is_file() {
            for row in read_spot_xlsx_rows(&spots_path, position)? {
                spots.entry(sample).or_default().push(row);
            }
        }
    }

    publish_sample_xlsx(
        workspace,
        &dirnames,
        &mut traces,
        "binding",
        &[
            "pos",
            "roi",
            "t",
            "minutes",
            "spot_count",
            "intensity_sum",
            "cumulative_count",
        ],
    )?;
    publish_sample_xlsx(
        workspace,
        &dirnames,
        &mut summaries,
        "binding_summary",
        &[
            "pos",
            "roi",
            "frames",
            "spots_mean",
            "intensity_mean",
            "cumulative_last",
        ],
    )?;
    publish_sample_xlsx(
        workspace,
        &dirnames,
        &mut spots,
        "spots",
        &[
            "pos",
            "roi",
            "t",
            "x",
            "y",
            "intensity",
            "photons",
            "fwhm",
            "sx",
            "sy",
            "snb",
            "probability",
        ],
    )
}

pub fn run_plot_traces(
    workspace: &Path,
    mapping: &SampleMapping,
    interval_minutes: f64,
) -> Result<(), String> {
    if interval_minutes <= 0.0 || !interval_minutes.is_finite() {
        return Err(format!("interval must be > 0, got {interval_minutes}"));
    }
    let csvs = discover_binding_csvs(&workspace.join("analysis"))?;
    let panels = load_binding_panels(&csvs, "spot_count", mapping)?;
    if panels.is_empty() {
        return Err("no binding traces to plot".to_string());
    }
    let intensity_panels = load_binding_panels(&csvs, "intensity_sum", mapping)?;
    let dirnames = sample_pack_dirnames(mapping);
    let sample_dir = |sample: usize| {
        let dirname = dirnames
            .get(&sample)
            .map(String::as_str)
            .unwrap_or("sample");
        workspace.join("results").join(dirname)
    };
    write_per_sample_metric_plots(
        &panels,
        &sample_dir,
        "binding_traces",
        "spots",
        interval_minutes,
        mapping,
    )?;
    if !intensity_panels.is_empty() {
        write_per_sample_metric_plots(
            &intensity_panels,
            &sample_dir,
            "binding_intensity_traces",
            "spot intensity",
            interval_minutes,
            mapping,
        )?;
    }
    Ok(())
}

fn detect_frame_spots(
    pixels: &[u16],
    height: usize,
    width: usize,
    time: u32,
    config: &BindingDetectConfig,
    position: u32,
    roi: u32,
) -> Result<Vec<BindingSpot>, String> {
    match catch_unwind(AssertUnwindSafe(|| {
        detect_spots_u16_slice(pixels, height, width, time, config)
    })) {
        Ok(Ok(spots)) => Ok(spots),
        Ok(Err(error)) => Err(error.to_string()),
        Err(payload) => {
            let detail = panic_payload_message(payload.as_ref());
            Err(format!(
                "Picasso identify aborted on Pos{position} Roi{roi} t={time}{detail}. A candidate on the detection-box border can panic picasso-rs. Use a smaller boxSize or a larger ROI."
            ))
        }
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        format!(": {message}")
    } else if let Some(message) = payload.downcast_ref::<String>() {
        format!(": {message}")
    } else {
        String::new()
    }
}

struct SpotRow {
    roi: u32,
    spot: BindingSpot,
}

struct CountRow {
    pos: u32,
    roi: u32,
    t: u32,
    minutes: f64,
    spot_count: u32,
    intensity_sum: f64,
    cumulative_count: u32,
}

struct SummaryRow {
    pos: u32,
    roi: u32,
    frames: u32,
    spots: f64,
    intensity: f64,
    cumulative_last: u32,
}

fn count_rows_for_roi(
    roi: u32,
    spots: &[BindingSpot],
    times: &[u32],
    interval_minutes: f64,
) -> Vec<CountRow> {
    let counts = if spots.is_empty() {
        Vec::new()
    } else {
        spot_counts(spots, Some(times), MATCH_DISTANCE_PX)
    };
    times
        .iter()
        .map(|&t| {
            let count = counts.iter().find(|row| row.time == t);
            let intensity_sum = spots
                .iter()
                .filter(|spot| spot.time == t)
                .map(|spot| spot.intensity)
                .sum::<f64>();
            CountRow {
                pos: 0,
                roi,
                t,
                minutes: f64::from(t) * interval_minutes,
                spot_count: count.map(|row| row.spot_count as u32).unwrap_or(0),
                intensity_sum,
                cumulative_count: count.map(|row| row.cumulative_count as u32).unwrap_or(0),
            }
        })
        .collect()
}

fn format_float(value: f64) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        "nan".to_string()
    }
}

fn write_csv_only(path: &Path, headers: &[&str], rows: &[Vec<String>]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);
    writer
        .write_record(headers)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(row)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    Ok(())
}

fn read_csv(path: &Path) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
    let mut reader = Reader::from_path(path).map_err(|error| error.to_string())?;
    let headers = reader
        .headers()
        .map_err(|error| error.to_string())?
        .iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let rows = reader
        .records()
        .map(|record| {
            record
                .map_err(|error| error.to_string())
                .map(|row| row.iter().map(str::to_string).collect())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((headers, rows))
}

fn column_index(headers: &[String], name: &str) -> Option<usize> {
    headers.iter().position(|header| header == name)
}

fn parse_f64(raw: &str) -> Option<f64> {
    raw.trim().parse::<f64>().ok()
}

fn write_spot_csv(path: &Path, rows: &[SpotRow]) -> Result<(), String> {
    let headers = [
        "roi",
        "t",
        "x",
        "y",
        "intensity",
        "photons",
        "fwhm",
        "sx",
        "sy",
        "snb",
        "probability",
    ];
    let body = rows
        .iter()
        .map(|row| {
            vec![
                row.roi.to_string(),
                row.spot.time.to_string(),
                format_float(row.spot.x),
                format_float(row.spot.y),
                format_float(row.spot.intensity),
                format_float(row.spot.photons),
                format_float(row.spot.fwhm),
                format_float(row.spot.sx),
                format_float(row.spot.sy),
                format_float(row.spot.snb),
                format_float(row.spot.probability),
            ]
        })
        .collect::<Vec<_>>();
    write_csv_only(path, &headers, &body)
}

fn write_count_csv(path: &Path, rows: &[CountRow]) -> Result<(), String> {
    let headers = [
        "roi",
        "t",
        "minutes",
        "spot_count",
        "intensity_sum",
        "cumulative_count",
    ];
    let body = rows
        .iter()
        .map(|row| {
            vec![
                row.roi.to_string(),
                row.t.to_string(),
                format!("{:.4}", row.minutes),
                row.spot_count.to_string(),
                format_float(row.intensity_sum),
                row.cumulative_count.to_string(),
            ]
        })
        .collect::<Vec<_>>();
    write_csv_only(path, &headers, &body)
}

fn write_summary_csv(path: &Path, rows: &[SummaryRow]) -> Result<(), String> {
    let headers = [
        "roi",
        "frames",
        "spots_mean",
        "intensity_mean",
        "cumulative_last",
    ];
    let body = rows
        .iter()
        .map(|row| {
            vec![
                row.roi.to_string(),
                row.frames.to_string(),
                format_float(row.spots),
                format_float(row.intensity),
                row.cumulative_last.to_string(),
            ]
        })
        .collect::<Vec<_>>();
    write_csv_only(path, &headers, &body)
}

fn discover_binding_csvs(dir: &Path) -> Result<Vec<PathBuf>, String> {
    if !dir.is_dir() {
        return Err(format!("Expected analysis directory at {}", dir.display()));
    }
    let mut csvs = Vec::new();
    for entry in fs::read_dir(dir).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let pos_dir = entry.path();
        if !pos_dir.is_dir() {
            continue;
        }
        let path = pos_dir.join("binding.csv");
        if path.is_file() {
            csvs.push(path);
        }
    }
    csvs.sort();
    if csvs.is_empty() {
        return Err(format!(
            "No position binding CSV files (expected Pos{{n}}/binding.csv) in {}",
            dir.display()
        ));
    }
    Ok(csvs)
}

fn binding_position(path: &Path) -> Result<u32, String> {
    path.parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix("Pos"))
        .and_then(|rest| rest.parse().ok())
        .ok_or_else(|| format!("Expected Pos{{n}}/binding.csv, got {}", path.display()))
}

#[derive(Debug, Clone)]
struct TracePanel {
    sample: usize,
    paths: Vec<PathBuf>,
    traces: Vec<Vec<(f64, f64)>>,
    y_values: Vec<f64>,
}

fn group_trace_rows(
    headers: &[String],
    rows: &[Vec<String>],
    y_column: &str,
) -> Result<BTreeMap<i64, Vec<(f64, f64)>>, String> {
    let t_index = column_index(headers, "t").ok_or("missing t column")?;
    let y_index =
        column_index(headers, y_column).ok_or_else(|| format!("missing {y_column} column"))?;
    let roi_index = column_index(headers, "roi").ok_or("missing roi column")?;

    let mut groups: BTreeMap<i64, Vec<(f64, f64)>> = BTreeMap::new();
    for row in rows {
        let roi = parse_f64(&row[roi_index]).ok_or("invalid roi")? as i64;
        let t = parse_f64(&row[t_index]).ok_or("invalid t")?;
        let y = parse_f64(&row[y_index]).ok_or("invalid y")?;
        groups.entry(roi).or_default().push((t, y));
    }
    Ok(groups)
}

fn load_binding_panels(
    csvs: &[PathBuf],
    y_column: &str,
    mapping: &SampleMapping,
) -> Result<Vec<TracePanel>, String> {
    let mut grouped = BTreeMap::<usize, TracePanel>::new();
    for path in csvs {
        let position = binding_position(path)?;
        let sample = mapping
            .iter()
            .position(|sample| sample.positions.contains(&position))
            .ok_or_else(|| format!("No sample owns Pos{position} ({})", path.display()))?;
        let (headers, rows) = read_csv(path)?;
        let groups = group_trace_rows(&headers, &rows, y_column)?;
        let mut y_values = Vec::new();
        let traces = groups
            .into_values()
            .map(|mut points| {
                points
                    .sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
                y_values.extend(points.iter().map(|(_, value)| *value));
                points
            })
            .collect::<Vec<_>>();
        let entry = grouped.entry(sample).or_insert_with(|| TracePanel {
            sample,
            paths: Vec::new(),
            traces: Vec::new(),
            y_values: Vec::new(),
        });
        entry.paths.push(path.clone());
        entry.traces.extend(traces);
        entry.y_values.extend(y_values);
    }
    Ok(grouped.into_values().collect())
}

fn read_count_rows(path: &Path, position: u32) -> Result<Vec<CountRow>, String> {
    let (headers, rows) = read_csv(path)?;
    let column = |name: &str| {
        column_index(&headers, name).ok_or_else(|| format!("{} is missing {name}", path.display()))
    };
    let roi = column("roi")?;
    let t = column("t")?;
    let minutes = column("minutes")?;
    let spot_count = column("spot_count")?;
    let intensity_sum = column("intensity_sum")?;
    let cumulative_count = column("cumulative_count")?;
    let mut parsed = Vec::with_capacity(rows.len());
    for row in &rows {
        let cell = |index: usize| -> Result<&str, String> {
            row.get(index)
                .map(String::as_str)
                .ok_or_else(|| format!("{} has a short row", path.display()))
        };
        let number = |index: usize| -> Result<u32, String> {
            cell(index)?
                .parse()
                .map_err(|error| format!("{}: {error}", path.display()))
        };
        parsed.push(CountRow {
            pos: position,
            roi: number(roi)?,
            t: number(t)?,
            minutes: cell(minutes)?
                .parse()
                .map_err(|error| format!("{}: {error}", path.display()))?,
            spot_count: number(spot_count)?,
            intensity_sum: cell(intensity_sum)?
                .parse()
                .map_err(|error| format!("{}: {error}", path.display()))?,
            cumulative_count: number(cumulative_count)?,
        });
    }
    Ok(parsed)
}

fn read_spot_xlsx_rows(path: &Path, position: u32) -> Result<Vec<Vec<String>>, String> {
    let (headers, rows) = read_csv(path)?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let mut xlsx = Vec::with_capacity(headers.len() + 1);
        xlsx.push(position.to_string());
        xlsx.extend(row);
        out.push(xlsx);
    }
    Ok(out)
}

fn summarize_rows(rows: &[CountRow]) -> Vec<SummaryRow> {
    let mut groups = Vec::<SummaryRow>::new();
    for row in rows {
        if let Some(group) = groups
            .iter_mut()
            .find(|group| group.pos == row.pos && group.roi == row.roi)
        {
            group.frames += 1;
            group.spots += f64::from(row.spot_count);
            group.intensity += row.intensity_sum;
            group.cumulative_last = row.cumulative_count;
        } else {
            groups.push(SummaryRow {
                pos: row.pos,
                roi: row.roi,
                frames: 1,
                spots: f64::from(row.spot_count),
                intensity: row.intensity_sum,
                cumulative_last: row.cumulative_count,
            });
        }
    }
    for group in &mut groups {
        if group.frames > 0 {
            group.spots /= f64::from(group.frames);
            group.intensity /= f64::from(group.frames);
        }
    }
    groups
}

fn count_xlsx_row(row: &CountRow) -> Vec<String> {
    vec![
        row.pos.to_string(),
        row.roi.to_string(),
        row.t.to_string(),
        format!("{:.4}", row.minutes),
        row.spot_count.to_string(),
        format_float(row.intensity_sum),
        row.cumulative_count.to_string(),
    ]
}

fn summary_xlsx_row(row: &SummaryRow) -> Vec<String> {
    vec![
        row.pos.to_string(),
        row.roi.to_string(),
        row.frames.to_string(),
        format_float(row.spots),
        format_float(row.intensity),
        row.cumulative_last.to_string(),
    ]
}

fn publish_sample_xlsx(
    workspace: &Path,
    dirnames: &BTreeMap<usize, String>,
    rows: &mut BTreeMap<usize, Vec<Vec<String>>>,
    stem: &str,
    headers: &[&str],
) -> Result<(), String> {
    for (sample, body) in rows {
        let dirname = dirnames.get(sample).map(String::as_str).unwrap_or("sample");
        let path = workspace
            .join("results")
            .join(dirname)
            .join(format!("{stem}.xlsx"));
        write_xlsx_only(&path, headers, body)?;
    }
    Ok(())
}

fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() == 1 {
        return values[0];
    }
    let Some(q) = q.is_finite().then_some(q.clamp(0.0, 1.0)) else {
        return 0.0;
    };
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return 0.0;
    }
    if finite.len() == 1 {
        return finite[0];
    }
    marray::np::percentile(&marray::np::array(finite), q * 100.0)
}

fn percentile_ylim(values: &[f64]) -> (f64, f64) {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return (0.0, 1.0);
    }
    let low = quantile(&finite, 0.0);
    let high = quantile(&finite, 1.0) / 0.9;
    if !low.is_finite() || !high.is_finite() {
        return (0.0, 1.0);
    }
    if low < high {
        (low, high)
    } else {
        let pad = if low == 0.0 { 1.0 } else { low.abs() * 0.05 };
        (low - pad, high + pad)
    }
}

fn figure_size_for_grid(nrows: usize, ncols: usize) -> (f64, f64) {
    let rows = nrows.max(1);
    let cols = ncols.max(1);
    if rows == 1 && cols == 1 {
        return (FIGURE_SINGLE_WIDTH_IN, FIGURE_SINGLE_HEIGHT_IN);
    }
    let panel_slots = rows * cols;
    let scale = if panel_slots >= 12 {
        0.75
    } else if panel_slots >= 8 {
        0.85
    } else {
        1.0
    };
    (
        PANEL_WIDTH_IN * cols as f64 * scale,
        PANEL_HEIGHT_IN * rows as f64 * scale,
    )
}

fn figure_builder_for_grid(nrows: usize, ncols: usize) -> FigureBuilder {
    let (width_in, height_in) = figure_size_for_grid(nrows, ncols);
    Figure::builder()
        .size(Size::inches(width_in, height_in))
        .label_fontsize(LABEL_FONTSIZE)
        .tick_fontsize(TICK_FONTSIZE)
        .title_fontsize(TITLE_FONTSIZE)
        .gaps(0.16, 0.22)
}

fn trace_line_style(color_name: &str, alpha: f64) -> LineStyle {
    LineStyle::new()
        .color(Color::hex(color_name))
        .alpha(alpha)
        .width(1.5)
}

fn save_figure(figure: &Figure, path: &Path, pad_inches: f64) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    figure
        .save(
            path,
            SaveOptions::new().dpi(FIGURE_DPI).pad_inches(pad_inches),
        )
        .map_err(|error| error.to_string())
}

fn sample_label(sample: usize, mapping: &SampleMapping) -> String {
    mapping
        .get(sample)
        .map(|entry| entry.name.clone())
        .unwrap_or_else(|| format!("sample {sample}"))
}

fn sample_subplot_title(sample: usize, trace_count: usize, mapping: &SampleMapping) -> String {
    let label = sample_label(sample, mapping);
    format!("{label} ({trace_count} traces)")
}

fn sample_trace_naming_haystack(
    sample: usize,
    paths: &[PathBuf],
    mapping: &SampleMapping,
) -> String {
    let mut parts = vec![sample_label(sample, mapping)];
    for path in paths {
        if let Some(name) = path.file_name().and_then(|value| value.to_str()) {
            parts.push(name.to_string());
        }
    }
    parts.join(" ")
}

fn trace_color_alpha(haystack: &str) -> (&'static str, f64) {
    let lower = haystack.to_lowercase();
    let color = if lower.contains("egfp") || lower.contains("gfp") {
        "green"
    } else if lower.contains("mcherry") {
        "red"
    } else if lower.contains("yfp") {
        "yellow"
    } else if lower.contains("bfp") {
        "blue"
    } else {
        "gray"
    };
    (color, 0.1)
}

fn subplot_grid_shape(panel_count: usize) -> (usize, usize) {
    let n = panel_count.max(1);
    match n {
        1 => (1, 1),
        2 => (1, 2),
        3 | 4 => (2, 2),
        5 | 6 => (2, 3),
        7..=9 => (3, 3),
        10..=12 => (3, 4),
        _ => {
            let cols = (n as f64).sqrt().ceil() as usize;
            let rows = n.div_ceil(cols);
            (rows.max(1), cols.max(1))
        }
    }
}

#[derive(Debug, Clone)]
struct SampleSummary {
    t_minutes: Vec<f64>,
    mean: Vec<f64>,
    median: Vec<f64>,
    q25: Vec<f64>,
    q75: Vec<f64>,
    trace_count: usize,
}

fn sample_summary_curves(traces: &[Vec<(f64, f64)>], interval: f64) -> Option<SampleSummary> {
    let mut by_time: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    let mut trace_count = 0usize;
    for trace in traces {
        let mut used = false;
        for &(t, y) in trace {
            if !t.is_finite() || !y.is_finite() {
                continue;
            }
            let minutes = t * interval;
            let key = (minutes * 1_000_000.0).round() as i64;
            by_time.entry(key).or_default().push(y);
            used = true;
        }
        if used {
            trace_count += 1;
        }
    }
    if by_time.is_empty() || trace_count == 0 {
        return None;
    }

    let mut t_minutes = Vec::with_capacity(by_time.len());
    let mut mean = Vec::with_capacity(by_time.len());
    let mut median = Vec::with_capacity(by_time.len());
    let mut q25 = Vec::with_capacity(by_time.len());
    let mut q75 = Vec::with_capacity(by_time.len());
    for (key, values) in by_time {
        t_minutes.push(key as f64 / 1_000_000.0);
        let n = values.len() as f64;
        mean.push(values.iter().sum::<f64>() / n);
        median.push(quantile(&values, 0.5));
        q25.push(quantile(&values, 0.25));
        q75.push(quantile(&values, 0.75));
    }
    Some(SampleSummary {
        t_minutes,
        mean,
        median,
        q25,
        q75,
        trace_count,
    })
}

fn summary_ylim(summary: Option<&SampleSummary>) -> (f64, f64) {
    let Some(summary) = summary else {
        return (0.0, 1.0);
    };
    let mut values = Vec::with_capacity(
        summary.mean.len() + summary.median.len() + summary.q25.len() + summary.q75.len(),
    );
    values.extend_from_slice(&summary.mean);
    values.extend_from_slice(&summary.median);
    values.extend_from_slice(&summary.q25);
    values.extend_from_slice(&summary.q75);
    percentile_ylim(&values)
}

fn companion_plot_path(primary: &Path, suffix: &str) -> PathBuf {
    let stem = primary
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("plot");
    primary.with_file_name(format!("{stem}_{suffix}.png"))
}

fn write_per_sample_metric_plots(
    panels: &[TracePanel],
    sample_dir: &dyn Fn(usize) -> PathBuf,
    file_stem: &str,
    y_label: &str,
    interval: f64,
    mapping: &SampleMapping,
) -> Result<(), String> {
    if panels.is_empty() {
        return Err("no panels to plot".to_string());
    }
    let mut all_y = Vec::new();
    for panel in panels {
        all_y.extend_from_slice(&panel.y_values);
    }
    let shared_ylim = percentile_ylim(&all_y);
    let summaries: Vec<Option<SampleSummary>> = panels
        .iter()
        .map(|panel| sample_summary_curves(&panel.traces, interval))
        .collect();
    let mut summary_values = Vec::new();
    for summary in summaries.iter().flatten() {
        summary_values.extend_from_slice(&summary.mean);
        summary_values.extend_from_slice(&summary.median);
        summary_values.extend_from_slice(&summary.q25);
        summary_values.extend_from_slice(&summary.q75);
    }
    let shared_summary = percentile_ylim(&summary_values);

    for (index, panel) in panels.iter().enumerate() {
        let dir = sample_dir(panel.sample);
        fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
        let primary = dir.join(format!("{file_stem}.png"));
        let own = percentile_ylim(&panel.y_values);
        let one = std::slice::from_ref(panel);
        let summary = summaries.get(index).unwrap_or(&None);
        let own_summary = summary_ylim(summary.as_ref());
        write_subplot_grid(one, &primary, y_label, interval, mapping, |_| own)?;
        write_subplot_grid(
            one,
            &companion_plot_path(&primary, "shared_y"),
            y_label,
            interval,
            mapping,
            |_| shared_ylim,
        )?;
        let summary_plot = companion_plot_path(&primary, "summary");
        write_summary_subplot_grid(
            one,
            std::slice::from_ref(summary),
            &summary_plot,
            y_label,
            mapping,
            |_| own_summary,
        )?;
        write_summary_subplot_grid(
            one,
            std::slice::from_ref(summary),
            &companion_plot_path(&summary_plot, "shared_y"),
            y_label,
            mapping,
            |_| shared_summary,
        )?;
    }
    Ok(())
}

fn write_subplot_grid(
    panels: &[TracePanel],
    output_plot: &Path,
    y_label: &str,
    interval: f64,
    mapping: &SampleMapping,
    ylim_for_panel: impl Fn(usize) -> (f64, f64),
) -> Result<(), String> {
    let (rows, cols) = subplot_grid_shape(panels.len().max(1));
    let mut builder = figure_builder_for_grid(rows, cols);

    for (index, panel) in panels.iter().enumerate() {
        let (y_low, y_high) = ylim_for_panel(index);
        let max_t = panel
            .traces
            .iter()
            .flat_map(|trace| trace.iter().map(|point| point.0))
            .fold(0.0f64, f64::max)
            * interval;
        let (color, alpha) = trace_color_alpha(&sample_trace_naming_haystack(
            panel.sample,
            &panel.paths,
            mapping,
        ));
        let title = sample_subplot_title(panel.sample, panel.traces.len(), mapping);
        let traces = panel.traces.clone();
        let y_label = y_label.to_string();
        let y_scientific = y_label.contains("intensity");

        builder = builder.panel(GridPos::new(rows, cols, index + 1), move |panel| {
            for trace in &traces {
                let x: Vec<f64> = trace.iter().map(|(time, _)| time * interval).collect();
                let y: Vec<f64> = trace.iter().map(|(_, value)| *value).collect();
                panel.line(&x, &y, trace_line_style(color, alpha));
            }
            let mut axes = AxesStyle::new()
                .title(title)
                .x_label("time (min)")
                .y_label(y_label)
                .y_range(y_low, y_high)
                .x_range(0.0, max_t.max(interval));
            if y_scientific {
                axes = axes.y_tick_format(TickFormat::Scientific);
            }
            panel.axes(axes);
        });
    }

    for index in panels.len()..(rows * cols) {
        builder = builder.panel(GridPos::new(rows, cols, index + 1), |panel| {
            panel.axes(AxesStyle::new().hide(true));
        });
    }

    let figure = builder.build().map_err(|error| error.to_string())?;
    let pad = if panels.len() <= 1 {
        SAVE_PAD_SINGLE_INCHES
    } else {
        SAVE_PAD_GRID_INCHES
    };
    save_figure(&figure, output_plot, pad)
}

fn write_summary_subplot_grid(
    panels: &[TracePanel],
    summaries: &[Option<SampleSummary>],
    output_plot: &Path,
    y_label: &str,
    mapping: &SampleMapping,
    ylim_for_panel: impl Fn(usize) -> (f64, f64),
) -> Result<(), String> {
    let (rows, cols) = subplot_grid_shape(panels.len().max(1));
    let mut builder = figure_builder_for_grid(rows, cols);

    for (index, panel) in panels.iter().enumerate() {
        let (y_low, y_high) = ylim_for_panel(index);
        let (color, _alpha) = trace_color_alpha(&sample_trace_naming_haystack(
            panel.sample,
            &panel.paths,
            mapping,
        ));
        let y_label = y_label.to_string();
        let y_scientific = y_label.contains("intensity");
        let summary = summaries.get(index).and_then(|value| value.as_ref());
        let title = sample_subplot_title(
            panel.sample,
            summary.map(|s| s.trace_count).unwrap_or(0),
            mapping,
        );
        let show_legend = index == 0;

        builder = builder.panel(GridPos::new(rows, cols, index + 1), move |p| {
            let max_t = if let Some(summary) = summary {
                p.fill_between(
                    &summary.t_minutes,
                    &summary.q25,
                    &summary.q75,
                    FillBetweenStyle::new()
                        .color(Color::hex(color))
                        .alpha(0.25)
                        .label("IQR"),
                );
                p.line(
                    &summary.t_minutes,
                    &summary.median,
                    trace_line_style(color, 1.0)
                        .width(1.8)
                        .dash(LineDash::Solid)
                        .label("median"),
                );
                p.line(
                    &summary.t_minutes,
                    &summary.mean,
                    trace_line_style(color, 1.0)
                        .width(1.5)
                        .dash(LineDash::Dashed)
                        .label("mean"),
                );
                summary
                    .t_minutes
                    .iter()
                    .copied()
                    .fold(0.0f64, f64::max)
                    .max(1.0)
            } else {
                1.0
            };

            let mut axes = AxesStyle::new()
                .title(title)
                .x_label("time (min)")
                .y_label(y_label)
                .y_range(y_low, y_high)
                .x_range(0.0, max_t);
            if y_scientific {
                axes = axes.y_tick_format(TickFormat::Scientific);
            }
            if show_legend && summary.is_some() {
                axes = axes.legend(LegendStyle::show());
            }
            p.axes(axes);
        });
    }

    for index in panels.len()..(rows * cols) {
        builder = builder.panel(GridPos::new(rows, cols, index + 1), |panel| {
            panel.axes(AxesStyle::new().hide(true));
        });
    }

    let figure = builder.build().map_err(|error| error.to_string())?;
    let pad = if panels.len() <= 1 {
        SAVE_PAD_SINGLE_INCHES
    } else {
        SAVE_PAD_GRID_INCHES
    };
    save_figure(&figure, output_plot, pad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::SampleAnalysis;
    use tiff::encoder::{colortype, TiffEncoder};

    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 32;

    fn paint_spot(frame: &mut [u16], cx: f64, cy: f64, amplitude: f64, sigma: f64) {
        let width = WIDTH as usize;
        let height = HEIGHT as usize;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let value = amplitude * (-0.5 * (dx * dx + dy * dy) / (sigma * sigma)).exp();
                let index = y * width + x;
                let summed = f64::from(frame[index]) + value;
                frame[index] = summed.round().clamp(0.0, 65535.0) as u16;
            }
        }
    }

    fn write_pages(path: &Path, pages: &[Vec<u16>]) {
        let file = File::create(path).expect("create tiff");
        let mut encoder = TiffEncoder::new(file).expect("encoder");
        for page in pages {
            let image = encoder
                .new_image::<colortype::Gray16>(WIDTH, HEIGHT)
                .expect("gray image");
            image.write_data(page).expect("write page");
        }
    }

    fn background_page(value: u16) -> Vec<u16> {
        vec![value; (WIDTH * HEIGHT) as usize]
    }

    fn write_position(workspace: &Path, position: u32, signal_pages: &[Vec<u16>]) {
        let pos_dir = workspace.join(format!("roi/Pos{position}"));
        fs::create_dir_all(&pos_dir).unwrap();
        let mut pages = Vec::new();
        for signal in signal_pages {
            pages.push(background_page(40));
            pages.push(signal.clone());
        }
        write_pages(&pos_dir.join("Roi0.tif"), &pages);
        let index = serde_json::json!({
            "position": position,
            "axisOrder": "TCZYX",
            "channelCount": 2,
            "timeCount": signal_pages.len(),
            "zCount": 1,
            "rois": [{
                "roi": 0,
                "fileName": "Roi0.tif",
                "bbox": { "roi": 0, "x": 0, "y": 0, "w": WIDTH, "h": HEIGHT }
            }]
        });
        fs::write(
            pos_dir.join("index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();
    }

    fn sample_mapping() -> SampleMapping {
        SampleMapping(vec![SampleAnalysis {
            name: "bound".into(),
            positions: vec![1],
            signal: vec![1],
            segmentation: 0,
        }])
    }

    #[test]
    fn detect_params_use_picasso_defaults_when_omitted() {
        let params = validate_detect_params(None, None).unwrap();
        assert_eq!(params.min_gradient, DEFAULT_MIN_GRADIENT);
        assert_eq!(params.box_size, DEFAULT_BOX_SIZE);
    }

    #[test]
    fn detect_params_reject_even_box_and_negative_threshold() {
        let even_err = validate_detect_params(None, Some(8)).unwrap_err();
        assert!(even_err.contains("odd integer"), "{even_err}");
        let neg_err = validate_detect_params(Some(-1.0), None).unwrap_err();
        assert!(neg_err.contains("minGradient"), "{neg_err}");
    }

    #[test]
    fn run_pipeline_rejects_non_positive_interval() {
        let dir = tempfile::tempdir().unwrap();
        let err = run_pipeline(
            dir.path(),
            &sample_mapping(),
            0.0,
            BindingDetectParams::default(),
        )
        .unwrap_err();
        assert!(err.contains("invalid interval"), "{err}");
    }

    #[test]
    fn synthetic_spots_are_counted_per_roi_and_frame() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let mut t0 = background_page(80);
        paint_spot(&mut t0, 16.0, 16.0, 8000.0, 1.4);
        let mut t1 = background_page(80);
        paint_spot(&mut t1, 16.0, 16.0, 8000.0, 1.4);
        paint_spot(&mut t1, 16.0, 24.0, 8000.0, 1.4);
        write_position(root, 1, &[t0, t1]);

        let params = BindingDetectParams {
            min_gradient: 80.0,
            box_size: 7,
        };
        run_pipeline(root, &sample_mapping(), 1.0, params).unwrap();

        let (headers, rows) = read_csv(&root.join("analysis/Pos1/binding.csv")).unwrap();
        assert_eq!(
            headers,
            [
                "roi",
                "t",
                "minutes",
                "spot_count",
                "intensity_sum",
                "cumulative_count"
            ]
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0], "0");
        assert_eq!(rows[0][1], "0");
        assert_eq!(rows[0][3], "1");
        assert_eq!(rows[1][1], "1");
        assert_eq!(rows[1][3], "2");
        let first_intensity: f64 = rows[0][4].parse().unwrap();
        let second_intensity: f64 = rows[1][4].parse().unwrap();
        assert!(first_intensity > 0.0);
        assert!(second_intensity > first_intensity);

        let (spot_headers, spot_rows) = read_csv(&root.join("analysis/Pos1/spots.csv")).unwrap();
        assert_eq!(spot_headers[0], "roi");
        assert_eq!(spot_rows.len(), 3);
        assert!(root.join("results/bound/binding.xlsx").is_file());
        assert!(root.join("results/bound/binding_traces.png").is_file());
        assert!(root
            .join("results/bound/binding_traces_summary_shared_y.png")
            .is_file());
        assert!(root
            .join("results/bound/binding_intensity_traces.png")
            .is_file());
        assert!(root
            .join("results/bound/binding_intensity_traces_summary_shared_y.png")
            .is_file());
    }

    #[test]
    fn empty_roi_writes_zero_counts() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write_position(root, 1, &[background_page(80), background_page(80)]);
        run_position(
            root,
            &sample_mapping(),
            1,
            1.0,
            BindingDetectParams::default(),
        )
        .unwrap();
        let (_, rows) = read_csv(&root.join("analysis/Pos1/binding.csv")).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][3], "0");
        assert_eq!(rows[1][3], "0");
        let (_, spots) = read_csv(&root.join("analysis/Pos1/spots.csv")).unwrap();
        assert!(spots.is_empty());
    }

    #[test]
    fn missing_roi_directory_fails_before_writing_files() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let err = run_position(
            root,
            &sample_mapping(),
            1,
            1.0,
            BindingDetectParams::default(),
        )
        .unwrap_err();
        assert!(err.contains("No ROI directory"), "{err}");
        assert!(!root.join("analysis/Pos1").exists());
    }

    #[test]
    fn a_border_candidate_does_not_abort_the_process() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let mut frame = background_page(80);
        paint_spot(&mut frame, 3.0, 16.0, 8000.0, 1.4);
        write_position(root, 1, &[frame]);
        let result = run_position(
            root,
            &sample_mapping(),
            1,
            1.0,
            BindingDetectParams::default(),
        );
        match result {
            Ok(()) => assert!(root.join("analysis/Pos1/binding.csv").is_file()),
            Err(error) => {
                assert!(
                    error.contains("Picasso") || error.contains("border"),
                    "{error}"
                );
                assert!(!root.join("analysis/Pos1").exists());
            }
        }
    }
}
