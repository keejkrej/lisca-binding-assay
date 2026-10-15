use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use csv::ReaderBuilder;
use marray::np::s;
use mplot::prelude::{AxesStyle, Color, Colormap, Figure, GridPos, ImageStyle, LegendLoc, LegendStyle, LineStyle, SaveOptions, ScatterStyle, Size};

use crate::filter_spots::read_spot_csv;
use crate::paths::traces_plot_output_path;
use crate::roi::load_roi_stack;

const PALETTE: [&str; 6] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
];

#[derive(Debug, Clone)]
pub struct ShowTracesResult {
    pub output_path: PathBuf,
    pub roi_sizes: Vec<usize>,
    pub x_axis: String,
    pub row_count: usize,
}

#[derive(Debug, Clone)]
pub struct PlotLnpResult {
    pub output_path: PathBuf,
    pub selected_roi: u32,
    pub selected_time: u32,
    pub spot_count: usize,
    pub cell_count: usize,
}

pub fn run_show_traces(
    input_file: &Path,
    output: Option<&Path>,
    use_time_real: bool,
) -> Result<ShowTracesResult, String> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(input_file)
        .map_err(|e| format!("Failed to open {}: {e}", input_file.display()))?;
    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| format!("Failed to read headers from {}: {e}", input_file.display()))?
        .iter()
        .map(str::to_string)
        .collect();

    let time_idx = headers.iter().position(|h| h == "time");
    let roi_size_idx = headers.iter().position(|h| h == "roi_size");
    let mean_idx = headers.iter().position(|h| h == "mean_intensity");
    if time_idx.is_none() || roi_size_idx.is_none() || mean_idx.is_none() {
        return Err(
            "Traces CSV must contain time, roi_size, and mean_intensity columns".to_string(),
        );
    }
    let time_real_idx = headers.iter().position(|h| h == "time_real");
    let has_time_real = time_real_idx.is_some();
    let use_real = use_time_real && has_time_real;

    let mut grouped: BTreeMap<usize, Vec<(f64, f64)>> = BTreeMap::new();
    let mut row_count = 0usize;

    for record in reader.records() {
        let rec = record.map_err(|e| format!("CSV row error: {e}"))?;
        let size: usize = rec
            .get(roi_size_idx.unwrap())
            .ok_or_else(|| "Missing roi_size".to_string())?
            .parse()
            .map_err(|e| format!("Invalid roi_size: {e}"))?;
        let x_val: f64 = if use_real {
            let s = rec
                .get(time_real_idx.unwrap())
                .ok_or_else(|| "Traces CSV is missing time_real values".to_string())?;
            if s.is_empty() {
                return Err("Traces CSV is missing time_real values".to_string());
            }
            s.parse()
                .map_err(|e| format!("Invalid time_real: {e}"))?
        } else {
            rec.get(time_idx.unwrap())
                .ok_or_else(|| "Missing time".to_string())?
                .parse()
                .map_err(|e| format!("Invalid time: {e}"))?
        };
        let y_val: f64 = rec
            .get(mean_idx.unwrap())
            .ok_or_else(|| "Missing mean_intensity".to_string())?
            .parse()
            .map_err(|e| format!("Invalid mean_intensity: {e}"))?;
        grouped.entry(size).or_default().push((x_val, y_val));
        row_count += 1;
    }

    if row_count == 0 {
        return Err("Traces CSV has no rows".to_string());
    }

    for pairs in grouped.values_mut() {
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    let output_path = traces_plot_output_path(input_file, output);
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create {}: {e}", parent.display()))?;
    }

    let x_label = if use_real { "time_real" } else { "time" };
    let roi_sizes: Vec<usize> = grouped.keys().copied().collect();

    let fig = Figure::builder()
        .size(Size::inches(10.0, 5.0))
        .panel(GridPos::new(1, 1, 1), |p| {
            for (idx, (&size, pairs)) in grouped.iter().enumerate() {
                let xs: Vec<f64> = pairs.iter().map(|&(x, _)| x).collect();
                let ys: Vec<f64> = pairs.iter().map(|&(_, y)| y).collect();
                let color = Color::hex(PALETTE[idx % PALETTE.len()]);
                p.line(
                    &xs,
                    &ys,
                    LineStyle::new()
                        .color(color)
                        .width(1.8)
                        .label(format!("{size}x{size}")),
                );
            }
            p.axes(
                AxesStyle::new()
                    .title("Region intensity traces")
                    .x_label(x_label)
                    .y_label("mean_intensity")
                    .legend(LegendStyle::show().loc(LegendLoc::UpperRight)),
            );
        })
        .build()
        .map_err(|e| format!("Failed to build traces figure: {e}"))?;

    fig.save(&output_path, SaveOptions::new().dpi(160))
        .map_err(|e| format!("Failed to save {}: {e}", output_path.display()))?;

    Ok(ShowTracesResult {
        output_path,
        roi_sizes,
        x_axis: x_label.to_string(),
        row_count,
    })
}

pub fn read_counts_csv(
    path: &Path,
) -> Result<(Vec<u32>, BTreeMap<u32, (Vec<f64>, Vec<usize>)>), String> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("Failed to open {}: {e}", path.display()))?;
    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| format!("Failed to read headers from {}: {e}", path.display()))?
        .iter()
        .map(str::to_string)
        .collect();

    let roi_idx = headers.iter().position(|h| h == "roi");
    let time_real_idx = headers.iter().position(|h| h == "time_real");
    let count_idx = headers.iter().position(|h| h == "spot_count");
    if roi_idx.is_none() || time_real_idx.is_none() || count_idx.is_none() {
        return Err("Counts CSV must contain roi, time_real, and spot_count columns".to_string());
    }

    let mut raw_grouped: BTreeMap<u32, Vec<(f64, usize)>> = BTreeMap::new();
    for record in reader.records() {
        let rec = record.map_err(|e| format!("Row error: {e}"))?;
        let roi: u32 = rec
            .get(roi_idx.unwrap())
            .unwrap()
            .parse()
            .map_err(|e| format!("Invalid roi: {e}"))?;
        let t_real: f64 = rec
            .get(time_real_idx.unwrap())
            .unwrap()
            .parse()
            .map_err(|e| format!("Invalid time_real: {e}"))?;
        let count: usize = rec
            .get(count_idx.unwrap())
            .unwrap()
            .parse()
            .map_err(|e| format!("Invalid spot_count: {e}"))?;
        raw_grouped.entry(roi).or_default().push((t_real, count));
    }

    if raw_grouped.is_empty() {
        return Err("Counts CSV has no rows".to_string());
    }

    let mut grouped = BTreeMap::new();
    for (roi, mut pairs) in raw_grouped {
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let times = pairs.iter().map(|&(t, _)| t).collect();
        let counts = pairs.iter().map(|&(_, c)| c).collect();
        grouped.insert(roi, (times, counts));
    }
    let rois: Vec<u32> = grouped.keys().copied().collect();
    Ok((rois, grouped))
}

pub fn auto_select_roi(grouped: &BTreeMap<u32, (Vec<f64>, Vec<usize>)>) -> u32 {
    grouped
        .iter()
        .max_by_key(|(_, (_, counts))| counts.last().copied().unwrap_or(0))
        .map(|(&roi, _)| roi)
        .unwrap_or(0)
}

pub fn auto_select_time(
    grouped: &BTreeMap<u32, (Vec<f64>, Vec<usize>)>,
    roi: u32,
) -> u32 {
    let Some((_, counts)) = grouped.get(&roi) else {
        return 0;
    };
    if counts.is_empty() {
        return 0;
    }
    let max_c = *counts.iter().max().unwrap_or(&0) as f64;
    let target = max_c * 0.7;
    counts
        .iter()
        .position(|&c| (c as f64) >= target)
        .unwrap_or(counts.len() - 1) as u32
}

pub fn intensity_to_radius(intensity: f64) -> f64 {
    (2.0 + 14.0 * (intensity - 2000.0) / 14000.0).clamp(2.0, 16.0)
}

pub fn resolve_spot_csv(filtered_dir: &Path, roi: u32, time_index: u32) -> Result<PathBuf, String> {
    let candidates = [
        filtered_dir.join(format!("roi{roi}_time{time_index:04}_filtered.csv")),
        filtered_dir.join(format!("roi{roi:02}_time{time_index:09}_filtered.csv")),
        filtered_dir.join(format!("roi{roi:02}_time{time_index:04}_filtered.csv")),
        filtered_dir.join(format!("roi{roi}_time{time_index:04}.csv")),
    ];
    for p in &candidates {
        if p.is_file() {
            return Ok(p.clone());
        }
    }
    Err(format!(
        "Filtered spot CSV not found: {}",
        candidates[0].display()
    ))
}

pub fn run_plot_lnp(
    input_dir: &Path,
    counts_csv: &Path,
    filtered_dir: &Path,
    output: &Path,
    position: u32,
    channel: u32,
    roi: Option<u32>,
    time: Option<u32>,
    time_unit: &str,
) -> Result<PlotLnpResult, String> {
    if time_unit != "sec" && time_unit != "min" {
        return Err("--time-unit must be 'sec' or 'min'".to_string());
    }
    let (rois, grouped) = read_counts_csv(counts_csv)?;
    let selected_roi = roi.unwrap_or_else(|| auto_select_roi(&grouped));
    let selected_time = time.unwrap_or_else(|| auto_select_time(&grouped, selected_roi));

    let fluo_stack = load_roi_stack(input_dir, position, selected_roi, channel)?;
    let _bf_stack = load_roi_stack(input_dir, position, selected_roi, 0)?;
    let image = fluo_stack
        .slice(s![selected_time as usize, .., ..])
        .to_owned();
    let (h, w) = image.dim();
    let flat_img: Vec<f64> = image.iter().copied().collect();

    let spot_csv = resolve_spot_csv(filtered_dir, selected_roi, selected_time)?;
    let (_, spot_rows) = read_spot_csv(&spot_csv)?;

    let mut spot_xs = Vec::with_capacity(spot_rows.len());
    let mut spot_ys = Vec::with_capacity(spot_rows.len());
    for row in &spot_rows {
        if let (Some(xs), Some(ys)) = (row.get("x"), row.get("y")) {
            if let (Ok(x), Ok(y)) = (xs.parse::<f64>(), ys.parse::<f64>()) {
                spot_xs.push(x);
                spot_ys.push(y);
            }
        }
    }

    let time_scale = if time_unit == "min" { 1.0 / 60.0 } else { 1.0 };
    let x_label = if time_unit == "min" { "t (min)" } else { "t (s)" };

    let mut mean_times: Vec<f64> = Vec::new();
    let mut mean_counts: Vec<f64> = Vec::new();
    for &r in &rois {
        if let Some((times, counts)) = grouped.get(&r) {
            if mean_times.is_empty() {
                mean_times = times.iter().map(|&t| t * time_scale).collect();
                mean_counts = counts.iter().map(|&c| c as f64).collect();
            } else {
                for (idx, &c) in counts.iter().enumerate() {
                    if idx < mean_counts.len() {
                        mean_counts[idx] += c as f64;
                    }
                }
            }
        }
    }
    if !rois.is_empty() {
        let n_rois = rois.len() as f64;
        for v in &mut mean_counts {
            *v /= n_rois;
        }
    }

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create {}: {e}", parent.display()))?;
    }

    let fig = Figure::builder()
        .size(Size::inches(12.0, 4.0))
        .panel(GridPos::new(1, 3, 1), |p| {
            p.image(
                flat_img,
                w,
                h,
                ImageStyle::new().colormap(Colormap::Grayscale).colorbar(false),
            );
            p.axes(AxesStyle::new().title("A: Fluorescence"));
        })
        .panel(GridPos::new(1, 3, 2), |p| {
            if !spot_xs.is_empty() {
                p.scatter(
                    &spot_xs,
                    &spot_ys,
                    ScatterStyle::new()
                        .color(Color::hex("#ffeb3b"))
                        .size(5.0)
                        .label("Detected LNPs"),
                );
            }
            p.axes(
                AxesStyle::new()
                    .title("B: Detections")
                    .x_label("x (px)")
                    .y_label("y (px)"),
            );
        })
        .panel(GridPos::new(1, 3, 3), |p| {
            for &r in &rois {
                if let Some((times, counts)) = grouped.get(&r) {
                    let pt: Vec<f64> = times.iter().map(|&t| t * time_scale).collect();
                    let pc: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
                    p.line(
                        &pt,
                        &pc,
                        LineStyle::new().color(Color::hex("#b8b8b8")).width(1.0),
                    );
                }
            }
            if !mean_times.is_empty() {
                p.line(
                    &mean_times,
                    &mean_counts,
                    LineStyle::new()
                        .color(Color::hex("#d7263d"))
                        .width(2.0)
                        .label("median"),
                );
            }
            p.axes(
                AxesStyle::new()
                    .title("C: Binding Kinetics")
                    .x_label(x_label)
                    .y_label("n"),
            );
        })
        .build()
        .map_err(|e| format!("Failed to build plot-lnp figure: {e}"))?;

    fig.save(output, SaveOptions::new().dpi(200))
        .map_err(|e| format!("Failed to save {}: {e}", output.display()))?;

    Ok(PlotLnpResult {
        output_path: output.to_path_buf(),
        selected_roi,
        selected_time,
        spot_count: spot_rows.len(),
        cell_count: rois.len(),
    })
}
