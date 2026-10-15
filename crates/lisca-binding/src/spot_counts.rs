use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use csv::ReaderBuilder;

use crate::paths::spot_counts_output_path;
use crate::roi::load_time_map;

#[derive(Debug, Clone)]
pub struct SpotCountsResult {
    pub output_path: PathBuf,
    pub row_count: usize,
    pub roi_count: usize,
    pub time_count: usize,
    pub cumulative: bool,
}

pub fn parse_spot_file(path: &Path) -> Result<(u32, u32), String> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| "Invalid filename".to_string())?;
    let lower = name.to_ascii_lowercase();
    let stem = lower
        .strip_suffix("_filtered.csv")
        .or_else(|| lower.strip_suffix(".csv"))
        .ok_or_else(|| format!("Unrecognized spot CSV filename: {name}"))?;
    let rest = stem
        .strip_prefix("roi")
        .ok_or_else(|| format!("Unrecognized spot CSV filename: {name}"))?;
    let (roi_str, time_str) = rest
        .split_once("_time")
        .ok_or_else(|| format!("Unrecognized spot CSV filename: {name}"))?;
    let roi: u32 = roi_str
        .parse()
        .map_err(|_| format!("Unrecognized spot CSV filename: {name}"))?;
    let time: u32 = time_str
        .parse()
        .map_err(|_| format!("Unrecognized spot CSV filename: {name}"))?;
    Ok((roi, time))
}

pub fn read_spot_coordinates(path: &Path) -> Result<Vec<(f64, f64)>, String> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("Failed to open {}: {e}", path.display()))?;
    let headers = reader
        .headers()
        .map_err(|e| format!("Failed to read headers from {}: {e}", path.display()))?
        .clone();
    let x_idx = headers
        .iter()
        .position(|h| h == "x")
        .ok_or_else(|| format!("Spot CSV {} must contain x and y columns", path.display()))?;
    let y_idx = headers
        .iter()
        .position(|h| h == "y")
        .ok_or_else(|| format!("Spot CSV {} must contain x and y columns", path.display()))?;

    let mut coords = Vec::new();
    for record in reader.records() {
        let rec = record.map_err(|e| format!("Row error in {}: {e}", path.display()))?;
        let x: f64 = rec
            .get(x_idx)
            .ok_or_else(|| "Missing x".to_string())?
            .parse()
            .map_err(|e| format!("Invalid x: {e}"))?;
        let y: f64 = rec
            .get(y_idx)
            .ok_or_else(|| "Missing y".to_string())?
            .parse()
            .map_err(|e| format!("Invalid y: {e}"))?;
        coords.push((x, y));
    }
    Ok(coords)
}

pub fn count_new_spots(
    previous: &[(f64, f64)],
    current: &[(f64, f64)],
    match_distance: f64,
) -> (usize, Vec<(f64, f64)>) {
    if current.is_empty() {
        return (0, previous.to_vec());
    }
    if previous.is_empty() {
        return (current.len(), current.to_vec());
    }

    let mut updated = previous.to_vec();
    let mut new_count = 0usize;
    for &(cx, cy) in current {
        let min_dist = previous
            .iter()
            .map(|&(px, py)| (px - cx).hypot(py - cy))
            .fold(f64::INFINITY, f64::min);
        if min_dist <= match_distance {
            continue;
        }
        updated.push((cx, cy));
        new_count += 1;
    }
    (new_count, updated)
}

pub fn collect_spot_files(input_dir: &Path) -> Result<BTreeMap<u32, BTreeMap<u32, PathBuf>>, String> {
    let entries = std::fs::read_dir(input_dir)
        .map_err(|e| format!("Failed to read {}: {e}", input_dir.display()))?;
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .map(|s| s.eq_ignore_ascii_case("csv"))
                    .unwrap_or(false)
        })
        .collect();
    paths.sort();

    let mut grouped: BTreeMap<u32, BTreeMap<u32, PathBuf>> = BTreeMap::new();
    for path in paths {
        if let Ok((roi, time_index)) = parse_spot_file(&path) {
            grouped.entry(roi).or_default().insert(time_index, path);
        }
    }
    if grouped.is_empty() {
        return Err(format!("No spot CSV files found in {}", input_dir.display()));
    }
    Ok(grouped)
}

fn fmt_py_float(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

pub fn run_spot_counts(
    input_dir: &Path,
    output: &Path,
    position: u32,
    channel: u32,
    time_interval: f64,
    time_map: Option<&Path>,
    cumulative: bool,
    match_distance: f64,
) -> Result<SpotCountsResult, String> {
    let grouped = collect_spot_files(input_dir)?;
    let all_times: Vec<u32> = grouped
        .values()
        .flat_map(|m| m.keys().copied())
        .collect::<BTreeSet<u32>>()
        .into_iter()
        .collect();

    let time_real_by_index: BTreeMap<u32, f64> = if let Some(tm_path) = time_map {
        load_time_map(tm_path)?
    } else {
        all_times
            .iter()
            .map(|&t| (t, f64::from(t) * time_interval))
            .collect()
    };

    let output_path = spot_counts_output_path(output, position, channel);
    std::fs::create_dir_all(output)
        .map_err(|e| format!("Failed to create {}: {e}", output.display()))?;
    let file = File::create(&output_path)
        .map_err(|e| format!("Failed to create {}: {e}", output_path.display()))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "roi,time,time_real,spot_count,frame_spots")
        .map_err(|e| format!("Failed to write header: {e}"))?;

    let mut row_count = 0usize;
    for (&roi, by_time) in &grouped {
        let mut known: Vec<(f64, f64)> = Vec::new();
        let mut cumulative_count = 0usize;
        for &time_index in &all_times {
            let current = if let Some(path) = by_time.get(&time_index) {
                read_spot_coordinates(path)?
            } else {
                Vec::new()
            };
            let frame_count = current.len();
            let count_value = if cumulative {
                let (new_count, next_known) = count_new_spots(&known, &current, match_distance);
                known = next_known;
                cumulative_count += new_count;
                cumulative_count
            } else {
                frame_count
            };
            let time_real = *time_real_by_index
                .get(&time_index)
                .ok_or_else(|| format!("time map missing entry for time={time_index}"))?;
            writeln!(
                writer,
                "{},{},{},{},{}",
                roi,
                time_index,
                fmt_py_float(time_real),
                count_value,
                frame_count
            )
            .map_err(|e| format!("Failed to write row: {e}"))?;
            row_count += 1;
        }
    }
    writer
        .flush()
        .map_err(|e| format!("Failed to flush {}: {e}", output_path.display()))?;

    Ok(SpotCountsResult {
        output_path,
        row_count,
        roi_count: grouped.len(),
        time_count: all_times.len(),
        cumulative,
    })
}
