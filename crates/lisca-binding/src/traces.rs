use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use marray::np::{s, Array2, Array3};

use crate::frames::{available_times, load_stack};
use crate::paths::traces_output_path;
use crate::roi::load_time_map;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projection {
    Mean,
    Max,
    Sum,
}

impl Projection {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "mean" => Ok(Self::Mean),
            "max" => Ok(Self::Max),
            "sum" => Ok(Self::Sum),
            other => Err(format!(
                "Invalid projection '{other}', expected 'mean', 'max', or 'sum'"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TracesResult {
    pub output_path: PathBuf,
    pub row_count: usize,
    pub time_count: usize,
    pub roi_sizes: Vec<usize>,
    pub center_y: usize,
    pub center_x: usize,
}

pub fn parse_sizes(sizes: &str) -> Result<Vec<usize>, String> {
    let mut values = Vec::new();
    let mut seen = BTreeSet::new();
    for part in sizes.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val: usize = trimmed
            .parse()
            .map_err(|_| format!("Invalid ROI size '{trimmed}'"))?;
        if val == 0 {
            return Err("ROI sizes must be positive integers".to_string());
        }
        if !seen.insert(val) {
            return Err("ROI sizes must be unique".to_string());
        }
        values.push(val);
    }
    if values.is_empty() {
        return Err("At least one ROI size is required".to_string());
    }
    values.sort_unstable();
    Ok(values)
}

pub fn project_stack(stack: &Array3<f64>, projection: Projection) -> Array2<f64> {
    let shape = stack.shape();
    let (nz, h, w) = (shape[0], shape[1], shape[2]);
    let mut out = Array2::<f64>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let val = match projection {
                Projection::Mean => {
                    let sum: f64 = (0..nz).map(|z| stack[[z, y, x]]).sum();
                    sum / (nz as f64)
                }
                Projection::Max => (0..nz)
                    .map(|z| stack[[z, y, x]])
                    .fold(f64::NEG_INFINITY, f64::max),
                Projection::Sum => (0..nz).map(|z| stack[[z, y, x]]).sum(),
            };
            out[[y, x]] = val;
        }
    }
    out
}

pub fn select_plane(stack: &Array3<f64>, z: usize) -> Result<Array2<f64>, String> {
    let nz = stack.shape()[0];
    if z >= nz {
        return Err(format!("z={z} is out of range for stack depth {nz}"));
    }
    Ok(stack.slice(s![z, .., ..]).to_owned())
}

pub fn square_roi_bounds(
    center_y: usize,
    center_x: usize,
    size: usize,
    height: usize,
    width: usize,
) -> Result<(usize, usize, usize, usize), String> {
    let half = size / 2;
    if center_y < half || center_x < half {
        return Err(format!(
            "ROI size={size} centered at ({center_y}, {center_x}) extends outside image bounds ({height}, {width})"
        ));
    }
    let y0 = center_y - half;
    let x0 = center_x - half;
    let y1 = y0 + size;
    let x1 = x0 + size;
    if y1 > height || x1 > width {
        return Err(format!(
            "ROI size={size} centered at ({center_y}, {center_x}) extends outside image bounds ({height}, {width})"
        ));
    }
    Ok((y0, y1, x0, x1))
}

pub fn sample_roi_mean(
    image: &Array2<f64>,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
) -> Result<f64, String> {
    if y1 <= y0 || x1 <= x0 {
        return Err(format!("Empty ROI region at y=[{y0},{y1}), x=[{x0},{x1})"));
    }
    let mut sum = 0.0;
    let count = ((y1 - y0) * (x1 - x0)) as f64;
    for y in y0..y1 {
        for x in x0..x1 {
            sum += image[[y, x]];
        }
    }
    Ok(sum / count)
}

fn fmt_py_float(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

pub fn run_traces(
    input_dir: &Path,
    position: u32,
    channel: u32,
    sizes: &str,
    center_y: Option<usize>,
    center_x: Option<usize>,
    z: Option<usize>,
    projection: Projection,
    time_map: Option<&Path>,
    output: &Path,
) -> Result<TracesResult, String> {
    let roi_sizes = parse_sizes(sizes)?;
    let times = available_times(input_dir, position, channel);
    if times.is_empty() {
        return Err(format!(
            "No frames found for position={position}, channel={channel} under {}",
            input_dir.display()
        ));
    }

    let time_real_by_index = if let Some(tm) = time_map {
        Some(load_time_map(tm)?)
    } else {
        None
    };

    let first_stack = load_stack(input_dir, position, channel, times[0])?;
    let reference_image = if let Some(z_idx) = z {
        select_plane(&first_stack, z_idx)?
    } else {
        project_stack(&first_stack, projection)
    };
    let (h, w) = reference_image.dim();
    let resolved_center_y = center_y.unwrap_or(h / 2);
    let resolved_center_x = center_x.unwrap_or(w / 2);

    let mut roi_bounds = Vec::with_capacity(roi_sizes.len());
    for &size in &roi_sizes {
        let bounds = square_roi_bounds(resolved_center_y, resolved_center_x, size, h, w)?;
        roi_bounds.push((size, bounds));
    }

    let output_path = traces_output_path(output, position, channel);
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create {}: {e}", parent.display()))?;
    }
    let file = File::create(&output_path)
        .map_err(|e| format!("Failed to create {}: {e}", output_path.display()))?;
    let mut writer = BufWriter::new(file);

    if time_real_by_index.is_some() {
        writeln!(
            writer,
            "time,time_real,roi_size,center_y,center_x,y0,y1,x0,x1,mean_intensity"
        )
        .map_err(|e| format!("Failed to write header: {e}"))?;
    } else {
        writeln!(
            writer,
            "time,roi_size,center_y,center_x,y0,y1,x0,x1,mean_intensity"
        )
        .map_err(|e| format!("Failed to write header: {e}"))?;
    }

    let mut row_count = 0usize;
    for &time_index in &times {
        let stack = load_stack(input_dir, position, channel, time_index)?;
        let image = if let Some(z_idx) = z {
            select_plane(&stack, z_idx)?
        } else {
            project_stack(&stack, projection)
        };
        if image.dim() != (h, w) {
            return Err(format!(
                "Frame time={time_index} shape {:?} does not match reference shape ({h}, {w})",
                image.dim()
            ));
        }

        for &(size, (y0, y1, x0, x1)) in &roi_bounds {
            let mean_val = sample_roi_mean(&image, y0, y1, x0, x1)?;
            if let Some(ref tm) = time_real_by_index {
                let t_real = *tm
                    .get(&time_index)
                    .ok_or_else(|| format!("time_map missing entry for time={time_index}"))?;
                writeln!(
                    writer,
                    "{},{},{},{},{},{},{},{},{},{}",
                    time_index,
                    fmt_py_float(t_real),
                    size,
                    resolved_center_y,
                    resolved_center_x,
                    y0,
                    y1,
                    x0,
                    x1,
                    fmt_py_float(mean_val)
                )
                .map_err(|e| format!("Failed to write row: {e}"))?;
            } else {
                writeln!(
                    writer,
                    "{},{},{},{},{},{},{},{},{}",
                    time_index,
                    size,
                    resolved_center_y,
                    resolved_center_x,
                    y0,
                    y1,
                    x0,
                    x1,
                    fmt_py_float(mean_val)
                )
                .map_err(|e| format!("Failed to write row: {e}"))?;
            }
            row_count += 1;
        }
    }
    writer
        .flush()
        .map_err(|e| format!("Failed to flush {}: {e}", output_path.display()))?;

    Ok(TracesResult {
        output_path,
        row_count,
        time_count: times.len(),
        roi_sizes,
        center_y: resolved_center_y,
        center_x: resolved_center_x,
    })
}
