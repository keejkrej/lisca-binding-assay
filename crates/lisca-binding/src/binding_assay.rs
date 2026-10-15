//! Workspace-level LNP binding assay analysis (`lnp-binding`) using Picasso.
//!
//! For each ROI crop and time frame on the first signal channel, Picasso detects
//! individual nanoparticles and fits a 2D Gaussian to each particle to retrieve
//! its intensity and radius (`sqrt(sx * sy)`).
//!
//! Outputs written under `analysis/Pos{n}/`:
//! - `particles.csv`: every detected particle (`roi,t,minutes,particle,x,y,intensity,radius,photons,sx,sy,fwhm,background,snb,net_gradient`)
//! - `binding.csv`: per-ROI time series (`roi,t,minutes,particles,intensity_mean,intensity_median,intensity_iqr,radius_mean,radius_median,radius_iqr`)
//! - `binding_summary.csv`: per-ROI summary across frames (`roi,frames,particles_mean,intensity_mean,intensity_median,intensity_iqr,radius_mean,radius_median,radius_iqr`)
//!
//! And per-sample workbooks under `results/<sample>/`:
//! - `particles.xlsx`
//! - `binding.xlsx`
//! - `binding_summary.xlsx`

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use csv::{ReaderBuilder, WriterBuilder};
use marray::np::{Array2, s};
use serde::{Deserialize, Serialize};

use crate::export::write_xlsx_only;
use crate::picasso_stage::{PicassoParams, detect_spots_picasso};
use crate::roi::{load_roi_index, load_roi_stack, roi_dir};
use crate::sample::{SampleMapping, sample_pack_dirnames};

pub const DEFAULT_MIN_GRADIENT: f64 = 250.0;
pub const DEFAULT_BOX_SIZE: usize = 7;
const NOISE_GRADIENT_SIGMAS: f64 = 25.0;

pub const PARTICLE_CSV_HEADERS: &[&str] = &[
    "roi",
    "t",
    "minutes",
    "particle",
    "x",
    "y",
    "intensity",
    "radius",
    "photons",
    "sx",
    "sy",
    "fwhm",
    "background",
    "snb",
    "net_gradient",
];

pub const PARTICLE_XLSX_HEADERS: &[&str] = &[
    "pos",
    "roi",
    "t",
    "minutes",
    "particle",
    "x",
    "y",
    "intensity",
    "radius",
    "photons",
    "sx",
    "sy",
    "fwhm",
    "background",
    "snb",
    "net_gradient",
];

pub const BINDING_CSV_HEADERS: &[&str] = &[
    "roi",
    "t",
    "minutes",
    "particles",
    "intensity_mean",
    "intensity_median",
    "intensity_iqr",
    "radius_mean",
    "radius_median",
    "radius_iqr",
];

pub const BINDING_XLSX_HEADERS: &[&str] = &[
    "pos",
    "roi",
    "t",
    "minutes",
    "particles",
    "intensity_mean",
    "intensity_median",
    "intensity_iqr",
    "radius_mean",
    "radius_median",
    "radius_iqr",
];

pub const BINDING_SUMMARY_CSV_HEADERS: &[&str] = &[
    "roi",
    "frames",
    "particles_mean",
    "intensity_mean",
    "intensity_median",
    "intensity_iqr",
    "radius_mean",
    "radius_median",
    "radius_iqr",
];

pub const BINDING_SUMMARY_XLSX_HEADERS: &[&str] = &[
    "pos",
    "roi",
    "frames",
    "particles_mean",
    "intensity_mean",
    "intensity_median",
    "intensity_iqr",
    "radius_mean",
    "radius_median",
    "radius_iqr",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParticleSpot {
    pub x: f64,
    pub y: f64,
    pub intensity: f64,
    pub radius: f64,
    pub photons: f64,
    pub sx: f64,
    pub sy: f64,
    pub fwhm: f64,
    pub background: f64,
    pub snb: f64,
    pub net_gradient: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParticleRecord {
    pub pos: u32,
    pub roi: u32,
    pub t: u32,
    pub minutes: f64,
    pub particle: u32,
    pub spot: ParticleSpot,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct BindingFrameStats {
    pub particles: u32,
    pub intensity_mean: f64,
    pub intensity_median: f64,
    pub intensity_iqr: f64,
    pub radius_mean: f64,
    pub radius_median: f64,
    pub radius_iqr: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BindingRow {
    pub pos: u32,
    pub roi: u32,
    pub t: u32,
    pub minutes: f64,
    pub stats: BindingFrameStats,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BindingSummaryRow {
    pub pos: u32,
    pub roi: u32,
    pub frames: u32,
    pub particles_mean: f64,
    pub intensity_mean: f64,
    pub intensity_median: f64,
    pub intensity_iqr: f64,
    pub radius_mean: f64,
    pub radius_median: f64,
    pub radius_iqr: f64,
}

/// Histogram of fluorescence values in `[0, 65535]` to estimate shared position noise.
#[derive(Clone)]
pub struct FluorescenceHistogram {
    bins: Vec<u32>,
    count: u64,
}

impl FluorescenceHistogram {
    pub fn new() -> Self {
        Self {
            bins: vec![0; 65536],
            count: 0,
        }
    }

    pub fn sample(&mut self, pixels: &[f64]) {
        for (index, value) in pixels.iter().enumerate() {
            if index % 4 != 0 || !value.is_finite() || *value < 0.0 {
                continue;
            }
            let bin = (*value).round() as usize;
            if bin >= self.bins.len() {
                continue;
            }
            self.bins[bin] += 1;
            self.count += 1;
        }
    }

    pub fn min_gradient(&self) -> f64 {
        if self.count == 0 {
            return DEFAULT_MIN_GRADIENT;
        }
        let low = percentile_bin(&self.bins, self.count, 0.5);
        let mut deviations = vec![0u32; self.bins.len()];
        for (bin, count) in self.bins.iter().copied().enumerate() {
            if count == 0 {
                continue;
            }
            let deviation = (bin as f64 - low).abs().round() as usize;
            let slot = deviation.min(deviations.len() - 1);
            deviations[slot] = deviations[slot].saturating_add(count);
        }
        let mad = percentile_bin(&deviations, self.count, 0.5);
        let sigma = 1.4826 * mad;
        (NOISE_GRADIENT_SIGMAS * sigma).max(DEFAULT_MIN_GRADIENT)
    }
}

impl Default for FluorescenceHistogram {
    fn default() -> Self {
        Self::new()
    }
}

fn percentile_bin(bins: &[u32], count: u64, quantile: f64) -> f64 {
    if count == 0 {
        return 0.0;
    }
    let target = ((count as f64 - 1.0) * quantile).round() as u64;
    let mut seen = 0u64;
    for (bin, count) in bins.iter().copied().enumerate() {
        seen += u64::from(count);
        if seen > target {
            return bin as f64;
        }
    }
    0.0
}

fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
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

fn round_dec(v: f64, decimals: i32) -> f64 {
    let factor = 10f64.powi(decimals);
    let r = (v * factor).round() / factor;
    if r == 0.0 { 0.0 } else { r }
}

pub fn detect_particles_frame(
    frame: &Array2<f64>,
    min_gradient: f64,
) -> Result<Vec<ParticleSpot>, String> {
    let params = PicassoParams {
        min_gradient,
        box_size: DEFAULT_BOX_SIZE,
        fit_method: "lq".to_string(),
        model: "elliptic".to_string(),
        baseline: 0.0,
        sensitivity: 1.0,
        gain: 1.0,
        gaussian_sigma: None,
    };
    let spots = detect_spots_picasso(frame, &params, "Pos0", "roi0", 0)?;
    Ok(spots
        .into_iter()
        .map(|spot| {
            let sx = round_dec(spot.sx.max(0.0), 4);
            let sy = round_dec(spot.sy.max(0.0), 4);
            let radius = round4((sx * sy).sqrt());
            ParticleSpot {
                x: round_dec(spot.x, 4),
                y: round_dec(spot.y, 4),
                intensity: round_dec(spot.intens_a, 3),
                radius,
                photons: round_dec(spot.photons, 2),
                sx,
                sy,
                fwhm: round_dec(spot.fwhm, 4),
                background: round_dec(spot.intens_b, 3),
                snb: round_dec(spot.snb, 3),
                net_gradient: round_dec(spot.net_gradient, 2),
            }
        })
        .collect())
}

pub fn summarize_particles(particles: &[ParticleSpot]) -> BindingFrameStats {
    if particles.is_empty() {
        return BindingFrameStats::default();
    }
    let count = particles.len() as u32;
    let n = particles.len() as f64;
    let intensities: Vec<f64> = particles.iter().map(|p| p.intensity).collect();
    let radii: Vec<f64> = particles.iter().map(|p| p.radius).collect();

    let intensity_mean = round4(intensities.iter().sum::<f64>() / n);
    let intensity_median = round4(quantile(&intensities, 0.50));
    let intensity_iqr = round4(quantile(&intensities, 0.75) - quantile(&intensities, 0.25));

    let radius_mean = round4(radii.iter().sum::<f64>() / n);
    let radius_median = round4(quantile(&radii, 0.50));
    let radius_iqr = round4(quantile(&radii, 0.75) - quantile(&radii, 0.25));

    BindingFrameStats {
        particles: count,
        intensity_mean,
        intensity_median,
        intensity_iqr,
        radius_mean,
        radius_median,
        radius_iqr,
    }
}

pub fn run_position_binding(
    workspace: &Path,
    mapping: &SampleMapping,
    position: u32,
    interval_minutes: f64,
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
        return Err(format!("No ROI entries found in {}", pos_dir.display()));
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
    let signal = *sample.signal.first().ok_or_else(|| {
        format!(
            "sample {} has no signal channel for lnp-binding",
            sample.name
        )
    })?;

    let mut roi_entries = index.rois.clone();
    roi_entries.sort_by_key(|e| e.roi);

    let mut stacks = Vec::with_capacity(roi_entries.len());
    let mut histogram = FluorescenceHistogram::new();
    for entry in &roi_entries {
        let stack = load_roi_stack(workspace, position, entry.roi, signal)?;
        if let Some(slice) = stack.as_slice() {
            histogram.sample(slice);
        } else {
            let owned: Vec<f64> = stack.iter().copied().collect();
            histogram.sample(&owned);
        }
        stacks.push((entry.roi, stack));
    }

    let min_gradient = histogram.min_gradient();
    let mut particle_rows = Vec::new();
    let mut binding_rows = Vec::new();

    for (roi, stack) in &stacks {
        let time_count = stack.shape()[0];
        let time_indices = index.resolved_time_indices(time_count)?;
        for stack_t in 0..time_count {
            let source_t = time_indices[stack_t];
            let minutes = f64::from(source_t) * interval_minutes;
            let frame = stack.slice(s![stack_t, .., ..]).to_owned();
            let spots = detect_particles_frame(&frame, min_gradient)?;
            let stats = summarize_particles(&spots);

            for (idx, spot) in spots.into_iter().enumerate() {
                particle_rows.push(ParticleRecord {
                    pos: position,
                    roi: *roi,
                    t: source_t,
                    minutes,
                    particle: idx as u32,
                    spot,
                });
            }

            binding_rows.push(BindingRow {
                pos: position,
                roi: *roi,
                t: source_t,
                minutes,
                stats,
            });
        }
    }

    if binding_rows.is_empty() {
        return Err(format!("position {position} has no ROI frames to score"));
    }

    particle_rows.sort_by(|a, b| {
        a.roi
            .cmp(&b.roi)
            .then(a.t.cmp(&b.t))
            .then(a.particle.cmp(&b.particle))
    });
    binding_rows.sort_by(|a, b| a.roi.cmp(&b.roi).then(a.t.cmp(&b.t)));

    write_particle_csv(&particles_csv_path(workspace, position), &particle_rows)?;
    write_binding_csv(&binding_csv_path(workspace, position), &binding_rows)
}

pub fn write_binding_summary(workspace: &Path, mapping: &SampleMapping) -> Result<(), String> {
    let csvs = discover_binding_csvs(&workspace.join("analysis"))?;
    let dirnames = sample_pack_dirnames(mapping);
    let mut binding_by_sample: BTreeMap<usize, Vec<Vec<String>>> = BTreeMap::new();
    let mut summary_by_sample: BTreeMap<usize, Vec<Vec<String>>> = BTreeMap::new();
    let mut particles_by_sample: BTreeMap<usize, Vec<Vec<String>>> = BTreeMap::new();

    for path in csvs {
        let position = binding_position(&path)?;
        let sample = mapping
            .iter()
            .position(|sample| sample.positions.contains(&position))
            .ok_or_else(|| format!("No sample owns Pos{position} ({})", path.display()))?;
        let rows = read_binding_rows(&path, position)?;
        let summary_rows = summarize_binding_rows(&rows);
        let pos_dir = path
            .parent()
            .ok_or_else(|| format!("binding csv has no directory: {}", path.display()))?;
        let summary_path = pos_dir.join("binding_summary.csv");
        write_binding_summary_csv(&summary_path, &summary_rows)?;

        for row in &rows {
            binding_by_sample
                .entry(sample)
                .or_default()
                .push(binding_xlsx_row(row));
        }
        for row in &summary_rows {
            summary_by_sample
                .entry(sample)
                .or_default()
                .push(binding_summary_xlsx_row(row));
        }

        let particles_path = pos_dir.join("particles.csv");
        if particles_path.is_file() {
            let particle_rows = read_particle_rows(&particles_path, position)?;
            let entry = particles_by_sample.entry(sample).or_default();
            for p_row in &particle_rows {
                entry.push(particle_xlsx_row(p_row));
            }
        } else {
            particles_by_sample.entry(sample).or_default();
        }
    }

    publish_sample_xlsx(
        workspace,
        &dirnames,
        &mut binding_by_sample,
        "binding",
        BINDING_XLSX_HEADERS,
    )?;
    publish_sample_xlsx(
        workspace,
        &dirnames,
        &mut summary_by_sample,
        "binding_summary",
        BINDING_SUMMARY_XLSX_HEADERS,
    )?;
    if !particles_by_sample.is_empty() {
        publish_sample_xlsx_allow_empty(
            workspace,
            &dirnames,
            &mut particles_by_sample,
            "particles",
            PARTICLE_XLSX_HEADERS,
        )?;
    }
    Ok(())
}

pub fn discover_binding_csvs(dir: &Path) -> Result<Vec<PathBuf>, String> {
    if !dir.is_dir() {
        return Err(format!("Expected binding traces at {}", dir.display()));
    }
    let mut csvs = Vec::new();
    for entry in fs::read_dir(dir).map_err(|error| error.to_string())? {
        let pos_dir = entry.map_err(|error| error.to_string())?.path();
        if !pos_dir.is_dir() {
            continue;
        }
        let Some(name) = pos_dir.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !(name.starts_with("Pos") && name[3..].chars().all(|c| c.is_ascii_digit())) {
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
            "No binding traces (expected Pos{{n}}/binding.csv) in {}",
            dir.display()
        ));
    }
    Ok(csvs)
}

pub fn discover_particle_csvs(dir: &Path) -> Result<Vec<PathBuf>, String> {
    if !dir.is_dir() {
        return Err(format!("Expected particle tables at {}", dir.display()));
    }
    let mut csvs = Vec::new();
    for entry in fs::read_dir(dir).map_err(|error| error.to_string())? {
        let pos_dir = entry.map_err(|error| error.to_string())?.path();
        if !pos_dir.is_dir() {
            continue;
        }
        let Some(name) = pos_dir.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !(name.starts_with("Pos") && name[3..].chars().all(|c| c.is_ascii_digit())) {
            continue;
        }
        let path = pos_dir.join("particles.csv");
        if path.is_file() {
            csvs.push(path);
        }
    }
    csvs.sort();
    Ok(csvs)
}

pub fn binding_position(path: &Path) -> Result<u32, String> {
    path.parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix("Pos"))
        .and_then(|rest| rest.parse().ok())
        .ok_or_else(|| format!("Expected Pos{{n}}/binding.csv, got {}", path.display()))
}

fn binding_csv_path(workspace: &Path, position: u32) -> PathBuf {
    workspace
        .join("analysis")
        .join(format!("Pos{position}"))
        .join("binding.csv")
}

fn particles_csv_path(workspace: &Path, position: u32) -> PathBuf {
    workspace
        .join("analysis")
        .join(format!("Pos{position}"))
        .join("particles.csv")
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
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|error| error.to_string())?;
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

fn write_particle_csv(path: &Path, rows: &[ParticleRecord]) -> Result<(), String> {
    let body = rows
        .iter()
        .map(|row| {
            vec![
                row.roi.to_string(),
                row.t.to_string(),
                format!("{:.4}", row.minutes),
                row.particle.to_string(),
                format!("{:.4}", row.spot.x),
                format!("{:.4}", row.spot.y),
                format!("{:.4}", row.spot.intensity),
                format!("{:.4}", row.spot.radius),
                format!("{:.2}", row.spot.photons),
                format!("{:.4}", row.spot.sx),
                format!("{:.4}", row.spot.sy),
                format!("{:.4}", row.spot.fwhm),
                format!("{:.4}", row.spot.background),
                format!("{:.4}", row.spot.snb),
                format!("{:.2}", row.spot.net_gradient),
            ]
        })
        .collect::<Vec<_>>();
    write_csv_only(path, PARTICLE_CSV_HEADERS, &body)
}

fn write_binding_csv(path: &Path, rows: &[BindingRow]) -> Result<(), String> {
    let body = rows
        .iter()
        .map(|row| {
            vec![
                row.roi.to_string(),
                row.t.to_string(),
                format!("{:.4}", row.minutes),
                row.stats.particles.to_string(),
                format!("{:.4}", row.stats.intensity_mean),
                format!("{:.4}", row.stats.intensity_median),
                format!("{:.4}", row.stats.intensity_iqr),
                format!("{:.4}", row.stats.radius_mean),
                format!("{:.4}", row.stats.radius_median),
                format!("{:.4}", row.stats.radius_iqr),
            ]
        })
        .collect::<Vec<_>>();
    write_csv_only(path, BINDING_CSV_HEADERS, &body)
}

fn write_binding_summary_csv(path: &Path, rows: &[BindingSummaryRow]) -> Result<(), String> {
    let body = rows
        .iter()
        .map(|row| {
            vec![
                row.roi.to_string(),
                row.frames.to_string(),
                format!("{:.4}", row.particles_mean),
                format!("{:.4}", row.intensity_mean),
                format!("{:.4}", row.intensity_median),
                format!("{:.4}", row.intensity_iqr),
                format!("{:.4}", row.radius_mean),
                format!("{:.4}", row.radius_median),
                format!("{:.4}", row.radius_iqr),
            ]
        })
        .collect::<Vec<_>>();
    write_csv_only(path, BINDING_SUMMARY_CSV_HEADERS, &body)
}

pub fn read_binding_rows(path: &Path, position: u32) -> Result<Vec<BindingRow>, String> {
    let (headers, rows) = read_csv(path)?;
    let column = |name: &str| {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("{} is missing {name}", path.display()))
    };
    let roi = column("roi")?;
    let t = column("t")?;
    let minutes = column("minutes")?;
    let particles = column("particles")?;
    let intensity_mean = column("intensity_mean")?;
    let intensity_median = column("intensity_median")?;
    let intensity_iqr = column("intensity_iqr")?;
    let radius_mean = column("radius_mean")?;
    let radius_median = column("radius_median")?;
    let radius_iqr = column("radius_iqr")?;

    let mut parsed = Vec::with_capacity(rows.len());
    for row in &rows {
        let cell = |index: usize| -> Result<&str, String> {
            row.get(index)
                .map(String::as_str)
                .ok_or_else(|| format!("{} has a short row", path.display()))
        };
        let u32_val = |index: usize| -> Result<u32, String> {
            cell(index)?
                .parse()
                .map_err(|error| format!("{}: {error}", path.display()))
        };
        let f64_val = |index: usize| -> Result<f64, String> {
            cell(index)?
                .parse()
                .map_err(|error| format!("{}: {error}", path.display()))
        };
        parsed.push(BindingRow {
            pos: position,
            roi: u32_val(roi)?,
            t: u32_val(t)?,
            minutes: f64_val(minutes)?,
            stats: BindingFrameStats {
                particles: u32_val(particles)?,
                intensity_mean: f64_val(intensity_mean)?,
                intensity_median: f64_val(intensity_median)?,
                intensity_iqr: f64_val(intensity_iqr)?,
                radius_mean: f64_val(radius_mean)?,
                radius_median: f64_val(radius_median)?,
                radius_iqr: f64_val(radius_iqr)?,
            },
        });
    }
    Ok(parsed)
}

pub fn read_particle_rows(path: &Path, position: u32) -> Result<Vec<ParticleRecord>, String> {
    let (headers, rows) = read_csv(path)?;
    let column = |name: &str| {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("{} is missing {name}", path.display()))
    };
    let roi = column("roi")?;
    let t = column("t")?;
    let minutes = column("minutes")?;
    let particle = column("particle")?;
    let x = column("x")?;
    let y = column("y")?;
    let intensity = column("intensity")?;
    let radius = column("radius")?;
    let photons = column("photons")?;
    let sx = column("sx")?;
    let sy = column("sy")?;
    let fwhm = column("fwhm")?;
    let background = column("background")?;
    let snb = column("snb")?;
    let net_gradient = column("net_gradient")?;

    let mut parsed = Vec::with_capacity(rows.len());
    for row in &rows {
        let cell = |index: usize| -> Result<&str, String> {
            row.get(index)
                .map(String::as_str)
                .ok_or_else(|| format!("{} has a short row", path.display()))
        };
        let u32_val = |index: usize| -> Result<u32, String> {
            cell(index)?
                .parse()
                .map_err(|error| format!("{}: {error}", path.display()))
        };
        let f64_val = |index: usize| -> Result<f64, String> {
            cell(index)?
                .parse()
                .map_err(|error| format!("{}: {error}", path.display()))
        };
        parsed.push(ParticleRecord {
            pos: position,
            roi: u32_val(roi)?,
            t: u32_val(t)?,
            minutes: f64_val(minutes)?,
            particle: u32_val(particle)?,
            spot: ParticleSpot {
                x: f64_val(x)?,
                y: f64_val(y)?,
                intensity: f64_val(intensity)?,
                radius: f64_val(radius)?,
                photons: f64_val(photons)?,
                sx: f64_val(sx)?,
                sy: f64_val(sy)?,
                fwhm: f64_val(fwhm)?,
                background: f64_val(background)?,
                snb: f64_val(snb)?,
                net_gradient: f64_val(net_gradient)?,
            },
        });
    }
    Ok(parsed)
}

pub fn summarize_binding_rows(rows: &[BindingRow]) -> Vec<BindingSummaryRow> {
    let mut groups = Vec::<BindingSummaryRow>::new();
    for row in rows {
        if let Some(group) = groups
            .iter_mut()
            .find(|group| group.pos == row.pos && group.roi == row.roi)
        {
            group.frames += 1;
            group.particles_mean += f64::from(row.stats.particles);
            group.intensity_mean += row.stats.intensity_mean;
            group.intensity_median += row.stats.intensity_median;
            group.intensity_iqr += row.stats.intensity_iqr;
            group.radius_mean += row.stats.radius_mean;
            group.radius_median += row.stats.radius_median;
            group.radius_iqr += row.stats.radius_iqr;
        } else {
            groups.push(BindingSummaryRow {
                pos: row.pos,
                roi: row.roi,
                frames: 1,
                particles_mean: f64::from(row.stats.particles),
                intensity_mean: row.stats.intensity_mean,
                intensity_median: row.stats.intensity_median,
                intensity_iqr: row.stats.intensity_iqr,
                radius_mean: row.stats.radius_mean,
                radius_median: row.stats.radius_median,
                radius_iqr: row.stats.radius_iqr,
            });
        }
    }
    for group in &mut groups {
        let frames = f64::from(group.frames.max(1));
        group.particles_mean = round4(group.particles_mean / frames);
        group.intensity_mean = round4(group.intensity_mean / frames);
        group.intensity_median = round4(group.intensity_median / frames);
        group.intensity_iqr = round4(group.intensity_iqr / frames);
        group.radius_mean = round4(group.radius_mean / frames);
        group.radius_median = round4(group.radius_median / frames);
        group.radius_iqr = round4(group.radius_iqr / frames);
    }
    groups.sort_by(|left, right| left.pos.cmp(&right.pos).then(left.roi.cmp(&right.roi)));
    groups
}

fn binding_xlsx_row(row: &BindingRow) -> Vec<String> {
    vec![
        row.pos.to_string(),
        row.roi.to_string(),
        row.t.to_string(),
        format!("{:.4}", row.minutes),
        row.stats.particles.to_string(),
        format!("{:.4}", row.stats.intensity_mean),
        format!("{:.4}", row.stats.intensity_median),
        format!("{:.4}", row.stats.intensity_iqr),
        format!("{:.4}", row.stats.radius_mean),
        format!("{:.4}", row.stats.radius_median),
        format!("{:.4}", row.stats.radius_iqr),
    ]
}

fn binding_summary_xlsx_row(row: &BindingSummaryRow) -> Vec<String> {
    vec![
        row.pos.to_string(),
        row.roi.to_string(),
        row.frames.to_string(),
        format!("{:.4}", row.particles_mean),
        format!("{:.4}", row.intensity_mean),
        format!("{:.4}", row.intensity_median),
        format!("{:.4}", row.intensity_iqr),
        format!("{:.4}", row.radius_mean),
        format!("{:.4}", row.radius_median),
        format!("{:.4}", row.radius_iqr),
    ]
}

fn particle_xlsx_row(row: &ParticleRecord) -> Vec<String> {
    vec![
        row.pos.to_string(),
        row.roi.to_string(),
        row.t.to_string(),
        format!("{:.4}", row.minutes),
        row.particle.to_string(),
        format!("{:.4}", row.spot.x),
        format!("{:.4}", row.spot.y),
        format!("{:.4}", row.spot.intensity),
        format!("{:.4}", row.spot.radius),
        format!("{:.2}", row.spot.photons),
        format!("{:.4}", row.spot.sx),
        format!("{:.4}", row.spot.sy),
        format!("{:.4}", row.spot.fwhm),
        format!("{:.4}", row.spot.background),
        format!("{:.4}", row.spot.snb),
        format!("{:.2}", row.spot.net_gradient),
    ]
}

fn publish_sample_xlsx(
    workspace: &Path,
    dirnames: &BTreeMap<usize, String>,
    frames: &mut BTreeMap<usize, Vec<Vec<String>>>,
    kind: &str,
    headers: &[&str],
) -> Result<(), String> {
    if frames.is_empty() {
        return Err(format!("no binding rows to publish as {kind}.xlsx"));
    }
    publish_sample_xlsx_allow_empty(workspace, dirnames, frames, kind, headers)
}

fn publish_sample_xlsx_allow_empty(
    workspace: &Path,
    dirnames: &BTreeMap<usize, String>,
    frames: &mut BTreeMap<usize, Vec<Vec<String>>>,
    kind: &str,
    headers: &[&str],
) -> Result<(), String> {
    for (sample, rows) in frames.iter_mut() {
        let dirname = dirnames
            .get(sample)
            .ok_or_else(|| format!("no results folder for sample {sample}"))?;
        rows.sort_by(|left, right| {
            let number = |row: &[String], index: usize| {
                row.get(index)
                    .and_then(|value| value.parse::<i64>().ok())
                    .unwrap_or(0)
            };
            number(left, 0)
                .cmp(&number(right, 0))
                .then(number(left, 1).cmp(&number(right, 1)))
                .then(number(left, 2).cmp(&number(right, 2)))
                .then(number(left, 4).cmp(&number(right, 4)))
        });
        let output = workspace
            .join("results")
            .join(dirname)
            .join(format!("{kind}.xlsx"));
        write_xlsx_only(&output, headers, rows)?;
    }
    Ok(())
}
