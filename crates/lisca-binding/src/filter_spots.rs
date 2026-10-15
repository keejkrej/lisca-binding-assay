use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use csv::{ReaderBuilder, WriterBuilder};

use crate::paths::filtered_spots_output_path;

#[derive(Debug, Clone)]
pub struct FilterSpotsResult {
    pub output_dir: PathBuf,
    pub file_count: usize,
    pub total_in: usize,
    pub total_out: usize,
}

#[derive(Debug, Clone)]
pub struct FilterCriteria {
    pub min_intensity: Option<f64>,
    pub max_intensity: Option<f64>,
    pub min_fwhm: Option<f64>,
    pub max_fwhm: Option<f64>,
    pub min_probability: Option<f64>,
}

impl Default for FilterCriteria {
    fn default() -> Self {
        Self {
            min_intensity: Some(2500.0),
            max_intensity: None,
            min_fwhm: Some(2.0),
            max_fwhm: Some(6.0),
            min_probability: Some(0.4),
        }
    }
}

pub fn filter_spot_frame(
    rows: &[BTreeMap<String, String>],
    criteria: &FilterCriteria,
) -> Result<Vec<BTreeMap<String, String>>, String> {
    let mut filtered = Vec::new();
    for row in rows {
        let intens: f64 = row
            .get("intensity")
            .ok_or_else(|| "Missing intensity column".to_string())?
            .parse()
            .map_err(|e| format!("Invalid intensity value: {e}"))?;
        if let Some(min_i) = criteria.min_intensity {
            if intens < min_i {
                continue;
            }
        }
        if let Some(max_i) = criteria.max_intensity {
            if intens > max_i {
                continue;
            }
        }
        if let Some(fwhm_str) = row.get("fwhm") {
            if !fwhm_str.is_empty() {
                let fwhm: f64 = fwhm_str
                    .parse()
                    .map_err(|e| format!("Invalid fwhm value: {e}"))?;
                if let Some(min_f) = criteria.min_fwhm {
                    if fwhm < min_f {
                        continue;
                    }
                }
                if let Some(max_f) = criteria.max_fwhm {
                    if fwhm > max_f {
                        continue;
                    }
                }
            }
        }
        let prob: f64 = row
            .get("probability")
            .ok_or_else(|| "Missing probability column".to_string())?
            .parse()
            .map_err(|e| format!("Invalid probability value: {e}"))?;
        if let Some(min_p) = criteria.min_probability {
            if prob < min_p {
                continue;
            }
        }
        filtered.push(row.clone());
    }
    Ok(filtered)
}

pub fn read_spot_csv(path: &Path) -> Result<(Vec<String>, Vec<BTreeMap<String, String>>), String> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("Failed to open {}: {e}", path.display()))?;
    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| format!("Spot CSV {} has no header: {e}", path.display()))?
        .iter()
        .map(str::to_string)
        .collect();
    if headers.is_empty() {
        return Err(format!("Spot CSV {} has no header", path.display()));
    }

    let mut rows = Vec::new();
    for record in reader.records() {
        let rec = record.map_err(|e| format!("Row error in {}: {e}", path.display()))?;
        let mut map = BTreeMap::new();
        for (k, v) in headers.iter().zip(rec.iter()) {
            map.insert(k.clone(), v.to_string());
        }
        rows.push(map);
    }
    Ok((headers, rows))
}

pub fn write_spot_csv(
    path: &Path,
    fieldnames: &[String],
    rows: &[BTreeMap<String, String>],
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create {}: {e}", parent.display()))?;
    }
    let mut writer = WriterBuilder::new()
        .from_path(path)
        .map_err(|e| format!("Failed to create {}: {e}", path.display()))?;
    writer
        .write_record(fieldnames)
        .map_err(|e| format!("Failed to write header to {}: {e}", path.display()))?;
    for row in rows {
        let rec: Vec<&str> = fieldnames
            .iter()
            .map(|f| row.get(f).map(String::as_str).unwrap_or(""))
            .collect();
        writer
            .write_record(&rec)
            .map_err(|e| format!("Failed to write record to {}: {e}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|e| format!("Failed to flush {}: {e}", path.display()))
}

pub fn run_filter_spots(
    input_dir: &Path,
    output: &Path,
    criteria: &FilterCriteria,
) -> Result<FilterSpotsResult, String> {
    let entries = std::fs::read_dir(input_dir)
        .map_err(|e| format!("Failed to read {}: {e}", input_dir.display()))?;
    let mut csv_files: Vec<PathBuf> = entries
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
    csv_files.sort();

    if csv_files.is_empty() {
        return Err(format!("No CSV files found in {}", input_dir.display()));
    }

    let mut total_in = 0usize;
    let mut total_out = 0usize;

    for csv_path in &csv_files {
        let (fieldnames, rows) = read_spot_csv(csv_path)?;
        for req in ["y", "x", "intensity", "probability"] {
            if !fieldnames.iter().any(|h| h == req) {
                return Err(format!(
                    "{} must contain y, x, intensity, and probability columns",
                    csv_path.display()
                ));
            }
        }
        let filtered_rows = filter_spot_frame(&rows, criteria)?;
        let out_name = filtered_spots_output_path(csv_path);
        let file_name = out_name
            .file_name()
            .ok_or_else(|| "Invalid output filename".to_string())?;
        let output_path = output.join(file_name);
        write_spot_csv(&output_path, &fieldnames, &filtered_rows)?;
        total_in += rows.len();
        total_out += filtered_rows.len();
    }

    Ok(FilterSpotsResult {
        output_dir: output.to_path_buf(),
        file_count: csv_files.len(),
        total_in,
        total_out,
    })
}
