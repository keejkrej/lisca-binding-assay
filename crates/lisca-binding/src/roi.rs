use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use csv::ReaderBuilder;
use marray::np::Array3;
use serde::{Deserialize, Serialize};
use tiff::decoder::{Decoder, DecodingResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoiBbox {
    pub w: usize,
    pub h: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoiEntry {
    pub roi: u32,
    #[serde(rename = "fileName")]
    pub file_name: String,
    #[serde(default)]
    pub shape: Option<[usize; 5]>,
    #[serde(default)]
    pub bbox: Option<RoiBbox>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoiIndex {
    #[serde(default)]
    pub position: Option<u32>,
    #[serde(rename = "axisOrder", default)]
    pub axis_order: Option<String>,
    #[serde(rename = "timeCount", default)]
    pub time_count: Option<usize>,
    #[serde(rename = "channelCount", default)]
    pub channel_count: Option<usize>,
    #[serde(rename = "zCount", default)]
    pub z_count: Option<usize>,
    #[serde(rename = "timeIndices", default)]
    pub time_indices: Option<Vec<u32>>,
    pub rois: Vec<RoiEntry>,
}

impl RoiIndex {
    pub fn resolved_shape(&self, entry: &RoiEntry) -> Result<[usize; 5], String> {
        if let Some(shape) = entry.shape {
            return Ok(shape);
        }
        let t = self
            .time_count
            .ok_or_else(|| "index.json missing timeCount and entry.shape".to_string())?;
        let c = self
            .channel_count
            .ok_or_else(|| "index.json missing channelCount and entry.shape".to_string())?;
        let z = self.z_count.unwrap_or(1);
        let bbox = entry
            .bbox
            .as_ref()
            .ok_or_else(|| format!("ROI {} missing bbox and shape in index.json", entry.roi))?;
        Ok([t, c, z, bbox.h, bbox.w])
    }

    pub fn resolved_time_indices(&self, time_count: usize) -> Result<Vec<u32>, String> {
        match &self.time_indices {
            None => Ok((0..time_count as u32).collect()),
            Some(indices) if indices.len() == time_count => Ok(indices.clone()),
            Some(indices) => Err(format!(
                "index.json timeIndices length {} does not match timeCount {time_count}",
                indices.len()
            )),
        }
    }
}

pub fn roi_dir(input_dir: &Path, position: u32) -> PathBuf {
    input_dir.join("roi").join(format!("Pos{position}"))
}

pub fn load_roi_index(input_dir: &Path, position: u32) -> Result<RoiIndex, String> {
    let index_path = roi_dir(input_dir, position).join("index.json");
    if !index_path.is_file() {
        return Err(format!(
            "No ROI directory found for position {position}: {}",
            index_path.display()
        ));
    }
    let contents = std::fs::read_to_string(&index_path)
        .map_err(|e| format!("Failed to read {}: {e}", index_path.display()))?;
    let index: RoiIndex = serde_json::from_str(&contents)
        .map_err(|e| format!("Invalid ROI index JSON {}: {e}", index_path.display()))?;
    if let Some(axis) = &index.axis_order {
        if !axis.eq_ignore_ascii_case("TCZYX") {
            return Err(format!(
                "{}: unsupported axisOrder '{axis}' (expected TCZYX)",
                index_path.display()
            ));
        }
    }
    Ok(index)
}

pub fn list_rois(input_dir: &Path, position: u32) -> Result<Vec<u32>, String> {
    let index = load_roi_index(input_dir, position)?;
    let mut rois: Vec<u32> = index.rois.into_iter().map(|e| e.roi).collect();
    rois.sort_unstable();
    Ok(rois)
}

pub fn roi_stack_path(input_dir: &Path, position: u32, roi: u32) -> Result<PathBuf, String> {
    let index = load_roi_index(input_dir, position)?;
    for entry in index.rois {
        if entry.roi == roi {
            return Ok(roi_dir(input_dir, position).join(entry.file_name));
        }
    }
    Err(format!("ROI {roi} not found for position={position}"))
}

pub(crate) fn decode_tiff_pages(path: &Path) -> Result<(Vec<Vec<f64>>, usize, usize), String> {
    let file = File::open(path).map_err(|e| format!("Failed to open {}: {e}", path.display()))?;
    let mut decoder =
        Decoder::new(file).map_err(|e| format!("Failed to decode TIFF {}: {e}", path.display()))?;
    let mut pages = Vec::new();
    let mut height;
    let mut width;

    loop {
        let (w, h) = decoder
            .dimensions()
            .map_err(|e| format!("TIFF dimensions error in {}: {e}", path.display()))?;
        width = w as usize;
        height = h as usize;
        let img = decoder
            .read_image()
            .map_err(|e| format!("TIFF read_image error in {}: {e}", path.display()))?;
        let page: Vec<f64> = match img {
            DecodingResult::U8(buf) => buf.into_iter().map(f64::from).collect(),
            DecodingResult::U16(buf) => buf.into_iter().map(f64::from).collect(),
            DecodingResult::U32(buf) => buf.into_iter().map(f64::from).collect(),
            DecodingResult::U64(buf) => buf.into_iter().map(|v| v as f64).collect(),
            DecodingResult::I8(buf) => buf.into_iter().map(f64::from).collect(),
            DecodingResult::I16(buf) => buf.into_iter().map(f64::from).collect(),
            DecodingResult::I32(buf) => buf.into_iter().map(f64::from).collect(),
            DecodingResult::I64(buf) => buf.into_iter().map(|v| v as f64).collect(),
            DecodingResult::F32(buf) => buf.into_iter().map(f64::from).collect(),
            DecodingResult::F64(buf) => buf,
            _ => return Err(format!("Unsupported TIFF sample format in {}", path.display())),
        };
        if page.len() != height * width {
            return Err(format!(
                "TIFF page in {} has {} samples, expected {}x{}",
                path.display(),
                page.len(),
                height,
                width
            ));
        }
        pages.push(page);
        if !decoder.more_images() {
            break;
        }
        decoder
            .next_image()
            .map_err(|e| format!("TIFF next_image error in {}: {e}", path.display()))?;
    }

    Ok((pages, height, width))
}

pub fn load_roi_stack(
    input_dir: &Path,
    position: u32,
    roi: u32,
    channel: u32,
) -> Result<Array3<f64>, String> {
    let index = load_roi_index(input_dir, position)?;
    let entry = index
        .rois
        .iter()
        .find(|e| e.roi == roi)
        .ok_or_else(|| format!("ROI {roi} not found for position={position}"))?;

    let [t_count, c_count, z_count, h_expected, w_expected] = index.resolved_shape(entry)?;
    let ch = channel as usize;
    if ch >= c_count {
        return Err(format!(
            "Channel {channel} is out of range for ROI {roi} with {c_count} channels"
        ));
    }

    let path = roi_dir(input_dir, position).join(&entry.file_name);
    let (pages, h, w) = decode_tiff_pages(&path)?;
    let expected_pages = t_count * c_count * z_count;
    if pages.len() != expected_pages || h != h_expected || w != w_expected {
        return Err(format!(
            "ROI {roi} has unsupported layout: {} pages of {}x{}, expected {:?}",
            pages.len(),
            h,
            w,
            [t_count, c_count, z_count, h_expected, w_expected]
        ));
    }

    let mut data = Vec::with_capacity(t_count * h * w);
    for t in 0..t_count {
        let page_idx = (t * c_count + ch) * z_count;
        data.extend_from_slice(&pages[page_idx]);
    }

    Array3::from_shape_vec((t_count, h, w), data)
        .map_err(|e| format!("Failed to construct Array3 for ROI {roi}: {e}"))
}

pub fn resolve_roi_times(
    input_dir: &Path,
    position: u32,
    channel: u32,
    roi: Option<u32>,
    all_rois: bool,
    all_times: bool,
    time: u32,
) -> Result<(Vec<u32>, Vec<u32>), String> {
    if roi.is_some() && all_rois {
        return Err("Use either --roi or --all-rois, not both".to_string());
    }
    let rois = if all_rois {
        list_rois(input_dir, position)?
    } else if let Some(r) = roi {
        vec![r]
    } else {
        return Err("Provide --roi or --all-rois when using ROI stacks".to_string());
    };
    if rois.is_empty() {
        return Err(format!("No ROIs found for position={position}"));
    }

    let first_stack = load_roi_stack(input_dir, position, rois[0], channel)?;
    let n_times = first_stack.shape()[0] as u32;
    let times = if all_times {
        (0..n_times).collect()
    } else {
        if time >= n_times {
            return Err(format!(
                "time={time} is out of range for ROI {} with {n_times} frames",
                rois[0]
            ));
        }
        vec![time]
    };

    Ok((rois, times))
}

pub fn load_time_map(path: &Path) -> Result<BTreeMap<u32, f64>, String> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("Failed to open time map {}: {e}", path.display()))?;
    let headers = reader
        .headers()
        .map_err(|e| format!("Failed to read headers from {}: {e}", path.display()))?
        .clone();
    let t_idx = headers
        .iter()
        .position(|h| h == "t")
        .ok_or_else(|| "Time map CSV must contain a t column".to_string())?;
    let real_idx = headers
        .iter()
        .position(|h| h == "t_real")
        .unwrap_or(t_idx);

    let mut map = BTreeMap::new();
    for record in reader.records() {
        let rec = record.map_err(|e| format!("CSV row error in {}: {e}", path.display()))?;
        let t: u32 = rec
            .get(t_idx)
            .ok_or_else(|| "Missing t value".to_string())?
            .trim()
            .parse()
            .map_err(|e| format!("Invalid t value: {e}"))?;
        let t_real: f64 = rec
            .get(real_idx)
            .ok_or_else(|| "Missing t_real value".to_string())?
            .trim()
            .parse()
            .map_err(|e| format!("Invalid t_real value: {e}"))?;
        map.insert(t, t_real);
    }
    if map.is_empty() {
        return Err(format!("Time map CSV {} has no rows", path.display()));
    }
    Ok(map)
}

pub fn build_time_map(time_count: u32, interval_sec: f64) -> BTreeMap<u32, f64> {
    (0..time_count)
        .map(|t| (t, f64::from(t) * interval_sec))
        .collect()
}
