use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use marray::np::Array3;

use crate::roi::decode_tiff_pages;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub path: PathBuf,
    pub position: u32,
    pub channel: u32,
    pub time: u32,
    pub z: u32,
}

fn parse_frame_filename(name: &str) -> Option<(u32, u32, u32, u32)> {
    let lower = name.to_ascii_lowercase();
    let stem = lower
        .strip_suffix(".tiff")
        .or_else(|| lower.strip_suffix(".tif"))?;
    let rest = stem.strip_prefix("img_channel")?;
    let (ch_str, rest) = rest.split_once("_position")?;
    let (pos_str, rest) = rest.split_once("_time")?;
    let (time_str, z_str) = rest.split_once("_z")?;
    Some((
        pos_str.parse().ok()?,
        ch_str.parse().ok()?,
        time_str.parse().ok()?,
        z_str.parse().ok()?,
    ))
}

fn parse_pos_dir(path: &Path) -> Option<u32> {
    let name = path.file_name()?.to_str()?;
    let digits = name.strip_prefix("Pos")?;
    digits.parse().ok()
}

pub fn find_frames(root: &Path) -> Vec<Frame> {
    let mut frames = Vec::new();
    let mut dirs_to_visit = vec![root.to_path_buf()];
    while let Some(dir) = dirs_to_visit.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut sorted_entries: Vec<_> = entries.flatten().map(|e| e.path()).collect();
        sorted_entries.sort();
        for path in sorted_entries {
            if path.is_dir() {
                dirs_to_visit.push(path);
            } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if let Some((pos_from_name, channel, time, z)) = parse_frame_filename(name) {
                    let position = path
                        .parent()
                        .and_then(parse_pos_dir)
                        .unwrap_or(pos_from_name);
                    frames.push(Frame {
                        path,
                        position,
                        channel,
                        time,
                        z,
                    });
                }
            }
        }
    }
    frames.sort_by_key(|f| (f.position, f.channel, f.time, f.z));
    frames
}

pub fn available_times(root: &Path, position: u32, channel: u32) -> Vec<u32> {
    let set: BTreeSet<u32> = find_frames(root)
        .into_iter()
        .filter(|f| f.position == position && f.channel == channel)
        .map(|f| f.time)
        .collect();
    set.into_iter().collect()
}

pub fn load_stack(root: &Path, position: u32, channel: u32, time: u32) -> Result<Array3<f64>, String> {
    let mut matching: Vec<Frame> = find_frames(root)
        .into_iter()
        .filter(|f| f.position == position && f.channel == channel && f.time == time)
        .collect();
    if matching.is_empty() {
        return Err(format!(
            "No frames found for position={position}, channel={channel}, time={time} under {}",
            root.display()
        ));
    }
    matching.sort_by_key(|f| f.z);

    let mut data = Vec::new();
    let mut height = 0usize;
    let mut width = 0usize;
    let z_count = matching.len();

    for (idx, frame) in matching.iter().enumerate() {
        let (pages, h, w) = decode_tiff_pages(&frame.path)?;
        if pages.is_empty() {
            return Err(format!("Empty TIFF file: {}", frame.path.display()));
        }
        if idx == 0 {
            height = h;
            width = w;
        } else if h != height || w != width {
            return Err(format!(
                "Inconsistent frame dimensions in {}: {}x{} vs {}x{}",
                frame.path.display(),
                h,
                w,
                height,
                width
            ));
        }
        data.extend_from_slice(&pages[0]);
    }

    Array3::from_shape_vec((z_count, height, width), data)
        .map_err(|e| format!("Failed to build 3D stack Array3: {e}"))
}
