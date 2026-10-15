use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use marray::np::{s, Array2, Array3};
use picasso::binding::{detect_spots_frame, BindingDetectConfig, BindingSpot};
use picasso::{CameraInfo, GaussModel, SpotFitMethod};

use crate::frames::load_stack;
use crate::paths::{picasso_roi_output_path, picasso_stack_output_path};
use crate::roi::{load_roi_stack, resolve_roi_times};

pub const PICASSO_SPOT_COLUMNS: [&str; 15] = [
    "y",
    "x",
    "intensity",
    "probability",
    "fwhm",
    "intens_A",
    "intens_B",
    "snb",
    "photons",
    "sx",
    "sy",
    "lpx",
    "lpy",
    "ellipticity",
    "net_gradient",
];

#[derive(Debug, Clone)]
pub struct PicassoParams {
    pub min_gradient: f64,
    pub box_size: usize,
    pub fit_method: String,
    pub model: String,
    pub baseline: f64,
    pub sensitivity: f64,
    pub gain: f64,
    pub gaussian_sigma: Option<f64>,
}

impl Default for PicassoParams {
    fn default() -> Self {
        Self {
            min_gradient: 500.0,
            box_size: 7,
            fit_method: "lq".to_string(),
            model: "elliptic".to_string(),
            baseline: 0.0,
            sensitivity: 1.0,
            gain: 1.0,
            gaussian_sigma: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PicassoRoiResult {
    pub output_dir: PathBuf,
    pub rois: Vec<u32>,
    pub time_count: usize,
    pub total_spots: usize,
    pub fit_method: String,
    pub model: String,
}

#[derive(Debug, Clone)]
pub struct PicassoStackResult {
    pub output_path: PathBuf,
    pub position: u32,
    pub channel: u32,
    pub time: u32,
    pub shape: Vec<usize>,
    pub fit_method: String,
    pub model: String,
}

pub fn parse_fit_method(method: &str) -> Result<SpotFitMethod, String> {
    match method {
        "lq" => Ok(SpotFitMethod::GaussLq),
        "mle" | "mle-integrated" => Ok(SpotFitMethod::GaussMle),
        other => Err(format!(
            "Unsupported fit_method '{other}', expected 'lq', 'mle', or 'mle-integrated'"
        )),
    }
}

pub fn parse_gauss_model(model: &str) -> Result<GaussModel, String> {
    match model {
        "spherical" => Ok(GaussModel::Spherical),
        "elliptic" => Ok(GaussModel::Elliptic),
        "rotated" => Ok(GaussModel::Rotated),
        other => Err(format!(
            "Unsupported model '{other}', expected 'spherical', 'elliptic', or 'rotated'"
        )),
    }
}

pub fn detect_spots_picasso(
    image: &Array2<f64>,
    params: &PicassoParams,
    pos_label: &str,
    roi_label: &str,
    time_index: u32,
) -> Result<Vec<BindingSpot>, String> {
    if params.box_size < 3 || params.box_size % 2 == 0 {
        return Err(format!(
            "box_size must be an odd integer >= 3, got {}",
            params.box_size
        ));
    }
    let fit_method = parse_fit_method(&params.fit_method)?;
    let gauss_model = parse_gauss_model(&params.model)?;
    let cfg = BindingDetectConfig {
        min_gradient: params.min_gradient,
        box_size: params.box_size,
        fit_method,
        gauss_model,
        gaussian_sigma: params.gaussian_sigma.filter(|&s| s > 0.0).map(|s| (s, s)),
        camera_info: CameraInfo::new(params.baseline, params.sensitivity, params.gain),
        pos: pos_label.to_string(),
        roi: roi_label.to_string(),
        parallel: true,
    };

    if params.fit_method == "mle-integrated" {
        let (h, w) = image.dim();
        let mut stack = Array3::<f64>::zeros((1, h, w));
        stack.slice_mut(s![0, .., ..]).assign(image);
        let filtered = if let Some(sig) = cfg.gaussian_sigma {
            picasso::localize::gaussian_filter_frame(image, sig).map_err(|e| e.to_string())?
        } else {
            image.clone()
        };
        let ids = picasso::localize::identify_in_frame(
            &filtered,
            0,
            cfg.min_gradient,
            cfg.box_size,
            None,
        );
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let spots = picasso::localize::get_spots(&stack, &ids, cfg.box_size, &cfg.camera_info)
            .map_err(|e| e.to_string())?;
        let mle_method = if gauss_model == GaussModel::Spherical {
            picasso::fitting::gaussmle::MleMethod::Sigma
        } else {
            picasso::fitting::gaussmle::MleMethod::SigmaYx
        };
        let (thetas, crlbs, ll, it) =
            picasso::fitting::gaussmle::gaussmle(&spots, 1e-3, 1000, mle_method, true)
                .map_err(|e| e.to_string())?;
        let mut locs = picasso::fitting::gaussmle::locs_from_fits(
            &ids,
            &thetas,
            &crlbs,
            &ll,
            &it,
            cfg.box_size,
        );
        for l in &mut locs {
            l.frame = time_index;
            let sx = l.sx.max(1e-6);
            let sy = l.sy.max(1e-6);
            l.peak = Some(l.photons.max(0.0) / (2.0 * std::f64::consts::PI * sx * sy));
        }
        Ok(locs
            .iter()
            .map(|l| BindingSpot::from_localization(l, &cfg.pos, &cfg.roi, cfg.min_gradient))
            .collect())
    } else {
        detect_spots_frame(image, time_index, &cfg).map_err(|e| e.to_string())
    }
}

fn round_dec(v: f64, decimals: i32) -> f64 {
    let factor = 10f64.powi(decimals);
    let r = (v * factor).round() / factor;
    if r == 0.0 { 0.0 } else { r }
}

pub fn write_picasso_spot_csv(output_path: &Path, spots: &[BindingSpot]) -> Result<(), String> {
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory {}: {e}", parent.display()))?;
    }
    let file = File::create(output_path)
        .map_err(|e| format!("Failed to create {}: {e}", output_path.display()))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", PICASSO_SPOT_COLUMNS.join(","))
        .map_err(|e| format!("Failed to write header to {}: {e}", output_path.display()))?;

    for s in spots {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            round_dec(s.y, 4),
            round_dec(s.x, 4),
            round_dec(s.intensity, 2),
            round_dec(s.probability, 4),
            round_dec(s.fwhm, 4),
            round_dec(s.intens_a, 3),
            round_dec(s.intens_b, 3),
            round_dec(s.snb, 3),
            round_dec(s.photons, 2),
            round_dec(s.sx, 4),
            round_dec(s.sy, 4),
            round_dec(s.lpx, 4),
            round_dec(s.lpy, 4),
            round_dec(s.ellipticity, 4),
            round_dec(s.net_gradient, 2),
        )
        .map_err(|e| format!("Failed to write row to {}: {e}", output_path.display()))?;
    }
    writer
        .flush()
        .map_err(|e| format!("Failed to flush {}: {e}", output_path.display()))
}

pub fn run_picasso_roi(
    input_dir: &Path,
    position: u32,
    channel: u32,
    time: u32,
    output: &Path,
    params: &PicassoParams,
    roi: Option<u32>,
    all_rois: bool,
    all_times: bool,
) -> Result<PicassoRoiResult, String> {
    let (rois, times) = resolve_roi_times(
        input_dir, position, channel, roi, all_rois, all_times, time,
    )?;

    let mut total_spots = 0usize;
    let pos_label = format!("Pos{position}");

    for &roi_index in &rois {
        let stack = load_roi_stack(input_dir, position, roi_index, channel)?;
        let n_frames = stack.shape()[0] as u32;
        let roi_label = format!("roi{roi_index}");
        for &time_index in &times {
            if time_index >= n_frames {
                return Err(format!(
                    "time={time_index} is out of range for ROI {roi_index} with {n_frames} frames"
                ));
            }
            let frame = stack.slice(s![time_index as usize, .., ..]).to_owned();
            let spots = detect_spots_picasso(&frame, params, &pos_label, &roi_label, time_index)?;
            let output_path = picasso_roi_output_path(output, roi_index, time_index);
            write_picasso_spot_csv(&output_path, &spots)?;
            total_spots += spots.len();
        }
    }

    Ok(PicassoRoiResult {
        output_dir: output.to_path_buf(),
        rois,
        time_count: times.len(),
        total_spots,
        fit_method: params.fit_method.clone(),
        model: params.model.clone(),
    })
}

pub fn run_picasso_stack(
    input_dir: &Path,
    position: u32,
    channel: u32,
    time: u32,
    output: &Path,
    params: &PicassoParams,
) -> Result<PicassoStackResult, String> {
    let output_path = picasso_stack_output_path(output, position, channel, time);
    let stack = load_stack(input_dir, position, channel, time)?;
    let shape = stack.shape().to_vec();
    if shape[0] != 1 {
        return Err(format!(
            "Multi-z stack shape {:?} is not supported on main. Use --roi-stacks for 2D per-cell LNP detection, or a single plane.",
            shape
        ));
    }
    let image = stack.slice(s![0, .., ..]).to_owned();
    let pos_label = format!("Pos{position}");
    let spots = detect_spots_picasso(&image, params, &pos_label, "stack", time)?;
    write_picasso_spot_csv(&output_path, &spots)?;

    Ok(PicassoStackResult {
        output_path,
        position,
        channel,
        time,
        shape,
        fit_method: params.fit_method.clone(),
        model: params.model.clone(),
    })
}
