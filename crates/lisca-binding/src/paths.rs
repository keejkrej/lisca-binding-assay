use std::path::{Path, PathBuf};

pub fn spotiflow_stack_output_path(output: &Path, position: u32, channel: u32, time: u32) -> PathBuf {
    output.join(format!(
        "spotiflow_position{position:03}_channel{channel:03}_time{time:09}.csv"
    ))
}

pub fn spotiflow_roi_output_path(output: &Path, roi: u32, time: u32) -> PathBuf {
    output.join(format!("roi{roi:02}_time{time:09}.csv"))
}

pub fn picasso_stack_output_path(output: &Path, position: u32, channel: u32, time: u32) -> PathBuf {
    output.join(format!(
        "picasso_position{position:03}_channel{channel:03}_time{time:09}.csv"
    ))
}

pub fn picasso_roi_output_path(output: &Path, roi: u32, time: u32) -> PathBuf {
    output.join(format!("roi{roi:02}_time{time:09}.csv"))
}

pub fn filtered_spots_output_path(input_csv: &Path) -> PathBuf {
    let stem = input_csv
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("spots");
    input_csv.with_file_name(format!("{stem}_filtered.csv"))
}

pub fn spot_counts_output_path(output: &Path, position: u32, channel: u32) -> PathBuf {
    output.join(format!(
        "spot_counts_position{position:03}_channel{channel:03}.csv"
    ))
}

pub fn traces_output_path(output: &Path, position: u32, channel: u32) -> PathBuf {
    let is_csv = output
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.eq_ignore_ascii_case("csv"))
        .unwrap_or(false);
    if is_csv {
        output.to_path_buf()
    } else {
        output.join(format!(
            "traces_position{position:03}_channel{channel:03}.csv"
        ))
    }
}

pub fn traces_plot_output_path(input_file: &Path, output: Option<&Path>) -> PathBuf {
    if let Some(out) = output {
        return out.to_path_buf();
    }
    let stem = input_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("traces");
    input_file.with_file_name(format!("{stem}_traces.png"))
}
