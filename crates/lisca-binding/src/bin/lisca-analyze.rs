use std::path::PathBuf;
use std::process::ExitCode;

use lisca_binding::{
    run_filter_spots, run_picasso_roi, run_picasso_stack, run_plot_lnp, run_show_traces,
    run_spot_counts, run_traces, FilterCriteria, PicassoParams, Projection,
};

const USAGE: &str = "\
lisca-analyze (lisca-binding) — Single-particle LNP binding assay CLI

USAGE:
  lisca-analyze picasso <INPUT_DIR> [options]
  lisca-analyze filter-spots <INPUT_DIR> [options]
  lisca-analyze spot-counts <INPUT_DIR> [options]
  lisca-analyze traces <INPUT_DIR> [options]
  lisca-analyze show-traces <INPUT_CSV> [options]
  lisca-analyze plot-lnp <INPUT_DIR> <COUNTS_CSV> --filtered-dir <DIR> [options]
  lisca-analyze pipeline <INPUT_DIR> [options]
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(subcommand) = args.first().map(String::as_str) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    if matches!(subcommand, "-h" | "--help" | "help") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let rest = &args[1..];
    let result = match subcommand {
        "picasso" => cmd_picasso(rest),
        "filter-spots" => cmd_filter_spots(rest),
        "spot-counts" => cmd_spot_counts(rest),
        "traces" => cmd_traces(rest),
        "show-traces" => cmd_show_traces(rest),
        "plot-lnp" => cmd_plot_lnp(rest),
        "pipeline" => cmd_pipeline(rest),
        other => Err(format!("unknown subcommand: {other}\n\n{USAGE}")),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_picasso(args: &[String]) -> Result<(), String> {
    let input_dir = first_positional(args)
        .map(PathBuf::from)
        .ok_or_else(|| "missing INPUT_DIR argument".to_string())?;
    let position = flag_u32_any(args, &["--position", "-p"])?.unwrap_or(0);
    let channel = flag_u32_any(args, &["--channel", "-c"])?.unwrap_or(0);
    let time = flag_u32_any(args, &["--time", "-t"])?.unwrap_or(0);
    let output = flag_path_any(args, &["--output", "-o"]).unwrap_or_else(|| PathBuf::from("picasso"));
    let min_gradient = flag_f64_any(args, &["--min-gradient", "-g"])?.unwrap_or(500.0);
    let box_size = flag_usize_any(args, &["--box-size", "-b"])?.unwrap_or(7);
    let fit_method = flag_value_any(args, &["--fit-method"])
        .unwrap_or("lq")
        .to_string();
    let model = flag_value_any(args, &["--model", "-m"])
        .unwrap_or("elliptic")
        .to_string();
    let baseline = flag_f64_any(args, &["--baseline"])?.unwrap_or(0.0);
    let sensitivity = flag_f64_any(args, &["--sensitivity"])?.unwrap_or(1.0);
    let gain = flag_f64_any(args, &["--gain"])?.unwrap_or(1.0);
    let gaussian_sigma = flag_f64_any(args, &["--gaussian-sigma"])?;
    let roi = flag_u32_any(args, &["--roi"])?;
    let all_rois = has_flag(args, "--all-rois");
    let all_times = has_flag(args, "--all-times");
    let use_roi_stacks = has_flag(args, "--roi-stacks") && !has_flag(args, "--stack");

    let params = PicassoParams {
        min_gradient,
        box_size,
        fit_method,
        model,
        baseline,
        sensitivity,
        gain,
        gaussian_sigma,
    };

    let roi_mode = use_roi_stacks || all_rois || roi.is_some();
    if roi_mode {
        let res = run_picasso_roi(
            &input_dir,
            position,
            channel,
            time,
            &output,
            &params,
            roi,
            all_rois,
            all_times,
        )?;
        println!(
            "Saved Picasso output to {} for rois={:?}, times={}, total_spots={}, fit_method={}, model={}",
            res.output_dir.display(),
            res.rois,
            res.time_count,
            res.total_spots,
            res.fit_method,
            res.model
        );
    } else {
        let res = run_picasso_stack(&input_dir, position, channel, time, &output, &params)?;
        println!(
            "Saved picasso output to {} (shape={:?}, fit_method={}, model={})",
            res.output_path.display(),
            res.shape,
            res.fit_method,
            res.model
        );
    }
    Ok(())
}

fn cmd_filter_spots(args: &[String]) -> Result<(), String> {
    let input_dir = first_positional(args)
        .map(PathBuf::from)
        .ok_or_else(|| "missing INPUT_DIR argument".to_string())?;
    let output = flag_path_any(args, &["--output", "-o"]).unwrap_or_else(|| PathBuf::from("filtered"));
    let min_intensity = Some(flag_f64_any(args, &["--min-intensity"])?.unwrap_or(2500.0));
    let max_intensity = flag_f64_any(args, &["--max-intensity"])?;
    let min_fwhm = Some(flag_f64_any(args, &["--min-fwhm"])?.unwrap_or(2.0));
    let max_fwhm = Some(flag_f64_any(args, &["--max-fwhm"])?.unwrap_or(6.0));
    let min_probability = Some(flag_f64_any(args, &["--min-probability"])?.unwrap_or(0.4));

    let criteria = FilterCriteria {
        min_intensity,
        max_intensity,
        min_fwhm,
        max_fwhm,
        min_probability,
    };
    let res = run_filter_spots(&input_dir, &output, &criteria)?;
    println!(
        "Filtered {} files into {}: spots {} -> {}",
        res.file_count,
        res.output_dir.display(),
        res.total_in,
        res.total_out
    );
    Ok(())
}

fn cmd_spot_counts(args: &[String]) -> Result<(), String> {
    let input_dir = first_positional(args)
        .map(PathBuf::from)
        .ok_or_else(|| "missing INPUT_DIR argument".to_string())?;
    let output = flag_path_any(args, &["--output", "-o"]).unwrap_or_else(|| PathBuf::from("."));
    let position = flag_u32_any(args, &["--position", "-p"])?.unwrap_or(0);
    let channel = flag_u32_any(args, &["--channel", "-c"])?.unwrap_or(1);
    let time_interval = flag_f64_any(args, &["--time-interval"])?.unwrap_or(40.0);
    let time_map = flag_path_any(args, &["--time-map"]);
    let cumulative = !has_flag(args, "--per-frame");
    let match_distance = flag_f64_any(args, &["--match-distance"])?.unwrap_or(5.0);

    let res = run_spot_counts(
        &input_dir,
        &output,
        position,
        channel,
        time_interval,
        time_map.as_deref(),
        cumulative,
        match_distance,
    )?;
    println!(
        "Saved {} with rows={}, rois={}, times={}, cumulative={}",
        res.output_path.display(),
        res.row_count,
        res.roi_count,
        res.time_count,
        res.cumulative
    );
    Ok(())
}

fn cmd_traces(args: &[String]) -> Result<(), String> {
    let input_dir = first_positional(args)
        .map(PathBuf::from)
        .ok_or_else(|| "missing INPUT_DIR argument".to_string())?;
    let position = flag_u32_any(args, &["--position", "-p"])?.unwrap_or(0);
    let channel = flag_u32_any(args, &["--channel", "-c"])?.unwrap_or(0);
    let sizes = flag_value_any(args, &["--sizes"]).unwrap_or("8,16,32,64,128");
    let center_y = flag_usize_any(args, &["--center-y"])?;
    let center_x = flag_usize_any(args, &["--center-x"])?;
    let z = flag_usize_any(args, &["--z"])?;
    let proj_str = flag_value_any(args, &["--projection"]).unwrap_or("mean");
    let projection = Projection::parse(proj_str)?;
    let time_map = flag_path_any(args, &["--time-map"]);
    let output = flag_path_any(args, &["--output", "-o"]).unwrap_or_else(|| PathBuf::from("."));

    let res = run_traces(
        &input_dir,
        position,
        channel,
        sizes,
        center_y,
        center_x,
        z,
        projection,
        time_map.as_deref(),
        &output,
    )?;
    println!(
        "Saved {} with rows={}, times={}, roi_sizes={:?}, center=({}, {})",
        res.output_path.display(),
        res.row_count,
        res.time_count,
        res.roi_sizes,
        res.center_y,
        res.center_x
    );
    Ok(())
}

fn cmd_show_traces(args: &[String]) -> Result<(), String> {
    let input_file = first_positional(args)
        .map(PathBuf::from)
        .ok_or_else(|| "missing INPUT_CSV argument".to_string())?;
    let output = flag_path_any(args, &["--output", "-o"]);
    let use_time_real = !has_flag(args, "--use-frame");

    let res = run_show_traces(&input_file, output.as_deref(), use_time_real)?;
    println!(
        "Saved {} with roi_sizes={:?}, x_axis={}, rows={}",
        res.output_path.display(),
        res.roi_sizes,
        res.x_axis,
        res.row_count
    );
    Ok(())
}

fn cmd_plot_lnp(args: &[String]) -> Result<(), String> {
    let pos_args = positional_args(args);
    if pos_args.len() < 2 {
        return Err("plot-lnp requires <INPUT_DIR> and <COUNTS_CSV>".to_string());
    }
    let input_dir = PathBuf::from(pos_args[0]);
    let counts_csv = PathBuf::from(pos_args[1]);
    let filtered_dir = flag_path_any(args, &["--filtered-dir"])
        .ok_or_else(|| "plot-lnp requires --filtered-dir <DIR>".to_string())?;
    let output =
        flag_path_any(args, &["--output", "-o"]).unwrap_or_else(|| PathBuf::from("fig5_panels.png"));
    let position = flag_u32_any(args, &["--position", "-p"])?.unwrap_or(0);
    let channel = flag_u32_any(args, &["--channel", "-c"])?.unwrap_or(1);
    let roi = flag_u32_any(args, &["--roi"])?;
    let time = flag_u32_any(args, &["--time", "-t"])?;
    let time_unit = flag_value_any(args, &["--time-unit"]).unwrap_or("min");

    let res = run_plot_lnp(
        &input_dir,
        &counts_csv,
        &filtered_dir,
        &output,
        position,
        channel,
        roi,
        time,
        time_unit,
    )?;
    println!(
        "Saved {} with roi={}, time={}, spots={}, cells={}",
        res.output_path.display(),
        res.selected_roi,
        res.selected_time,
        res.spot_count,
        res.cell_count
    );
    Ok(())
}

fn cmd_pipeline(args: &[String]) -> Result<(), String> {
    let input_dir = first_positional(args)
        .map(PathBuf::from)
        .ok_or_else(|| "missing INPUT_DIR argument".to_string())?;
    let output_dir = flag_path_any(args, &["--output", "-o"]).unwrap_or_else(|| input_dir.join("analysis"));
    let position = flag_u32_any(args, &["--position", "-p"])?.unwrap_or(0);
    let channel = flag_u32_any(args, &["--channel", "-c"])?.unwrap_or(1);
    let min_gradient = flag_f64_any(args, &["--min-gradient", "-g"])?.unwrap_or(500.0);
    let time_interval = flag_f64_any(args, &["--time-interval"])?.unwrap_or(40.0);

    let picasso_dir = output_dir.join("picasso");
    let filtered_dir = output_dir.join("filtered");
    let params = PicassoParams {
        min_gradient,
        ..PicassoParams::default()
    };
    run_picasso_roi(
        &input_dir,
        position,
        channel,
        0,
        &picasso_dir,
        &params,
        None,
        true,
        true,
    )?;
    let filter_res = run_filter_spots(&picasso_dir, &filtered_dir, &FilterCriteria::default())?;
    let counts_res = run_spot_counts(
        &filtered_dir,
        &output_dir,
        position,
        channel,
        time_interval,
        None,
        true,
        5.0,
    )?;
    let fig_path = output_dir.join("fig5_panels.png");
    run_plot_lnp(
        &input_dir,
        &counts_res.output_path,
        &filtered_dir,
        &fig_path,
        position,
        channel,
        None,
        None,
        "min",
    )?;
    println!(
        "Pipeline finished into {}: filtered_spots={}, counts={}",
        output_dir.display(),
        filter_res.total_out,
        counts_res.output_path.display()
    );
    Ok(())
}

fn is_bool_flag(arg: &str) -> bool {
    matches!(
        arg,
        "--all-rois"
            | "--all-times"
            | "--roi-stacks"
            | "--stack"
            | "--cumulative"
            | "--per-frame"
            | "--use-time-real"
            | "--use-frame"
            | "-h"
            | "--help"
    )
}

fn positional_args(args: &[String]) -> Vec<&str> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--" {
            out.extend(args[i + 1..].iter().map(String::as_str));
            break;
        }
        if arg.starts_with('-') {
            if arg.contains('=') || is_bool_flag(arg) {
                i += 1;
            } else {
                i += 2;
            }
            continue;
        }
        out.push(arg);
        i += 1;
    }
    out
}

fn first_positional(args: &[String]) -> Option<&str> {
    positional_args(args).into_iter().next()
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn flag_value_any<'a>(args: &'a [String], names: &[&str]) -> Option<&'a str> {
    for &name in names {
        let mut i = 0;
        while i < args.len() {
            let arg = args[i].as_str();
            if let Some(rest) = arg.strip_prefix(&format!("{name}=")) {
                return Some(rest);
            }
            if arg == name {
                return args.get(i + 1).map(String::as_str);
            }
            i += 1;
        }
    }
    None
}

fn flag_path_any(args: &[String], names: &[&str]) -> Option<PathBuf> {
    flag_value_any(args, names).map(PathBuf::from)
}

fn flag_f64_any(args: &[String], names: &[&str]) -> Result<Option<f64>, String> {
    match flag_value_any(args, names) {
        None => Ok(None),
        Some(raw) => raw
            .parse::<f64>()
            .map(Some)
            .map_err(|_| format!("invalid {:?} value: {raw}", names)),
    }
}

fn flag_u32_any(args: &[String], names: &[&str]) -> Result<Option<u32>, String> {
    match flag_value_any(args, names) {
        None => Ok(None),
        Some(raw) => raw
            .parse::<u32>()
            .map(Some)
            .map_err(|_| format!("invalid {:?} value: {raw}", names)),
    }
}

fn flag_usize_any(args: &[String], names: &[&str]) -> Result<Option<usize>, String> {
    match flag_value_any(args, names) {
        None => Ok(None),
        Some(raw) => raw
            .parse::<usize>()
            .map(Some)
            .map_err(|_| format!("invalid {:?} value: {raw}", names)),
    }
}
