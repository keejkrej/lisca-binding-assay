use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;

use csv::ReaderBuilder;
use lisca_binding::{
    run_filter_spots, run_picasso_roi, run_picasso_stack, run_plot_lnp, run_show_traces,
    run_spot_counts, run_traces, FilterCriteria, PicassoParams, Projection, PICASSO_SPOT_COLUMNS,
};
use tempfile::TempDir;
use tiff::encoder::{colortype::Gray16, TiffEncoder};

const H: usize = 48;
const W: usize = 48;

fn render_gaussian_frame(spots: &[(f64, f64, f64, f64, f64)], baseline: f64) -> Vec<u16> {
    let mut img = vec![baseline; H * W];
    for &(xc, yc, amp, sx, sy) in spots {
        for y in 0..H {
            for x in 0..W {
                let dx = (x as f64) - xc;
                let dy = (y as f64) - yc;
                let g = amp * (-0.5 * (dx * dx / (sx * sx) + dy * dy / (sy * sy))).exp();
                img[y * W + x] += g;
            }
        }
    }
    img.into_iter()
        .map(|v| v.round().clamp(0.0, 65535.0) as u16)
        .collect()
}

fn render_bf_cell_frame() -> Vec<u16> {
    let mut img = vec![1000u16; H * W];
    for y in 12..36 {
        for x in 12..36 {
            // Textured interior so variation filter + Otsu segments cleanly
            let checker = if (x + y) % 2 == 0 { 2200u16 } else { 1400u16 };
            img[y * W + x] = checker;
        }
    }
    img
}

fn write_tiff_pages(path: &Path, pages: &[Vec<u16>]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent dir");
    }
    let file = File::create(path).expect("create tiff");
    let mut encoder = TiffEncoder::new(file).expect("tiff encoder");
    for page in pages {
        encoder
            .write_image::<Gray16>(W as u32, H as u32, page)
            .expect("write tiff page");
    }
}

fn build_synthetic_workspace(ws: &Path) {
    let roi_pos0 = ws.join("roi").join("Pos0");
    fs::create_dir_all(&roi_pos0).expect("mkdir roi/Pos0");

    // Time-series spot schedules for ROI 0 and ROI 1 (4 frames each, 2 channels: ch0=BF, ch1=Fluo)
    let roi0_spots: Vec<Vec<(f64, f64, f64, f64, f64)>> = vec![
        vec![(16.25, 18.40, 1200.0, 1.35, 1.45)],
        vec![
            (16.30, 18.35, 1250.0, 1.35, 1.45),
            (30.60, 22.15, 1500.0, 1.40, 1.30),
        ],
        vec![
            (16.28, 18.38, 1220.0, 1.35, 1.45),
            (30.55, 22.20, 1480.0, 1.40, 1.30),
            (22.10, 33.75, 1800.0, 1.50, 1.40),
        ],
        vec![
            (16.26, 18.42, 1210.0, 1.35, 1.45),
            (30.58, 22.18, 1510.0, 1.40, 1.30),
            (22.12, 33.72, 1790.0, 1.50, 1.40),
        ],
    ];

    let roi1_spots: Vec<Vec<(f64, f64, f64, f64, f64)>> = vec![
        vec![(20.15, 20.80, 1350.0, 1.40, 1.40)],
        vec![(20.18, 20.78, 1360.0, 1.40, 1.40)],
        vec![
            (20.20, 20.82, 1340.0, 1.40, 1.40),
            (32.40, 28.60, 1650.0, 1.35, 1.50),
        ],
        vec![
            (20.17, 20.79, 1355.0, 1.40, 1.40),
            (32.42, 28.58, 1640.0, 1.35, 1.50),
        ],
    ];

    let bf_page = render_bf_cell_frame();
    for (roi_idx, schedule) in [(0u32, &roi0_spots), (1u32, &roi1_spots)] {
        let mut pages = Vec::with_capacity(4 * 2);
        for spots in schedule {
            pages.push(bf_page.clone()); // ch 0
            pages.push(render_gaussian_frame(spots, 100.0)); // ch 1
        }
        write_tiff_pages(&roi_pos0.join(format!("Roi{roi_idx}.tif")), &pages);
    }

    let index_json = serde_json::json!({
        "rois": [
            { "roi": 0, "fileName": "Roi0.tif", "shape": [4, 2, 1, H, W] },
            { "roi": 1, "fileName": "Roi1.tif", "shape": [4, 2, 1, H, W] }
        ]
    });
    fs::write(
        roi_pos0.join("index.json"),
        serde_json::to_string_pretty(&index_json).unwrap(),
    )
    .expect("write index.json");

    // Also write Pos0 full-frame single-plane TIFFs for stack mode and traces
    let pos0_dir = ws.join("Pos0");
    fs::create_dir_all(&pos0_dir).expect("mkdir Pos0");
    for (t, spots) in roi0_spots.iter().enumerate() {
        let ch0_name = format!("img_channel000_position000_time{t:09}_z000.tif");
        let ch1_name = format!("img_channel001_position000_time{t:09}_z000.tif");
        write_tiff_pages(&pos0_dir.join(ch0_name), &[bf_page.clone()]);
        write_tiff_pages(
            &pos0_dir.join(ch1_name),
            &[render_gaussian_frame(spots, 100.0)],
        );
    }
}

fn binding_repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run_python_binding(repo: &Path, args: &[&str]) {
    let venv_bin = repo.join(".venv").join("bin").join("binding");
    let mut cmd = if venv_bin.is_file() {
        let mut c = Command::new(&venv_bin);
        c.current_dir(repo);
        c
    } else {
        let mut c = Command::new("uv");
        c.current_dir(repo).arg("run").arg("binding");
        c
    };
    cmd.env("BINDING_CONTOUR_BACKEND", "otsu");
    for a in args {
        cmd.arg(*a);
    }
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("Failed to spawn Python binding CLI: {e}"));
    assert!(
        output.status.success(),
        "Python `binding {}` failed:\nstdout: {}\nstderr: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn read_csv_rows(path: &Path) -> (Vec<String>, Vec<BTreeMap<String, String>>) {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let headers: Vec<String> = reader
        .headers()
        .expect("headers")
        .iter()
        .map(str::to_string)
        .collect();
    let mut rows = Vec::new();
    for rec in reader.records() {
        let rec = rec.expect("record");
        let mut map = BTreeMap::new();
        for (k, v) in headers.iter().zip(rec.iter()) {
            map.insert(k.clone(), v.to_string());
        }
        rows.push(map);
    }
    (headers, rows)
}

fn approx_eq(a: f64, b: f64, rel_tol: f64, abs_tol: f64) -> bool {
    let diff = (a - b).abs();
    if diff <= abs_tol {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(1e-12);
    diff / scale <= rel_tol
}

fn compare_csv_files(py_csv: &Path, rs_csv: &Path, columns: &[&str], rel_tol: f64, abs_tol: f64) {
    let (py_headers, py_rows) = read_csv_rows(py_csv);
    let (rs_headers, rs_rows) = read_csv_rows(rs_csv);
    assert_eq!(
        py_headers,
        rs_headers,
        "CSV header mismatch between {} and {}",
        py_csv.display(),
        rs_csv.display()
    );
    assert_eq!(
        py_rows.len(),
        rs_rows.len(),
        "CSV row count mismatch between {} and {}",
        py_csv.display(),
        rs_csv.display()
    );
    for (row_idx, (py_r, rs_r)) in py_rows.iter().zip(rs_rows.iter()).enumerate() {
        for &col in columns {
            let py_v: f64 = py_r
                .get(col)
                .unwrap_or_else(|| panic!("missing col {col}"))
                .parse()
                .expect("parse py float");
            let rs_v: f64 = rs_r
                .get(col)
                .unwrap_or_else(|| panic!("missing col {col}"))
                .parse()
                .expect("parse rs float");
            assert!(
                approx_eq(py_v, rs_v, rel_tol, abs_tol),
                "Row {row_idx} col '{col}' mismatch in {}: Python={py_v} vs Rust={rs_v}",
                py_csv.file_name().unwrap().to_string_lossy()
            );
        }
    }
}

#[test]
fn python_and_rust_binding_pipeline_match_on_synthetic_workspace() {
    let tmp = TempDir::new().expect("tempdir");
    let ws = tmp.path().join("workspace");
    build_synthetic_workspace(&ws);

    let py_out = tmp.path().join("py_out");
    let rs_out = tmp.path().join("rs_out");
    fs::create_dir_all(&py_out).unwrap();
    fs::create_dir_all(&rs_out).unwrap();

    let repo = binding_repo_root();
    let ws_str = ws.to_str().unwrap();

    // 1. Run Python Picasso ROI detection (GaussLQ elliptic)
    let py_picasso_dir = py_out.join("picasso");
    run_python_binding(
        &repo,
        &[
            "picasso",
            ws_str,
            "--roi-stacks",
            "--all-rois",
            "--all-times",
            "-p",
            "0",
            "-c",
            "1",
            "--min-gradient",
            "200",
            "--fit-method",
            "lq",
            "--model",
            "elliptic",
            "-o",
            py_picasso_dir.to_str().unwrap(),
        ],
    );

    // Run Rust Picasso ROI detection
    let rs_picasso_dir = rs_out.join("picasso");
    let lq_params = PicassoParams {
        min_gradient: 200.0,
        box_size: 7,
        fit_method: "lq".to_string(),
        model: "elliptic".to_string(),
        baseline: 0.0,
        sensitivity: 1.0,
        gain: 1.0,
        gaussian_sigma: None,
    };
    let rs_roi_res = run_picasso_roi(
        &ws,
        0,
        1,
        0,
        &rs_picasso_dir,
        &lq_params,
        None,
        true,
        true,
    )
    .expect("rust run_picasso_roi");
    assert_eq!(rs_roi_res.rois, vec![0, 1]);
    assert_eq!(rs_roi_res.time_count, 4);
    assert_eq!(rs_roi_res.total_spots, 15);

    for roi in 0..2 {
        for t in 0..4 {
            let fname = format!("roi{roi:02}_time{t:09}.csv");
            compare_csv_files(
                &py_picasso_dir.join(&fname),
                &rs_picasso_dir.join(&fname),
                &PICASSO_SPOT_COLUMNS,
                1e-2,
                1e-2,
            );
        }
    }

    // 2. Also verify stack mode with both GaussMLE (`mle`) and pixel-integrated MLE (`mle-integrated`)
    for (method, model, subdir) in [
        ("mle", "spherical", "stack_mle"),
        ("mle-integrated", "elliptic", "stack_mle_int"),
    ] {
        let py_stack_dir = py_out.join(subdir);
        let rs_stack_dir = rs_out.join(subdir);
        run_python_binding(
            &repo,
            &[
                "picasso",
                ws_str,
                "--stack",
                "-p",
                "0",
                "-c",
                "1",
                "-t",
                "2",
                "--min-gradient",
                "200",
                "--fit-method",
                method,
                "--model",
                model,
                "-o",
                py_stack_dir.to_str().unwrap(),
            ],
        );
        let stack_params = PicassoParams {
            min_gradient: 200.0,
            box_size: 7,
            fit_method: method.to_string(),
            model: model.to_string(),
            ..PicassoParams::default()
        };
        run_picasso_stack(&ws, 0, 1, 2, &rs_stack_dir, &stack_params)
            .expect("rust run_picasso_stack");
        compare_csv_files(
            &py_stack_dir.join("picasso_position000_channel001_time000000002.csv"),
            &rs_stack_dir.join("picasso_position000_channel001_time000000002.csv"),
            &PICASSO_SPOT_COLUMNS,
            1e-2,
            1e-2,
        );
    }

    // 3. Filter spots parity
    let py_filtered_dir = py_out.join("filtered");
    let rs_filtered_dir = rs_out.join("filtered");
    run_python_binding(
        &repo,
        &[
            "filter-spots",
            py_picasso_dir.to_str().unwrap(),
            "-o",
            py_filtered_dir.to_str().unwrap(),
            "--min-intensity",
            "500",
            "--min-fwhm",
            "1.5",
            "--max-fwhm",
            "6.0",
            "--min-probability",
            "0.5",
        ],
    );
    let filter_res = run_filter_spots(
        &rs_picasso_dir,
        &rs_filtered_dir,
        &FilterCriteria {
            min_intensity: Some(500.0),
            max_intensity: None,
            min_fwhm: Some(1.5),
            max_fwhm: Some(6.0),
            min_probability: Some(0.5),
        },
    )
    .expect("rust run_filter_spots");
    assert_eq!(filter_res.total_in, 15);
    assert_eq!(filter_res.total_out, 15);

    for roi in 0..2 {
        for t in 0..4 {
            let fname = format!("roi{roi:02}_time{t:09}_filtered.csv");
            compare_csv_files(
                &py_filtered_dir.join(&fname),
                &rs_filtered_dir.join(&fname),
                &PICASSO_SPOT_COLUMNS,
                1e-2,
                1e-2,
            );
        }
    }

    // 4. Spot counts parity (cumulative & per-frame)
    run_python_binding(
        &repo,
        &[
            "spot-counts",
            py_filtered_dir.to_str().unwrap(),
            "-o",
            py_out.to_str().unwrap(),
            "-p",
            "0",
            "-c",
            "1",
            "--time-interval",
            "40",
            "--cumulative",
            "--match-distance",
            "4.0",
        ],
    );
    let rs_counts = run_spot_counts(&rs_filtered_dir, &rs_out, 0, 1, 40.0, None, true, 4.0)
        .expect("rust run_spot_counts");
    compare_csv_files(
        &py_out.join("spot_counts_position000_channel001.csv"),
        &rs_counts.output_path,
        &["roi", "time", "time_real", "spot_count", "frame_spots"],
        1e-9,
        1e-9,
    );

    // 5. Traces & show-traces parity
    run_python_binding(
        &repo,
        &[
            "traces",
            ws_str,
            "-p",
            "0",
            "-c",
            "1",
            "--sizes",
            "8,16,32",
            "-o",
            py_out.to_str().unwrap(),
        ],
    );
    let rs_traces = run_traces(
        &ws,
        0,
        1,
        "8,16,32",
        None,
        None,
        None,
        Projection::Mean,
        None,
        &rs_out,
    )
    .expect("rust run_traces");
    compare_csv_files(
        &py_out.join("traces_position000_channel001.csv"),
        &rs_traces.output_path,
        &[
            "time",
            "roi_size",
            "center_y",
            "center_x",
            "y0",
            "y1",
            "x0",
            "x1",
            "mean_intensity",
        ],
        1e-7,
        1e-7,
    );

    let py_traces_png = py_out.join("traces_plot.png");
    let rs_traces_png = rs_out.join("traces_plot.png");
    run_python_binding(
        &repo,
        &[
            "show-traces",
            py_out
                .join("traces_position000_channel001.csv")
                .to_str()
                .unwrap(),
            "-o",
            py_traces_png.to_str().unwrap(),
        ],
    );
    run_show_traces(&rs_traces.output_path, Some(&rs_traces_png), true)
        .expect("rust run_show_traces");
    assert!(py_traces_png.is_file() && py_traces_png.metadata().unwrap().len() > 0);
    assert!(rs_traces_png.is_file() && rs_traces_png.metadata().unwrap().len() > 0);

    // 6. plot-lnp parity
    let py_fig_png = py_out.join("fig5_panels.png");
    let rs_fig_png = rs_out.join("fig5_panels.png");
    run_python_binding(
        &repo,
        &[
            "plot-lnp",
            ws_str,
            py_out
                .join("spot_counts_position000_channel001.csv")
                .to_str()
                .unwrap(),
            "--filtered-dir",
            py_filtered_dir.to_str().unwrap(),
            "-o",
            py_fig_png.to_str().unwrap(),
        ],
    );
    let rs_plot_res = run_plot_lnp(
        &ws,
        &rs_counts.output_path,
        &rs_filtered_dir,
        &rs_fig_png,
        0,
        1,
        None,
        None,
        "min",
    )
    .expect("rust run_plot_lnp");
    assert_eq!(rs_plot_res.selected_roi, 0);
    assert_eq!(rs_plot_res.selected_time, 2);
    assert_eq!(rs_plot_res.spot_count, 3);
    assert_eq!(rs_plot_res.cell_count, 2);
    assert!(py_fig_png.is_file() && py_fig_png.metadata().unwrap().len() > 0);
    assert!(rs_fig_png.is_file() && rs_fig_png.metadata().unwrap().len() > 0);
}
