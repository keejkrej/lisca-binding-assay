//! Single-particle LNP binding assay analysis for LiSCA workspaces.
//!
//! Provides on-par Rust implementations of the `lisca-binding-assay` pipeline stages
//! importing [`picasso`] (`picasso-rs`), [`marray`], and [`mplot`]:
//! - [`binding_assay`]: Workspace-level LNP binding assay (`run_position_binding`, `write_binding_summary`, `detect_particles_frame`, `summarize_particles`).
//! - [`picasso_stage`]: Single-particle spot identification and 2D Gaussian PSF fitting (`run_picasso_roi`, `run_picasso_stack`, `detect_spots_picasso`).
//! - [`filter_spots`]: Spot filtering by intensity, FWHM, and confidence score (`run_filter_spots`).
//! - [`spot_counts`]: Per-frame and cumulative unique spot counting across time (`run_spot_counts`).
//! - [`traces`]: Concentric square ROI intensity trace extraction (`run_traces`).
//! - [`plot`]: Region trace plots (`run_show_traces`) and 3-panel LNP binding figures (`run_plot_lnp`).

pub mod binding_assay;
pub mod export;
pub mod filter_spots;
pub mod frames;
pub mod paths;
pub mod picasso_stage;
pub mod plot;
pub mod roi;
pub mod sample;
pub mod spot_counts;
pub mod traces;

pub use binding_assay::{
    BINDING_CSV_HEADERS, BINDING_SUMMARY_CSV_HEADERS, BindingFrameStats, BindingRow,
    BindingSummaryRow, DEFAULT_BOX_SIZE, DEFAULT_MIN_GRADIENT, FluorescenceHistogram,
    PARTICLE_CSV_HEADERS, ParticleRecord, ParticleSpot, detect_particles_frame,
    discover_binding_csvs, discover_particle_csvs, run_position_binding, summarize_particles,
    write_binding_summary,
};
pub use export::write_xlsx_only;
pub use filter_spots::{
    FilterCriteria, FilterSpotsResult, filter_spot_frame, read_spot_csv, run_filter_spots,
    write_spot_csv,
};
pub use frames::{Frame, available_times, find_frames, load_stack};
pub use paths::{
    filtered_spots_output_path, picasso_roi_output_path, picasso_stack_output_path,
    spot_counts_output_path, spotiflow_roi_output_path, spotiflow_stack_output_path,
    traces_output_path, traces_plot_output_path,
};
pub use picasso_stage::{
    PICASSO_SPOT_COLUMNS, PicassoParams, PicassoRoiResult, PicassoStackResult,
    detect_spots_picasso, run_picasso_roi, run_picasso_stack, write_picasso_spot_csv,
};
pub use plot::{
    PlotLnpResult, ShowTracesResult, auto_select_roi, auto_select_time, intensity_to_radius,
    read_counts_csv, resolve_spot_csv, run_plot_lnp, run_show_traces,
};
pub use roi::{
    RoiEntry, RoiIndex, build_time_map, list_rois, load_roi_index, load_roi_stack, load_time_map,
    resolve_roi_times, roi_dir, roi_stack_path,
};
pub use sample::{SampleAnalysis, SampleMapping, sample_pack_dirnames};
pub use spot_counts::{
    SpotCountsResult, collect_spot_files, count_new_spots, parse_spot_file, read_spot_coordinates,
    run_spot_counts,
};
pub use traces::{
    Projection, TracesResult, parse_sizes, project_stack, run_traces, sample_roi_mean,
    select_plane, square_roi_bounds,
};
