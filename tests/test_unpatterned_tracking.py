"""Exercise the exported btrack link-screening script on small known tracks."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/exploratory/unpatterned_lnp/btrack/audit_links.py"
)


@pytest.mark.parametrize("only_short_tracks", [False, True])
def test_link_screen_splits_jumps_before_filtering(tmp_path, only_short_tracks):
    rows = []
    tracks = (
        [(0, list(range(4)), list(range(4)))]
        if only_short_tracks
        else [
            (0, list(range(10)), list(range(5)) + list(range(100, 105))),
            (1, [0, 1, 2, 3, 8, 9, 10, 11, 12], list(range(9))),
            (2, list(range(4)), list(range(4))),
            (3, [0, 1, 4, 5, 6], [0, 1, 10, 11, 12]),
        ]
    )
    for particle, frames, positions in tracks:
        rows.extend(
            {"particle": particle, "frame": frame, "x": x, "y": 0, "time_s": 8 * frame}
            for frame, x in zip(frames, positions, strict=True)
        )
    original = pd.DataFrame(rows)
    for recording in ("005", "006", "007"):
        out = tmp_path / f"LISCA_review_A549_exp_60x{recording}"
        out.mkdir()
        original.to_csv(out / "global_all_tracks.csv", index=False)
    subprocess.run(
        [sys.executable, str(SCRIPT)],
        env={**os.environ, "LNP_RESULTS_DIR": str(tmp_path)},
        check=True,
        capture_output=True,
        text=True,
    )
    for out in tmp_path.iterdir():
        pd.testing.assert_frame_equal(
            pd.read_csv(out / "global_all_tracks.csv"), original
        )
        actual = pd.read_csv(out / "audited_tracks.csv")
        summary = json.loads((out / "audit.json").read_text())
        if only_short_tracks:
            assert actual.empty
            assert summary["median_span_s"] is None
            assert summary["retained_fragments"] == 0
            continue
        assert summary["rejected_connections"] == 2
        assert summary["retained_fragments"] == 4
        assert actual.groupby("particle").size().eq(5).all()
        assert set(actual.source_btrack_particle) == {0, 1, 3}
        assert actual[actual.source_btrack_particle == 0].particle.nunique() == 2
        # The three-frame gap is allowed; the five-frame gap is split.
        assert actual[actual.source_btrack_particle == 3].frame.tolist() == [
            0,
            1,
            4,
            5,
            6,
        ]
        assert actual[actual.source_btrack_particle == 1].frame.tolist() == [
            8,
            9,
            10,
            11,
            12,
        ]
