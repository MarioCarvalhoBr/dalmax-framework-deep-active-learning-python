"""`dalmax.reporting.pilot_report`: the read-only pilot summary (2026-11-12)."""

from __future__ import annotations

import json
from pathlib import Path

from dalmax.reporting.pilot_report import (
    PilotRow,
    degeneracy_flags,
    read_campaign_log_elapsed,
    read_total_time,
    render_pilot_report,
)


def test_campaign_log_elapsed_takes_the_last_done_of_that_job(tmp_path: Path) -> None:
    log = tmp_path / "campaign.log"
    log.write_text(
        "2026-11-12T00:00:00Z [1/3] p1/VAAL/nq100 seed=1 DONE elapsed=100s\n"
        "2026-11-12T00:00:00Z [2/3] p1/VAAL/nq100 seed=2 DONE elapsed=200s\n"
        "2026-11-12T00:00:00Z [1/3] p1/VAAL/nq100 seed=1 DONE elapsed=150s\n"
    )
    assert read_campaign_log_elapsed(log, "p1/VAAL/nq100", 1) == 150.0
    assert read_campaign_log_elapsed(log, "p1/WAAL/nq100", 1) is None
    assert read_campaign_log_elapsed(tmp_path / "nope.log", "p1/VAAL/nq100", 1) is None


def test_total_time_fallback(tmp_path: Path) -> None:
    (tmp_path / "log-dalmax.log").write_text("x - WARNING - Total time: 61.5 seconds\n")
    assert read_total_time(tmp_path) == 61.5
    assert read_total_time(tmp_path / "missing") is None


def test_degeneracy_flags() -> None:
    flags = degeneracy_flags({
        "pool_order_fraction": [0.02, 0.9],
        "vaal_score_n_unique": [500, 1],
        "waal_uncertainty_std": [0.1, 0.0],
        "vaal_discriminator_auc": [None, 0.7],
    })
    assert len(flags) == 3
    assert any("pool head" in f for f in flags) and any("constant scores" in f for f in flags)
    assert not degeneracy_flags({"pool_order_fraction": [0.03, 0.04], "vaal_score_n_unique": [400]})


def test_render_reports_times_projection_diagnostics_and_missing_leaves(tmp_path: Path) -> None:
    leaf = tmp_path / "leaf"
    leaf.mkdir()
    (leaf / "results.json").write_text(json.dumps({
        "all_f1_score": [0.3, 0.5],
        "strategy_diagnostics": {"pool_order_fraction": [0.95], "vaal_score_std": [0.12]},
    }))
    rows = [
        PilotRow("p1/VAAL/nq100 seed=1", "VAAL", leaf, 3600.0, 3000.0, runs_in_selection=3),
        PilotRow("p1/WAAL/nq100 seed=1", "WAAL", None, None, 2700.0, runs_in_selection=3),
    ]
    text = render_pilot_report(rows)
    assert "wall time 60.0 min [campaign.log]; estimate 50.0 min; measured/estimate = 1.20" in text
    assert "VAAL: 3 runs in this selection = 3.0 h at this pace (estimate 2.5 h)" in text
    assert "weighted F1 per round: 0.3, 0.5" in text
    assert "pool_order_fraction: 0.95" in text and "FLAG:" in text
    assert "p1/WAAL/nq100 seed=1" in text and "no leaf yet" in text
