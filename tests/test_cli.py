import json

import pytest

from hdcompass.cli import main


def test_sweep_quick_writes_report(tmp_path):
    main(["sweep", "--quick", "--out", str(tmp_path)])
    report = json.loads((tmp_path / "report.json").read_text())
    assert len(report["config_hash"]) == 12
    (row,) = report["results"]
    assert {"manifold_mae", "decode_mae", "is_ring", "ring_score", "runtime_s"} <= set(row)
    assert (tmp_path / "sweep.png").exists()
    with pytest.raises(FileExistsError):
        main(["sweep", "--quick", "--out", str(tmp_path)])


def test_pipeline_on_synthetic_session(tmp_path):
    from hdcompass.cli import run_pipeline
    from hdcompass.synth import simulate_session

    s, _ = simulate_session(n_cells=16, wake=150, rem=60, sws=120, rem_epoch=30, sws_epoch=60)
    report, _, details = run_pipeline(s, tmp_path, quick=True)
    assert set(report["dynamics"]) == {"wake", "rem", "sws"}
    assert report["decode_wake_validation"]["hmm_head_tuning_mae_rad"] < 0.5
    assert set(details["results"]) == {"wake", "rem", "sws"}
    for name in ("ring", "barcode", "decode_wake", "dynamics"):
        assert (tmp_path / f"{name}.png").exists()
    json.loads((tmp_path / "report.json").read_text())


def test_pipeline_skips_a_missing_sleep_state(tmp_path):
    import pynapple as nap

    from hdcompass.cli import run_pipeline
    from hdcompass.datasets import Session
    from hdcompass.synth import simulate_session

    s, _ = simulate_session(n_cells=16, wake=150, rem=60, sws=120, rem_epoch=30, sws_epoch=60)
    no_rem = Session(s.spikes, s.angle, s.wake, nap.IntervalSet(start=[], end=[]), s.sws)
    report, _, _ = run_pipeline(no_rem, tmp_path, quick=True)
    assert report["skipped_states"] == ["rem"] and set(report["dynamics"]) == {"wake", "sws"}


def test_returned_report_is_what_is_saved(tmp_path):
    from hdcompass.cli import run_pipeline
    from hdcompass.synth import simulate_session

    s, _ = simulate_session(n_cells=12, wake=130, rem=40, sws=60, rem_epoch=40, sws_epoch=60)
    report, _, _ = run_pipeline(s, tmp_path, quick=True)
    assert json.loads(json.dumps(report)) == json.loads((tmp_path / "report.json").read_text())
