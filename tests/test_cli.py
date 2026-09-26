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
