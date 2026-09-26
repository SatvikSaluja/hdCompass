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
