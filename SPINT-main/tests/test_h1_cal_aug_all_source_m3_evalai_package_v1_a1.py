from pathlib import Path


def test_a1_is_proxy_only_and_preserves_submission_boundary() -> None:
    root = Path(__file__).resolve().parents[2]
    amendment = (root / "tfpd_exploration/h1_series_20260830/docs/AMENDMENT_H1_CAL_AUG_ALL_SOURCE_M3_EVALAI_PACKAGE_V1_A1.md").read_text()
    runner = (root / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_all_source_m3_evalai_package_v1_a1.py").read_text()
    assert "7897" in amendment and "17897" in amendment and "Docker is not restarted" in amendment
    assert "No scientific or packaging contract changes" in amendment
    assert "socat" in runner and "finally" in runner
    for forbidden in ("systemctl", "docker push", "evalai push", "--train"):
        assert forbidden not in runner

