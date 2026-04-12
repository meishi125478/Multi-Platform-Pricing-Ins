from pathlib import Path

from ins_pricing.frontend.image_catalog import build_generated_image_choices


def test_build_generated_image_choices_classifies_plot_types(tmp_path: Path) -> None:
    pre_dir = tmp_path / "Results" / "plot" / "od_bc" / "oneway" / "pre"
    post_dir = tmp_path / "ResultsXGB" / "plot" / "od_bc_auto"
    compare_dir = tmp_path / "Results" / "plot" / "od_bc_auto" / "double_lift"
    pre_dir.mkdir(parents=True)
    post_dir.mkdir(parents=True)
    compare_dir.mkdir(parents=True)

    pre_path = pre_dir / "driver_age_train.png"
    oneway_path = post_dir / "00_od_bc_auto_driver_age_oneway_xgb.png"
    lift_path = post_dir / "01_od_bc_auto_xgb_lift.png"
    compare_path = compare_dir / "double_lift_compare_xgb_od_bc_auto_xgb_raw_vs_xgb_ft_embed.png"

    for path in (pre_path, oneway_path, lift_path, compare_path):
        path.write_bytes(b"fake-png")

    items = build_generated_image_choices([str(pre_path), str(oneway_path), str(lift_path), str(compare_path)])

    assert [item["category"] for item in items] == [
        "Pre-Oneway",
        "Oneway",
        "Lift Curve",
        "FT-Embed Compare",
    ]
    assert items[0]["title"] == "Driver Age [Train]"
    assert items[1]["option_label"].startswith("Oneway | Driver Age")
    assert items[2]["option_label"].startswith("Lift Curve | XGB")
    assert items[3]["option_label"].startswith("FT-Embed Compare |")
