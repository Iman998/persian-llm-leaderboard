from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))
from core import parser


def test_scan_result_maps(tmp_path, monkeypatch):
    tmp_results = tmp_path
    ds_dir = tmp_results / "ds"
    ds_dir.mkdir()
    main = ds_dir / "Qwen2.5-7B-Instruct.csv"
    raw = ds_dir / "Qwen2.5-7B-Instruct_raw.csv"
    cat = ds_dir / "Qwen2.5-7B-Instruct_cat1.csv"
    for f in (main, raw, cat):
        f.write_text("")

    monkeypatch.setattr(parser, "RESULTS_DIR", tmp_results)
    datasets, main_map, raw_map, cat_map = parser.scan_result_maps()
    assert datasets == ["ds"]
    assert main_map[("ds", "Qwen2.5-7B-Instruct")] == main
    assert raw_map[("ds", "Qwen2.5-7B-Instruct")] == raw
    assert cat_map[("ds", "Qwen2.5-7B-Instruct", "cat1")] == cat

