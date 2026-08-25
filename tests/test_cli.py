from __future__ import annotations

from pathlib import Path

import pytest

from swath.cli import _collect_inputs, build_parser, main


def test_train_defaults_are_sane():
    args = build_parser().parse_args(["train"])
    assert args.task == "landcover"
    assert args.epochs == 30
    assert args.device == "auto"


def test_predict_requires_a_checkpoint():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["predict", "--input", "x.tif"])


def test_serve_accepts_several_checkpoints():
    args = build_parser().parse_args(
        ["serve", "--checkpoint", "a.pt", "--checkpoint", "b.pt", "--port", "9000"]
    )
    assert args.checkpoint == [Path("a.pt"), Path("b.pt")]
    assert args.port == 9000


def test_info_lists_the_registry(capsys):
    assert main(["info"]) == 0
    output = capsys.readouterr().out
    assert "landcover" in output
    assert "buildings" in output


def test_info_describes_a_checkpoint(capsys, checkpoint: Path):
    assert main(["info", "--checkpoint", str(checkpoint)]) == 0
    assert "test" in capsys.readouterr().out


def test_collect_inputs_filters_by_extension(tmp_path: Path):
    (tmp_path / "a.png").write_bytes(b"")
    (tmp_path / "b.tif").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("skip me", encoding="utf-8")
    found = _collect_inputs(tmp_path)
    assert [path.name for path in found] == ["a.png", "b.tif"]


def test_collect_inputs_accepts_a_single_file(tmp_path: Path):
    path = tmp_path / "single.tif"
    path.write_bytes(b"")
    assert _collect_inputs(path) == [path]


def test_collect_inputs_on_a_missing_path(tmp_path: Path):
    assert _collect_inputs(tmp_path / "absent") == []


def test_predict_end_to_end(tmp_path: Path, checkpoint: Path, tiles: Path, capsys):
    output = tmp_path / "out"
    code = main(
        [
            "predict",
            "--checkpoint", str(checkpoint),
            "--input", str(tiles / "images"),
            "--output", str(output),
            "--tile", "64",
            "--overlap", "16",
            "--device", "cpu",
            "--overlay",
        ]
    )
    assert code == 0
    assert (output / "tile_00_mask.png").is_file()
    assert (output / "tile_00_overlay.png").is_file()
    assert "tile_00.png" in capsys.readouterr().out
