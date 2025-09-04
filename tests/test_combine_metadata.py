from pathlib import Path
import json

from combine_metadata import combine_to_metadata


def test_combine_and_log(tmp_path: Path):
    # Arrange: create frames
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(1, 4):
        (frames / f"a_tif_{i:03d}.png").write_text("x")

    # Arrange: sidecars in output dir
    outdir = tmp_path / "out"
    outdir.mkdir()

    dpi_payload = {
        "source_dpi_x": 300.0,
        "source_dpi_y": 300.0,
        "normalized_dpi_x": 72.0,
        "normalized_dpi_y": 72.0,
        "source_path": str(frames / "a_tif_001.png"),
    }
    (outdir / "a_tif.dpi.json").write_text(json.dumps(dpi_payload))

    meta_payload = {"foo": "bar"}
    (outdir / "a_tif.json").write_text(json.dumps(meta_payload))

    # Act
    out_path = combine_to_metadata(outdir, "a_tif", output_name="metadata.json")

    # Assert
    data = json.loads(out_path.read_text())
    assert data["foo"] == "bar"
    assert data["source_folder"] == str(frames)
    assert data["frame_count"] == 3
    assert isinstance(data.get("processing_log"), list)
    steps = [e["step"] for e in data["processing_log"]]
    assert "validate_inputs" in steps
    assert "load_json" in steps
    assert "merge" in steps
    assert "derive_source" in steps
    assert "write_output" in steps

