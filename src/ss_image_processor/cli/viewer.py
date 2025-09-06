#!/usr/bin/env python3
"""
Build a simple HTML viewer that layers all sequences and allows per-sequence scrubbing.

The script scans a frames root (default: output/individual_frames) and groups
images by directory. Each directory is treated as one sequence; all image files
in that directory (with supported extensions) are sorted numerically by name.

Outputs:
  - viewer/index.html (static viewer)
  - viewer/manifest.json (lists sequences and their frames as relative paths)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

SUPPORTED_EXTS = {".webp", ".avif", ".png", ".jpg", ".jpeg"}


def natural_key(s: str) -> list[object]:
    """Split string into list of strings/ints for natural sorting."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_sequences(frames_root: Path) -> list[tuple[Path, list[Path]]]:
    """Return list of (sequence_dir, frames) where frames are sorted by natural order."""
    out: list[tuple[Path, list[Path]]] = []
    if not frames_root.exists():
        return out
    for dirpath, _dirnames, filenames in os_walk(frames_root):
        d = Path(dirpath)
        frames = [d / f for f in filenames if (d / f).suffix.lower() in SUPPORTED_EXTS]
        if not frames:
            continue
        frames.sort(key=lambda p: natural_key(p.name))
        out.append((d, frames))
    return out


def os_walk(root: Path):
    """Wrapper around os.walk yielding (dirpath, dirnames, filenames)."""
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden dirs
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
        yield dirpath, dirnames, filenames


def _read_metadata(seq_dir: Path) -> dict[str, int | None]:
    """Read offset metadata from metadata.json if present.

    Returns a dict with keys: offsetX, offsetY, origW, origH (ints or None).
    """
    mpath = seq_dir / "metadata.json"
    if not mpath.exists():
        return {"offsetX": 0, "offsetY": 0, "origW": None, "origH": None}
    try:
        data = json.loads(mpath.read_text(encoding="utf-8"))
        ox = int(data.get("offset_x", 0))
        oy = int(data.get("offset_y", 0))
        ow = data.get("original_width")
        oh = data.get("original_height")
        ow = int(ow) if isinstance(ow, int | float | str) and str(ow).isdigit() else None
        oh = int(oh) if isinstance(oh, int | float | str) and str(oh).isdigit() else None
        return {"offsetX": ox, "offsetY": oy, "origW": ow, "origH": oh}
    except Exception:
        return {"offsetX": 0, "offsetY": 0, "origW": None, "origH": None}


def write_manifest(
    viewer_dir: Path, frames_root: Path, sequences: list[tuple[Path, list[Path]]], mount_name: str = "frames"
) -> Path:
    """Write viewer/manifest.json with sequences, frames, and offset metadata.

    Frames are referenced under a mount inside the viewer directory (default 'frames').
    It is recommended to create a symlink `viewer/frames -> frames_root`.
    """
    manifest: list[dict] = []
    for seq_dir, frames in sequences:
        rel_name = seq_dir.relative_to(frames_root).as_posix() or "."
        # reference frames via the mount inside the viewer dir
        rel_frames = [f"{mount_name}/" + p.relative_to(frames_root).as_posix() for p in frames]
        meta = _read_metadata(seq_dir)
        manifest.append(
            {
                "id": rel_name,
                "name": rel_name,
                "frames": rel_frames,
                "offsetX": meta["offsetX"],
                "offsetY": meta["offsetY"],
                "origW": meta["origW"],
                "origH": meta["origH"],
            }
        )

    viewer_dir.mkdir(parents=True, exist_ok=True)
    out = viewer_dir / "manifest.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return out


def path_relative_to(p: Path, base: Path) -> Path:
    """Return relative path from base to p."""
    try:
        return p.relative_to(base)
    except Exception:
        return Path(os_path_relpath(p, start=base))


def os_path_relpath(p: Path, start: Path) -> str:
    import os

    return os.path.relpath(str(p), str(start))


def write_index_html(viewer_dir: Path) -> Path:
    """Create a simple static index.html that loads manifest.json and provides UI."""
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sequence Layer Viewer</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; display: flex; height: 100vh; }
    #sidebar { width: 320px; overflow-y: auto; border-right: 1px solid #ddd; padding: 12px; box-sizing: border-box; }
    #main { position: relative; flex: 1; display: flex; align-items: center; justify-content: center; background:#111; }
    #stage { position: relative; }
    .layer { position: absolute; top: 0; left: 0; }
    .panel { border-bottom: 1px solid #eee; padding: 8px 0; }
    .row { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
    .name { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    input[type="range"] { width: 100%; }
    label { font-size: 12px; color: #444; }
    #controls { padding-bottom: 8px; border-bottom: 1px solid #ddd; margin-bottom: 8px; }
    #status { color: #888; font-size: 12px; }
  </style>
</head>
<body>
  <div id="sidebar">
    <div id="controls">
      <div class="row"><button id="fit">Fit to first frame</button></div>
      <div class="row"><label><input type="checkbox" id="sync"> Sync all sliders</label></div>
      <div id="status"></div>
    </div>
    <div id="layers"></div>
  </div>
  <div id="main">
    <div id="stage"></div>
  </div>

  <script>
  const state = { sequences: [], sync: false };

  async function loadManifest() {
    const res = await fetch('manifest.json');
    const data = await res.json();
    state.sequences = data.map((seq, idx) => ({ ...seq, index: 0, visible: true, idx, offsetX: seq.offsetX||0, offsetY: seq.offsetY||0 }));
    renderUI();
    await renderStage();
  }

  function createImg(seq) {
    const img = document.createElement('img');
    img.className = 'layer';
    img.style.display = seq.visible ? 'block' : 'none';
    img.style.position = 'absolute';
    img.style.left = '0px';
    img.style.top = '0px';
    img.style.transform = `translate(${(seq.offsetX||0)}px, ${(seq.offsetY||0)}px)`;
    img.style.zIndex = String(seq.idx || 0);
    img.src = seq.frames[seq.index];
    img.dataset.seqId = seq.id;
    img.decoding = 'async';
    return img;
  }

  async function renderStage() {
    const stage = document.getElementById('stage');
    stage.innerHTML = '';
    // Determine stage size from max of origW/origH if provided
    const maxW = Math.max(0, ...state.sequences.map(s => s.origW||0));
    const maxH = Math.max(0, ...state.sequences.map(s => s.origH||0));
    if (maxW > 0 && maxH > 0) {
      stage.style.width = maxW + 'px';
      stage.style.height = maxH + 'px';
    }

    for (const seq of state.sequences) {
      const img = createImg(seq);
      stage.appendChild(img);
      // If stage has no explicit size, size it from the first loaded image
      if (stage.childElementCount === 1 && !(maxW>0 && maxH>0)) {
        await img.decode().catch(()=>{});
        stage.style.width = img.naturalWidth + 'px';
        stage.style.height = img.naturalHeight + 'px';
      }
    }
    updateStatus();
  }

  function updateStatus() {
    const st = document.getElementById('status');
    const total = state.sequences.length;
    const frames = state.sequences.map(s => s.frames.length).reduce((a,b)=>a+b,0);
    st.textContent = `${total} sequences, ${frames} frames`;
  }

  function renderUI() {
    const layers = document.getElementById('layers');
    layers.innerHTML = '';
    state.sequences.forEach((seq, i) => {
      const panel = document.createElement('div');
      panel.className = 'panel';

      const title = document.createElement('div');
      title.className = 'row name';
      title.textContent = seq.name;
      panel.appendChild(title);

      const row1 = document.createElement('div');
      row1.className = 'row';
      const vis = document.createElement('input');
      vis.type = 'checkbox';
      vis.checked = seq.visible;
      vis.addEventListener('change', () => {
        seq.visible = vis.checked;
        const img = document.querySelector(`img.layer[data-seq-id="${seq.id}"]`);
        if (img) img.style.display = seq.visible ? 'block' : 'none';
      });
      const visLabel = document.createElement('label'); visLabel.textContent = 'Visible';
      row1.appendChild(vis); row1.appendChild(visLabel);
      panel.appendChild(row1);

      const row2 = document.createElement('div');
      row2.className = 'row';
      const slider = document.createElement('input');
      slider.type = 'range';
      slider.min = 0; slider.max = Math.max(0, seq.frames.length - 1);
      slider.value = seq.index;
      const lbl = document.createElement('span');
      lbl.textContent = `${seq.index+1}/${seq.frames.length}`;
      slider.addEventListener('input', () => {
        const idx = parseInt(slider.value, 10) || 0;
        if (state.sync) {
          state.sequences.forEach(s => { s.index = Math.min(idx, s.frames.length-1); });
          document.querySelectorAll('#layers input[type=range]').forEach((el, j) => {
            const s = state.sequences[j]; el.value = s.index; el.nextSibling.textContent = `${s.index+1}/${s.frames.length}`;
          });
          document.querySelectorAll('img.layer').forEach((img, j) => {
            const s = state.sequences[j]; img.src = s.frames[s.index];
          });
        } else {
          seq.index = idx; lbl.textContent = `${seq.index+1}/${seq.frames.length}`;
          const img = document.querySelector(`img.layer[data-seq-id="${seq.id}"]`);
          if (img) img.src = seq.frames[seq.index];
        }
      });
      row2.appendChild(slider); row2.appendChild(lbl);
      panel.appendChild(row2);

      layers.appendChild(panel);
    });

    document.getElementById('sync').addEventListener('change', (e) => {
      state.sync = e.target.checked;
    });

    document.getElementById('fit').addEventListener('click', async () => {
      const first = document.querySelector('#stage img.layer');
      if (!first) return;
      await first.decode().catch(()=>{});
      const stage = document.getElementById('stage');
      stage.style.width = first.naturalWidth + 'px';
      stage.style.height = first.naturalHeight + 'px';
    });
  }

  loadManifest();
  </script>
</body>
<link rel="preload" as="fetch" href="manifest.json" crossorigin="anonymous" />
</html>
"""
    viewer_dir.mkdir(parents=True, exist_ok=True)
    out = viewer_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


def ensure_frames_mount(viewer_dir: Path, frames_root: Path, mount_name: str = "frames") -> Path:
    """Ensure a symlink inside `viewer_dir` that points to `frames_root`.

    Returns the path to the mount inside viewer_dir.
    """
    mount = viewer_dir / mount_name
    try:
        if mount.exists() or mount.is_symlink():
            try:
                if mount.is_symlink():
                    target = os.readlink(mount)
                    target_path = (mount.parent / target).resolve()
                    if target_path == frames_root.resolve():
                        return mount
                # Remove stale symlink
                if mount.is_symlink():
                    mount.unlink()
            except Exception:
                pass
        os.symlink(frames_root, mount)
    except Exception:
        # If symlink creation fails, the viewer can still work if served from
        # the project root and manifest uses relative paths that walk up, but
        # most simple servers forbid that. Prefer running `python -m http.server -d .`
        pass
    return mount


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build a simple layered viewer for image sequences")
    p.add_argument(
        "--frames-root",
        type=Path,
        default=Path("output/individual_frames"),
        help="Root directory containing per-sequence frames",
    )
    p.add_argument("--viewer-dir", type=Path, default=Path("viewer"), help="Directory to write the viewer files")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    frames_root = args.frames_root.resolve()
    viewer_dir = args.viewer_dir.resolve()

    sequences = find_sequences(frames_root)
    write_index_html(viewer_dir)
    mount = ensure_frames_mount(viewer_dir, frames_root, mount_name="frames")
    write_manifest(viewer_dir, frames_root, sequences, mount_name=mount.name)
    print(viewer_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
