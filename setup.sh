#!/bin/sh
set -eu
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
  echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
  exit 1
fi
if [ ! -x .venv/bin/python ]; then uv venv --python 3.12 .venv; fi
uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p models data outputs
if [ ! -f models/yolo11s.pt ]; then
  curl -L --fail --retry 5 --retry-all-errors --continue-at - \
    -o models/yolo11s.pt.part \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
  .venv/bin/python - <<'PY'
from pathlib import Path
import zipfile
p=Path('models/yolo11s.pt.part')
if p.stat().st_size != 19313732:
    raise RuntimeError('Model download is incomplete')
with zipfile.ZipFile(p) as archive:
    if archive.testzip() is not None:
        raise RuntimeError('Model download is corrupt')
p.rename('models/yolo11s.pt')
PY
fi
.venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
path = Path('models/yolo11s.pt')
expected = '85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5'
if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
    raise RuntimeError('Model checksum mismatch. Remove models/yolo11s.pt and rerun setup.sh.')
PY
echo 'Ready. Put camera videos and camera CSVs in data/, then run ./run.sh --check-data and ./run.sh'
