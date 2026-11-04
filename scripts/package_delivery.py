#!/usr/bin/env python3
"""Build a deterministic local delivery ZIP from an explicit file allowlist.

No inference, evaluation, downloads, git operations or third-party imports.
Reference trajectory input is never copied, even with --include-inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile


PREFIX = 'peleng-hackathon/'
SOURCE_DIRS = ('peleng', 'static', 'tests', 'scripts', 'docs')
SOURCE_SUFFIXES = {'.py', '.html', '.css', '.js', '.md', '.txt', '.in', '.sh', '.json'}
ROOT_FILES = ('README.md', 'START_HERE.md', 'run.sh', 'setup.sh', 'run.bat', 'setup.bat',
              'requirements.txt', 'requirements.in',
              'LICENSE', 'LICENSE.txt', 'LICENSE.md', 'THIRD_PARTY',
              'THIRD_PARTY.md', 'THIRD_PARTY.txt', '.gitignore')
EXCLUDED_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache', '.superpowers',
                 'node_modules', 'data', 'outputs', 'distribution'}
REQUIRED_RUN_FILES = ('config.json', 'observations.jsonl', 'trajectory.csv', 'summary.json')
OPTIONAL_RUN_FILES = ('evaluation.json', 'errors.csv', 'report.html', 'preview.jpg')


def _regular_file(path, base):
    """Reject symlinks at every level, including directory symlinks."""
    try:
        relative = path.relative_to(base)
    except ValueError:
        return False
    current = base
    if base.is_symlink():
        return False
    for part in relative.parts:
        current = current/part
        if current.is_symlink():
            return False
    return path.is_file()


def _required(path, base):
    if not _regular_file(path, base):
        raise FileNotFoundError(f'Required regular file is missing (symlinks excluded): {path}')
    return path


def build_bundle(root, output, *, run=None, include_inputs=False, screenshot=None):
    root, output = Path(root).absolute(), Path(output).absolute()
    selected = {}
    for name in ROOT_FILES:
        path = root/name
        if _regular_file(path, root):
            selected[name] = path
    for directory in SOURCE_DIRS:
        base = root/directory
        if not base.is_dir() or base.is_symlink():
            continue
        for path in base.rglob('*'):
            relative = path.relative_to(root)
            if (path.suffix.lower() in SOURCE_SUFFIXES
                    and not any(part in EXCLUDED_DIRS or part.startswith('.') for part in relative.parts)
                    and _regular_file(path, root)):
                selected[relative.as_posix()] = path
    model = root/'models/yolo11s.pt'
    selected['models/yolo11s.pt'] = _required(model, root)
    if run is not None:
        run = Path(run).absolute()
        for name in REQUIRED_RUN_FILES:
            selected[f'outputs/example-run/{name}'] = _required(run/name, run)
        for name in OPTIONAL_RUN_FILES:
            if _regular_file(run/name, run):
                selected[f'outputs/example-run/{name}'] = run/name
        for path in run.glob('preview-*s.jpg'):
            if _regular_file(path, run):
                selected[f'outputs/example-run/{path.name}'] = path
    if include_inputs:
        for camera in range(1, 6):
            for name in (f'Way1.FinalImage.CineCameraActor{camera}.mp4',
                         f'object_transform_CineCameraActor{camera}.csv'):
                selected[f'data/{name}'] = _required(root/'data'/name, root)
    if screenshot is not None:
        screenshot = Path(screenshot).absolute()
        if screenshot.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
            raise ValueError('Screenshot must be PNG, JPEG or WebP')
        selected[f'docs/demo-screenshot{screenshot.suffix.lower()}'] = _required(screenshot, screenshot.parent)
    if any(path == output for path in selected.values()):
        raise ValueError('Output ZIP must not overwrite an input')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement keeps a previous complete bundle if packing fails.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.zip.part', delete=False) as handle:
            temporary = Path(handle.name)
        entries = []
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative, path in sorted(selected.items()):
                info = zipfile.ZipInfo(PREFIX+relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (0o100755 if path.suffix == '.sh' else 0o100644) << 16
                digest = hashlib.sha256()
                size = 0
                with path.open('rb') as source, archive.open(info, 'w', force_zip64=True) as target:
                    while chunk := source.read(1024*1024):
                        digest.update(chunk)
                        size += len(chunk)
                        target.write(chunk)
                entries.append({'path': relative, 'bytes': size, 'sha256': digest.hexdigest()})
            manifest = {'format_version': 1, 'model': 'models/yolo11s.pt',
                        'example_run': 'outputs/example-run' if run is not None else None,
                        'camera_inputs_included': include_inputs,
                        'reference_trajectory_input_included': False,
                        'note': 'Evaluation/report artifacts may contain post-run reference values. Inference inputs exclude object trajectory.',
                        'files': entries}
            info = zipfile.ZipInfo(PREFIX+'DELIVERY_MANIFEST.json', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {'archive': str(output), 'files': len(selected)+1, 'bytes': output.stat().st_size,
            'camera_inputs_included': include_inputs, 'reference_trajectory_input_included': False}


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Completed saved run to include as outputs/example-run')
    parser.add_argument('--include-inputs', action='store_true', help='Include only the five camera MP4 + five camera CSV files; never target truth')
    parser.add_argument('--screenshot', type=Path, help='Optional demonstration screenshot')
    parser.add_argument('--output', type=Path, default=root/'distribution/peleng-hackathon.zip')
    args = parser.parse_args()
    try:
        result = build_bundle(root, args.output, run=args.run,
                              include_inputs=args.include_inputs, screenshot=args.screenshot)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(2, f'Packaging error: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
