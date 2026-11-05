"""Diagnostics without reference coordinates: coverage is not accuracy."""
import csv
import html
import json
import math
from collections import Counter
from pathlib import Path
from .scoring_contract import matches_reference, coverage_contract


def write_prediction_report(run_dir, expected_frames):
    run_dir = Path(run_dir)
    with (run_dir / 'trajectory_m.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    config_path = run_dir/'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    reference = config.get('reference_point') or ('front' if any(
        r['reference_point']=='front_face_center' for r in rows) else 'center')
    if len({int(r['Frame']) for r in rows}) != len(rows):
        raise ValueError('Duplicate source frames in trajectory')
    rows = [r for r in rows if 0 <= int(r['Frame']) < expected_frames]
    points = []
    for row in rows:
        xyz = [float(row[k + '_m']) if row[k + '_m'] else None for k in 'XYZ']
        valid = (all(v is not None and math.isfinite(v) for v in xyz)
                 and matches_reference(reference, row.get('reference_point'), row['state']))
        points.append((int(row['Frame']), float(row['Time_s']), xyz if valid else None))
    valid = sum(p[2] is not None for p in points)
    steps = [math.dist(a[2], b[2]) for a, b in zip(points, points[1:])
             if a[2] is not None and b[2] is not None and b[0] == a[0] + 1]
    summary_path = run_dir / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    reference_label = 'переда кабины грузовика' if reference=='front' else 'центра грузовика'
    diagnostics = {'ground_truth_available': False, 'accuracy': None,
                   'expected_frames': expected_frames, 'processed_frames': len(rows),
                   'coordinate_frames': valid, 'coordinate_coverage_pct': 100 * valid / expected_frames if expected_frames else None,
                   'states': dict(Counter(r['state'] for r in rows)),
                   'methods': dict(Counter(r['method'] for r in rows)),
                   'requested_reference_point': reference, 'units': 'metres',
                   'actual_reference_points': dict(Counter(r['reference_point'] for r in rows)),
                   'max_adjacent_step_m': max(steps, default=None),
                   'adjacent_steps_over_10m': sum(v > 10 for v in steps),
                   'detection_frames': summary.get('steps_with_detection'),
                   'multiview_metric_frames': summary.get('metric_triangulation_steps'),
                   'detections_by_camera': summary.get('detections_by_camera')}
    diagnostics.update(coverage_contract([f for f,t,p in points if p is not None],
        expected_frames, reference, summary.get('source_fps', 30.)))
    (run_dir / 'diagnostics.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    def chart(axis):
        values = [p[2][axis] for p in points if p[2] is not None]
        if not values:
            return '<p>No coordinates available</p>'
        lo, hi = min(values), max(values)
        span = max(hi - lo, 1)
        end = max((p[1] for p in points), default=1) or 1
        paths, current, last = [], [], None
        for frame, t, xyz in points:
            if xyz is None or (last is not None and frame != last + 1):
                if current: paths.append(current)
                current = []
            if xyz is not None:
                current.append(f'{70 + 880*t/end:.2f},{200 - 160*(xyz[axis]-lo)/span:.2f}')
            last = frame
        if current: paths.append(current)
        lines = ''.join('<polyline fill="none" stroke="#1763a6" stroke-width="2" points="' + ' '.join(p) + '"/>' for p in paths)
        return f'<svg viewBox="0 0 1000 250" role="img" aria-label="{"XYZ"[axis]} metres versus time"><path d="M70 25V210H960" fill="none" stroke="#888"/><text x="5" y="45">{hi:.1f} m</text><text x="5" y="200">{lo:.1f} m</text><text x="70" y="235">0 s</text><text x="900" y="235">{end:.1f} s</text>{lines}</svg>'
    content = '<!doctype html><html lang="ru"><meta charset="utf-8"><title>Scene trajectory</title><style>body{font:17px system-ui;color:#172b3a;background:#f4f7fa;max-width:1100px;margin:30px auto;padding:24px}section{background:white;padding:20px;margin:18px 0;border-radius:12px}svg{width:100%}pre{white-space:pre-wrap}</style><h1>Предсказанная траектория центра грузовика</h1><p>Координаты в метрах. Эталон отсутствует: точность и ошибка в метрах неизвестны. Покрытие означает наличие координат, включая прогноз фильтра.</p>'
    content = content.replace('Предсказанная траектория центра грузовика', 'Предсказанная траектория '+reference_label)
    content += f'<p>Обработано: {len(rows)} / {expected_frames}. Координаты: {valid} кадров.</p>'
    content += '<p>Участки ground_plane используют предположение Z=0: это приближение контакта с землёй, а не измеренная высота центра. Тип опорной точки сохранён в CSV в поле reference_point.</p>'
    content += ''.join(f'<section><h2>{axis}, м</h2>{chart(i)}</section>' for i, axis in enumerate('XYZ'))
    content += '<section><h2>Диагностика</h2><pre>' + html.escape(json.dumps(diagnostics, ensure_ascii=False, indent=2)) + '</pre></section></html>'
    (run_dir / 'report.html').write_text(content, encoding='utf-8')
    return diagnostics
