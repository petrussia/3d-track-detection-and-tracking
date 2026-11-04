"""Self-contained offline report for saved exact-frame evaluation results."""
import html
import json
from pathlib import Path


def _number(value, digits=2):
    return '—' if value is None else f'{value:,.{digits}f}'.replace(',', ' ').replace('.', ',')


def _percent(value):
    return '—' if value is None else _number(100*value, 1)+'%'


def _quality_rows(result):
    rows = []
    for label, key in [('Наблюдения', 'observed_metrics'), ('Прогнозы', 'predicted_metrics'),
                       ('Все доступные оценки', 'all_estimate_metrics')]:
        metrics = result[key]
        values = [str(metrics['count']), _number(metrics['horizontal']['mean_m']),
                  _number(metrics['horizontal']['rmse_m']), _number(metrics['three_d']['mean_m']),
                  _number(metrics['three_d']['rmse_m']), _number(metrics['three_d']['median_m']),
                  _number(metrics['three_d']['p95_m']), _number(metrics['three_d']['max_m'])]
        rows.append('<tr><th>'+label+'</th>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
    return ''.join(rows)


def _group_table(groups):
    rows = []
    for label, group in groups.items():
        metric = group['all_estimate_metrics']
        values = [str(group['processed_steps']), str(group['estimate_steps']),
                  str(group['observed_metrics']['count']), str(group['predicted_metrics']['count']),
                  _percent(group['coverage']['estimate_of_truth']),
                  _number(metric['horizontal']['rmse_m']), _number(metric['three_d']['rmse_m']),
                  _number(metric['three_d']['p95_m'])]
        values.extend(_number(metric['axes'][axis]['mae_m'])+' / '+_number(metric['axes'][axis]['rmse_m'])
                      for axis in 'xyz')
        rows.append('<tr><th>'+html.escape(label)+'</th>'+''.join('<td>'+value+'</td>' for value in values)+'</tr>')
    return '''<div class="table-scroll"><table><thead><tr><th>Группа</th><th>Записей</th><th>Оценок XYZ</th>
<th>Набл. с эталоном</th><th>Прогн. с эталоном</th><th>Оценки / весь эталон</th><th>XY RMSE, м</th>
<th>3D RMSE, м</th><th>3D p95, м</th><th>X MAE / RMSE, м</th><th>Y MAE / RMSE, м</th><th>Z MAE / RMSE, м</th>
</tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>'''


def _baseline_section(result):
    baseline = result.get('baseline_comparison')
    if not baseline:
        return ''
    rows = []
    for label, key in [('Наблюдения в обоих прогонах', 'observed_paired'),
                       ('Оценки в обоих прогонах, включая прогнозы', 'all_estimate_paired')]:
        pair = baseline[key]
        rows.append('<tr><th>'+label+'</th><td>'+str(pair['count'])+'</td><td>'+
                    _number(pair['current']['horizontal']['rmse_m'])+'</td><td>'+
                    _number(pair['baseline']['horizontal']['rmse_m'])+'</td><td>'+
                    _number(pair['current']['three_d']['rmse_m'])+'</td><td>'+
                    _number(pair['baseline']['three_d']['rmse_m'])+'</td></tr>')
    coverage = baseline['baseline_coverage']
    return '''<section class="panel"><h2>Сравнение на общих кадрах</h2>
<p>Базовый прогон: <code>'''+html.escape(baseline['run'])+'''</code>. Показатели в каждой строке рассчитаны по одному и тому же набору кадров для обоих прогонов. Сдвиг времени и подбор параметров по эталону не выполнялись.</p>
<div class="table-scroll"><table><thead><tr><th>Парная выборка</th><th>Кадров</th><th>XY RMSE, текущий</th><th>XY RMSE, базовый</th><th>3D RMSE, текущий</th><th>3D RMSE, базовый</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<p class="note">Покрытие базового прогона: наблюдения / эталон '''+_percent(coverage['observed_of_truth'])+'''; все оценки / эталон '''+_percent(coverage['estimate_of_truth'])+'''. Сравнение качества без учёта покрытия может вводить в заблуждение.</p></section>'''


def write_report(path: Path, result, points, run_name):
    """Render UTF-8 HTML with local canvas plots and JSON escaped for script context."""
    data = json.dumps({'result': result, 'points': points}, ensure_ascii=False, allow_nan=False)
    for character, escaped in [('&', '\\u0026'), ('<', '\\u003c'), ('>', '\\u003e'),
                               ('\u2028', '\\u2028'), ('\u2029', '\\u2029')]:
        data = data.replace(character, escaped)
    axis_rows = ''.join('<tr><th>'+axis.upper()+'</th><td>'+_number(result['axis_mae_m'][axis])+
                        '</td><td>'+_number(result['axis_rmse_m'][axis])+'</td></tr>' for axis in 'xyz')
    coverage = result['coverage']
    method_rows = ''.join('<tr><th>'+html.escape(method)+'</th><td>'+str(count)+'</td></tr>'
                          for method, count in result['method_counts'].items())
    diagnostics = result['discontinuities']
    within = result['within_segment_discontinuities']
    template = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Peleng · Отчёт о точности</title>
<style>
:root{color-scheme:light;--ink:#172d3b;--muted:#526879;--line:#d8e1e5;--blue:#1265b0;--orange:#b66b09;--truth:#52616c;--paper:#f2f5f6}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,sans-serif}main{max-width:1320px;margin:auto;padding:36px 28px 60px}h1{font-size:34px;letter-spacing:-1px;line-height:1.2;margin:10px 0 12px}h2{font-size:21px;margin:0 0 12px}h3{font-size:16px;margin:0 0 12px}.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:12px;color:var(--muted);font-weight:650}p{max-width:1050px;margin:8px 0 16px}.muted,.note{color:var(--muted)}.note{font-size:13px}.badge{display:inline-block;border:1px solid var(--line);padding:4px 10px;border-radius:20px;background:white;margin:0 6px 6px 0;font-size:12px}.panel{border:1px solid var(--line);border-radius:14px;background:white;padding:22px;margin-top:20px;box-shadow:0 2px 3px #162c3910}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-top:24px}.card{background:white;border:1px solid var(--line);border-radius:12px;padding:18px}.value{font-size:30px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.3;margin:5px 0}.label{font-size:13px;color:var(--muted)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.grid>.panel{min-width:0}.table-scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:10px;text-align:right;border-bottom:1px solid #e5ebee;white-space:nowrap}th:first-child{text-align:left}thead th{color:var(--muted);font-size:12px;font-weight:600}tbody th{font-weight:500}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin:12px 0}.legend label{cursor:pointer}.legend input{vertical-align:middle}.legend .truth{color:var(--truth)}.legend .observed{color:var(--blue)}.legend .predicted{color:var(--orange)}.controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:13px}.controls input{width:100px;padding:7px;border:1px solid var(--line);border-radius:6px;font:inherit}button{font:inherit;cursor:pointer;border:1px solid #b6c9d7;border-radius:6px;background:#eef5fa;color:#164d73;padding:7px 12px}button:hover{background:#e1edf6}.chart{position:relative;margin-top:18px}canvas{display:block;width:100%;height:265px;touch-action:pan-y}.xy canvas{height:410px}.tooltip{white-space:pre-line;position:absolute;pointer-events:none;display:none;background:#132c3ef2;color:white;border-radius:7px;padding:10px 12px;font:12px/1.5 system-ui;max-width:330px;z-index:2}.chart-title{font-size:14px;font-weight:650;margin-left:6px}.coverage-bar{display:flex;height:18px;border-radius:7px;overflow:hidden;background:#e6ecef;margin:14px 0}.coverage-bar span{height:100%}.observed-bar{background:var(--blue)}.predicted-bar{background:#d79536}.lost-bar{background:#a7b4bd}code{font-size:12px;overflow-wrap:anywhere;white-space:normal}details{margin:16px 0}summary{cursor:pointer;font-weight:600}a{color:#1265b0}footer{margin-top:26px;font-size:12px;color:var(--muted)}@media(max-width:760px){main{padding:22px 12px}.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr;gap:0}.panel{padding:16px}h1{font-size:27px}.value{font-size:25px}canvas{height:235px}}
</style></head><body><main>
<div class="eyebrow">Peleng / Сохранённый прогон</div><h1>Координаты и точность траектории</h1>
<p class="muted">Постфактум: прямое сопоставление по номеру исходного кадра. Все координаты и ошибки ниже — в метрах, время — в секундах исходного видео.</p>
<span class="badge">Эталон используется только при оценке</span><span class="badge">Без временного сдвига и интерполяции</span><span class="badge">Автономный HTML</span>
<p class="note">Прогон: <code>@@RUN@@</code><br>@@FPS@@ кадр/с · @@STEPS@@ обработанных кадров · @@TRUTH@@ корректных кадров эталона. Эталон относится к pivot объекта; точка оценки по изображению может ему не соответствовать. Результаты не доказывают качество на новом проверочном наборе.</p>
<div class="cards">
<div class="card"><div class="label">Покрытие наблюдениями / эталон</div><div class="value">@@COV@@</div><div class="label">@@MATCHED@@ кадров с координатами и эталоном</div></div>
<div class="card"><div class="label">XY RMSE · наблюдения</div><div class="value">@@XY@@ <small>м</small></div><div class="label">Горизонтальная ошибка</div></div>
<div class="card"><div class="label">3D RMSE · наблюдения</div><div class="value">@@XYZ@@ <small>м</small></div><div class="label">С учётом высоты</div></div>
<div class="card"><div class="label">Z MAE · наблюдения</div><div class="value">@@Z@@ <small>м</small></div><div class="label">Средняя абсолютная ошибка высоты</div></div>
</div>
<section class="panel"><h2>Покрытие и состояния</h2>
<div class="coverage-bar" id="coverage-bar" aria-label="Покрытие среди обработанных кадров"></div>
<div class="table-scroll"><table><thead><tr><th>Состояние</th><th>Кадров в прогоне</th><th>Доля обработанных кадров</th><th>Доля кадров эталона</th></tr></thead><tbody>
<tr><th>Наблюдение · observed</th><td>@@OBS@@</td><td>@@OBS_PROC@@</td><td>@@COV@@</td></tr>
<tr><th>Прогноз · coasting / predicted</th><td>@@PRED@@</td><td>@@PRED_PROC@@</td><td>@@PRED_TRUTH@@</td></tr>
<tr><th>Потеря · lost</th><td>@@LOST@@</td><td>@@LOST_PROC@@</td><td>—</td></tr>
<tr><th>Все конечные оценки XYZ</th><td>@@AVAILABLE@@</td><td>@@ALL_PROC@@</td><td>@@ALL_TRUTH@@</td></tr>
</tbody></table></div><p class="note">В процентах наблюдений и прогнозов учитываются только конечные XYZ. Счётчик состояния включает также записи без корректной позиции. Покрытие — доступность мировой оценки, а не recall детектора. Пропущенные исходные кадры не включены в знаменатель «обработанных». Доля обработанных кадров эталона: @@PROC_TRUTH@@. Некорректных оценок: @@BAD_EST@@.</p>
</section>
<section class="panel"><h2>Истинные и оценённые координаты X / Y / Z</h2>
<div class="legend"><label class="truth"><input type="checkbox" data-series="truth" checked> Эталон</label><label class="observed"><input type="checkbox" data-series="observed" checked> Наблюдение</label><label class="predicted"><input type="checkbox" data-series="predicted" checked> Прогноз</label></div>
<div class="controls"><label>От <input id="time-start" type="number" step="any" aria-label="Начало интервала, с"></label><label>до <input id="time-end" type="number" step="any" aria-label="Конец интервала, с"></label><span>с</span><button id="apply" type="button">Показать</button><button id="reset" type="button">Весь прогон</button></div>
<p class="note">Наведите курсор для значений. Интервал применяется ко всем графикам. Пробелы показывают отсутствующие координаты; линии оценок не соединяют пропущенные исходные кадры, смены режима или сегмента и явные сбросы. Эталон остаётся непрерывным на соседних корректных кадрах. Серый пунктир отмечает явный сброс режима. Точки показывают в том числе одиночные оценки.</p>
<div class="chart" data-kind="axis" data-axis="0"><div class="chart-title">X, м</div><canvas aria-label="Координата X по времени"></canvas><div class="tooltip"></div></div>
<div class="chart" data-kind="axis" data-axis="1"><div class="chart-title">Y, м</div><canvas aria-label="Координата Y по времени"></canvas><div class="tooltip"></div></div>
<div class="chart" data-kind="axis" data-axis="2"><div class="chart-title">Z, м</div><canvas aria-label="Координата Z по времени"></canvas><div class="tooltip"></div></div>
</section>
<div class="grid"><section class="panel"><h2>Путь в плоскости XY</h2><p class="note">Одинаковый масштаб по осям. Разрывы сохранены; прогноз отмечен оранжевым. Видимый временной интервал тот же, что на графиках координат.</p><div class="chart xy" data-kind="xy"><canvas aria-label="Истинный и оценённый путь XY"></canvas><div class="tooltip"></div></div></section>
<section class="panel"><h2>Ошибка по времени</h2><p class="note">Только точные пары «кадр оценки — кадр эталона». Наблюдения и прогнозы показаны раздельно.</p><div class="chart" data-kind="error"><div class="chart-title">3D ошибка, м</div><canvas aria-label="Трёхмерная ошибка по времени"></canvas><div class="tooltip"></div></div><div class="chart" data-kind="horizontal"><div class="chart-title">Горизонтальная ошибка, м</div><canvas aria-label="Горизонтальная ошибка по времени"></canvas><div class="tooltip"></div></div></section></div>
<section class="panel"><h2>Качество на кадрах с эталоном</h2><div class="table-scroll"><table><thead><tr><th>Выборка</th><th>Кадров</th><th>XY средняя, м</th><th>XY RMSE, м</th><th>3D средняя, м</th><th>3D RMSE, м</th><th>3D медиана, м</th><th>3D p95, м</th><th>3D max, м</th></tr></thead><tbody>@@QUALITY@@</tbody></table></div><p class="note">RMSE = √среднего квадрата ошибки. Процентили: линейная интерполяция между отсортированными ошибками. «—» означает отсутствие подходящих пар, а не нулевую ошибку. Все верхние показатели по умолчанию относятся к наблюдениям.</p></section>
<div class="grid"><section class="panel"><h2>Ошибки по осям · наблюдения</h2><table><thead><tr><th>Ось</th><th>MAE, м</th><th>RMSE, м</th></tr></thead><tbody>@@AXES@@</tbody></table><p class="note">Подробная статистика осей для наблюдений, прогнозов и всех оценок сохранена в evaluation.json.</p></section><section class="panel"><h2>Методы доступных оценок</h2><div class="table-scroll"><table><thead><tr><th>Метод</th><th>Кадров</th></tr></thead><tbody>@@METHODS@@</tbody></table></div><p class="note">Метод берётся из сохранённого world.method. Потерянные состояния и некорректные позиции не учитываются.</p></section></div>
<section class="panel"><h2>Качество и покрытие по режимам</h2><p class="note"><strong>assumed_ground</strong> — предположение плоской поверхности с Z=0 до надёжной многокамерной фиксации или после истечения поддержки глубины; <strong>metric_3d</strong> — режим после многокамерной 3D-фиксации, с ограниченным продолжением по направлениям камер. <strong>unknown</strong> — режим не сохранён в старом журнале. Все наземные оценки входят в общую статистику выше, в том числе с большой ошибкой высоты.</p>@@MODES@@<p class="note">Ошибки в таблице относятся ко всем доступным оценкам группы, включая прогнозы. Доля эталона использует все корректные кадры эталона, а не только эту группу. Раздельная статистика наблюдений/прогнозов, полные процентили и покрытие группы сохранены в JSON.</p></section>
<section class="panel"><h2>Качество и покрытие по методам</h2>@@METHOD_QUALITY@@<p class="note">Каждая запись относится к своему сохранённому методу. Потери учитываются в числе записей, но не получают вымышленных координат и ошибок.</p></section>
@@BASELINE@@
<section class="panel"><h2>Непрерывность и проверка данных</h2><h3>Все оценки, включая смены режима и сегмента</h3><p>Переходов между доступными оценками: <strong>@@PAIRS@@</strong>. Скачков &gt; 10 м: <strong>@@JUMPS@@</strong>; скоростей &gt; 50 м/с: <strong>@@SPEEDS@@</strong>. Максимальный скачок: <strong>@@MAX_JUMP@@ м</strong>, максимальная скорость: <strong>@@MAX_SPEED@@ м/с</strong>.</p><h3>Только внутри одного сегмента</h3><p>Переходов: <strong>@@WITHIN_PAIRS@@</strong>; скачков &gt; 10 м: <strong>@@WITHIN_JUMPS@@</strong>; скоростей &gt; 50 м/с: <strong>@@WITHIN_SPEEDS@@</strong>. Максимальный скачок: <strong>@@WITHIN_MAX_JUMP@@ м</strong>; максимальная скорость: <strong>@@WITHIN_MAX_SPEED@@ м/с</strong>. Исключено переходов через сброс или смену сегмента/режима: <strong>@@BOUNDARIES@@</strong>. Сегментов с сохранённым номером: <strong>@@SEGMENTS@@</strong>; записей с явным сбросом: <strong>@@RESETS@@</strong>.</p><p class="note">Смена режима корректирует систему предположений о положении, поэтому скачок на границе не обязательно является физическим движением. Он сохранён в общей диагностике. Внутренняя диагностика исключает такие границы, включая сбросы в промежуточных потерянных записях. Пороговые значения служат диагностикой, а не критерием точности. Скорость рассчитана по реальному source_time (при его отсутствии — source_frame / FPS), включая переходы через пробелы. Пары с неположительным Δt не получают скорость. Таких пар в общей диагностике: @@BAD_DT@@. Размах исходных кадров: @@SPAN@@ с. Длительности состояний в JSON — число записей / FPS; они не заполняют пропущенные интервалы.</p><p>Некорректных строк эталона: <strong>@@BAD_TRUTH@@</strong>. Пропущено некорректных строк журнала: <strong>@@BAD_TRACE@@</strong>. Дубликаты кадров не увеличивают покрытие; сохраняется первая корректная запись.</p><details><summary>Диагностика строк</summary><pre id="diagnostics" style="white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px"></pre></details></section>
<footer>Сохранённые файлы: <a href="evaluation.json">evaluation.json</a> · <a href="errors.csv">errors.csv</a>. Полный набор данных графиков встроен в HTML. Для просмотра не нужен сервер или подключение к интернету.</footer>
</main><script type="application/json" id="report-data">@@DATA@@</script>
<script>
'use strict';
const payload=JSON.parse(document.getElementById('report-data').textContent), points=payload.points, result=payload.result;
const colors={truth:'#52616c',observed:'#1265b0',predicted:'#b66b09'};
const visible={truth:true,observed:true,predicted:true};
const start=points.length?points[0].time_s:0, end=points.length?points[points.length-1].time_s:1;
let view={lo:start,hi:end>start?end:start+1};
const first=document.getElementById('time-start'), last=document.getElementById('time-end');
function inputs(){first.value=view.lo.toFixed(3);last.value=view.hi.toFixed(3)}inputs();
const fmt=(n,d=2)=>n===null||!Number.isFinite(n)?'—':n.toLocaleString('ru-RU',{maximumFractionDigits:d});
const pred=p=>p.state==='coasting'||p.state==='predicted';
function seriesPoint(p,s){if(s==='truth')return p.truth_m;if(s==='observed'&&p.state==='observed')return p.estimate_m;if(s==='predicted'&&pred(p))return p.estimate_m;return null}
function bounds(values){if(!values.length)return[-1,1];let lo=Infinity,hi=-Infinity;for(const value of values){lo=Math.min(lo,value);hi=Math.max(hi,value)}const pad=Math.max((hi-lo)*.08,Math.abs(lo)*.001,.5);return[lo-pad,hi+pad]}
function nice(n){const magnitude=10**Math.floor(Math.log10(n||1)),f=n/magnitude;return(f<=1?1:f<=2?2:f<=5?5:10)*magnitude}
function ticks(lo,hi){const step=nice((hi-lo)/5),out=[];for(let n=Math.ceil(lo/step)*step;n<=hi+step*1e-8;n+=step)out.push(n);return out}
const charts=[...document.querySelectorAll('.chart')].map(el=>({el,canvas:el.querySelector('canvas'),tip:el.querySelector('.tooltip'),kind:el.dataset.kind,axis:Number(el.dataset.axis||0)}));
function draw(chart){
 const canvas=chart.canvas,box=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
 canvas.width=Math.round(box.width*dpr);canvas.height=Math.round(box.height*dpr);
 const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const w=box.width,h=box.height,m={l:76,r:18,t:18,b:38};
 const rows=points.filter(p=>p.time_s>=view.lo&&p.time_s<=view.hi),series=[];
 for(const s of ['truth','observed','predicted']){if(!visible[s]||(s==='truth'&&chart.kind!=='axis'&&chart.kind!=='xy'))continue;
  const values=rows.map(p=>{const xyz=seriesPoint(p,s);if(!xyz)return{p,v:null};if(chart.kind==='xy')return{p,v:[xyz[0],xyz[1]]};if(chart.kind==='axis')return{p,v:[p.time_s,xyz[chart.axis]]};const e=chart.kind==='error'?p.three_d_error_m:p.horizontal_error_m;return{p,v:e===null?null:[p.time_s,e]}});series.push({name:s,values});}
 let xs=[],ys=[];for(const s of series)for(const item of s.values)if(item.v){xs.push(item.v[0]);ys.push(item.v[1])}
 let [xmin,xmax]=chart.kind==='xy'?bounds(xs):[view.lo,view.hi], [ymin,ymax]=bounds(ys);
 if(chart.kind==='error'||chart.kind==='horizontal')ymin=0;
 if(chart.kind==='xy'){const scale=Math.max((xmax-xmin)/(w-m.l-m.r),(ymax-ymin)/(h-m.t-m.b));const xc=(xmin+xmax)/2,yc=(ymin+ymax)/2;xmin=xc-scale*(w-m.l-m.r)/2;xmax=xc+scale*(w-m.l-m.r)/2;ymin=yc-scale*(h-m.t-m.b)/2;ymax=yc+scale*(h-m.t-m.b)/2;}
 const sx=x=>m.l+(x-xmin)/(xmax-xmin)*(w-m.l-m.r),sy=y=>h-m.b-(y-ymin)/(ymax-ymin)*(h-m.t-m.b);
 chart.geometry={xmin,xmax,ymin,ymax,sx,sy,m,w,h,rows};
 ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);ctx.font='11px system-ui';ctx.lineWidth=1;
 for(const y of ticks(ymin,ymax)){const py=sy(y);ctx.strokeStyle='#e5ebef';ctx.beginPath();ctx.moveTo(m.l,py);ctx.lineTo(w-m.r,py);ctx.stroke();ctx.fillStyle='#536978';ctx.textAlign='right';ctx.fillText(fmt(y,1),m.l-9,py+4)}
 for(const x of ticks(xmin,xmax)){const px=sx(x);ctx.strokeStyle='#edf1f3';ctx.beginPath();ctx.moveTo(px,m.t);ctx.lineTo(px,h-m.b);ctx.stroke();ctx.fillStyle='#536978';ctx.textAlign='center';ctx.fillText(fmt(x,1),px,h-m.b+20)}
 ctx.textAlign='right';ctx.fillText(chart.kind==='xy'?'X, м':'Время, с',w-m.r,h-3);
 if(chart.kind==='xy'){ctx.textAlign='left';ctx.fillText('Y, м',m.l,m.t-5)}
 ctx.save();ctx.beginPath();ctx.rect(m.l,m.t,w-m.l-m.r,h-m.t-m.b);ctx.clip();
 if(chart.kind!=='xy'){ctx.strokeStyle='#b5bec5';ctx.lineWidth=1;ctx.setLineDash([3,5]);for(const p of rows)if(p.mode_reset){const x=sx(p.time_s);ctx.beginPath();ctx.moveTo(x,m.t);ctx.lineTo(x,h-m.b);ctx.stroke()}}
 for(const s of series){ctx.strokeStyle=colors[s.name];ctx.fillStyle=colors[s.name];ctx.lineWidth=s.name==='truth'?1.5:2;ctx.setLineDash(s.name==='predicted'?[5,4]:[]);let previous=null;
  for(const item of s.values){if(!item.v){previous=null;continue}const x=sx(item.v[0]),y=sy(item.v[1]);if(previous&&item.p.frame===previous.p.frame+1&&(s.name==='truth'||item.p.connect_estimate)){ctx.beginPath();ctx.moveTo(sx(previous.v[0]),sy(previous.v[1]));ctx.lineTo(x,y);ctx.stroke()}ctx.beginPath();ctx.arc(x,y,s.name==='truth'?1:2,0,Math.PI*2);ctx.fill();previous=item;}}
 ctx.restore();ctx.setLineDash([]);
 if(!ys.length){ctx.fillStyle='#536978';ctx.textAlign='center';ctx.fillText('Нет доступных координат в выбранном интервале',w/2,h/2)}
}
function redraw(){for(const chart of charts)draw(chart)}
let pending=false;window.addEventListener('resize',()=>{if(!pending){pending=true;requestAnimationFrame(()=>{pending=false;redraw()})}});
for(const input of document.querySelectorAll('[data-series]'))input.addEventListener('change',()=>{visible[input.dataset.series]=input.checked;redraw()});
document.getElementById('apply').addEventListener('click',()=>{const lo=Number(first.value),hi=Number(last.value);if(Number.isFinite(lo)&&Number.isFinite(hi)&&hi>lo){view={lo,hi};redraw()}});
document.getElementById('reset').addEventListener('click',()=>{view={lo:start,hi:end>start?end:start+1};inputs();redraw()});
for(const chart of charts){chart.canvas.addEventListener('pointermove',event=>{const g=chart.geometry;if(!g||!g.rows.length)return;const box=chart.canvas.getBoundingClientRect(),x=event.clientX-box.left,y=event.clientY-box.top;let best=null,distance=Infinity;for(const p of g.rows){let d;if(chart.kind==='xy'){d=Infinity;for(const s of ['truth','observed','predicted']){if(!visible[s])continue;const v=seriesPoint(p,s);if(v)d=Math.min(d,Math.hypot(g.sx(v[0])-x,g.sy(v[1])-y))}}else d=Math.abs(g.sx(p.time_s)-x);if(d<distance){distance=d;best=p}}if(!best)return;const xyz=v=>v?v.map(n=>fmt(n)).join(' / '):'нет';chart.tip.textContent='Кадр '+best.frame+' · '+fmt(best.time_s,3)+' с\nСостояние: '+best.state+'\nМетод: '+(best.method||'—')+'\nРежим: '+(best.mode||'—')+' · сегмент '+(best.segment_id===null?'—':best.segment_id)+(best.mode_reset?' · сброс':'')+'\nЭталон X / Y / Z: '+xyz(best.truth_m)+' м\nОценка X / Y / Z: '+xyz(best.estimate_m)+' м\nXY ошибка: '+fmt(best.horizontal_error_m)+' м · 3D: '+fmt(best.three_d_error_m)+' м';chart.tip.style.display='block';chart.tip.style.left=Math.max(0,Math.min(x+12,box.width-320))+'px';chart.tip.style.top=Math.max(24,y-80)+'px'});chart.canvas.addEventListener('pointerleave',()=>{chart.tip.style.display='none'})}
const bar=document.getElementById('coverage-bar');for(const [key,klass]of [['observed_of_processed','observed-bar'],['predicted_of_processed','predicted-bar'],['lost_of_processed','lost-bar']]){const part=document.createElement('span');part.className=klass;part.style.width=100*(result.coverage[key]||0)+'%';bar.appendChild(part)}
document.getElementById('diagnostics').textContent=JSON.stringify({invalid_truth_rows:result.invalid_truth_rows,invalid_observation_rows:result.invalid_observation_rows},null,2);
redraw();
</script></body></html>'''
    replacements = {
        'RUN': html.escape(run_name), 'FPS': _number(result['source_fps']),
        'STEPS': str(result['processed_steps']), 'TRUTH': str(result['truth_frames']),
        'COV': _percent(coverage['observed_of_truth']), 'MATCHED': str(result['matched_observed_steps']),
        'XY': _number(result['horizontal_rmse_m']), 'XYZ': _number(result['three_d_rmse_m']),
        'Z': _number(result['axis_mae_m']['z']), 'OBS': str(result['observed_steps']),
        'PRED': str(result['predicted_steps']), 'LOST': str(result['lost_steps']),
        'AVAILABLE': str(result['estimate_steps']), 'BAD_EST': str(result['invalid_estimate_steps']),
        'OBS_PROC': _percent(coverage['observed_of_processed']),
        'PRED_PROC': _percent(coverage['predicted_of_processed']),
        'PRED_TRUTH': _percent(coverage['predicted_of_truth']),
        'LOST_PROC': _percent(coverage['lost_of_processed']),
        'ALL_PROC': _percent(coverage['estimate_of_processed']),
        'ALL_TRUTH': _percent(coverage['estimate_of_truth']),
        'PROC_TRUTH': _percent(coverage['processed_of_truth']),
        'QUALITY': _quality_rows(result), 'AXES': axis_rows,
        'MODES': _group_table(result['metrics_by_mode']),
        'METHOD_QUALITY': _group_table(result['metrics_by_method']),
        'METHODS': method_rows or '<tr><td colspan="2">Нет доступных оценок</td></tr>',
        'BASELINE': _baseline_section(result), 'PAIRS': str(diagnostics['pair_count']),
        'JUMPS': str(diagnostics['large_jump_count']), 'SPEEDS': str(diagnostics['large_speed_count']),
        'MAX_JUMP': _number(diagnostics['max_jump_m']), 'MAX_SPEED': _number(diagnostics['max_speed_m_s']),
        'BAD_DT': str(diagnostics['nonpositive_dt_pairs']), 'SPAN': _number(result['source_span_s']),
        'WITHIN_PAIRS': str(within['pair_count']), 'WITHIN_JUMPS': str(within['large_jump_count']),
        'WITHIN_SPEEDS': str(within['large_speed_count']),
        'WITHIN_MAX_JUMP': _number(within['max_jump_m']),
        'WITHIN_MAX_SPEED': _number(within['max_speed_m_s']),
        'BOUNDARIES': str(within['excluded_boundary_pairs']),
        'SEGMENTS': str(result['segment_count']), 'RESETS': str(result['segment_reset_steps']),
        'BAD_TRUTH': str(len(result['invalid_truth_rows'])),
        'BAD_TRACE': str(len(result['invalid_observation_rows'])), 'DATA': data,
    }
    # A single substitution pass keeps user-supplied strings from becoming template tokens.
    import re
    document = re.sub(r'@@([A-Z_]+)@@', lambda match: replacements[match.group(1)], template)
    path.write_text(document, encoding='utf-8')
