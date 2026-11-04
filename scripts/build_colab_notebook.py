"""Populate the scaffolded Colab notebook; no outputs or credentials are embedded."""
import json
from pathlib import Path
from textwrap import dedent

path=Path(__file__).resolve().parents[1]/'notebooks/peleng_colab.ipynb'
nb=json.loads(path.read_text(encoding='utf-8'))
cells=[]
def md(text): cells.append({'cell_type':'markdown','metadata':{},'source':dedent(text).strip().splitlines(True)})
def code(text): cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[], 'source':dedent(text).strip().splitlines(True)})

md('''# Пеленг: трекинг на GPU и сравнение моделей
Все вычисления выполняются в Colab. Локальная видеокарта не нужна.

1. Подключить GPU T4 и выполнить установку.
2. Разрешить подключение Drive; данные уже лежат в `Мой диск/Peleng-Colab`.
3. Выбрать набор и веса, выполнить обработку. Кадры появляются прямо в ноутбуке.
4. Результаты автоматически сохраняются в Drive, включая CSV в метрах и HTML.

**Целевая точка: геометрический центр грузовика.** Алгоритм пока приближает его по изображениям; точка в эталоне организаторов не подтверждена. Старый результат с `front` нельзя напрямую объявлять базой для `center`.

Это перенос текущего алгоритма, а не заявление об исправлении его геометрии. Обучение ниже отключено по умолчанию.''')
code('''# @title 1. Получить код и установить зависимости
from pathlib import Path
import os, sys, subprocess, json, shutil, hashlib
REPO = Path('/content/peleng')
if not REPO.exists():
    subprocess.run(['git','clone','https://github.com/petrussia/3d-track-detection-and-tracking.git',str(REPO)],check=True)
else:
    subprocess.run(['git','-C',str(REPO),'pull','--ff-only'],check=True)
os.chdir(REPO)
install = subprocess.run([sys.executable,'-m','pip','install','-q','-r','requirements-colab.txt'],capture_output=True,text=True)
if install.returncode:
    raise RuntimeError(install.stdout + install.stderr)
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
import torch
assert torch.cuda.is_available(), 'Включите GPU: Среда выполнения → Сменить среду выполнения → T4'
print('GPU:',torch.cuda.get_device_name(0))
print('VRAM, GB:',round(torch.cuda.get_device_properties(0).total_memory/1024**3,1))
print('Commit:',subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
''')
code('''# @title 2. Подключить Drive и распаковать данные
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = Path('/content/drive/MyDrive/Peleng-Colab')
assert (DRIVE_ROOT/'bundle-manifest.json').exists(), f'Не найдены данные: {DRIVE_ROOT}'
from peleng.colab_support import unpack_bundles, discover_datasets
DATASETS_ROOT = unpack_bundles(DRIVE_ROOT, Path('/content/peleng-data'))
DATASETS = discover_datasets(DATASETS_ROOT)
(REPO/'models').mkdir(exist_ok=True)
source_weights = DRIVE_ROOT/'yolo11s.pt'
assert source_weights.exists(), 'В папке Drive нужны исходные веса yolo11s.pt'
assert hashlib.sha256(source_weights.read_bytes()).hexdigest() == '85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5', 'Не совпала контрольная сумма весов'
shutil.copy2(source_weights, REPO/'models/yolo11s.pt')
print('Доступные наборы:',list(DATASETS))
''')
code('''# @title 3. Настройки прогона
DATASET = 'organizer' # @param ['organizer','Peleng-01-valley','Peleng-02-occlusion','Peleng-03-moving','Peleng-04-stop-go','Peleng-05-reverse']
MODEL = 'yolo11s.pt' # @param ['yolo11n.pt','yolo11s.pt','yolo11m.pt','custom']
CUSTOM_WEIGHTS = '' # @param {type:'string'}
IMAGE_SIZE = 960 # @param [640,960,1280] {type:'raw'}
REFERENCE = 'center' # @param ['center','front']
assert DATASET in DATASETS
if MODEL == 'custom':
    WEIGHTS=Path(CUSTOM_WEIGHTS)
    assert WEIGHTS.is_file(), 'Укажите полный путь к best.pt'
else:
    WEIGHTS=REPO/'models'/MODEL
    if not WEIGHTS.exists():
        from ultralytics import YOLO
        previous=Path.cwd()
        try:
            os.chdir(REPO/'models'); downloaded=YOLO(MODEL); del downloaded
        finally: os.chdir(previous)
    assert WEIGHTS.exists()
from peleng.inputs import inspect_dataset
print(json.dumps(inspect_dataset(DATASETS[DATASET]['videos']),indent=2,ensure_ascii=False))
print('Целевая точка:',REFERENCE,'; эталон:',DATASETS[DATASET]['truth_reference'])
''')
code('''# @title 4. Запустить обработку и просмотр камер
from peleng.colab_support import run_notebook
RUN_DIR = run_notebook(DATASETS[DATASET], WEIGHTS, REPO/'outputs', DRIVE_ROOT/'runs',
                       imgsz=IMAGE_SIZE, reference=REFERENCE)
print('Готово:',RUN_DIR)
print('Копия в Drive:',DRIVE_ROOT/'runs'/RUN_DIR.name)
''')
md('''## Сравнение моделей
Для следующей модели повторите ячейки 3 и 4. На одном GPU выполняйте прогоны последовательно. Сравнивайте строки с одинаковым набором и опорной точкой. `within_5m_pct` учитывает все корректные кадры эталона внутри видео; отсутствие координат не считается успехом.''')
code('''# @title 5. Сводная таблица всех прогонов
from peleng.colab_support import compare_runs
from IPython.display import display
comparison=compare_runs(DRIVE_ROOT/'runs')
display(comparison)
comparison.to_csv(DRIVE_ROOT/'comparison.csv',index=False)
''')
code('''# @title 6. Скачать последний результат (по желанию)
DOWNLOAD = False # @param {type:'boolean'}
if DOWNLOAD:
    from google.colab import files
    archive=shutil.make_archive(str(RUN_DIR), 'zip', RUN_DIR)
    files.download(archive)
else:
    print('Результаты уже сохранены в Drive. Для скачивания включите DOWNLOAD.')
''')
md('''## Дополнительный эксперимент: дообучение детектора
Не запускается автоматически. Обучаем один класс `truck` по рамкам синтетики.
Train: valley/occlusion/moving. Validation: stop-go. Test: reverse.
Оригинальный набор организаторов не входит в обучение.

Это обучение детекции рамки, **не** обучение проекции 3D-центра. После обучения отдельно проверяем 3D-траекторию. Первый запуск подготовки данных извлекает кадры на локальный диск Colab.''')
code('''# @title 7. Подготовить и обучить детектор (опционально)
TRAIN = False # @param {type:'boolean'}
TRAIN_MODEL = 'yolo11s.pt' # @param ['yolo11s.pt','yolo11m.pt']
EPOCHS = 30 # @param {type:'integer'}
BATCH = 4 # @param {type:'integer'}
TRAIN_IMAGE_SIZE = 960 # @param [640,960] {type:'raw'}
if TRAIN:
    from peleng.colab_support import prepare_detection_data
    from ultralytics import YOLO
    dataset_yaml=prepare_detection_data(DATASETS_ROOT,Path('/content/peleng-training'))
    base=REPO/'models'/TRAIN_MODEL
    training_model=YOLO(str(base) if base.exists() else TRAIN_MODEL)
    training_model.train(data=str(dataset_yaml),epochs=EPOCHS,time=0.75,
        imgsz=TRAIN_IMAGE_SIZE,batch=BATCH,device=0,amp=True,workers=2,cache=False,
        patience=8,seed=42,project=str(DRIVE_ROOT/'training'),name=TRAIN_MODEL.replace('.pt',''),exist_ok=False)
    BEST_WEIGHTS=Path(training_model.trainer.best)
    print('Для тестирования выберите MODEL=custom, CUSTOM_WEIGHTS=',BEST_WEIGHTS)
else:
    print('Обучение выключено. Для запуска включите TRAIN.')
''')
md('''## Выходные файлы и ограничения
- `trajectory.csv`: совместимый исходный экспорт, координаты в сантиметрах.
- `trajectory_m.csv`: основной удобный экспорт X/Y/Z в метрах.
- `observations.jsonl`: покадровые измерения, состояния и способы оценки.
- `scores.json`, `evaluation.json`, `errors.csv`, `report.html`: оценка после прогона.
- `environment.json`, `pip-freeze.txt`: GPU, веса, версия кода и окружение.

У исходного CSV два повреждённых значения. Они исключаются только из оценки. У синтетики эталон — геометрический центр; у исходного набора смысл точки требует проверки. Высота Z=0 и сбросы сегментов пока остаются ограничениями алгоритма. Если среда будет принудительно удалена, текущий незавершённый прогон может не успеть сохраниться; завершённые прогоны находятся в Drive.''')
for i,cell in enumerate(cells): cell['id']=f'peleng-{i:02}'
nb['cells']=cells
nb['metadata'].update({'accelerator':'GPU','colab':{'name':'Хакатон Пеленг.ipynb','provenance':[],'gpuType':'T4'},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
print(path)
