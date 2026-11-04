from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

from .engine import Engine


def create_app(engine: Engine):
    @asynccontextmanager
    async def lifespan(app):
        yield
        engine.stop()
        engine.wait()

    app = FastAPI(title='Peleng multi-camera tracking',lifespan=lifespan)

    @app.get('/')
    def index():
        return FileResponse(Path(__file__).parent.parent/'static/index.html')

    @app.get('/api/state')
    def state():
        return engine.snapshot()

    @app.post('/api/start')
    def start():
        if not engine.start():
            raise HTTPException(409,'A run is already active')
        return {'started':True}

    @app.post('/api/stop')
    def stop():
        engine.stop()
        return {'stopping':True}

    @app.get('/api/result/{filename}')
    def result(filename: str):
        if filename not in {'trajectory.csv', 'summary.json', 'config.json',
                            'report.html', 'evaluation.json', 'errors.csv', 'preview.jpg'}:
            raise HTTPException(404, 'Unknown result')
        summary = engine.snapshot().get('summary')
        if not summary:
            raise HTTPException(404, 'Run has not finished')
        path = Path(summary['run_dir'])/filename
        if not path.is_file():
            raise HTTPException(404, 'Result not generated yet')
        return FileResponse(path, headers={'Cache-Control':'no-store'})

    @app.get('/api/frame/{camera_id}.jpg')
    def frame(camera_id:int):
        image = engine.image(camera_id)
        if image is None:
            raise HTTPException(404,'No frame yet')
        return Response(image,media_type='image/jpeg',headers={'Cache-Control':'no-store'})

    return app
