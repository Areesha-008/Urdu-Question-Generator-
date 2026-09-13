import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.inference import Generator
from app.tokenizer import mark_answer
from config import ROOT, RUN_DIR


class GenerateRequest(BaseModel):
    sentence: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=500)


def create_app(run_dir=None):
    run_dir = Path(run_dir or os.environ.get('QG_RUN_DIR', RUN_DIR))
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.generator = Generator(run_dir) if (run_dir / 'best.pt').exists() else None
        yield

    app = FastAPI(title='Urdu Question Generator', lifespan=lifespan)
    frontend = ROOT / 'frontend'
    app.mount('/static', StaticFiles(directory=frontend), name='static')

    @app.get('/')
    def index():
        return FileResponse(frontend / 'index.html', headers={'Cache-Control': 'no-store'})

    @app.get('/health')
    def health():
        return {'ready': app.state.generator is not None}

    @app.post('/generate')
    def generate(request: GenerateRequest):
        try:
            source = mark_answer(request.sentence, request.answer)
            if app.state.generator is None:
                raise HTTPException(503, 'The model is not available.')
            with lock:
                return {'source': source, 'greedy': app.state.generator.generate(source),
                    'beam': app.state.generator.generate(source, beam_size=3)}
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    return app


app = create_app()
