"""FastAPI app: JSON API under /api plus the static web UI."""
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .engine import SearchEngine


def create_app(engine: Optional[SearchEngine] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or SearchEngine()
        yield

    app = FastAPI(title="NewsLens", description="Hybrid semantic search over news articles", lifespan=lifespan)

    def eng(request: Request) -> SearchEngine:
        return request.app.state.engine

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/stats")
    def stats(request: Request):
        return eng(request).stats()

    @app.get("/api/search")
    def search(request: Request,
               q: str = Query(..., min_length=1, max_length=300),
               mode: Literal["hybrid", "semantic", "keyword"] = "hybrid",
               k: int = Query(10, ge=1, le=50),
               category: Optional[str] = None):
        try:
            return eng(request).search(q.strip(), mode, k, category)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/compare")
    def compare(request: Request,
                q: str = Query(..., min_length=1, max_length=300),
                k: int = Query(10, ge=1, le=50),
                category: Optional[str] = None):
        try:
            return eng(request).compare(q.strip(), k, category)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/similar/{doc_id}")
    def similar(request: Request, doc_id: int, k: int = Query(10, ge=1, le=50)):
        try:
            return eng(request).similar(doc_id, k)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(config.FRONTEND_DIR / "index.html")

    return app


app = create_app()
