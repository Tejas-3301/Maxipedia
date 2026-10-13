"""Single-process server: API under /api and the built workspace (frontend/dist) at /. No Node/Vite needed, so this
is also the fallback for a failed live demo (see scripts/demo_serve.py).

    python -m uvicorn backend.serve:app --port 8000      # after: cd frontend && npm run build
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.config import ROOT
from backend.main import app as api
from backend.main import startup

DIST = ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_: FastAPI):
    startup()  # mounted sub-apps do not run their own lifespan
    yield


app = FastAPI(title="Media Forensics Workspace", lifespan=lifespan)
app.mount("/api", api)
if (DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
else:
    @app.get("/", response_class=HTMLResponse)
    def no_ui() -> str:
        return ("<h3>Frontend not built.</h3><p>Run <code>cd frontend && npm install && npm run build</code>, "
                "or use the API at <a href='/api/docs'>/api/docs</a>.</p>")
