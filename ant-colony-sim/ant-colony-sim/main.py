"""Ant Colony Simulation — uvicorn entrypoint."""
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from ant_sim.api import router, auth_router

app = FastAPI(title="Ant Colony Simulation", version="3.0")
app.include_router(auth_router)  # unprotected: /api/auth/check
app.include_router(router)       # protected by dependency

# Parse --api-key from argv (optional)
app.state.api_key = None
for i, arg in enumerate(sys.argv):
    if arg == "--api-key" and i + 1 < len(sys.argv):
        app.state.api_key = sys.argv[i + 1]
        print(f"[auth] API key protection enabled")
        break

# Serve static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(static_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
