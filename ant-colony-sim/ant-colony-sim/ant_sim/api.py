"""FastAPI routes — all endpoints per spec."""
from __future__ import annotations

import asyncio
import io
import json
import shutil
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Depends
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse

from .analysis import AnalysisEngine
from .metrics import compute_batch_scores
from .queue import SimQueue, SimStatus
from .validation import validate_config, load_agent_class, expand_batch


# --- Auth dependency (no BaseHTTPMiddleware) ---

async def check_api_key(request: Request):
    """Dependency: checks api key via header or query param. Skips if no key configured."""
    key = getattr(request.app.state, "api_key", None)
    if key is None:
        return
    auth = request.headers.get("authorization", "")
    if auth == f"Bearer {key}":
        return
    if request.query_params.get("api_key") == key:
        return
    raise HTTPException(status_code=401, detail="invalid api key")


# Unprotected router (auth check endpoint)
auth_router = APIRouter()

@auth_router.get("/api/auth/check")
async def auth_check(request: Request):
    key = getattr(request.app.state, "api_key", None)
    return {"required": key is not None}


# Protected router (all other endpoints)
router = APIRouter(dependencies=[Depends(check_api_key)])

DATA_DIR = Path("data/simulations")
DATA_DIR.mkdir(parents=True, exist_ok=True)

sim_queue = SimQueue(DATA_DIR)
analysis_engine = AnalysisEngine(DATA_DIR)


def _get_entry(sim_id: str) -> SimEntry | None:
    """Lookup by id first, then by name, then lazy-load from disk."""
    entry = sim_queue.entries.get(sim_id)
    if entry:
        return entry
    # Try disk (sim_id might be the folder name)
    return sim_queue.load_from_disk(sim_id)


from .queue import SimEntry  # noqa: E402

# --- Config ---

@router.post("/api/config/validate")
async def validate_config_endpoint(request: Request):
    body = await request.json()
    errors = validate_config(body)
    return {"valid": len(errors) == 0, "errors": errors}


# --- Simulation ---

@router.post("/api/simulation/start")
async def start_simulation(request: Request):
    body = await request.json()
    config = body.get("config", body)
    agent_code = body.get("agent_code", "")

    errors = validate_config(config)
    if errors:
        raise HTTPException(400, detail={"errors": errors})

    agent_class = None
    if agent_code:
        cls, errs = load_agent_class(agent_code)
        if errs:
            raise HTTPException(400, detail={"errors": errs})
        agent_class = cls

    sim_id = sim_queue.add(config, agent_class, agent_code)
    asyncio.create_task(sim_queue.run_next())

    return {"sim_id": sim_id, "name": config.get("name", sim_id)}


@router.post("/api/simulation/start-batch")
async def start_batch(request: Request):
    body = await request.json()

    if "batch" in body:
        configs = expand_batch(body["batch"])
    elif "configs" in body:
        configs = body["configs"]
    else:
        raise HTTPException(400, detail="Need 'batch' or 'configs'")

    agent_code = body.get("agent_code", "")
    agent_class = None
    if agent_code:
        cls, errs = load_agent_class(agent_code)
        if errs:
            raise HTTPException(400, detail={"errors": errs})
        agent_class = cls

    sim_ids = []
    for cfg in configs:
        errors = validate_config(cfg)
        if errors:
            continue
        sid = sim_queue.add(cfg, agent_class, agent_code)
        sim_ids.append(sid)

    # Start as many as possible
    for _ in range(min(len(sim_ids), sim_queue.MAX_CONCURRENT)):
        asyncio.create_task(sim_queue.run_next())

    return {"playlist_id": "batch_" + (sim_ids[0] if sim_ids else "empty"), "sim_ids": sim_ids}


@router.get("/api/simulation/{sim_id}/stream")
async def stream_simulation(sim_id: str):
    entry = _get_entry(sim_id)
    if not entry:
        raise HTTPException(404, detail="Simulation not found")

    async def event_stream():
        # Disk-loaded completed sims: stream from store
        if entry.status == SimStatus.COMPLETED and not entry.tick_states and entry.store:
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(None, list, entry.store.iter_chunks_raw())
            for raw in chunks:
                yield f"data: {raw}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'status': 'completed'})}\n\n"
            return

        # Live / in-memory sims: stream as ticks arrive
        sent = 0
        while True:
            if sent < len(entry.tick_states):
                for ts in entry.tick_states[sent:]:
                    yield f"data: {json.dumps(ts)}\n\n"
                    sent += 1
            elif entry.status in (SimStatus.COMPLETED, SimStatus.FAILED):
                yield f"data: {json.dumps({'event': 'done', 'status': entry.status.value})}\n\n"
                break
            else:
                await asyncio.sleep(0.05)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/api/simulation/{sim_id}/tick/{n}")
async def get_tick(sim_id: str, n: int):
    entry = _get_entry(sim_id)
    if not entry:
        raise HTTPException(404)
    if n < len(entry.tick_states):
        return entry.tick_states[n]
    # Try disk
    if entry.store:
        td = entry.store.get_tick(n)
        if td:
            return td
    raise HTTPException(404, detail=f"Tick {n} not found")


@router.get("/api/simulation/{sim_id}/status")
async def sim_status(sim_id: str):
    status = sim_queue.get_status(sim_id)
    if not status:
        raise HTTPException(404)
    entry = _get_entry(sim_id)
    if entry:
        status["config"] = entry.config
    return status


@router.get("/api/simulation/{sim_id}/config")
async def sim_config(sim_id: str):
    entry = _get_entry(sim_id)
    if not entry:
        raise HTTPException(404)
    return entry.config


@router.post("/api/simulation/{sim_id}/pause")
async def pause_sim(sim_id: str):
    ok = sim_queue.pause(sim_id)
    return {"paused": ok}


@router.post("/api/simulation/{sim_id}/resume")
async def resume_sim(sim_id: str):
    entry = _get_entry(sim_id)
    if not entry:
        raise HTTPException(404)
    if entry.engine:
        entry.engine.paused = False
        entry.status = SimStatus.RUNNING
        asyncio.create_task(sim_queue.run_next())
    return {"resumed": True}


@router.post("/api/simulation/{sim_id}/continue")
async def continue_sim(sim_id: str, request: Request):
    body = await request.json()
    extra_ticks = body.get("ticks", 100)

    entry = _get_entry(sim_id)
    if not entry:
        raise HTTPException(404)
    if entry.status != SimStatus.COMPLETED:
        raise HTTPException(400, detail="Sim not completed")

    # Modify config and re-queue
    new_config = dict(entry.config)
    new_config["simulation"]["max_ticks"] = entry.config["simulation"].get("max_ticks", 1000) + extra_ticks
    new_config["name"] = entry.name + f"_+{extra_ticks}"

    sim_id_new = sim_queue.add(new_config, entry.agent_class, entry.agent_code)
    asyncio.create_task(sim_queue.run_next())

    return {"sim_id": sim_id_new, "extra_ticks": extra_ticks}


@router.get("/api/simulations")
async def list_simulations(search: str = ""):
    sims = sim_queue.list_all()
    known_names = {s.get("name") for s in sims}

    # Load completed sims from disk that aren't in memory
    if DATA_DIR.exists():
        for d in DATA_DIR.iterdir():
            if not d.is_dir() or d.name in known_names:
                continue
            cfg_path = d / "config.json"
            if not cfg_path.exists():
                continue
            try:
                cfg = json.loads(cfg_path.read_text())
                sims.append({
                    "id": d.name,
                    "name": cfg.get("name", d.name),
                    "status": "completed",
                    "config": cfg,
                    "ticks": cfg.get("simulation", {}).get("max_ticks", 0),
                })
            except (json.JSONDecodeError, OSError):
                continue

    if search:
        sims = [s for s in sims if search.lower() in s.get("name", "").lower()]
    return sims


@router.get("/api/simulation/{sim_id}/metrics")
async def get_metrics(sim_id: str):
    entry = _get_entry(sim_id)
    if not entry:
        raise HTTPException(404)
    if not entry.metrics:
        raise HTTPException(404, detail="Metrics not available yet")
    return entry.metrics


@router.get("/api/simulation/{sim_id}/download")
async def download_sim(sim_id: str):
    entry = _get_entry(sim_id)
    if not entry:
        raise HTTPException(404)

    sim_dir = DATA_DIR / entry.name
    if not sim_dir.exists():
        raise HTTPException(404, detail="Simulation data not found on disk")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sim_dir.rglob("*"):
            if fp.is_file():
                zf.write(fp, fp.relative_to(sim_dir))
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{entry.name}.sim.zip"'},
    )


@router.post("/api/simulation/upload")
async def upload_sim(file: UploadFile = File(...)):
    content = await file.read()
    buf = io.BytesIO(content)

    try:
        with zipfile.ZipFile(buf, "r") as zf:
            # Extract name from config.json
            if "config.json" in zf.namelist():
                cfg = json.loads(zf.read("config.json"))
                name = cfg.get("name", file.filename.replace(".sim.zip", ""))
            else:
                name = file.filename.replace(".sim.zip", "")

            target = DATA_DIR / name
            target.mkdir(parents=True, exist_ok=True)
            zf.extractall(target)

        return {"name": name, "uploaded": True}
    except zipfile.BadZipFile:
        raise HTTPException(400, detail="Invalid zip file")


# --- Analysis ---

@router.post("/api/analysis/run")
async def run_analysis(request: Request):
    body = await request.json()
    sim_id = body.get("sim_id")
    sim_ids = body.get("sim_ids", [])
    plot_type = body.get("type", "food_time")
    custom_code = body.get("code")

    # Normalize: single sim_id → list, deduplicated
    if sim_id and sim_id not in sim_ids:
        sim_ids.insert(0, sim_id)

    entry = _get_entry(sim_ids[0]) if sim_ids else None

    if plot_type == "custom" and custom_code:
        contexts = []
        for sid in sim_ids:
            e = _get_entry(sid)
            if e:
                contexts.append({
                    "metrics": e.metrics or {},
                    "ticks": e.tick_states,
                    "config": e.config,
                })
        plot_id, _ = analysis_engine.run_custom(custom_code, contexts)
    elif entry and entry.metrics:
        m = entry.metrics
        if plot_type == "food_time":
            plot_id, _ = analysis_engine.builtin_food_over_time(m)
        elif plot_type == "alive_time":
            plot_id, _ = analysis_engine.builtin_alive_over_time(m)
        elif plot_type == "steps_food":
            plot_id, _ = analysis_engine.builtin_steps_to_food(m)
        elif plot_type == "batch_scores":
            all_m = [e.metrics for e in sim_queue.entries.values() if e.metrics]
            scores = compute_batch_scores(all_m)
            plot_id, _ = analysis_engine.builtin_batch_scores([s.to_dict() for s in scores])
        else:
            raise HTTPException(400, detail=f"Unknown plot type: {plot_type}")
    else:
        raise HTTPException(400, detail="No metrics available")

    return {"plot_id": plot_id}


@router.get("/api/analysis/{plot_id}/plot.png")
async def get_plot_png(plot_id: str):
    p = analysis_engine.get_plot_path(plot_id, "png")
    if not p:
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@router.get("/api/analysis/{plot_id}/plot.svg")
async def get_plot_svg(plot_id: str):
    p = analysis_engine.get_plot_path(plot_id, "svg")
    if not p:
        raise HTTPException(404)
    return FileResponse(p, media_type="image/svg+xml")

@router.get("/api/analysis/{plot_id}/data.csv")
async def get_plot_csv(plot_id: str):
    p = analysis_engine.get_plot_path(plot_id, "csv")
    if not p:
        raise HTTPException(404)
    return FileResponse(p, media_type="text/csv",
                        filename=f"{plot_id}.csv")

@router.post("/api/analysis/compare")
async def compare_sims(request: Request):
    body = await request.json()
    sim_ids = body.get("sim_ids", [])
    metric = body.get("metric", "food")

    all_m = []
    for sid in sim_ids:
        entry = _get_entry(sid)
        if entry and entry.metrics:
            all_m.append(entry.metrics)

    if len(all_m) < 2:
        raise HTTPException(400, detail="Need >= 2 completed sims to compare")

    plot_id, _ = analysis_engine.compare_sims(all_m, metric)
    return {"plot_id": plot_id}


# --- Queue ---

@router.get("/api/queue")
async def get_queue():
    return sim_queue.get_queue_info()


# --- Playlists (batch groupings) ---

@router.get("/api/playlists")
async def list_playlists():
    # Group by name prefix
    groups: dict[str, list[dict]] = {}
    for e in sim_queue.entries.values():
        prefix = e.name.rsplit("_", 1)[0] if "_" in e.name else e.name
        groups.setdefault(prefix, []).append({
            "id": e.id, "name": e.name, "status": e.status.value,
        })
    return [{"name": k, "sims": v} for k, v in groups.items()]
