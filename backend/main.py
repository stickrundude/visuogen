import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from uuid import uuid4
from pathlib import Path
from typing import Optional
import random
import os
import sys
import threading
import time

from dotenv import load_dotenv

# apply PYTHONPATH from environment variables
load_dotenv()
python_paths = os.environ.get("PYTHONPATH", "").split(":")
for path in python_paths:
    if path and path not in sys.path:
        sys.path.insert(0, path)

from .models.sd.sdxl_story_textinv import storybot_flow as sdxl_story_flow
from .models.sd.sdxl_diagram_gen import diagram_flow as sdxl_diagram_flow
from .models.kandinsky.kandinsky_story_gen import generate_story_images_kandinsky as kandinsky_story_flow
from .models.kandinsky.kandinsky_diagram_gen import generate_diagram_kandinsky3_full as kandinsky_diagram_flow

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = (Path(__file__).parent / "static").resolve()
print(f"[INFO] Mounting static directory: {static_dir}")
from starlette.staticfiles import StaticFiles as StarletteStaticFiles
app.mount("/static", StarletteStaticFiles(directory=static_dir, html=True, check_dir=False), name="static")

class GenerateRequest(BaseModel):
    prompt: str
    seed: Optional[int] = None

active_jobs = {}
global_logs = []

def log_job(job_id: str, message: str):
    formatted = f"[{job_id[:8]}] {message}"
    print(formatted)
    global_logs.append(formatted)

@app.get("/")
def home():
    print("[INFO] VisuoGen backend is running")
    return {"message": "VisuoGen backend is running"}

@app.get("/logs/stream")
def stream_logs():
    def event_generator():
        last_index = 0
        while True:
            time.sleep(1)
            new_logs = global_logs[last_index:]
            if new_logs:
                for line in new_logs:
                    yield f"data: {line}\n\n"
                last_index = len(global_logs)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/generate-story-sdxl")
async def generate_story_sdxl(request: GenerateRequest):
    seed = request.seed or random.randint(0, 2**32 - 1)
    job_id = str(uuid4())
    out_dir = static_dir / "sd" / "story" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log_job(job_id, f"[INFO] Starting SDXL Story Generation: Job ID = {job_id}, Seed = {seed}, Prompt = {request.prompt}")
    cancel_event = threading.Event()

    def run_generation():
        try:
            sdxl_story_flow(request.prompt, out_dir, seed, cancel_event=cancel_event, log_fn=lambda msg: log_job(job_id, msg))
        except Exception as e:
            log_job(job_id, f"[ERROR] Story generation failed for job {job_id}: {e}")
        finally:
            active_jobs.pop(job_id, None)
            log_job(job_id, f"[INFO] Job {job_id} finished or removed")

    thread = threading.Thread(target=run_generation)
    thread.start()
    active_jobs[job_id] = {"thread": thread, "cancel_event": cancel_event}

    return {"job_id": job_id, "status": "started", "seed": seed, "message": "Story generation started in background"}

@app.post("/generate-diagram-sdxl")
async def generate_diagram_sdxl(request: GenerateRequest):
    seed = request.seed or random.randint(0, 2**32 - 1)
    job_id = str(uuid4())
    output_path = static_dir / "sd" / "diagrams" / job_id / "diagram.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_job(job_id, f"[INFO] Starting SDXL Diagram Generation: Job ID = {job_id}, Seed = {seed}, Prompt = {request.prompt}")
    cancel_event = threading.Event()

    def run_generation():
        try:
            sdxl_diagram_flow(request.prompt, output_path, seed, 60, 8.0, 0.8, cancel_event, lambda msg: log_job(job_id, msg))
        except Exception as e:
            log_job(job_id, f"[ERROR] SDXL diagram generation failed for job {job_id}: {str(e)}")
        finally:
            active_jobs.pop(job_id, None)
            log_job(job_id, f"[INFO] Job {job_id} finished or removed")

    thread = threading.Thread(target=run_generation)
    thread.start()
    active_jobs[job_id] = {"thread": thread, "cancel_event": cancel_event}

    image_url = f"/static/{output_path.relative_to(static_dir).as_posix()}"
    return {"job_id": job_id, "image_url": image_url, "seed": seed, "message": "Diagram generation started in background"}

@app.post("/generate-story-kandinsky")
async def generate_story_kandinsky(request: GenerateRequest):
    seed = request.seed or random.randint(0, 2**32 - 1)
    job_id = str(uuid4())
    out_dir = static_dir / "kandinsky" / "story" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log_job(job_id, f"[INFO] Starting Kandinsky Story Generation: Job ID = {job_id}, Seed = {seed}, Prompt = {request.prompt}")
    cancel_event = threading.Event()

    def run_generation():
        try:
            kandinsky_story_flow(request.prompt, out_dir, seed=seed, cancel_event=cancel_event, log_fn=lambda msg: log_job(job_id, msg))
        except Exception as e:
            log_job(job_id, f"[ERROR] Kandinsky story generation failed for job {job_id}: {str(e)}")
        finally:
            active_jobs.pop(job_id, None)
            log_job(job_id, f"[INFO] Job {job_id} finished or removed")

    thread = threading.Thread(target=run_generation)
    thread.start()
    active_jobs[job_id] = {"thread": thread, "cancel_event": cancel_event}

    return {"job_id": job_id, "status": "started", "seed": seed, "message": "Story generation started in background"}

@app.post("/generate-diagram-kandinsky")
async def generate_diagram_kandinsky(request: GenerateRequest):
    seed = request.seed or random.randint(0, 2**32 - 1)
    job_id = str(uuid4())
    output_path = static_dir / "kandinsky" / "diagrams" / job_id / "diagram.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_job(job_id, f"[INFO] Starting Kandinsky Diagram Generation: Job ID = {job_id}, Seed = {seed}, Prompt = {request.prompt}")
    cancel_event = threading.Event()

    def run_generation():
        try:
            kandinsky_diagram_flow(request.prompt, output_path, seed, cancel_event, log_fn=lambda msg: log_job(job_id, msg))
        except Exception as e:
            log_job(job_id, f"[ERROR] Kandinsky diagram generation failed for job {job_id}: {str(e)}")
        finally:
            active_jobs.pop(job_id, None)
            log_job(job_id, f"[INFO] Job {job_id} finished or removed")

    thread = threading.Thread(target=run_generation)
    thread.start()
    active_jobs[job_id] = {"thread": thread, "cancel_event": cancel_event}

    image_url = f"/static/{output_path.relative_to(static_dir).as_posix()}"
    return {"job_id": job_id, "image_url": image_url, "seed": seed, "message": "Diagram generation started in background"}

@app.post("/cancel-job/{job_id}")
def cancel_job(job_id: str):
    job = active_jobs.get(job_id)
    if job:
        cancel_event = job.get("cancel_event")
        if cancel_event and not cancel_event.is_set():
            cancel_event.set()
            return {"job_id": job_id, "status": "cancelled", "message": "Cancellation signal sent"}
    return {"job_id": job_id, "status": "not_found_or_finished", "message": "No running job with that ID"}

@app.get("/list-story-jobs/{model}")
def list_story_jobs(model: str):
    story_path = static_dir / model / "story"
    if not story_path.exists():
        raise HTTPException(status_code=404, detail="Model not found or no story directory")

    folders = [
        f.name for f in story_path.iterdir()
        if f.is_dir() and f.name != "__pycache__"
    ]
    return {"folders": sorted(folders)}

@app.get("/list-story-images/{model}/{folder}")
def list_story_images(model: str, folder: str):
    story_dir = static_dir / model / "story" / folder / "story"
    if not story_dir.exists() or not story_dir.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")

    image_files = sorted([
        f.name for f in story_dir.iterdir()
        if f.is_file() and f.suffix == ".png" and f.name.startswith("story_")
    ])

    return {"images": image_files}

@app.get("/story-log/{model}/{folder}")
def get_story_log(model: str, folder: str):
    log_path = static_dir / model / "story" / folder / "story" / "story_log.json"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log not found")

    with open(log_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    return JSONResponse(content=log_data)

@app.get("/list-diagrams-jobs/{model}")
def list_diagram_jobs(model: str):
    diagram_path = static_dir / model / "diagrams"
    if not diagram_path.exists():
        raise HTTPException(status_code=404, detail="Model not found or no diagrams directory")

    folders = [
        f.name for f in diagram_path.iterdir()
        if f.is_dir() and f.name != "__pycache__"
    ]
    return {"folders": sorted(folders)}

@app.get("/diagram-log/{model}/{folder}")
def get_diagram_log(model: str, folder: str):
    log_path = static_dir / model / "diagrams" / folder / "diagram_log.json"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Diagram log not found")

    with open(log_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    return JSONResponse(content=log_data)

@app.get("/diagram-image/{model}/{folder}")
def get_diagram_image(model: str, folder: str):
    diagram_path = static_dir / model / "diagrams" / folder / "diagram.png"
    if not diagram_path.exists():
        raise HTTPException(status_code=404, detail="Diagram image not found")

    image_url = f"/static/{diagram_path.relative_to(static_dir).as_posix()}"
    return {"image": image_url}