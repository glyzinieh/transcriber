import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .whisper import transcribe_audio

app = FastAPI(title="Transcriber")
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

JOBS: dict[str, dict[str, str]] = {}
JOBS_LOCK = threading.Lock()


def update_job(job_id: str, **fields: str) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(fields)


def process_transcription(job_id: str, source_path: str, output_path: str) -> None:
    try:
        update_job(job_id, status="processing")

        # progress callback receives a float 0.0-1.0; store as integer percent string
        def _progress(p: float) -> None:
            try:
                pct = str(int(max(0.0, min(1.0, p)) * 100))
                update_job(job_id, progress=pct)
            except Exception:
                pass

        text = transcribe_audio(source_path, progress_callback=_progress)
        Path(output_path).write_text(text, encoding="utf-8")
        update_job(job_id, status="done", output_path=output_path, progress="100")
    except Exception as exc:  # noqa: BLE001
        update_job(job_id, status="failed", error=str(exc), progress="0")
    finally:
        Path(source_path).unlink(missing_ok=True)


@app.get("/")
async def index(request: Request):
    with JOBS_LOCK:
        jobs_list = [
            {
                "job_id": jid,
                "status": job.get("status", "queued"),
                "filename": job.get("filename", ""),
            }
            for jid, job in list(JOBS.items())[::-1]
        ]

    return templates.TemplateResponse(
        request, "index.html", {"request": request, "jobs": jobs_list}
    )


@app.post("/transcribe")
async def transcribe(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="ファイル名を取得できませんでした")

    suffix = Path(file.filename).suffix or ".wav"
    job_id = uuid.uuid4().hex
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    output_path = str(Path(temp_path).with_suffix(".txt"))
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "filename": file.filename,
            "output_path": output_path,
            "progress": "0",
        }

    background_tasks.add_task(process_transcription, job_id, temp_path, output_path)

    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}")
async def job_status(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")

    return templates.TemplateResponse(
        request,
        "status.html",
        {
            "request": request,
            "job_id": job_id,
            "filename": job.get("filename", ""),
            "status": job.get("status", "queued"),
            "error": job.get("error", ""),
            "progress": job.get("progress", "0"),
        },
    )


@app.get("/download/{job_id}")
async def download(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")

    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail="まだ文字起こし中です")

    output_path = job.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="出力ファイルが見つかりません")

    filename = f"{Path(job.get('filename', 'transcript')).stem or 'transcript'}.txt"
    return FileResponse(
        output_path,
        media_type="text/plain; charset=utf-8",
        filename=filename,
    )
