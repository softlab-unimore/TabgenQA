"""
TabQA Generator — FastAPI backend
Wraps the Gradino generation library and serves the React frontend.
"""

import asyncio
import csv
import io
import json
import os
import re
import sys
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="TabQA Generator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Directories ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(_HERE, "..", "output"))
TASKS_DIR = os.path.join(OUTPUT_DIR, "tasks")
EVALS_DIR = os.path.join(OUTPUT_DIR, "evals")

# ── In-memory task stores ────────────────────────────────────────────────────
# { task_id: { status, progress, current, total, instances, ... } }
tasks: dict = {}
# { eval_id: { status, progress, accuracy, ... } }
eval_tasks: dict = {}


# ── Pydantic models ─────────────────────────────────────────────────────────

class GenerationParams(BaseModel):
    domain: str = "environmental"
    question_type: str = "sum"
    num_tables: int = 3
    num_samples: int = 10
    col_cardinality: int = 20
    num_columns: int = 21
    sequential: bool = False
    api_key: str = ""


class EvalParams(BaseModel):
    task_id: str
    model_type: str = "openai"   # openai | claude | gemini | huggingface
    model_name: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = ""           # vLLM base URL for huggingface model type


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_tables(raw) -> list:
    """Normalize the 'Table' field from Gradino output into a list of HTML strings."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(t) for t in raw if t]
    s = str(raw)
    found = re.findall(r'<table[\s\S]*?</table>', s, re.IGNORECASE)
    return found if found else ([s] if s.strip() else [])


def _to_list(raw) -> list:
    """Ensure a value is always returned as a list."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    return [raw]


def _flatten_instances(data: dict) -> list:
    """Convert Gradino's nested {nt: {method: {pert: [records]}}} into a flat list."""
    instances = []
    idx = 0
    for nt_key in sorted(data.keys(), key=lambda k: int(k) if k.lstrip("-").isdigit() else k):
        for method_key, pert_dict in data[nt_key].items():
            for pert_key, records in pert_dict.items():
                for rec in records:
                    instances.append({
                        "id": idx,
                        "num_tables": nt_key,
                        "method": method_key,
                        "perturbation": pert_key,
                        "question": str(rec.get("Question") or ""),
                        "tables": _extract_tables(rec.get("Table")),
                        "answer": str(rec.get("Label") or ""),
                        "sql_queries": _to_list(rec.get("SQL Query")),
                        "constraints": rec.get("Constraints") or {},
                    })
                    idx += 1
    return instances


def _get_instance_count(task: dict) -> int:
    instances = task.get("instances", [])
    return len(instances) if instances else task.get("instance_count", 0)


# ── Persistence ──────────────────────────────────────────────────────────────

def _persist_task(task_id: str):
    task = tasks.get(task_id)
    if not task:
        return
    try:
        os.makedirs(TASKS_DIR, exist_ok=True)
        instances = task.get("instances", [])
        meta = {k: v for k, v in task.items() if k not in ("instances", "process", "stop_requested")}
        meta["task_id"] = task_id
        meta["instance_count"] = len(instances)
        with open(os.path.join(TASKS_DIR, f"{task_id}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        if instances:
            with open(os.path.join(TASKS_DIR, f"{task_id}_instances.json"), "w", encoding="utf-8") as f:
                json.dump(instances, f, ensure_ascii=False)
    except Exception as exc:
        print(f"[WARNING] persist_task {task_id}: {exc}", file=sys.stderr)


def _persist_eval(eval_id: str):
    task = eval_tasks.get(eval_id)
    if not task:
        return
    try:
        eval_dir = os.path.join(EVALS_DIR, eval_id)
        os.makedirs(eval_dir, exist_ok=True)
        meta = {k: v for k, v in task.items() if k not in ("process", "stop_requested")}
        meta["eval_id"] = eval_id
        with open(os.path.join(eval_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
    except Exception as exc:
        print(f"[WARNING] persist_eval {eval_id}: {exc}", file=sys.stderr)


# ── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    os.makedirs(TASKS_DIR, exist_ok=True)
    os.makedirs(EVALS_DIR, exist_ok=True)

    for fname in sorted(os.listdir(TASKS_DIR)):
        if not fname.endswith("_meta.json"):
            continue
        task_id = fname[:-10]  # strip "_meta.json"
        if task_id in tasks:
            continue
        try:
            with open(os.path.join(TASKS_DIR, fname), encoding="utf-8") as f:
                meta = json.load(f)
            tasks[task_id] = {
                **{k: v for k, v in meta.items() if k != "task_id"},
                "instances": [],
                "process": None,
                "stop_requested": False,
            }
        except Exception as exc:
            print(f"[WARNING] load task {task_id}: {exc}", file=sys.stderr)

    for eval_id in os.listdir(EVALS_DIR):
        if eval_id in eval_tasks:
            continue
        meta_path = os.path.join(EVALS_DIR, eval_id, "meta.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            eval_tasks[eval_id] = {
                **{k: v for k, v in meta.items() if k != "eval_id"},
                "process": None,
                "stop_requested": False,
            }
        except Exception as exc:
            print(f"[WARNING] load eval {eval_id}: {exc}", file=sys.stderr)


# ── Background generation task ───────────────────────────────────────────────

async def _run_generation(task_id: str, params: dict):
    task = tasks[task_id]
    env = os.environ.copy()
    api_key = params.get("api_key", "")
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    elif not env.get("OPENAI_API_KEY"):
        here = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(here, "..", "gradino", ".env")
        if os.path.isfile(dotenv_path):
            with open(dotenv_path) as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line.startswith("OPENAI_API_KEY=") and not _line.startswith("#"):
                        env["OPENAI_API_KEY"] = _line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

    params_json = json.dumps(params)
    script_path = os.path.join(os.path.dirname(__file__), "generate_script.py")

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, script_path, params_json,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=16 * 1024 * 1024,
        )
        task["process"] = proc

        stderr_chunks: list = []
        async def _read_stderr():
            async for chunk in proc.stderr:
                stderr_chunks.append(chunk)
        stderr_task = asyncio.create_task(_read_stderr())

        async for raw_line in proc.stdout:
            if task.get("stop_requested"):
                proc.terminate()
                await proc.wait()
                await stderr_task
                task["status"] = "stopped"
                _persist_task(task_id)
                return

            line = raw_line.decode().strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                mtype = msg.get("type")
                if mtype == "progress":
                    task["progress"] = msg["progress"]
                    task["current"] = msg["current"]
                    task["total"] = msg["total"]
                    task["status_message"] = msg.get("desc", "Generating…")
                elif mtype == "status":
                    task["status_message"] = msg.get("message", "")
                elif mtype == "result":
                    task["progress"] = 1.0
                    task["instances"] = _flatten_instances(msg.get("data", {}))
                    task["generation_errors"] = msg.get("errors", [])
                elif mtype == "error":
                    task["status"] = "error"
                    task["error"] = msg.get("message", "Unknown error")
                    task["traceback"] = msg.get("traceback", "")
                    await stderr_task
                    _persist_task(task_id)
                    return
            except json.JSONDecodeError:
                pass

        await proc.wait()
        await stderr_task
        stderr_output = b"".join(stderr_chunks).decode(errors="replace").strip()
        if stderr_output:
            print(f"[generate_script stderr]\n{stderr_output[:4000]}", file=sys.stderr, flush=True)

        if task["status"] == "running":
            if proc.returncode == 0:
                task["status"] = "completed"
                task["progress"] = 1.0
            else:
                task["status"] = "error"
                task["error"] = (stderr_output or "subprocess exited with non-zero code")[-2000:]
        _persist_task(task_id)

    except Exception as exc:
        import traceback as _tb
        task["status"] = "error"
        task["error"] = str(exc)
        task["traceback"] = _tb.format_exc()
        _persist_task(task_id)


# ── Background evaluation task ───────────────────────────────────────────────

async def _run_evaluation(eval_id: str, params: dict):
    task = eval_tasks[eval_id]
    env = os.environ.copy()
    api_key = params.get("api_key", "")
    model_type = params.get("model_type", "openai")
    if api_key:
        key_map = {
            "openai": "OPENAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "huggingface": "OPENAI_API_KEY",
        }
        if model_type in key_map:
            env[key_map[model_type]] = api_key

    params_json = json.dumps(params)
    script_path = os.path.join(_HERE, "eval_script.py")

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, script_path, params_json,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        task["process"] = proc

        stderr_chunks: list = []
        async def _read_stderr():
            async for chunk in proc.stderr:
                stderr_chunks.append(chunk)
        stderr_task = asyncio.create_task(_read_stderr())

        async for raw_line in proc.stdout:
            line = raw_line.decode().strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                mtype = msg.get("type")
                if mtype == "progress":
                    task["progress"] = msg["progress"]
                    task["current"] = msg["current"]
                    task["total"] = msg["total"]
                    task["status_message"] = msg.get("desc", "Evaluating…")
                elif mtype == "status":
                    task["status_message"] = msg.get("message", "")
                elif mtype == "result":
                    task["accuracy"] = msg.get("accuracy")
                    task["avg_f1"] = msg.get("avg_f1")
                    task["correct"] = msg.get("correct")
                elif mtype == "error":
                    task["status"] = "error"
                    task["error"] = msg.get("message")
                    await stderr_task
                    _persist_eval(eval_id)
                    return
            except json.JSONDecodeError:
                pass

        await proc.wait()
        await stderr_task
        stderr_output = b"".join(stderr_chunks).decode(errors="replace").strip()
        if stderr_output:
            print(f"[eval_script stderr]\n{stderr_output[:2000]}", file=sys.stderr, flush=True)

        if task["status"] == "running":
            if proc.returncode == 0:
                task["status"] = "completed"
                task["progress"] = 1.0
            else:
                task["status"] = "error"
                task["error"] = (stderr_output or "subprocess failed")[-1000:]
        _persist_eval(eval_id)

    except Exception as exc:
        task["status"] = "error"
        task["error"] = str(exc)
        _persist_eval(eval_id)


# ── API routes ───────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/generate")
async def start_generation(params: GenerationParams):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "running",
        "progress": 0.0,
        "current": 0,
        "total": params.num_samples,
        "instances": [],
        "status_message": "Initializing…",
        "error": None,
        "traceback": None,
        "generation_errors": [],
        "process": None,
        "stop_requested": False,
        "created_at": datetime.utcnow().isoformat(),
        "params": params.model_dump(exclude={"api_key"}),
    }
    asyncio.create_task(_run_generation(task_id, params.model_dump()))
    return {"task_id": task_id}


@app.get("/api/generate/{task_id}/stream")
async def stream_progress(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")

    async def _events():
        last_payload = None
        while True:
            task = tasks.get(task_id)
            if not task:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            payload = json.dumps({
                "status": task["status"],
                "progress": task["progress"],
                "current": task["current"],
                "total": task["total"],
                "instance_count": len(task.get("instances", [])),
                "status_message": task.get("status_message", ""),
                "error": task.get("error"),
            })
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload

            if task["status"] in ("completed", "stopped", "error"):
                break
            await asyncio.sleep(0.4)

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/generate/{task_id}")
async def stop_generation(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    tasks[task_id]["stop_requested"] = True
    tasks[task_id]["status"] = "stopped"
    return {"status": "stop_requested"}


@app.get("/api/tasks")
async def list_tasks():
    result = []
    for tid, task in tasks.items():
        result.append({
            "task_id": tid,
            "status": task["status"],
            "progress": task["progress"],
            "instance_count": _get_instance_count(task),
            "created_at": task.get("created_at"),
            "params": task.get("params"),
        })
    return {"tasks": result}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    task = tasks[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "current": task["current"],
        "total": task["total"],
        "instance_count": _get_instance_count(task),
        "status_message": task.get("status_message", ""),
        "error": task.get("error"),
        "params": task.get("params"),
        "created_at": task.get("created_at"),
    }


@app.get("/api/tasks/{task_id}/instances")
async def get_instances(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    task = tasks[task_id]
    instances = task.get("instances", [])
    if not instances:
        inst_path = os.path.join(TASKS_DIR, f"{task_id}_instances.json")
        if os.path.isfile(inst_path):
            try:
                with open(inst_path, encoding="utf-8") as f:
                    instances = json.load(f)
                task["instances"] = instances  # cache in memory
            except Exception as exc:
                print(f"[WARNING] load instances {task_id}: {exc}", file=sys.stderr)
    return {
        "instances": instances,
        "generation_errors": task.get("generation_errors", []),
    }


@app.put("/api/tasks/{task_id}/instances/{instance_id}")
async def update_instance(task_id: str, instance_id: int, data: dict):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    task = tasks[task_id]
    instances = task.get("instances", [])
    for inst in instances:
        if inst["id"] == instance_id:
            for field in ("question", "answer", "tables"):
                if field in data:
                    inst[field] = data[field]
            _persist_task(task_id)
            return {"status": "ok"}
    raise HTTPException(404, "Instance not found")


@app.get("/api/tasks/{task_id}/download")
async def download_dataset(task_id: str, fmt: str = "csv"):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    task = tasks[task_id]
    instances = task.get("instances", [])
    if not instances:
        inst_path = os.path.join(TASKS_DIR, f"{task_id}_instances.json")
        if os.path.isfile(inst_path):
            try:
                with open(inst_path, encoding="utf-8") as f:
                    instances = json.load(f)
            except Exception:
                pass

    if fmt == "json":
        content = json.dumps(instances, indent=2, ensure_ascii=False)
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=tabqa_{task_id[:8]}.json"},
        )

    out = io.StringIO()
    writer = csv.DictWriter(
        out,
        fieldnames=["id", "question", "answer", "tables", "method", "perturbation", "num_tables"],
        extrasaction="ignore",
    )
    writer.writeheader()
    for inst in instances:
        writer.writerow({
            "id": inst["id"],
            "question": inst["question"],
            "answer": inst["answer"],
            "tables": json.dumps(inst["tables"], ensure_ascii=False),
            "method": inst["method"],
            "perturbation": inst["perturbation"],
            "num_tables": inst["num_tables"],
        })
    return StreamingResponse(
        io.BytesIO(out.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=tabqa_{task_id[:8]}.csv"},
    )


# ── History ───────────────────────────────────────────────────────────────────

@app.get("/api/history")
async def list_history(
    domain: Optional[str] = None,
    question_type: Optional[str] = None,
    num_tables: Optional[int] = None,
    status: Optional[str] = None,
):
    result = []
    for tid, task in tasks.items():
        p = task.get("params") or {}
        if domain and p.get("domain") != domain:
            continue
        if question_type and p.get("question_type") != question_type:
            continue
        if num_tables is not None and p.get("num_tables") != num_tables:
            continue
        if status and task.get("status") != status:
            continue
        result.append({
            "task_id": tid,
            "status": task.get("status"),
            "progress": task.get("progress", 0),
            "instance_count": _get_instance_count(task),
            "created_at": task.get("created_at"),
            "params": p,
            "error": task.get("error"),
        })
    result.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"tasks": result}


# ── Evaluation ────────────────────────────────────────────────────────────────

@app.post("/api/evaluate")
async def start_evaluation(ep: EvalParams):
    if ep.task_id not in tasks:
        raise HTTPException(404, "Task not found")
    task = tasks[ep.task_id]

    instances = task.get("instances", [])
    if not instances:
        inst_path = os.path.join(TASKS_DIR, f"{ep.task_id}_instances.json")
        if os.path.isfile(inst_path):
            try:
                with open(inst_path, encoding="utf-8") as f:
                    instances = json.load(f)
                task["instances"] = instances
            except Exception as exc:
                raise HTTPException(500, f"Failed to load instances: {exc}")
    if not instances:
        raise HTTPException(400, "No instances found for this task")

    eval_id = str(uuid.uuid4())
    os.makedirs(EVALS_DIR, exist_ok=True)
    eval_dir = os.path.join(EVALS_DIR, eval_id)
    os.makedirs(eval_dir, exist_ok=True)
    instances_file = os.path.join(eval_dir, "instances.json")
    with open(instances_file, "w", encoding="utf-8") as f:
        json.dump(instances, f, ensure_ascii=False)

    eval_tasks[eval_id] = {
        "status": "running",
        "progress": 0.0,
        "current": 0,
        "total": len(instances),
        "task_id": ep.task_id,
        "model_type": ep.model_type,
        "model_name": ep.model_name,
        "accuracy": None,
        "avg_f1": None,
        "correct": None,
        "created_at": datetime.utcnow().isoformat(),
        "error": None,
        "status_message": "Initializing…",
        "process": None,
        "stop_requested": False,
    }

    asyncio.create_task(_run_evaluation(eval_id, {
        "instances_file": instances_file,
        "model_type": ep.model_type,
        "model_name": ep.model_name,
        "api_key": ep.api_key,
        "base_url": ep.base_url,
        "output_dir": eval_dir,
        "eval_id": eval_id,
    }))
    return {"eval_id": eval_id}


@app.get("/api/evaluate/{eval_id}/stream")
async def stream_eval_progress(eval_id: str):
    if eval_id not in eval_tasks:
        raise HTTPException(404, "Eval task not found")

    async def _events():
        last_payload = None
        while True:
            task = eval_tasks.get(eval_id)
            if not task:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break
            payload = json.dumps({
                "status": task["status"],
                "progress": task["progress"],
                "current": task["current"],
                "total": task["total"],
                "status_message": task.get("status_message", ""),
                "accuracy": task.get("accuracy"),
                "avg_f1": task.get("avg_f1"),
                "correct": task.get("correct"),
                "error": task.get("error"),
            })
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if task["status"] in ("completed", "stopped", "error"):
                break
            await asyncio.sleep(0.4)

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/evaluate/{eval_id}")
async def get_eval(eval_id: str):
    if eval_id not in eval_tasks:
        raise HTTPException(404, "Eval task not found")
    task = eval_tasks[eval_id]
    return {
        "eval_id": eval_id,
        "status": task["status"],
        "progress": task["progress"],
        "current": task["current"],
        "total": task["total"],
        "task_id": task.get("task_id"),
        "model_type": task.get("model_type"),
        "model_name": task.get("model_name"),
        "accuracy": task.get("accuracy"),
        "avg_f1": task.get("avg_f1"),
        "correct": task.get("correct"),
        "status_message": task.get("status_message", ""),
        "error": task.get("error"),
        "created_at": task.get("created_at"),
    }


@app.get("/api/evaluate/{eval_id}/predictions")
async def get_eval_predictions(eval_id: str):
    if eval_id not in eval_tasks:
        raise HTTPException(404, "Eval task not found")
    preds_file = os.path.join(EVALS_DIR, eval_id, "predictions.json")
    if not os.path.isfile(preds_file):
        raise HTTPException(404, "Predictions not yet available (evaluation may still be running)")
    with open(preds_file, encoding="utf-8") as f:
        predictions = json.load(f)
    return {"eval_id": eval_id, "predictions": predictions}


@app.get("/api/leaderboard")
async def get_leaderboard(task_id: Optional[str] = None):
    result = []
    for eid, et in eval_tasks.items():
        if task_id and et.get("task_id") != task_id:
            continue
        if et.get("status") != "completed":
            continue
        gen_task = tasks.get(et.get("task_id", ""))
        result.append({
            "eval_id": eid,
            "task_id": et.get("task_id"),
            "model_type": et.get("model_type"),
            "model_name": et.get("model_name"),
            "accuracy": et.get("accuracy"),
            "avg_f1": et.get("avg_f1"),
            "correct": et.get("correct"),
            "total": et.get("total"),
            "created_at": et.get("created_at"),
            "task_params": gen_task.get("params") if gen_task else None,
        })
    result.sort(key=lambda x: x.get("accuracy") or 0, reverse=True)
    return {"leaderboard": result}


# ── Static frontend ──────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")
