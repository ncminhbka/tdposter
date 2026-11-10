#!/usr/bin/env python3
"""MÁY CHỦ TEXTFIX: form / prompt -> LLM (prompt FLUX + câu khách + các hướng thiết kế) -> mỗi phiên bản: FLUX.2-klein vẽ nháp ->
Hi-SAM tìm nét chữ -> dòng (OCR + nét chữ) -> gỡ chữ (LaMa) -> VLM điền chữ -> dựng (Chromium) -> kiểm -> vòng sửa -> poster.

  python server/app.py --port 8088                  # bản distill (4 bước)
  python server/app.py --port 8088 --model base     # bản base (50 bước, CFG 4)

API (giao diện server/ui.html dùng job + hỏi tiến độ; script dùng /api/generate chờ xong):
  POST /api/jobs         GenerateRequest -> {job_id}
  GET  /api/jobs/{id}    {status: queued|running|done|error, progress, result, error}
  POST /api/generate     GenerateRequest -> kết quả (chờ xong)
  GET  /api/health       model đã nạp, LLM / VLM đang dùng, hàng đợi
Ảnh từng lần chạy: output/runs/<run_id>/ (request.json, brief.json, v<i>/draft.png, ink.png (mặt nạ nét chữ Hi-SAM), ink.jpg (nét + mặt nạ gỡ chữ), base.jpg (nền đã gỡ chữ),
slots.jpg, poster.png, poster_first.png (lượt đầu, khi vòng sửa đổi), steps.jpg, ops.jpg, plan.json), phục vụ ở /outputs/<run_id>/...

FLUX ở máy khác (TEXTFIX_FLUX_URL, docs/DEPLOY.md): không nạp FLUX; số luồng VẼ = TEXTFIX_FLUX_CONCURRENCY, card chỉ chạy Hi-SAM / LaMa.

DÂY CHUYỀN: mỗi card GPU một FLUX + một luồng VẼ (chung hàng đợi, TEXTFIX_GPUS="0,1" mặc định mọi card), nhóm luồng THIẾT KẾ
(TEXTFIX_DESIGN_WORKERS=4, mỗi luồng một Chromium -- Playwright đồng bộ không chạy trong vòng lặp asyncio). Nhiều phiên bản / nhiều
yêu cầu gối đầu nhau; job ghi kết quả từng phiên bản ngay khi xong (UI hiện dần).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import queue
import shutil
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

from textfix.config import load_env  # noqa: E402

load_env()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("textfix.server")

RUNS = Path(os.environ.get("TEXTFIX_RUNS", ROOT / "output" / "runs"))
RUNS.mkdir(parents=True, exist_ok=True)
MAX_RUNS = int(os.environ.get("TEXTFIX_MAX_RUNS", "300"))
UI = Path(__file__).resolve().parent / "ui.html"
STATE: dict[str, Any] = {"flux": [], "llm": None, "vlm": None, "models": None, "error": None, "args": None, "started": None}
JOBS: dict[str, dict] = {}
ORDER: list[str] = []   # hàng đợi (job chưa xong), để báo vị trí

# ---------------------------------------------------------------------------------------------------- form
# hợp đồng API (form từng loại, kiểm đầu vào, mã lỗi): server/contract.py
from contract import (DESIGN_STAGES, ApiError, GenerateRequest, decode_image, draw_error, failure,  # noqa: E402
                      fetch_image, form_of, ignored_fields, missing_fields, validate)
import contract_v1 as v1  # noqa: E402   API v1 cho chatbot: POST /api/v1/images/generations (docs/API_V1.md)


# ---------------------------------------------------------------------------------------------------- dây chuyền
# Mỗi phiên bản đi qua hai trạm, các phiên bản (và các yêu cầu) gối đầu nhau:
#   VẼ (GPU)        mỗi card một FLUX + một Hi-SAM + một luồng, chung một hàng đợi          nháp -> nét chữ
#   THIẾT KẾ        nhóm luồng, mỗi luồng một Chromium; VLM ở máy khác nên gọi song song    ô -> VLM -> dựng -> kiểm   ~15 s
# LLM viết brief (một lời gọi / yêu cầu, kèm các hướng thiết kế) chạy ở luồng điều phối của yêu cầu.
DRAW_Q: "queue.Queue" = queue.Queue()
DESIGN = ThreadPoolExecutor(max_workers=int(os.environ.get("TEXTFIX_DESIGN_WORKERS", "4")), thread_name_prefix="design")
JOBRUN = ThreadPoolExecutor(max_workers=int(os.environ.get("TEXTFIX_JOB_WORKERS", "8")), thread_name_prefix="job")
LOCAL = threading.local()   # Chromium của từng luồng thiết kế (Playwright đồng bộ: mỗi luồng một bản)
LOAD_LOCK = threading.Lock()   # nạp FLUX từng card một (xem gpu_worker)


def gpu_list() -> list[int]:
    """Card dùng được: TEXTFIX_GPUS="0,1" nếu đặt, không thì mọi card máy thấy (không card nào: [])."""
    env = os.environ.get("TEXTFIX_GPUS")
    if env:
        return [int(x) for x in env.split(",") if x.strip()]
    import torch
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


def draw_devices() -> list[str]:
    """Mỗi phần tử = một luồng VẼ và thiết bị của nó.
    FLUX tại chỗ: mỗi card một luồng (mỗi card một FLUX). FLUX ở máy khác (TEXTFIX_FLUX_URL): số luồng = số nháp gửi sang
    cùng lúc (TEXTFIX_FLUX_CONCURRENCY), chia vòng lên các card chỉ để chạy Hi-SAM / LaMa; không card nào thì chạy CPU."""
    from textfix.draw import remote
    gpus = [f"cuda:{k}" for k in gpu_list()]
    if not remote.enabled():
        if not gpus:
            raise RuntimeError("không thấy GPU để nạp FLUX (hoặc đặt TEXTFIX_FLUX_URL để gọi FLUX ở máy khác)")
        return gpus
    return [(gpus or ["cpu"])[i % max(1, len(gpus))] for i in range(remote.concurrency())]


def gpu_worker(dev: str, args, ready: threading.Event) -> None:
    """Một luồng VẼ trên thiết bị dev: có FLUX (tại chỗ: nạp lên card; ở xa: bộ gọi API) + Hi-SAM, lấy việc từ hàng đợi chung."""
    from textfix import config
    from textfix.draw import remote, hisam
    from textfix.engine import draw
    try:
        if remote.enabled():
            F = remote.RemoteFlux(device=dev)
            with LOAD_LOCK:
                hisam.load(dev)   # nét chữ: nạp sẵn để lỗi thiếu tệp lộ ra lúc khởi động
        else:
            import torch
            torch.cuda.set_device(dev)
            # NẠP LẦN LƯỢT từng card: transformers đổi kiểu số mặc định của torch (toàn cục, không theo luồng) khi nạp bộ mã
            # hoá chữ -> hai card nạp cùng lúc giẫm nhau, một bản lẫn float32 (Flux.load còn ép bf16 lần nữa). Vẽ vẫn song song.
            with LOAD_LOCK:
                t0 = time.time()
                if config.draw_profile() == "qwen":   # start_server.sh --draw qwen: Qwen-Image tại chỗ thay FLUX
                    from textfix.draw.qwen import QwenDraw
                    F = QwenDraw(device=dev).load()
                else:
                    from textfix.draw.flux import Flux
                    F = Flux(args.model, steps=args.steps, weights_dir=args.weights_dir, device=dev).load()
                hisam.load(dev)
            log.info("%s %s sẵn sàng (%.0fs, %.1f GB)", F.info()["model"], dev, time.time() - t0,
                     torch.cuda.memory_allocated(dev) / 2 ** 30)
        STATE["flux"].append(F)
    except BaseException as e:   # gồm SystemExit của flux2.util khi thiếu trọng số
        STATE["error"] = f"{dev}: {type(e).__name__}: {e}"
        log.error("KHÔNG nạp được luồng vẽ %s: %s", dev, traceback.format_exc())
        return
    finally:
        ready.set()
    while True:
        task = DRAW_Q.get()
        if task is None:
            return
        v, fut = task["v"], task["fut"]
        if fut.set_running_or_notify_cancel():
            try:
                v["gpu"] = dev
                fut.set_result(draw(F, task["B"], task["W"], task["H"], v["seed"], task["product"], progress=task["progress"]))
            except BaseException as e:
                fut.set_exception(e)


def init_models(args) -> None:
    """LLM / VLM (gọi qua mạng), một luồng VẼ cho mỗi card, làm nóng OCR. Chạy ở luồng phụ (lúc khởi động)."""
    from textfix import config
    from textfix.brief import SCHEMA
    from textfix.llm import json_call
    from textfix.lines.ocr import _engine
    if not os.environ.get("TEXTFIX_LLM_URL") or not os.environ.get("TEXTFIX_MODEL"):
        raise RuntimeError("chưa đặt TEXTFIX_LLM_URL / TEXTFIX_MODEL (export trong terminal, docs/SERVER.md)")
    from textfix import erase
    from textfix.draw import hisam
    lost = hisam.missing() + ([] if erase.lama_ready() else [erase.LAMA_PATH])
    if lost:   # người dùng chốt 14/10: thiếu tệp thì báo rõ và dừng, không lùi về cách nào khác
        raise RuntimeError("thiếu tệp model gỡ chữ (docs/SERVER.md, mục Hi-SAM và LaMa): " + "; ".join(lost))
    m = config.models()
    STATE["models"] = m
    STATE["llm"] = json_call(m["llm"]["url"], m["llm"]["model"], m["llm"]["key_env"], tag="brief", schema=SCHEMA)
    STATE["vlm"] = json_call(m["vlm"]["url"], m["vlm"]["model"], m["vlm"]["key_env"], tag="designer")
    _engine()(np.full((64, 256, 3), 255, np.uint8))   # làm nóng OCR (lần đầu nạp mô hình ONNX)
    devs = draw_devices()
    evs = []
    for i, dev in enumerate(devs):
        ev = threading.Event()
        threading.Thread(target=gpu_worker, args=(dev, args, ev), name=f"draw{i}", daemon=True).start()
        evs.append(ev)
    for ev in evs:
        ev.wait()
    if not STATE["flux"]:
        raise RuntimeError(STATE["error"] or "không nạp được luồng vẽ nào")
    log.info("Sẵn sàng: %d luồng vẽ %s (model vẽ %s), %d luồng thiết kế; LLM %s | VLM %s", len(STATE["flux"]), devs,
             STATE["flux"][0].info()["model"], DESIGN._max_workers, m["llm"]["model"], m["vlm"]["model"])


def page():
    """Chromium của luồng thiết kế hiện tại (mở lần đầu dùng)."""
    if getattr(LOCAL, "browser", None) is None:
        from textfix.render.browser import Browser
        LOCAL.browser = Browser().__enter__()
    return LOCAL.browser.page


def save(img: np.ndarray, path: Path, quality: int | None = None) -> None:
    from PIL import Image
    Image.fromarray(np.asarray(img)).save(path, **({"quality": quality} if quality else {}))


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return None
    return o


def design_one(job: dict, v: dict, d: dict, Bv: dict, product: bool, vd: Path, url: str, t_llm: float) -> None:
    """TRẠM THIẾT KẾ cho một phiên bản + lưu ảnh / JSON debug. Lỗi -> v["error"] (không dừng phiên bản khác)."""
    from textfix.lines.view import marked_image
    from textfix.engine import design
    from textfix.render import ops_image
    try:
        from textfix.draw import hisam
        save(d["draft"], vd / "draft.png")   # có ngay khi vẽ xong: UI xem được nháp trong lúc thiết kế
        hisam.to_png(d["ink"], vd / "ink.png")   # mặt nạ nét chữ: đủ để dựng lại tại máy cùng nháp + plan.json
        v.update(draft_url=f"{url}/draft.png")
        r = design(d, Bv, STATE["vlm"], page(), product=product, progress=lambda m: v.update(stage=m))
        save(r["poster"], vd / "poster.png")
        save(r["hisam_image"], vd / "hisam.jpg", 88)   # Hi-SAM: nét chữ thô
        save(r["ink_image"], vd / "ink.jpg", 90)       # mặt nạ gỡ chữ (nét, quầng, dòng giữ, nét không thuộc dòng nào)
        save(r["clean"], vd / "clean.jpg", 92)         # LaMa: gỡ mọi dòng -- nền VLM nhìn
        save(r["base"], vd / "base.jpg", 92)           # nền dựng: dòng "keep" đã trả về điểm ảnh nháp
        save(r["steps"], vd / "steps.jpg", 88)
        P = r["plan"]
        M = {**P["slots"], "size": [d["draft"].shape[1], d["draft"].shape[0]],
             "by": {m["id"]: m for m in P["slots"]["L"]}}
        save(np.asarray(ops_image(r["draft"], M, P)), vd / "ops.jpg", 88)
        save(np.asarray(marked_image(r["draft"], M)), vd / "slots.jpg", 88)
        first = r["poster_first"] is not r["poster"]
        if first:
            save(r["poster_first"], vd / "poster_first.png")
        timing = {"llm": t_llm, **r["timing"]}
        (vd / "plan.json").write_text(json.dumps(jsonable({"seed": v["seed"], "direction": v.get("direction"), "prompt_en": Bv["prompt_en"], "kho": Bv.get("kho") or [],
                                                           "timing": timing, **P}), ensure_ascii=False, indent=1), encoding="utf-8")
        v.update(status="done", stage="", poster_url=f"{url}/poster.png", steps_url=f"{url}/steps.jpg", ops_url=f"{url}/ops.jpg",
                 slots_url=f"{url}/slots.jpg", plan_url=f"{url}/plan.json", hisam_url=f"{url}/hisam.jpg", ink_url=f"{url}/ink.jpg", clean_url=f"{url}/clean.jpg", base_url=f"{url}/base.jpg", timing=timing, missing=P["missing"],
                 errors=len(P["errors"]), ops=sum(o.get("kind") not in ("skip", "keep") for o in P["ops"]),
                 **({"poster_first_url": f"{url}/poster_first.png"} if first else {}))
        if any(e.get("type") == "empty_plan" for e in P["errors"]):   # D1: VLM không ghép được gì -> lỗi, không trả nháp
            v.update(status="error", error={"code": "vlm_empty_plan", "stage": "VLM ghép câu",
                                            "message": "VLM không ghép câu khách nào vào dòng nào"})
        log.info("%s v%d xong (gpu %s) %s", job["run_id"], v["index"], v.get("gpu"), timing)
    except Exception as e:
        log.error("%s v%d lỗi thiết kế: %s", job["run_id"], v["index"], traceback.format_exc())
        st = v.get("stage") or ""
        code = next((c for k, c in DESIGN_STAGES if k in st), "design_failed")
        v.update(status="error", stage="", error=failure(code, st, e))


def image_one(job: dict, v: dict, d: dict, Bv: dict, vd: Path, url: str, t_llm: float) -> None:
    """ẢNH THƯỜNG (B["mode"] == "image": không chữ khách): ảnh FLUX vẽ là kết quả, không dò chữ / VLM / dựng."""
    try:
        save(d["draft"], vd / "poster.png")
        timing = {"llm": t_llm, **{k: round(x, 1) for k, x in d["timing"].items()}}
        (vd / "plan.json").write_text(json.dumps({"mode": "image", "seed": v["seed"], "direction": v.get("direction"), "kho": Bv.get("kho") or [],
                                                  "prompt_en": Bv["prompt_en"], "timing": timing}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
        v.update(status="done", stage="", poster_url=f"{url}/poster.png", timing=timing, missing=[], errors=0, ops=0)
        log.info("%s v%d xong (ảnh thường, gpu %s) %s", job["run_id"], v["index"], v.get("gpu"), timing)
    except Exception as e:
        log.error("%s v%d lỗi lưu ảnh: %s", job["run_id"], v["index"], traceback.format_exc())
        v.update(status="error", stage="", error=failure("internal_error", "lưu ảnh", e))


def run_job(job: dict, req: GenerateRequest) -> dict:
    """Một yêu cầu: LLM một lần (brief + các hướng thiết kế), rồi n phiên bản vào dây chuyền VẼ -> THIẾT KẾ. Kết quả từng phiên
    bản ghi vào job["result"]["posters"] ngay khi xong (UI hiện dần)."""
    from textfix.brief import make_brief
    from textfix.draw.flux import SIZES
    t_all = time.time()
    run_id = job["run_id"]
    rd = RUNS / run_id
    rd.mkdir(parents=True, exist_ok=True)
    form = form_of(req)
    prompt = (req.prompt or "").strip() or None
    product = decode_image(req.image_base64)
    design_hint = {"style": req.style_pref or "auto", "color": req.primary_color or None}
    (rd / "request.json").write_text(json.dumps({**req.model_dump(exclude={"image_base64"}), "has_product_image": product is not None},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    if product is not None:
        save(product, rd / "product.png")
    n = max(1, min(4, int(req.num_images or 1)))
    job["progress"] = "LLM viết prompt" + (f" + {n - 1} hướng thiết kế" if n > 1 else "")
    t0 = time.time()
    try:
        B = make_brief(STATE["llm"], prompt, form if len(form) > 1 else None, design_hint, product is not None, n_variants=n - 1,
                       aspect=req.aspect_ratio)
    except Exception as e:
        raise ApiError("llm_failed", "LLM viết prompt", f"{type(e).__name__}: {e}") from e
    t_llm = round(time.time() - t0, 1)
    (rd / "brief.json").write_text(json.dumps(B, ensure_ascii=False, indent=1), encoding="utf-8")
    plain = B.get("mode") == "image"   # không chữ khách: chỉ FLUX vẽ
    W, H = SIZES.get(req.aspect_ratio, SIZES["1:1"])
    base_seed = req.seed if req.seed is not None else int(time.time()) % 100000
    directions = [{"name": "chính", "prompt_en": B["prompt_en"]}] + B.get("variants", [])
    posters = []
    job["result"] = {"status": "running", "job_id": job["id"], "run_id": run_id, "mode": B.get("mode", "poster"), "width": W, "height": H, "posters": posters,
                     "ignored_fields": ignored_fields(req), "debug_zip_url": f"api/runs/{run_id}/zip",
                     "brief": {"prompt_en": B["prompt_en"], "texts": B["texts"], "variants": B.get("variants", []), "kho": B.get("kho") or [],
                               "log": B["log"]}}
    futs = []
    for i in range(n):
        dirn = directions[i] if i < len(directions) else directions[0]   # thiếu hướng: dùng prompt chính, seed khác
        v = {"index": i, "seed": base_seed + i * 9973, "direction": dirn["name"], "status": "drawing", "stage": "chờ GPU"}
        posters.append(v)
        vd = rd / f"v{i}"
        vd.mkdir(exist_ok=True)
        Bv = {**B, "prompt_en": dirn["prompt_en"]}
        fut: Future = Future()
        DRAW_Q.put({"v": v, "fut": fut, "B": Bv, "W": W, "H": H, "product": product,
                    "progress": lambda m, v=v: v.update(stage=m)})

        def drawn(f, v=v, Bv=Bv, vd=vd, i=i):
            if f.exception() is not None:
                log.error("%s v%d lỗi vẽ: %s", run_id, i, f.exception())
                v.update(status="error", stage="", error=draw_error(f.exception(), v.get("stage") or "FLUX vẽ nháp"))
                return
            if plain:
                futs.append(DESIGN.submit(image_one, job, v, f.result(), Bv, vd, f"outputs/{run_id}/v{i}", t_llm))
                return
            v.update(status="designing", stage="chờ thiết kế")
            futs.append(DESIGN.submit(design_one, job, v, f.result(), Bv, product is not None, vd, f"outputs/{run_id}/v{i}", t_llm))
        fut.add_done_callback(drawn)
        futs.append(fut)
    while True:   # chờ mọi phiên bản (danh sách futs lớn dần khi bản vẽ xong được đưa sang thiết kế)
        pending = [f for f in list(futs) if not f.done()]
        if not pending and all(v["status"] in ("done", "error") for v in posters):
            break
        done = sum(v["status"] == "done" for v in posters)
        job["progress"] = f"{done}/{n} xong · " + " · ".join(f"ảnh {v['index'] + 1}: {v['stage']}" for v in posters
                                                                if v["status"] not in ("done", "error"))
        time.sleep(0.3)
    ok = [p for p in posters if p["status"] == "done"]
    if not ok:
        e = posters[0].get("error") or failure("internal_error", "", None)
        raise ApiError(e["code"], e["stage"], e["message"])
    prune()
    job["result"].update(status="success", elapsed_seconds=round(time.time() - t_all, 1), final_poster_url=ok[0]["poster_url"],
                         poster=f"/api/jobs/{job['id']}/poster")
    return job["result"]


def prune() -> None:
    ds = sorted([d for d in RUNS.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
    for d in ds[:-MAX_RUNS]:
        shutil.rmtree(d, ignore_errors=True)


def check(req: GenerateRequest) -> None:
    """Đầu vào sai -> 422, model chưa nạp -> 503 (trước khi tải ảnh theo link hay xếp hàng)."""
    validate(req)
    form = form_of(req)
    miss = missing_fields(req, form)
    if miss:
        raise HTTPException(422, {"code": "missing_required_fields", "category": form["category"], "fields": miss})
    if not STATE["flux"] or STATE.get("vlm") is None:
        raise HTTPException(503, {"code": "model_not_ready", "message": STATE["error"] or "đang nạp model"})


async def admit(req: GenerateRequest) -> dict:
    """Kiểm -> tải ảnh theo link (luồng phụ: không chặn vòng sự kiện) -> job mới. Ảnh tải về đi tiếp như image_base64."""
    check(req)
    if req.image_url:
        raw = await asyncio.to_thread(fetch_image, req.image_url)
        req.image_base64 = base64.b64encode(raw).decode()
    return new_job(req)


def new_job(req: GenerateRequest) -> dict:
    """Ghi job vào hàng đợi (đầu vào đã qua check)."""
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "run_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{jid[:4]}", "status": "queued",
           "progress": "đang xếp hàng", "created": time.time(), "result": None, "error": None}
    JOBS[jid] = job
    ORDER.append(jid)
    return job


def work(job: dict, req: GenerateRequest) -> None:
    job["status"], job["started"] = "running", time.time()
    try:
        job["result"] = run_job(job, req)
        job["status"] = "done"
    except ApiError as e:
        log.error("job %s lỗi %s: %s", job["id"], e.code, traceback.format_exc())
        job["status"], job["error"] = "error", e.as_dict()
    except Exception as e:
        log.error("job %s lỗi: %s", job["id"], traceback.format_exc())
        job["status"], job["error"] = "error", failure("internal_error", job.get("progress") or "", e)
    finally:
        job["progress"] = ""
        if job["id"] in ORDER:
            ORDER.remove(job["id"])
        for k in [k for k, j in JOBS.items() if j["status"] in ("done", "error") and time.time() - j["created"] > 6 * 3600]:
            JOBS.pop(k, None)


def sync_kho() -> None:
    """Kho ảnh tham chiếu (textfix/kho_sync.py): đồng bộ từ kho đối tượng về thư mục đệm. Không bao giờ ném lỗi."""
    from textfix.kho_sync import sync_from_env
    STATE["kho"] = r = sync_from_env()
    (log.warning if r["status"] == "error" else log.info)("Kho ảnh tham chiếu: %s", r)


# ---------------------------------------------------------------------------------------------------- HTTP
@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["started"] = time.time()
    asyncio.get_running_loop().run_in_executor(None, sync_kho)   # song song với nạp model; hỏng thì vẽ như chưa có kho
    try:
        await asyncio.get_running_loop().run_in_executor(None, init_models, STATE["args"])
    except BaseException as e:   # máy chủ vẫn lên, /api/health báo lỗi
        if isinstance(e, KeyboardInterrupt):
            raise
        STATE["error"] = STATE["error"] or f"{type(e).__name__}: {e}"
        log.error("KHÔNG nạp được model: %s", traceback.format_exc())
    yield
    for _ in STATE["flux"]:
        DRAW_Q.put(None)


app = FastAPI(title="TextFix", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=str(RUNS)), name="outputs")


@app.get("/")
async def ui():
    return FileResponse(UI, media_type="text/html", headers={"Cache-Control": "no-store"})   # pull mã mới là thấy UI mới


@app.get("/api/health")
async def health():
    F, m = STATE["flux"], STATE.get("models")
    return {"ready": bool(F) and STATE.get("vlm") is not None, "error": STATE["error"],
            "flux": F[0].info() if F else None, "gpus": [f.dev_dit for f in F],
            "models": {k: {"url": v["url"], "model": v["model"]} for k, v in m.items()} if m else None,
            "queue": len(ORDER), "draw_queue": DRAW_Q.qsize(), "design_workers": DESIGN._max_workers, "kho": STATE.get("kho"),
            "uptime_s": round(time.time() - (STATE["started"] or time.time()))}


@app.post("/api/jobs")
async def create_job(req: GenerateRequest):
    job = await admit(req)
    JOBRUN.submit(work, job, req)
    return {"job_id": job["id"], "run_id": job["run_id"], "queue": len(ORDER)}


@app.get("/api/jobs/{jid}")
async def get_job(jid: str):
    job = JOBS.get(jid)
    if job is None:
        raise HTTPException(404, "không có job này (máy chủ đã khởi động lại?)")
    pos = ORDER.index(jid) if jid in ORDER else None
    out = {k: job[k] for k in ("id", "run_id", "status", "progress", "result", "error")} | \
        {"position": pos, "elapsed": round(time.time() - job.get("started", job["created"]), 1)}
    if job["status"] == "done":   # lấy ảnh: không phải ghép đường dẫn tương đối, chỉ cần job_id
        out["poster"] = f"/api/jobs/{jid}/poster"
    return out


@app.get("/api/jobs/{jid}/poster")
def get_poster(jid: str, index: Optional[int] = None, format: str = "png"):
    """Ảnh poster của job: PNG (mặc định) hoặc JSON base64 (format=base64). index: poster thứ mấy (num_images > 1); bỏ trống =
    poster xong đầu tiên. Chưa xong -> 409 not_ready; poster lỗi -> 409 poster_failed (kèm error); tệp đã bị dọn -> 410."""
    job = JOBS.get(jid)
    if job is None:
        raise HTTPException(404, {"code": "job_not_found", "message": "không có job này (máy chủ đã khởi động lại?)"})
    if format not in ("png", "base64"):
        raise HTTPException(422, {"code": "invalid_format", "value": format, "allowed": ["png", "base64"]})
    if job["status"] == "error":
        raise HTTPException(409, {"code": "job_failed", "error": job["error"]})
    posters = (job.get("result") or {}).get("posters") or []
    if index is None:
        p = next((p for p in posters if p.get("status") == "done"), None)
        if p is None:
            raise HTTPException(409, {"code": "not_ready", "status": job["status"], "progress": job["progress"]})
    else:
        if job["status"] == "done" and not 0 <= index < len(posters):
            raise HTTPException(404, {"code": "poster_not_found", "index": index, "count": len(posters)})
        p = posters[index] if 0 <= index < len(posters) else None
        if p is None or p.get("status") not in ("done", "error"):
            raise HTTPException(409, {"code": "not_ready", "status": job["status"], "progress": job["progress"]})
        if p["status"] == "error":
            raise HTTPException(409, {"code": "poster_failed", "index": index, "error": p.get("error")})
    f = RUNS / p["poster_url"].split("/", 1)[1]   # "outputs/<run_id>/v<i>/poster.png" -> RUNS/<run_id>/v<i>/poster.png
    if not f.is_file():
        raise HTTPException(410, {"code": "poster_expired", "message": "ảnh đã bị dọn (máy chủ chỉ giữ các lượt gần nhất)"})
    if format == "base64":
        r = job["result"]
        return {"job_id": jid, "index": p["index"], "width": r["width"], "height": r["height"], "mime": "image/png",
                "image_base64": base64.b64encode(f.read_bytes()).decode()}
    return FileResponse(f, media_type="image/png", filename=f"{jid}_{p['index']}.png")


ZIP_LIMIT = int(float(os.environ.get("TEXTFIX_ZIP_MB", "22")) * 1024 * 1024)   # < 24 MB (như textfix/tools/ship.py)
ZIP_JPG = ("poster.png", "poster_first.png")   # ảnh để xem -> JPG; nháp + mặt nạ nét chữ giữ PNG (đo lại được)


def run_zip(rd: Path, all_jpg: bool = False) -> bytes:
    """Mọi tệp debug của một lượt chạy (request, brief, ảnh sản phẩm, mỗi phiên bản: nháp, nét chữ, nền đã gỡ, dòng, lệnh, các bước,
    poster, plan.json) trong MỘT zip, thư mục gốc <run_id>/ -- giải nén vào output/runs/ ở máy cá nhân. all_jpg: mọi PNG thành JPG
    95 (khi zip vượt giới hạn)."""
    import zipfile
    from PIL import Image
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in sorted(x for x in rd.rglob("*") if x.is_file()):
            arc = f"{rd.name}/{f.relative_to(rd).as_posix()}"
            if f.suffix == ".png" and (all_jpg or f.name in ZIP_JPG):
                b = io.BytesIO()
                Image.open(f).convert("RGB").save(b, "JPEG", quality=95 if all_jpg else 92)
                z.writestr(arc[:-4] + ".jpg", b.getvalue(), compress_type=zipfile.ZIP_STORED)
            else:
                z.write(f, arc, compress_type=zipfile.ZIP_STORED if f.suffix in (".png", ".jpg") else zipfile.ZIP_DEFLATED)
    return buf.getvalue()


@app.get("/api/runs/{run_id}/zip")
def download_run(run_id: str):
    import re
    rd = RUNS / run_id
    if not re.fullmatch(r"run_[0-9A-Za-z_]+", run_id) or not rd.is_dir():
        raise HTTPException(404, "không có lượt chạy này")
    data = run_zip(rd)
    if len(data) >= ZIP_LIMIT:
        data = run_zip(rd, all_jpg=True)
    if len(data) >= ZIP_LIMIT:
        raise HTTPException(413, f"zip {len(data) / 2**20:.1f} MB vượt giới hạn {ZIP_LIMIT / 2**20:.0f} MB")
    return Response(data, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{run_id}.zip"'})


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    job = await admit(req)
    await asyncio.wrap_future(JOBRUN.submit(work, job, req))
    if job["status"] != "done":
        return JSONResponse({"status": "error", "error": job["error"], "run_id": job["run_id"]}, status_code=500)
    return job["result"]


# ---------------------------------------------------------------------------------------------------- API v1 (chatbot)
V1_PATH = "/api/v1/images/generations"


def v1_error_response(e: "v1.V1Error", rid: str) -> JSONResponse:
    hdr = {"X-Request-Id": rid, **({"Retry-After": str(e.retry_after)} if e.retry_after else {})}
    return JSONResponse(v1.body_error(e, rid), status_code=e.http, headers=hdr)


@app.exception_handler(RequestValidationError)
async def _validation(request: Request, exc: RequestValidationError):
    """Thân sai khuôn: API v1 trả 400 theo khuôn lỗi của nó; các đường còn lại giữ 422 mặc định của FastAPI."""
    if request.url.path.endswith(V1_PATH):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        return v1_error_response(v1.V1Error(400, "ERR_INVALID_REQUEST", f"thân yêu cầu sai khuôn: {exc.errors()[:3]}"), rid)
    return await request_validation_exception_handler(request, exc)


@app.post(V1_PATH)
async def generations_v1(req: v1.V1Request, request: Request):
    """MỘT lời gọi đồng bộ: prompt (+ ảnh theo link) -> một poster. Lõi = job của API demo (run_job), chờ tối đa
    TEXTFIX_V1_DEADLINE giây (55); quá hạn -> 504, job vẫn chạy nốt (zip debug theo run_id trong log)."""
    t0 = time.time()
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    aspect = req.aspect_ratio or "1:1"
    try:
        v1.check(req)
        if not STATE["flux"] or STATE.get("vlm") is None:   # chi tiết lỗi nạp (có thể chứa địa chỉ nội bộ): /api/health, log
            raise v1.V1Error(503, "ERR_SERVICE_UNAVAILABLE", "lỗi nạp model" if STATE["error"] else "đang nạp model",
                             retry_after=30)
        if len(ORDER) >= int(os.environ.get("TEXTFIX_MAX_QUEUE", "8")):
            raise v1.V1Error(503, "ERR_SERVICE_BUSY", f"hàng đợi đầy ({len(ORDER)} yêu cầu)", retry_after=15)
        urls = [u for u in req.image_urls if (u or "").strip()]
        image_b64 = None
        if urls:   # bản đầu: ảnh đầu là ảnh sản phẩm (FLUX image 1); các ảnh sau chưa dùng
            try:
                image_b64 = base64.b64encode(await asyncio.to_thread(fetch_image, urls[0])).decode()
            except HTTPException as e:
                raise v1.image_error(e.detail if isinstance(e.detail, dict) else {"message": str(e.detail)}) from e
        greq = GenerateRequest(category="promo", prompt=req.prompt.strip(), aspect_ratio=aspect, num_images=1,
                               image_base64=image_b64, v1_request_id=rid, v1_image_urls=urls)
        job = new_job(greq)
        log.info("v1 %s -> %s (ảnh: %d link%s; khoá lạ: %s)", rid, job["run_id"], len(urls), ", chỉ dùng ảnh đầu" if len(urls) > 1 else "",
                 sorted((req.model_extra or {}).keys()))
        fut = asyncio.wrap_future(JOBRUN.submit(work, job, greq))
        try:
            await asyncio.wait_for(asyncio.shield(fut), float(os.environ.get("TEXTFIX_V1_DEADLINE", "55")))
        except asyncio.TimeoutError:
            raise v1.V1Error(504, "ERR_UPSTREAM_TIMEOUT", f"quá hạn, run {job['run_id']} vẫn chạy nốt") from None
        if job["status"] != "done":
            log.info("v1 %s run %s lỗi chi tiết: %s", rid, job["run_id"], job["error"])
            raise v1.job_error(job["error"])
        r = job["result"]
        p = next(p for p in r["posters"] if p.get("status") == "done")
        png = (RUNS / p["poster_url"].split("/", 1)[1]).read_bytes()
    except v1.V1Error as e:
        log.info("v1 %s lỗi %d %s (%.1fs): %s", rid, e.http, e.error_code, time.time() - t0, e.message)
        return v1_error_response(e, rid)
    except Exception as e:   # không để lọt lỗi khuôn FastAPI / chuỗi rỗng
        log.error("v1 %s lỗi: %s", rid, traceback.format_exc())
        return v1_error_response(v1.V1Error(500, "ERR_IMAGE_GENERATION_FAILED", f"internal_error ({type(e).__name__})"), rid)
    ms = int((time.time() - t0) * 1000)
    log.info("v1 %s xong %s (%.1fs)", rid, job["run_id"], ms / 1000)
    if "image/png" in (request.headers.get("accept") or ""):
        return Response(png, media_type="image/png", headers=v1.headers(rid, aspect, ms))
    return JSONResponse(v1.body_success(rid, base64.b64encode(png).decode(), aspect, r["width"], r["height"], ms),
                        headers={"X-Request-Id": rid})


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser(description="Máy chủ TextFix")
    ap.add_argument("--host", default=os.environ.get("TEXTFIX_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("TEXTFIX_PORT", "8088")))
    ap.add_argument("--model", choices=["distill", "base"], default=os.environ.get("TEXTFIX_FLUX", "distill"))
    ap.add_argument("--steps", type=int, default=int(os.environ["TEXTFIX_STEPS"]) if os.environ.get("TEXTFIX_STEPS") else None,
                    help="số bước khử nhiễu (mặc định BFL: distill 4, base 50)")
    ap.add_argument("--weights-dir", default=os.environ.get("TEXTFIX_WEIGHTS"),
                    help="thư mục trọng số FLUX.2-klein (nếu không ở ~/persistent-data)")
    STATE["args"] = ap.parse_args()
    uvicorn.run(app, host=STATE["args"].host, port=STATE["args"].port)


if __name__ == "__main__":
    main()
