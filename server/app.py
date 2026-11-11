#!/usr/bin/env python3
"""MÁY CHỦ TEXTFIX -- MỘT API cho chatbot (docs/API_V1.md):

  POST /api/v1/images/generations   lời chat (+ ảnh sản phẩm) -> MỘT poster, đồng bộ; gửi lại poster đã trả + lời sửa = SỬA
  GET  /api/health                  {ready, queue}

Dây chuyền mỗi yêu cầu: LLM (prompt FLUX + câu khách) -> FLUX.2-klein vẽ nháp -> Hi-SAM tìm nét chữ -> dòng (OCR + nét chữ) ->
gỡ chữ (LaMa, song song với VLM) -> VLM ghép câu -> dựng (Chromium) -> kiểm -> vòng sửa -> poster.

  python server/app.py --port 8088                  # FLUX tại chỗ, bản distill (4 bước)
  TEXTFIX_FLUX_URL=... python server/app.py         # FLUX ở máy khác (docs/DEPLOY.md): card chỉ chạy Hi-SAM / LaMa

Mỗi lượt chạy ghi vào TEXTFIX_RUNS/<run_id>/ (header X-Run-Id của câu trả lời): request.json, brief.json, v0/draft.png, ink.png,
poster.png, plan.json (có timing) -- và ảnh debug (steps.jpg, ops.jpg, slots.jpg, ...) ghi SAU khi đã trả poster. Không có
đường HTTP nào đọc các tệp này: xem trên máy chủ (scripts/dev_batch.py gói zip).

DÂY CHUYỀN: mỗi card GPU một FLUX + một luồng VẼ (chung hàng đợi; FLUX ở xa: TEXTFIX_FLUX_CONCURRENCY luồng), nhóm luồng THIẾT KẾ
(TEXTFIX_DESIGN_WORKERS=8, mỗi luồng một Chromium -- Playwright đồng bộ không chạy trong vòng lặp asyncio). Các yêu cầu gối đầu nhau.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
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
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

from textfix.config import load_env  # noqa: E402

load_env()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("textfix.server")

import contract as C  # noqa: E402   hợp đồng API: khuôn yêu cầu / trả lời, mã lỗi, tải ảnh theo link

RUNS = Path(os.environ.get("TEXTFIX_RUNS", ROOT / "output" / "runs"))
RUNS.mkdir(parents=True, exist_ok=True)
MAX_RUNS = int(os.environ.get("TEXTFIX_MAX_RUNS", "300"))
STATE: dict[str, Any] = {"flux": [], "llm": None, "vlm": None, "models": None, "error": None, "args": None, "started": None,
                         "ready": False}   # ready: CHỈ sau khi nạp + làm nóng xong (11/11: làm nóng hỏng mà vẫn nhận việc)
ORDER: list[str] = []   # các lượt đang chạy (run_id): đầy TEXTFIX_MAX_QUEUE -> 503 bận

# ---------------------------------------------------------------------------------------------------- dây chuyền
#   VẼ (GPU)        mỗi card một FLUX + một Hi-SAM + một luồng, chung một hàng đợi          nháp -> nét chữ
#   THIẾT KẾ        nhóm luồng, mỗi luồng một Chromium; VLM ở máy khác nên gọi song song    dòng -> VLM -> dựng -> kiểm
# LLM viết brief ở luồng điều phối của yêu cầu (JOBRUN). Ảnh debug ghi ở SAVE, sau khi đã trả poster.
DRAW_Q: "queue.Queue" = queue.Queue()
# 8 = TEXTFIX_MAX_QUEUE (12/11): luồng thiết kế phần lớn thời gian CHỜ VLM qua mạng (~6 s); 4 luồng thì 8 yêu cầu cùng lúc phải
# chờ luồng tới 8 s (notebook, vòng tải). Mỗi luồng thêm một Chromium (~200 MB RAM), gần như không thêm CPU.
DESIGN = ThreadPoolExecutor(max_workers=int(os.environ.get("TEXTFIX_DESIGN_WORKERS", "8")), thread_name_prefix="design")
JOBRUN = ThreadPoolExecutor(max_workers=int(os.environ.get("TEXTFIX_JOB_WORKERS", "8")), thread_name_prefix="job")
SAVE = ThreadPoolExecutor(max_workers=2, thread_name_prefix="save")
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
    from textfix.draw import hisam, remote
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
        fut = task["fut"]
        if fut.set_running_or_notify_cancel():
            try:
                task["timing"]["cho_ve"] = round(time.time() - task["t0"], 1)
                # draft có sẵn (ảnh lạ cần đọc chữ): chỉ Hi-SAM; refs (sửa poster): [poster cũ, ảnh sản phẩm mới?]
                fut.set_result(draw(F, task["B"], task["W"], task["H"], task["seed"], task["product"], draft=task.get("draft"),
                                    refs=task.get("refs"), progress=task["progress"]))
            except BaseException as e:
                fut.set_exception(e)


def init_models(args) -> None:
    """LLM / VLM (gọi qua mạng), một luồng VẼ cho mỗi card, làm nóng. Chạy ở luồng phụ (lúc khởi động)."""
    from textfix import config, erase
    from textfix.brief import SCHEMA
    from textfix.draw import hisam
    from textfix.llm import json_call
    if not os.environ.get("TEXTFIX_LLM_URL") or not os.environ.get("TEXTFIX_MODEL"):
        raise RuntimeError("chưa đặt TEXTFIX_LLM_URL / TEXTFIX_MODEL (export trong terminal, docs/SERVER.md)")
    lost = hisam.missing() + ([] if erase.lama_ready() else [erase.LAMA_PATH])
    if lost:   # người dùng chốt 14/10: thiếu tệp thì báo rõ và dừng, không lùi về cách nào khác
        raise RuntimeError("thiếu tệp model gỡ chữ (docs/SERVER.md, mục Hi-SAM và LaMa): " + "; ".join(lost))
    m = config.models()
    STATE["models"] = m
    STATE["llm"] = json_call(m["llm"]["url"], m["llm"]["model"], m["llm"]["key_env"], tag="brief", schema=SCHEMA)
    STATE["vlm"] = json_call(m["vlm"]["url"], m["vlm"]["model"], m["vlm"]["key_env"], tag="designer")
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
    warm(sorted(set(devs)))
    STATE["ready"] = True
    log.info("Sẵn sàng: %d luồng vẽ %s (model vẽ %s), %d luồng thiết kế; LLM %s | VLM %s", len(STATE["flux"]), devs,
             STATE["flux"][0].info(), DESIGN._max_workers, m["llm"]["model"], m["vlm"]["model"])


def warm(devs: list[str]) -> None:
    """LÀM NÓNG trước khi báo "Sẵn sàng" (11/11: ba yêu cầu đầu sau khi bật máy chủ cùng quá hạn 55 s vì phải nạp hộ):
    - mỗi thiết bị: Hi-SAM + LaMa chạy thử (TorchScript tối ưu ở vài lượt đầu: chạy hai lượt);
    - MỖI luồng thiết kế: mở Chromium + nạp OCR của luồng đó (rào chắn: mỗi việc làm nóng nằm trên một luồng khác nhau).
    Hỏng (Chromium không mở được...) -> ném lỗi: /api/health báo chưa sẵn sàng, API trả 503, thay vì nhận việc rồi lỗi."""
    from textfix.draw import hisam
    from textfix.erase.lama import inpaint
    from textfix.lines.ocr import _engine
    t0 = time.time()
    img = np.full((1024, 1024, 3), 200, np.uint8)
    img[400:520, 200:820] = 30
    mask = np.zeros(img.shape[:2], bool)
    mask[400:520, 200:820] = True
    for dev in devs:
        hisam.strokes(img, None if dev == "cpu" else dev)
        for _ in range(2):
            inpaint(img, mask, None if dev == "cpu" else dev)
    n = DESIGN._max_workers
    gate = threading.Barrier(n)

    def one() -> None:
        page()
        _engine()(img[380:540, 180:840])
        gate.wait(timeout=180)
    for f in [DESIGN.submit(one) for _ in range(n)]:
        f.result()
    # bước LLM: tokenizer FLUX (đếm token prompt, ~20 s lần đầu) + chỉ mục kho -- nạp lười ở yêu cầu đầu tiên (12/11)
    from textfix import kho
    from textfix.brief import count_tokens
    count_tokens("warm")
    kho.match_detail("phở bò")
    log.info("làm nóng xong (%.0fs): Hi-SAM + LaMa trên %s, Chromium + OCR trên %d luồng thiết kế", time.time() - t0, devs, n)


def ready() -> bool:
    """Nhận việc được: luồng vẽ + VLM đã nạp, làm nóng xong (Chromium mở được...), không có lỗi nạp."""
    return bool(STATE["ready"] and STATE["flux"] and STATE.get("vlm") is not None and not STATE["error"])


def page():
    """Chromium của luồng thiết kế hiện tại (mở lần đầu dùng)."""
    if getattr(LOCAL, "browser", None) is None:
        from textfix.render.browser import Browser
        LOCAL.browser = Browser().__enter__()
    return LOCAL.browser.page


def png_bytes(img: np.ndarray, info: dict | None = None) -> bytes:
    """PNG nén mức 3: cùng điểm ảnh, cỡ tệp gần bằng mức 6 mặc định, mã hoá nhanh gấp đôi (1 MP: 0,1 s thay 0,18 s).
    info: thông tin poster giấu trong file (textfix/meta.py) -- gửi lại ảnh này kèm lời sửa là sửa được."""
    from textfix.meta import save_png
    return save_png(img, info)


def save(img: np.ndarray, path: Path, quality: int | None = None) -> None:
    from PIL import Image
    if path.suffix == ".png":
        path.write_bytes(png_bytes(img))
    else:
        Image.fromarray(np.asarray(img)).save(path, quality=quality or 90)


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


def save_debug(r: dict, d: dict, vd: Path) -> None:
    """Ảnh debug của một lượt (SAU khi đã trả poster, luồng SAVE). Hỏng chỉ ghi log."""
    try:
        from textfix.draw import hisam
        from textfix.engine import debug_images
        from textfix.lines.view import marked_image
        from textfix.render import ops_image
        save(d["draft"], vd / "draft.png")
        hisam.to_png(d["ink"], vd / "ink.png")   # mặt nạ nét chữ: đủ để dựng lại tại máy cùng nháp + plan.json
        if r is None:
            return
        r = debug_images(r)
        save(r["hisam_image"], vd / "hisam.jpg", 88)   # Hi-SAM: nét chữ thô
        save(r["ink_image"], vd / "ink.jpg", 90)       # mặt nạ gỡ chữ (nét, quầng, dòng giữ, nét không thuộc dòng nào)
        save(r["clean"], vd / "clean.jpg", 92)         # LaMa: gỡ mọi dòng
        save(r["base"], vd / "base.jpg", 92)           # nền dựng: dòng "keep" đã trả về điểm ảnh nháp
        save(r["steps"], vd / "steps.jpg", 88)
        P = r["plan"]
        M = {**P["slots"], "size": [d["draft"].shape[1], d["draft"].shape[0]], "by": {m["id"]: m for m in P["slots"]["L"]}}
        save(np.asarray(ops_image(r["draft"], M, P)), vd / "ops.jpg", 88)
        save(np.asarray(marked_image(r["draft"], M)), vd / "slots.jpg", 88)
        if r["poster_first"] is not r["poster"]:
            save(r["poster_first"], vd / "poster_first.png")
    except Exception:
        log.error("%s: lỗi ghi ảnh debug: %s", vd, traceback.format_exc())


def design_step(d: dict, B: dict, product: bool, vd: Path, timing: dict, t_q: float, info: dict | None = None) -> bytes:
    """TRẠM THIẾT KẾ (luồng DESIGN) -> PNG poster. Lỗi -> ApiError với mã theo bước hỏng."""
    from textfix.engine import design
    from textfix.llm import collect
    timing["cho_thiet_ke"] = round(time.time() - t_q, 1)
    stage = {"s": ""}
    try:
        with collect() as calls:   # số token của lời gọi VLM -> plan.json
            r = design(d, B, STATE["vlm"], page(), product=product, progress=lambda m: stage.update(s=m), debug=False)
    except Exception as e:
        code = next((c for k, c in C.DESIGN_STAGES if k in stage["s"]), "design_failed")
        raise C.ApiError(code, stage["s"], f"{type(e).__name__}: {e}") from e
    t0 = time.time()
    P = r["plan"]
    if info is not None:   # câu viết ở nhiều chỗ (nút lặp...): lần sửa sau phải biết để FLUX không vẽ chỗ thứ hai trống
        from textfix.meta import set_shown
        set_shown(info, P["ops"])
    png = png_bytes(r["poster"], info)
    (vd / "poster.png").write_bytes(png)
    timing.update(r["timing"], luu=round(time.time() - t0, 1))
    (vd / "plan.json").write_text(json.dumps(jsonable({"seed": B.get("seed"), "prompt_en": B["prompt_en"], "kho": B.get("kho") or [],
                                                       "timing": timing, "calls": B.get("calls", []) + calls,
                                                       **({"edit": B["edit"]} if B.get("edit") else {}), **P}),
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    SAVE.submit(save_debug, r, d, vd)
    if any(e.get("type") == "empty_plan" for e in P["errors"]):   # VLM không ghép được gì -> lỗi, không trả nháp
        raise C.ApiError("vlm_empty_plan", "VLM ghép câu", "VLM không ghép câu khách nào vào dòng nào")
    return png



def generate(run_id: str, req: C.Request, product: np.ndarray | None, rid: str, old: dict | None = None,
             old_img: np.ndarray | None = None) -> tuple[bytes, int, int, dict, str]:
    """MỘT yêu cầu (luồng JOBRUN): LLM -> VẼ -> THIẾT KẾ -> (PNG, W, H, timing, khung). Lỗi -> ApiError.
    old / old_img: poster của mình cần sửa (khối thông tin trong file + ảnh). Ảnh kèm theo không có khối thông tin mà LLM
    brief bảo là yêu cầu SỬA ảnh đó -> ảnh lạ: đọc chữ trên ảnh (read_foreign) rồi đi chung luồng sửa (11/11)."""
    from textfix.brief import make_brief
    from textfix.draw.flux import SIZES
    from textfix.edit import make_edit, nearest_ratio
    from textfix.llm import collect
    from textfix.meta import info as poster_info
    t_all = time.time()
    rd = RUNS / run_id
    vd = rd / "v0"
    vd.mkdir(parents=True, exist_ok=True)
    aspect = req.aspect_ratio or "1:1"
    ignored = sorted((req.model_extra or {}).keys())
    if old is not None and "aspect_ratio" in req.model_fields_set:
        ignored.append("aspect_ratio")   # sửa: giữ khung cũ (chatbot có thể luôn gửi 1:1), đổi khung chỉ khi lời sửa nói

    def request_json(mode: str) -> None:
        (rd / "request.json").write_text(json.dumps({"request_id": rid, "prompt": req.prompt, "aspect_ratio": aspect,
                                                     "n_images": len(req.b64_images()), "has_product_image": product is not None,
                                                     "mode": mode, "parent": (old or {}).get("run_id"),
                                                     "depth": (old or {}).get("depth"), "ignored_fields": ignored},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    request_json("edit" if old is not None else "new")
    if product is not None:
        save(product, rd / "product.png")
    if old_img is not None:
        save(old_img, rd / "source.png")
    timing: dict = {"llm": 0.0}

    def llm_step(f):
        t0 = time.time()
        try:
            return f()
        except Exception as e:
            raise C.ApiError("llm_failed", "LLM viết prompt" if old is None else "LLM sửa", f"{type(e).__name__}: {e}") from e
        finally:
            timing["llm"] = round(timing["llm"] + time.time() - t0, 1)

    with collect() as calls:   # số token của lời gọi LLM / VLM (prompt / trả lời) -> brief.json, plan.json
        if old is None:
            B = llm_step(lambda: make_brief(STATE["llm"], req.prompt, product is not None, aspect=aspect))
            if B["mode"] == "edit":   # ảnh lạ cần sửa: ảnh kèm theo là ảnh cũ, không phải ảnh sản phẩm
                old_img, product = product, None
                old = read_foreign(old_img, rd, timing)
                if "aspect_ratio" in req.model_fields_set:
                    ignored.append("aspect_ratio")
                request_json("edit_foreign")
        if old is not None:
            B = llm_step(lambda: make_edit(STATE["llm"], req.prompt, old, new_product=product is not None))
    B["calls"] = calls
    editing = old is not None
    has_product = product is not None or bool(editing and old.get("product"))
    if editing:
        aspect = B["edit"]["aspect"] or (old.get("aspect") if old.get("aspect") in SIZES else None) or \
            nearest_ratio(old_img.shape[1], old_img.shape[0])
    B["seed"] = int(time.time() * 1000) % 100000
    (rd / "brief.json").write_text(json.dumps(B, ensure_ascii=False, indent=1), encoding="utf-8")
    W, H = SIZES.get(aspect, SIZES["1:1"])
    info = poster_info(run_id, B, aspect, has_product, parent=old)
    fut: Future = Future()
    stage = {"s": "chờ GPU"}
    DRAW_Q.put({"fut": fut, "B": B, "W": W, "H": H, "seed": B["seed"], "product": product, "timing": timing, "t0": time.time(),
                "refs": [old_img] + ([product] if product is not None else []) if editing else None,
                "progress": lambda m: stage.update(s=m)})
    try:
        d = fut.result()
    except Exception as e:
        raise C.draw_error(e, stage["s"]) from e
    if B.get("mode") == "image":   # không chữ khách: ảnh FLUX vẽ là kết quả
        png = png_bytes(d["draft"], info)
        (vd / "poster.png").write_bytes(png)
        timing.update({k: round(x, 1) for k, x in d["timing"].items()})
        SAVE.submit(save_debug, None, d, vd)
    else:
        png = DESIGN.submit(design_step, d, B, has_product, vd, timing, time.time(), info).result()
    timing["tong"] = round(time.time() - t_all, 1)
    if B.get("mode") == "image":
        (vd / "plan.json").write_text(json.dumps({"mode": "image", "seed": B["seed"], "prompt_en": B["prompt_en"], "timing": timing,
                                                  "calls": B.get("calls", []), **({"edit": B["edit"]} if B.get("edit") else {})},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    prune()
    return png, W, H, timing, aspect


def read_foreign(img: np.ndarray, rd: Path, timing: dict) -> dict:
    """ẢNH LẠ: Hi-SAM (trạm VẼ, cùng hàng đợi GPU) -> dòng -> VLM chép chữ -> "poster cũ" cho luồng sửa. Lỗi -> read_failed."""
    from textfix.edit import nearest_ratio, read_image
    t0 = time.time()
    fut: Future = Future()
    H, W = img.shape[:2]
    DRAW_Q.put({"fut": fut, "B": {"mode": "poster"}, "W": W, "H": H, "seed": 0, "product": None, "draft": img, "timing": {},
                "t0": time.time(), "progress": lambda m: None})
    try:
        ink = fut.result()["ink"]
        # dòng (OCR) + VLM trên luồng THIẾT KẾ: OCR / Chromium mỗi luồng ở đó đã làm nóng (luồng JOBRUN thì chưa)
        R = DESIGN.submit(read_image, img, ink, STATE["vlm"]).result()
    except Exception as e:
        raise C.ApiError("read_failed", "đọc chữ ảnh lạ", f"{type(e).__name__}: {e}") from e
    finally:
        timing["doc_anh"] = round(time.time() - t0, 1)
    (rd / "read.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"texts": R["texts"], "prompt_en": R["prompt_en"], "product": R["product"], "aspect": nearest_ratio(W, H),
            "run_id": None, "depth": 0}


def prune() -> None:
    ds = sorted([d for d in RUNS.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
    for d in ds[:-MAX_RUNS]:
        shutil.rmtree(d, ignore_errors=True)


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
    except BaseException as e:   # máy chủ vẫn lên, /api/health báo chưa sẵn sàng, chi tiết ở log
        if isinstance(e, KeyboardInterrupt):
            raise
        STATE["error"] = STATE["error"] or f"{type(e).__name__}: {e}"
        log.error("KHÔNG nạp được model: %s", traceback.format_exc())
    yield
    for _ in STATE["flux"]:
        DRAW_Q.put(None)


app = FastAPI(title="TextFix", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
PATH = "/api/v1/images/generations"


@app.get("/api/health")
async def health():
    """Chỉ trạng thái (không địa chỉ nội bộ, không chi tiết lỗi nạp: các thứ đó ở log máy chủ)."""
    return {"ready": ready(), "failed": bool(STATE["error"]), "queue": len(ORDER)}


def error_response(e: C.V1Error, rid: str, run_id: str | None = None) -> JSONResponse:
    hdr = {"X-Request-Id": rid, **({"Retry-After": str(e.retry_after)} if e.retry_after else {}),
           **({"X-Run-Id": run_id} if run_id else {})}
    return JSONResponse(C.body_error(e, rid), status_code=e.http, headers=hdr)


@app.exception_handler(RequestValidationError)
async def _validation(request: Request, exc: RequestValidationError):
    """Thân sai khuôn (không phải JSON, prompt sai kiểu...) -> 400 theo khuôn lỗi của API."""
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    return error_response(C.V1Error(400, "ERR_INVALID_REQUEST", f"thân yêu cầu sai khuôn: {exc.errors()[:3]}"), rid)


@app.post(PATH)
async def generations(req: C.Request, request: Request):
    """MỘT lời gọi đồng bộ: prompt (+ ảnh sản phẩm base64) -> một poster. Chờ tối đa TEXTFIX_V1_DEADLINE giây (55, tính từ lúc
    nhận yêu cầu); quá hạn -> 504, lượt chạy vẫn chạy nốt (tệp ở TEXTFIX_RUNS/<run_id>)."""
    t0 = time.time()
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    aspect = req.aspect_ratio or "1:1"
    run_id = None
    try:
        C.check(req)
        if not ready():
            raise C.V1Error(503, "ERR_SERVICE_UNAVAILABLE", "lỗi nạp model" if STATE["error"] else "đang nạp model", retry_after=30)
        if len(ORDER) >= int(os.environ.get("TEXTFIX_MAX_QUEUE", "8")):
            raise C.V1Error(503, "ERR_SERVICE_BUSY", f"hàng đợi đầy ({len(ORDER)} yêu cầu)", retry_after=15)
        b64s = req.b64_images()
        # ảnh đầu tiên có khối thông tin poster (textfix/meta.py) = poster cần SỬA; ảnh đầu tiên không có = ảnh sản phẩm (FLUX
        # image 1; khi sửa: ảnh sản phẩm mới, image 2); các ảnh sau chưa dùng (vẫn kiểm cho đúng khuôn)
        from textfix.meta import read as read_info
        old = old_img = product = None
        for s in b64s:
            raw = C.decode_b64(s)
            m = read_info(raw) if old is None else None
            if m is not None:
                old, old_img = m, C.to_image(raw, 1536)
            elif product is None:
                product = C.to_image(raw)
            else:
                C.to_image(raw)
        left = float(os.environ.get("TEXTFIX_V1_DEADLINE", "55")) - (time.time() - t0)
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        n_img = len(b64s)
        log.info("%s -> %s (ảnh: %d%s; khoá lạ: %s)", rid, run_id, n_img,
                 f", SỬA {old.get('run_id')} (lần {old.get('depth', 0) + 1})" if old is not None else "",
                 sorted((req.model_extra or {}).keys()))
        ORDER.append(run_id)
        fut = JOBRUN.submit(generate, run_id, req, product, rid, old=old, old_img=old_img)
        fut.add_done_callback(lambda f, r=run_id: ORDER.remove(r) if r in ORDER else None)
        try:
            png, W, H, timing, aspect = await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(fut)), left)
        except asyncio.TimeoutError:
            raise C.V1Error(504, "ERR_UPSTREAM_TIMEOUT", f"quá hạn, run {run_id} vẫn chạy nốt") from None
        except C.ApiError as e:
            log.error("%s run %s lỗi %s (%s): %s", rid, run_id, e.code, e.stage, e.message)
            raise C.job_error(e) from None
    except C.V1Error as e:
        log.info("%s lỗi %d %s (%.1fs): %s", rid, e.http, e.error_code, time.time() - t0, e.message)
        return error_response(e, rid, run_id)
    except Exception as e:   # không để lọt lỗi khuôn FastAPI / chuỗi rỗng
        log.error("%s lỗi: %s", rid, traceback.format_exc())
        return error_response(C.V1Error(500, "ERR_IMAGE_GENERATION_FAILED", f"internal_error ({type(e).__name__})"), rid, run_id)
    ms = int((time.time() - t0) * 1000)
    log.info("%s xong %s (%.1fs) %s", rid, run_id, ms / 1000, timing)
    extra = {"X-Run-Id": run_id, "X-Timing": C.timing_header(timing)}
    if "image/png" in (request.headers.get("accept") or ""):
        return Response(png, media_type="image/png", headers={**C.headers(rid, aspect, ms), **extra})
    return JSONResponse(C.body_success(rid, base64.b64encode(png).decode(), aspect, W, H, ms), headers={"X-Request-Id": rid, **extra})


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
