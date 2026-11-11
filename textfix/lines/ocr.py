"""Dò dòng chữ (kể cả chữ giun) bằng rapidocr (ONNX nhỏ, CPU). Mỗi dòng: khung, đa giác 4 điểm, chữ đọc được, độ tin, góc."""

from __future__ import annotations

import math
import os
import threading

import numpy as np

_LOCAL = threading.local()   # MỖI LUỒNG một RapidOCR (máy chủ: luồng vẽ dọn chữ sót và các luồng thiết kế đọc nháp cùng lúc)
_LOCK = threading.Lock()     # bộ nhớ đệm dùng chung
_MAKE = threading.Lock()     # tạo RapidOCR từng luồng một


def _engine():
    if getattr(_LOCAL, "ocr", None) is None:
        import rapidocr_onnxruntime as ro
        with _MAKE:   # bọc InferenceSession theo luồng rồi tạo ngay: các luồng làm nóng cùng lúc không giẫm card của nhau
            _limit_threads()
            _use_gpu()
            _LOCAL.ocr = ro.RapidOCR()
    return _LOCAL.ocr


_GPU_N = [0]   # đếm luồng đã nhận card (chia vòng)


def gpu_devices() -> list[int]:
    """TEXTFIX_OCR_GPU="0,1": các card chạy OCR (mỗi luồng nhận một card, chia vòng); trống / "off": CPU."""
    v = (os.environ.get("TEXTFIX_OCR_GPU") or "").strip().lower()
    return [] if v in ("", "off", "no", "cpu") else [int(x) for x in v.split(",") if x.strip().isdigit()]


def _use_gpu() -> None:
    """OCR TRÊN GPU (12/11, cần gói onnxruntime-gpu thay onnxruntime): bọc InferenceSession của rapidocr để đưa CUDA lên đầu.
    Không dùng use_cuda của rapidocr 1.2.3: nó ghim card 0 và dò thuật toán cuDNN EXHAUSTIVE -- mỗi cỡ ảnh / dòng mới dò lại
    từ đầu, mà cỡ dòng chữ của mình lần nào cũng khác. Ở đây: HEURISTIC, card theo luồng. Không nạp được CUDA: onnxruntime tự
    lùi về CPU (kết quả như cũ, chỉ chậm)."""
    devs = gpu_devices()
    from rapidocr_onnxruntime import utils as ru
    base = getattr(ru.InferenceSession, "textfix_base", ru.InferenceSession)
    if not devs:
        ru.InferenceSession = base
        return
    with _LOCK:
        dev = devs[_GPU_N[0] % len(devs)]
        _GPU_N[0] += 1
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):   # onnxruntime >= 1.21: nạp cuDNN / cuBLAS từ gói pip nvidia-* (đi kèm torch)
            ort.preload_dlls()
    except Exception:
        pass

    def session(path, sess_options=None, providers=None, **kw):
        cuda = ("CUDAExecutionProvider", {"device_id": dev, "cudnn_conv_algo_search": "HEURISTIC",
                                          "arena_extend_strategy": "kSameAsRequested"})
        return base(path, sess_options=sess_options, providers=[cuda, "CPUExecutionProvider"], **kw)
    session.textfix_base = base
    ru.InferenceSession = session


def _limit_threads() -> None:
    """TEXTFIX_THREADS (docker/run.sh đặt theo hạn mức CPU của container): số luồng của onnxruntime. onnxruntime không đọc
    OMP_NUM_THREADS mà lấy số lõi cả máy, còn rapidocr 1.2.3 không có tuỳ chọn số luồng -> thay SessionOptions nó dùng."""
    n = int(os.environ.get("TEXTFIX_THREADS") or 0)
    from rapidocr_onnxruntime import utils as ru
    if n <= 0 or getattr(ru.SessionOptions, "textfix_threads", None) == n:
        return
    base = getattr(ru.SessionOptions, "textfix_base", ru.SessionOptions)

    def options():
        o = base()
        o.intra_op_num_threads = n
        o.inter_op_num_threads = 1
        return o
    options.textfix_threads, options.textfix_base = n, base
    ru.SessionOptions = options


MULTI_SCALE = 0.5   # lượt dò thứ hai ở ảnh thu nhỏ: chữ RẤT TO / chữ trang trí bị sót ở cỡ gốc
MULTI_MIN = 600     # chỉ chạy lượt thu nhỏ khi cạnh ngắn ảnh >= 600 px


def _once(img, k: float = 1.0) -> list[dict]:
    res, _ = _engine()(img)
    out = []
    for poly, text, score in res or []:
        p = np.asarray(poly, dtype=float) / k
        # góc cạnh trên (điểm 0 -> 1): dòng nghiêng / phối cảnh là dấu hiệu chữ thuộc cảnh
        ang = math.degrees(math.atan2(p[1, 1] - p[0, 1], p[1, 0] - p[0, 0]))
        out.append({"box": [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())],
                    "poly": p.tolist(), "text": text, "score": float(score), "angle": ang})
    return out


def _cover(a, b) -> float:
    """Phần diện tích khung a nằm trong khung b."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h) / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


_CACHE: dict = {}   # nội dung ảnh -> dòng (dọn chữ sót và tách ô cùng đọc nháp); giữ vài ảnh gần nhất


def read_lines(img: np.ndarray | str, multi: bool = True) -> list[dict]:
    """Dòng chữ ở cỡ gốc + (multi) dòng chỉ dò được ở ảnh thu nhỏ MULTI_SCALE, không chồng dòng đã có (khung chồng < 30% cả hai phía)."""
    import copy
    import hashlib
    key = None
    if isinstance(img, np.ndarray):
        key = (img.shape, multi, hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest())
        with _LOCK:
            if key in _CACHE:
                return copy.deepcopy(_CACHE[key])
    out = _read(img, multi)
    if key is not None:
        with _LOCK:
            while len(_CACHE) >= 16:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = copy.deepcopy(out)
    return out


def _read(img: np.ndarray | str, multi: bool) -> list[dict]:
    out = _once(img)
    if not multi or isinstance(img, str) or min(img.shape[:2]) < MULTI_MIN:
        return out
    from PIL import Image
    H, W = img.shape[:2]
    small = np.asarray(Image.fromarray(img).resize((int(W * MULTI_SCALE), int(H * MULTI_SCALE)), Image.LANCZOS))
    for x in _once(small, MULTI_SCALE):
        if all(_cover(x["box"], o["box"]) < 0.3 and _cover(o["box"], x["box"]) < 0.3 for o in out):
            out.append(x)
    return out


REGION_THRESH = 0.3   # ngưỡng bản đồ vùng chữ, cùng số bộ dò dùng (Det.thresh)
REGION_MIN = 0.0005   # vùng >= 0.05% ảnh (bỏ đốm)
UNCLIP = 1.6          # bộ dò co vùng chữ lại khi học; nới trả như chính nó làm (Det.unclip_ratio): khoảng nới = diện tích * 1.6 / chu vi


def regions(img: np.ndarray) -> list[dict]:
    """VÙNG CHỮ HÌNH DẠNG TỰ DO (12/10, chữ khó): bản đồ xác suất của chính bộ dò trong RapidOCR (DBNet), lấy đường viền từng vùng
    TRƯỚC khi nó bị ép thành khung 4 điểm. Bộ dò thấy đủ dòng cong / xiên / chữ trang trí; chúng chỉ bị mất ở khâu ép khung (vùng
    cong chiếm ít diện tích trong hình chữ nhật bao nó -> điểm thấp -> bị loại: vòm "KHAI TRƯƠNG", nửa "TIỆM BÁNH" của con dấu,
    cả dòng "FLASH SALE"). -> [{mask (vùng LÕI, bool cỡ ảnh), pad (px cần nới ra mỗi phía để ôm chữ), area}]; [] nếu bản RapidOCR
    đang cài không cho lấy bản đồ (khi đó phần chữ khó tự tắt, mọi thứ khác chạy như cũ)."""
    import cv2
    try:
        det = _engine().text_detector
        post, cap = det.postprocess_op, {}

        class _Tap:   # nghe lén đầu vào khâu ép khung: không phụ thuộc cách tiền xử lý của từng phiên bản
            def __call__(self, pred, *a, **k):
                cap["pred"] = np.asarray(pred)
                return post(pred, *a, **k)

            def __getattr__(self, name):
                return getattr(post, name)
        det.postprocess_op = _Tap()
        try:
            det(np.ascontiguousarray(img))
        finally:
            det.postprocess_op = post
        pm = cap["pred"]
        pm = pm.reshape(pm.shape[-2], pm.shape[-1]).astype(np.float32)
    except Exception:
        return []
    H, W = img.shape[:2]
    m = (cv2.resize(pm, (W, H), interpolation=cv2.INTER_LINEAR) > REGION_THRESH).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        if st[i][4] < REGION_MIN * W * H:
            continue
        mk = cc == i
        cs, _h = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        per = float(sum(cv2.arcLength(c, True) for c in cs)) or 1.0
        out.append({"mask": mk, "area": int(st[i][4]), "pad": float(st[i][4]) * UNCLIP / per})
    return out


def read_region(img: np.ndarray, rect) -> str:
    """Đọc chữ trong một hình chữ nhật xoay (cv2.minAreaRect) bằng bộ đọc của RapidOCR -- để biết một vùng chữ mà bộ dò thấy nhưng
    khâu ép khung đã bỏ có thật là CHỮ không (hàng sao, hoạ tiết đọc ra rỗng). Lỗi / phiên bản không hỗ trợ: ""."""
    (cx, cy), (w, h), a = rect
    if w < h:
        w, h, a = h, w, a + 90
    text, score = read_rect(img, (cx, cy), w, h, a)
    return text if score >= MIN_REC else ""


def read_rect(img: np.ndarray, c, w: float, h: float, a: float) -> tuple[str, float]:
    """Đọc MỘT dòng nằm trong hình chữ nhật tâm c, dài w, cao h, nghiêng a độ (dương = xuôi kim đồng hồ trên ảnh) -> (chữ, độ tin).
    Dùng để chọn chiều đọc của dòng viết DỌC (a = +90 hay -90). Lỗi: ("", 0)."""
    import cv2
    try:
        cx, cy = c
        w, h = int(w * 1.08) + 2, int(h * 1.25) + 2
        Mx = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
        Mx[0, 2] += w / 2 - cx
        Mx[1, 2] += h / 2 - cy
        crop = cv2.warpAffine(np.ascontiguousarray(img), Mx, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        res = _engine().text_recognizer([crop])
        rec = res[0] if isinstance(res, tuple) else res
        return (str(rec[0][0]), float(rec[0][1])) if rec and rec[0] else ("", 0.0)
    except Exception:
        return "", 0.0


MIN_REC = 0.3   # độ tin tối thiểu của bộ đọc (chữ méo của nháp đọc ra độ tin thấp vẫn là chữ)
