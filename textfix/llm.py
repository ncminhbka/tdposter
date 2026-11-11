"""Gọi LLM / VLM qua API OpenAI-compatible (vLLM): trả JSON, nhiều ảnh, có cache theo nội dung (~/.cache/textfix/<tag>).

  call = json_call(url, model, key_env, tag="designer", schema=None)
  out = call(system, [text | png_bytes, ...])   # -> dict
  out = call(system, parts, meta := {})         # meta nhận: prompt_tokens, completion_tokens, s (giây), cached, images
  out = call(system, parts, schema=S)           # khuôn JSON riêng cho lời gọi này (VLM ghép câu: khuôn dựng theo từng poster)
  with collect() as U: ...                      # U nhận {tag, prompt_tokens, completion_tokens, s, attempts, cached} của mọi
                                                # lời gọi trên LUỒNG này (máy chủ ghi vào plan.json)
Tham số sinh theo thẻ model Qwen3.8-27B (chế độ không suy nghĩ), trừ presence_penalty: xem sampling().
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("textfix.llm")

# 11/11 (notebook, API v1): một lời gọi VLM mất 306 s -- lần đầu model sinh lặp tới hết hạn chờ 300 s, gọi lại ~6 s là xong.
# Lời gọi bình thường: LLM ~10 s, VLM ~9 s, trả lời ~1-2 nghìn token. Nên: chờ ngắn, trần độ dài trả lời, gọi lại do CHÍNH vòng
# dưới đây quyết (thư viện openai không tự gọi lại thêm bên trong).
RETRIES = int(os.environ.get("TEXTFIX_LLM_RETRIES", "2"))   # gọi lại khi trả lời hỏng / cụt / quá giờ / lỗi mạng (audit A2 / D2)


def timeout() -> float:
    """Giây chờ MỘT lần gọi (TEXTFIX_LLM_TIMEOUT, mặc định 40; trước 11/11 là 300)."""
    return float(os.environ.get("TEXTFIX_LLM_TIMEOUT", "40"))


def max_tokens() -> int:
    """Trần độ dài trả lời (TEXTFIX_LLM_MAX_TOKENS, mặc định 4096): model sinh lặp thì dừng sớm, JSON cụt -> gọi lại."""
    return int(os.environ.get("TEXTFIX_LLM_MAX_TOKENS", "4096"))


def sampling() -> dict:
    """Tham số sinh, đặt RÕ thay vì để mặc định của máy vLLM (12/11). Thẻ model Qwen3.8-27B, chế độ không suy nghĩ:
    temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5. presence_penalty ở đây mặc định 0: việc của LLM / VLM
    là CHÉP NGUYÊN VĂN câu khách (texts rồi chép lại trong prompt_en; VLM chép câu vào dòng) -- phạt token đã xuất hiện đẩy model
    chép sai; sinh lặp đã có khuôn JSON strict + trần max_tokens chặn. Đổi bằng TEXTFIX_LLM_TEMPERATURE / _TOP_P / _TOP_K /
    _PRESENCE_PENALTY."""
    e = os.environ.get
    return {"temperature": float(e("TEXTFIX_LLM_TEMPERATURE", "0.7")), "top_p": float(e("TEXTFIX_LLM_TOP_P", "0.8")),
            "top_k": int(e("TEXTFIX_LLM_TOP_K", "20")), "presence_penalty": float(e("TEXTFIX_LLM_PRESENCE_PENALTY", "0"))}


_SINK = threading.local()


@contextmanager
def collect():
    """Gom số token / giây của mọi lời gọi json_call trên luồng hiện tại (kể cả lời gọi đọc từ cache: cached=True)."""
    prev = getattr(_SINK, "items", None)
    _SINK.items = out = []
    try:
        yield out
    finally:
        _SINK.items = prev


def client(base_url: str, key_env: str = "TEXTFIX_LLM_API_KEY"):
    from openai import OpenAI
    key = os.environ.get(key_env) or os.environ.get("TEXTFIX_LLM_API_KEY") or "none"
    return OpenAI(base_url=base_url, api_key=key, timeout=timeout(), max_retries=0)


def extra_body() -> dict:
    """vLLM + Qwen: tắt chế độ suy nghĩ (JSON ép khuôn, không cần chuỗi suy luận; nhanh hơn nhiều); top_k (không có trong khuôn
    OpenAI) đi theo đường này."""
    return {"chat_template_kwargs": {"enable_thinking": False}, "top_k": sampling()["top_k"]}


def brief_temperature() -> float:
    """Nhiệt độ của LLM đầu luồng (brief + sửa poster): TEXTFIX_BRIEF_TEMPERATURE, mặc định 0.3. Việc chính là CHÉP câu khách;
    0.7 (thẻ model) làm mỗi lần chạy lệch một kiểu -- 11/11 edit1 lượt 2, Qwen3.5-35B-A3B cùng lời chat: lượt 1 "Bò ba chỉ",
    lượt 2 "Bò chỉ"; "... part-time cho quán cơm" thành "... part-time cho"."""
    return float(os.environ.get("TEXTFIX_BRIEF_TEMPERATURE", "0.3"))


def json_call(base_url: str, model: str, key_env: str = "TEXTFIX_LLM_API_KEY", tag: str = "llm", schema: dict | None = None,
              temperature: float | None = None):
    """temperature: thay nhiệt độ chung (sampling) cho mọi lời gọi của bản này (LLM đầu luồng: brief_temperature)."""
    schema0 = schema
    if not base_url or not model:
        raise RuntimeError("chưa đặt LLM / VLM: export TEXTFIX_LLM_URL và TEXTFIX_MODEL (docs/SERVER.md)")
    cache = Path.home() / ".cache" / "textfix" / tag
    cl = None

    def call(system: str, parts: list, meta: dict | None = None, fresh: bool = False, schema: dict | None = None) -> dict:
        """fresh: không đọc cache (vẫn ghi) -- gọi lại để lấy câu trả lời KHÁC cho cùng đầu vào. schema: khuôn JSON của riêng
        lời gọi này (thay khuôn chung của json_call)."""
        nonlocal cl
        schema = schema or schema0
        if meta is not None:
            meta.update(images=sum(isinstance(p, bytes) for p in parts), cached=False)
        cache.mkdir(parents=True, exist_ok=True)
        sp = sampling()
        if temperature is not None:
            sp["temperature"] = temperature   # trong khoá đệm (h băm sp)
        h = hashlib.sha1((model + base_url + system + json.dumps(schema or {}) + json.dumps(sp)).encode())
        for p in parts:
            h.update(p if isinstance(p, bytes) else str(p).encode())
        f = cache / f"{h.hexdigest()}.json"
        sink = getattr(_SINK, "items", None)
        if f.exists() and not fresh:
            if meta is not None:
                meta["cached"] = True
            if sink is not None:
                sink.append({"tag": tag, "cached": True})
            return json.loads(f.read_text(encoding="utf-8"))
        cl = cl or client(base_url, key_env)
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(p).decode()}}
                   if isinstance(p, bytes) else {"type": "text", "text": str(p)} for p in parts]
        fmt = ({"type": "json_schema", "json_schema": {"name": tag, "schema": schema, "strict": True}} if schema
               else {"type": "json_object"})
        t0 = time.time()
        # trả lời hỏng (JSON cụt / sai) hoặc lỗi mạng / quá giờ: gọi lại, tối đa RETRIES lần; hết lượt thì ném lỗi, không vào cache
        for k in range(RETRIES + 1):
            t1 = time.time()
            try:
                r = cl.chat.completions.create(model=model, messages=[{"role": "system", "content": system},
                                                                      {"role": "user", "content": content}],
                                               response_format=fmt, max_tokens=max_tokens(), temperature=sp["temperature"],
                                               top_p=sp["top_p"], presence_penalty=sp["presence_penalty"],
                                               extra_body=extra_body())
                if r.choices[0].finish_reason == "length":   # chạm trần độ dài: gần như luôn là sinh lặp -> log đoạn cuối (trường nào lặp)
                    tail = (r.choices[0].message.content or "")[-240:]
                    raise ValueError(f"trả lời chạm trần {max_tokens()} token (sinh lặp?), đoạn cuối: {tail!r}")
                out = json.loads(r.choices[0].message.content)
                if not isinstance(out, dict):
                    raise ValueError(f"trả lời không phải JSON object: {type(out).__name__}")
                break
            except Exception as e:
                log.warning("%s lần %d/%d hỏng sau %.1fs: %s: %s", tag, k + 1, RETRIES + 1, time.time() - t1,
                            type(e).__name__, str(e)[:400])
                if k == RETRIES:
                    raise
        u = getattr(r, "usage", None)
        got = {"attempts": k + 1, "s": round(time.time() - t0, 1), "prompt_tokens": getattr(u, "prompt_tokens", None),
               "completion_tokens": getattr(u, "completion_tokens", None)}
        if meta is not None:
            meta.update(got)
        if sink is not None:
            sink.append({"tag": tag, "cached": False, **got})
        f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return out
    return call
