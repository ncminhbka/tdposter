"""textfix.llm.json_call với máy LLM giả: trần độ dài trả lời, gọi lại do chính vòng của json_call quyết, không vào cache khi hỏng."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from textfix import llm


class FakeClient:
    """Trả lần lượt các câu trả lời dựng sẵn: (nội dung, finish_reason) hoặc một ngoại lệ."""

    def __init__(self, answers):
        self.answers, self.calls = list(answers), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        text, finish = a
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)], usage=None)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(llm.Path, "home", lambda: tmp_path)   # cache trong thư mục tạm

    def make(answers):
        cl = FakeClient(answers)
        monkeypatch.setattr(llm, "client", lambda *a, **k: cl)
        return cl, llm.json_call("http://llm.test/v1", "m", tag="t")
    return make


def test_runaway_answer_is_cut_and_retried(fake):
    cl, call = fake([('{"a": ' + " " * 50, "length"), (json.dumps({"a": 1}), "stop")])
    meta = {}
    assert call("sys", ["x"], meta) == {"a": 1}
    assert meta["attempts"] == 2 and all(c["max_tokens"] == llm.max_tokens() for c in cl.calls)


def test_timeout_is_retried_then_raises(fake, monkeypatch):
    monkeypatch.setattr(llm, "RETRIES", 1)
    cl, call = fake([TimeoutError("quá giờ"), TimeoutError("quá giờ")])
    with pytest.raises(TimeoutError):
        call("sys", ["y"])
    assert len(cl.calls) == 2


def test_failed_answer_is_not_cached(fake, monkeypatch):
    monkeypatch.setattr(llm, "RETRIES", 0)
    _, call = fake([("không phải json", "stop")])
    with pytest.raises(ValueError):
        call("sys", ["z"])
    _, call = fake([(json.dumps({"ok": True}), "stop")])
    assert call("sys", ["z"]) == {"ok": True}   # lần trước hỏng không bị đọc lại từ cache


def test_openai_client_does_not_retry_on_its_own(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("TEXTFIX_LLM_TIMEOUT", "12")
    c = llm.client("http://llm.test/v1")
    assert c.max_retries == 0 and c.timeout == 12
