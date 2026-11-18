"""Chromium (Playwright bản đồng bộ) để dựng HTML lên nền. Máy chủ không tải được Chromium của Playwright: dùng bản có sẵn
(PLAYWRIGHT_CHROME_PATH, hoặc chromium / google-chrome trong PATH).

  with Browser() as B:
      B.page.set_content(...)
"""

from __future__ import annotations

import os
import shutil


class Browser:
    def __enter__(self) -> "Browser":
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        exe = os.environ.get("PLAYWRIGHT_CHROME_PATH") or next(
            (shutil.which(b) for b in ("chromium-browser", "chromium", "google-chrome", "google-chrome-stable") if shutil.which(b)), None)
        self._br = self._pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
                                            **({"executable_path": exe} if exe else {}))
        self.page = self._br.new_page()
        self.crashed = False   # tab sập (hết RAM...): lời gọi sau báo "Target crashed" mà page.is_closed() vẫn False
        self.page.on("crash", lambda *_: setattr(self, "crashed", True))
        return self

    def alive(self) -> bool:
        """Còn dùng được: tiến trình Chromium còn nối, tab chưa đóng / chưa sập."""
        try:
            return not self.crashed and self._br.is_connected() and not self.page.is_closed()
        except Exception:
            return False

    def __exit__(self, *exc) -> None:
        try:
            self._br.close()
        except Exception:   # Chromium đã chết: đóng báo lỗi, vẫn phải dừng Playwright
            pass
        finally:
            self._pw.stop()
