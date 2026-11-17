"""Ảnh debug của bước gỡ chữ: nét Hi-SAM (hồng), mặt nạ gỡ / quầng / dòng giữ.
"""

from __future__ import annotations

import numpy as np

from .mask import GROW_HOLD, line_region, stroke_cap


def hisam_image(draft: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Đầu ra thô của Hi-SAM: nháp nhạt đi, nét chữ tô HỒNG."""
    vis = draft.astype(np.float32) * 0.55 + 255 * 0.45
    ink = np.asarray(ink, bool)
    vis[ink] = vis[ink] * 0.15 + np.asarray((235, 0, 200), np.float32) * 0.85
    return vis.round().clip(0, 255).astype(np.uint8)


def ink_image(draft: np.ndarray, ink: np.ndarray, mask: np.ndarray, M: dict, kept: set) -> np.ndarray:
    """Bước gỡ chữ: nét bị gỡ XANH LÁ, phần nới / quầng VÀNG, nét của dòng giữ XANH DƯƠNG, nét không thuộc dòng nào HỒNG."""
    H, W = draft.shape[:2]
    import cv2
    ink = np.asarray(ink, bool)
    _, cc, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    hold = np.zeros((H, W), bool)
    for l in M["L"]:
        if l["id"] in kept:
            hold |= line_region(l, H, W, GROW_HOLD, stroke_cap(l, cc, stats))
    vis = draft.astype(np.float32) * 0.55 + 255 * 0.45
    for sel, col, a in ((mask & ~ink, (255, 210, 0), 0.55), (mask & ink, (0, 170, 60), 0.85), (ink & ~mask & hold, (30, 110, 240), 0.85),
                        (ink & ~mask & ~hold, (235, 0, 200), 0.85)):
        vis[sel] = vis[sel] * (1 - a) + np.asarray(col, np.float32) * a
    return vis.round().clip(0, 255).astype(np.uint8)
