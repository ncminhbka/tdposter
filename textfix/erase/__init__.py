"""BƯỚC 4b -- GỠ CHỮ GIUN khỏi nháp: mặt nạ từ nét chữ Hi-SAM, nền dưới chữ do LaMa vẽ lại -- MỘT lượt LaMa cho cả poster.

  ink = hisam.strokes(draft)                        # nét chữ (trạm vẽ)
  clean, mask = erase(draft, ink, M)                # gỡ MỌI dòng của M["L"] -> nền VLM nhìn
  base, mask = settle(draft, clean, ink, M, plan)   # dòng VLM chọn "keep" trở về điểm ảnh nháp (không gọi LaMa lần nữa)

Chất lượng gỡ chữ do mặt nạ quyết: LaMa sạch khi mặt nạ ôm sát nét, hỏng khi mặt nạ trùm cả nút / icon. Mỗi dòng cùng một cách:
  1. NÉT    các MẢNH nét Hi-SAM của dòng: mỗi mảnh thuộc đúng một dòng (dòng có nhiều điểm của nó nhất, >= nửa mảnh --
            lines.ink.own, cùng luật bước đo dòng), nên đuôi chữ / dấu thanh thò khỏi khung đi cùng chữ và hai dòng sát nhau không
            gỡ chung một mảnh; cộng nét không dòng nào sở hữu nằm trong lõi dòng. Hi-SAM đã trả lời "có phải chữ không", nên vòng
            con dấu, viền nút, hoạ tiết không bị kéo vào. Mảnh TRANG TRÍ đầu dòng (chấm "•", lines.lead) không phải chữ: ở lại.
  2. DỰ PHÒNG  Hi-SAM thấy quá ít ở dòng (chữ rất nhỏ): thêm các mảng cùng màu mực cỡ một chữ cái; vẫn không có thì lấy lõi dòng.
  3. QUẦNG  Hi-SAM chỉ thấy thân nét. Trong dải sát nét, điểm khác nền của dòng là viền dày / bóng đổ / quầng sáng; nền phẳng thì
            lan tiếp theo đuôi mờ. Quầng phải dính vào nét (icon đứng cách chữ một khe nền thì không), không lấn sang dòng khác.
  4. DẤU CÂU  mảng đậm nhỏ nằm gọn trong lõi dòng, giữa nét đầu và nét cuối (gạch nối, dấu chấm Hi-SAM bỏ qua).
Nét của dòng giữ (chữ in trên sản phẩm) và mảnh trang trí không bao giờ vào mặt nạ. Còn sót quầng mờ / nét nhỏ thì chữ dựng mới đè lên.
Phần: mask.py (mặt nạ theo từng dòng), lama.py (LaMa), view.py (ảnh debug).
Số đo và các hướng đã thử, đã bỏ: docs/TUNING.md (mục gỡ chữ), công cụ đo scripts/erase_bench.py.
"""

from __future__ import annotations

import numpy as np

from .lama import LAMA_PATH, _blend, inpaint, lama_ready  # noqa: F401
from .mask import _dilate, erase_mask, line_region  # noqa: F401
from .view import hisam_image, ink_image  # noqa: F401

# ---------------------------------------------------------------- API

def erase(draft: np.ndarray, ink: np.ndarray, M: dict, device: str | None = None,
          log: list | None = None, plates: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Gỡ mọi dòng của M["L"]: LaMa lấp mặt nạ, ngoài mặt nạ giữ nguyên điểm ảnh nháp. -> (nền sạch chữ, mặt nạ).
    device: card chạy LaMa (nên là card đã vẽ nháp, để chia tải). Thiếu big-lama.pt: báo lỗi và dừng."""
    if not lama_ready():
        raise RuntimeError(f"thiếu tệp LaMa {LAMA_PATH} (docs/SERVER.md, mục LaMa; đổi chỗ: export TEXTFIX_LAMA=<đường dẫn>)")
    mask = erase_mask(draft, ink, M, set(), log, plates)
    if not mask.any():
        return draft.copy(), mask
    return _blend(inpaint(draft, mask, device), draft, mask), mask


def settle(draft: np.ndarray, clean: np.ndarray, ink: np.ndarray, M: dict, P: dict | None = None,
           log: list | None = None, done: tuple | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Nền dựng của một kế hoạch P: nền sạch chữ trong mặt nạ của các dòng KHÔNG giữ, nháp gốc ở mọi chỗ khác. Chỗ hai mặt nạ
    chồng nhau thuộc dòng bị gỡ. Không gọi LaMa. -> (nền, mặt nạ)."""
    kept = kept_of(P)
    if not kept and done is not None:   # done = (mặt nạ, nhật ký) của erase(): không dòng nào giữ -> mặt nạ y hệt, khỏi tính lại
        if log is not None:             # (~1 s / poster trên CPU, 12/11)
            log.extend(done[1])
        return clean, done[0]
    mask = erase_mask(draft, ink, M, kept, log)
    return (clean if not kept else _blend(clean, draft, mask)), mask


PLATE_MAX = 4.0     # hình chứa dòng rộng > 4 lần khung dòng: tấm bảng / khung lớn (nhiều dòng), không phải nút / thẻ -> không xoá
PLATE_PAD = 0.15    # nới mặt nạ hình theo cao dòng: mép khử răng cưa, bóng đổ ngắn dưới nút


def empty_plates(plates: dict, M: dict, used: set) -> tuple[np.ndarray | None, list[str]]:
    """NÚT / THẺ TRỐNG (12/11, sme1 s03: FLUX vẽ 7 thẻ giá cho 4 dịch vụ, chữ thẻ lặp bị gỡ, thẻ trắng nằm lại -- người dùng: "xoá luôn
    thẻ trống"): dòng nằm trong một hình (plates, từ erase_mask) mà không đơn vị nào viết / giữ (used) -> xoá cả hình. Không xoá khi
    hình lớn (tấm bảng) hoặc chạm dòng khác đang dùng. -> (mặt nạ cần LaMa vẽ lại, hoặc None; nhật ký)."""
    import cv2
    out, log = None, []
    by = M["by"]
    for lid, pm in plates.items():
        if lid in used or lid not in by:
            continue
        b = by[lid]["box"]
        h = max(1.0, float(b[3] - b[1]))
        ratio = float(pm.sum()) / max(1.0, (b[2] - b[0]) * h)
        if ratio > PLATE_MAX:
            log.append(f"{lid}: hình chứa dòng lớn ({ratio:.1f} lần khung dòng), không xoá")
            continue
        k = max(3, int(round(PLATE_PAD * h)))
        grown = cv2.dilate(pm.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))).astype(bool)
        busy = np.zeros(pm.shape, bool)   # khung các dòng đang dùng (chữ viết / chữ giữ trên sản phẩm)
        for o in used:
            if o in by:
                ob = by[o]["box"]
                busy[max(0, int(ob[1])):int(ob[3]) + 1, max(0, int(ob[0])):int(ob[2]) + 1] = True
        # lòng hình đè lên dòng đang dùng (tấm bảng chứa cả dòng khác): không xoá. Chỉ phần NỚI chạm (12/11 quick1 q10: nút sát đáy
        # gói Nescafe có chữ giữ) thì xoá hình, chừa khung dòng kia
        if float((pm & busy).sum()) > 0.02 * float(pm.sum()):
            log.append(f"{lid}: hình trống đè lên dòng đang dùng, không xoá")
            continue
        grown &= ~busy
        out = grown if out is None else out | grown
        log.append(f"{lid}: nút / thẻ không còn chữ -> xoá cả hình ({100 * grown.mean():.1f}% ảnh)")
    return out, log


def kept_of(P: dict | None) -> set:
    """Các dòng lệnh "keep" của kế hoạch (giữ nguyên điểm ảnh nháp)."""
    return {m for op in (P or {}).get("ops") or [] if op.get("kind") == "keep" for m in op.get("marks") or op.get("slots") or []}

__all__ = ["LAMA_PATH", "_dilate", "erase_mask", "hisam_image", "ink_image", "inpaint", "lama_ready", "line_region"]
