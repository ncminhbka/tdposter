"""Mặt nạ gỡ chữ theo từng dòng: mảnh nét của dòng, dự phòng màu mực, quầng, dấu câu (xem erase/__init__.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ..color import hex_lab, lab
from ..lines.ink import Ink, own

# --- vùng của dòng (theo cao dòng)
GROW_HOLD = 0.3     # vùng "thuộc dòng" (giữ / đếm chồng lấn)


GROW_SEARCH = 0.6   # vùng tìm nét / quầng


GROW_CORE = 0.1     # lõi dòng


HEIGHT_CAP = 2.2    # cỡ dòng <= 2,2 lần cao mảng nét cao (khung OCR trùm hai hàng thì "cỡ" phình, dải quầng nuốt icon)


# --- 1. nét
MASK_GROW = float(os.environ.get("TEXTFIX_MASK_GROW", "0.12"))   # nới quanh nét (viền khử răng cưa còn lại thì LaMa vẽ lại hình chữ)


# --- 2. dự phòng
WEAK = 0.08         # Hi-SAM phủ < 8% lõi dòng = thấy quá ít (dòng thường 21-30%)


INK_SAME = 25.0     # cùng màu mực: <= 25 ΔE


INK_IN = 0.7        # mảng cùng màu mực là nét khi >= 70% nằm trong vùng dòng (vòng con dấu, viền nút phần lớn ở ngoài)


GLYPH_MAX = 2.0     # ... và cạnh dài <= 2 cao dòng (một chữ cái)


# --- 3. quầng
HALO_REACH = 0.35   # dải sát nét rộng 0,35 cao dòng


HALO_DE = 14.0      # điểm trong dải khác nền của dòng > 14 ΔE là quầng (nền ước từ vành ngay ngoài dải, trong vùng dòng: chữ


                    # trong nút thì vành là lòng nút, lòng nút không thành quầng)
HALO_LO = 7.0       # nền phẳng: quầng lan tiếp qua điểm 7-14 ΔE liền với nó (đuôi bóng mờ dần); điểm đậm hơn ngoài dải là hình khác


HALO_FLAT = 8.0     # nền phẳng = vành lệch trung vị <= 8 ΔE; nền có vân / ảnh chụp thì không lan


PLATE_FILL = float(os.environ.get("TEXTFIX_PLATE_FILL", "0.85"))   # >= 85% khe giữa các nét trong lõi dòng một màu, khác màu vành:
                    # chữ nằm trong LÒNG HÌNH (nút / ruy băng / thẻ có lề hẹp hơn dải) -> nền = màu lòng hình, không phải vành
PLATE_TAIL = 0.2    # lòng hình: đuôi bóng mờ chỉ lan trong 0,2 cao dòng quanh nét
PLATE_DE = 9.0      # lòng hình đồng màu (đo được màu chuẩn): quầng = khác lòng hình > 9 ΔE, đuôi mờ > 4 ΔE (bóng mềm trên thẻ
PLATE_LO = 4.0      # trắng để lại vết khi dùng ngưỡng nền thường 14 / 7)
HALO_LINE = 0.25    # dòng "có quầng" (nhật ký) khi quầng > 25% phần nét đã nới


def line_region(l: dict, H: int, W: int, grow: float) -> np.ndarray:
    """Vùng một dòng (bool cỡ ảnh): đa giác OCR, hoặc dải dọc đường chân với dòng cong; nới `grow` cao dòng, dọc nhiều hơn ngang
    (dấu thanh tiếng Việt nhô khỏi khung OCR)."""
    import cv2
    m = np.zeros((H, W), np.uint8)
    if l.get("path"):
        sz = float(l.get("size") or 1.0)
        for (ax, ay), (bx, by) in zip(l["path"][:-1], l["path"][1:]):
            ln = float(np.hypot(bx - ax, by - ay)) or 1.0
            nx, ny = (by - ay) / ln, -(bx - ax) / ln
            cv2.fillConvexPoly(m, np.int32([[ax - nx * 0.3 * sz, ay - ny * 0.3 * sz], [bx - nx * 0.3 * sz, by - ny * 0.3 * sz],
                                            [bx + nx * 1.2 * sz, by + ny * 1.2 * sz], [ax + nx * 1.2 * sz, ay + ny * 1.2 * sz]]), 1)
        h = sz
    else:
        b = l["box"]
        poly = l.get("poly") or [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
        cv2.fillPoly(m, [np.asarray(poly, np.int32)], 1)
        h = float(min(b[3] - b[1], b[2] - b[0])) if abs(l.get("angle") or 0) < 2 else float(l.get("size") or (b[3] - b[1]))
    k = max(1, int(round(grow * h)))
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * int(1.6 * k) + 1))).astype(bool)


@dataclass
class _Page:
    """Những gì mọi dòng dùng chung: ảnh Lab, nét Hi-SAM và các mảng nét, vùng từng dòng, vùng giữ, số dòng phủ mỗi điểm."""
    H: int
    W: int
    lab: np.ndarray
    ink: np.ndarray
    n_cc: int
    cc: np.ndarray
    stats: np.ndarray
    regions: dict
    count: np.ndarray
    hold: np.ndarray
    owner: np.ndarray    # mảnh nét -> số thứ tự dòng sở hữu trong M["L"] (-1: không dòng nào)
    index: dict          # id dòng -> số thứ tự
    keep: np.ndarray     # điểm không bao giờ gỡ: mảnh của dòng giữ, mảnh trang trí (nới 2 px: viền khử răng cưa)


@dataclass
class _Window:
    """Một dòng, cắt theo khung của vùng tìm: s = lát cắt; r2 vùng tìm, r vùng dòng, r0 lõi; hgt cao dòng."""
    s: tuple
    r2: np.ndarray
    r: np.ndarray
    r0: np.ndarray
    hgt: float


def erase_mask(draft: np.ndarray, ink: np.ndarray, M: dict, kept: set, log: list | None = None,
               plates: dict | None = None) -> np.ndarray:
    """Mặt nạ gỡ chữ của mọi dòng không giữ (bool cỡ ảnh) -- các bước ở đầu tệp.
    plates: nhận {id dòng: mặt nạ LÒNG HÌNH cỡ ảnh} của các dòng nằm trong nút / thẻ (_plate) -- dòng không còn chữ thì xoá cả
    hình (erase.empty_plates)."""
    pg = _page(draft, ink, M["L"], kept)
    mask = np.zeros((pg.H, pg.W), bool)
    sources, n_halo = {"nét": 0, "màu mực": 0, "lõi dòng": 0}, 0
    for l in M["L"]:
        if l["id"] in kept:
            continue
        w = _window(pg, l)
        if w is None:
            continue
        got, src = _strokes(pg, l, w)
        got &= ~pg.hold[w.s]
        grown, halo, inside = _halo(pg, w, got)
        mask[w.s] |= grown | halo
        if plates is not None and inside is not None:
            pm = np.zeros((pg.H, pg.W), bool)
            pm[w.s] = inside
            plates[l["id"]] = pm
        sources[src] += 1
        n_halo += bool(halo.sum() > HALO_LINE * max(1.0, float(grown.sum())))
    mask &= ~(pg.ink & pg.hold) & ~pg.keep
    if log is not None:
        free = pg.ink & ~mask & ~pg.hold
        log.append("nền = nháp trừ chữ (Hi-SAM + LaMa): mặt nạ %.1f%% ảnh; dòng theo %s; %d dòng có quầng; %d dòng giữ nguyên; nét "
                   "không thuộc dòng nào %.2f%% ảnh" % (100 * mask.mean(), ", ".join(f"{k} {v}" for k, v in sources.items() if v),
                                                       n_halo, len(kept), 100 * free.mean()))
    return mask


def _page(draft: np.ndarray, ink: np.ndarray, lines: list[dict], kept: set) -> _Page:
    import cv2
    H, W = draft.shape[:2]
    ink = np.asarray(ink, bool)
    n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    regions = {l["id"]: line_region(l, H, W, GROW_HOLD) for l in lines}
    count, hold = np.zeros((H, W), np.uint8), np.zeros((H, W), bool)
    for l in lines:
        count += regions[l["id"]]
        if l["id"] in kept:
            hold |= regions[l["id"]]
    owner = np.full(n_cc, -1, int)
    for k, ids in enumerate(own(Ink(ink), [regions[l["id"]] for l in lines])):
        owner[ids] = k
    index = {l["id"]: k for k, l in enumerate(lines)}
    keep = np.isin(cc, [i for i in range(1, n_cc) if owner[i] >= 0 and lines[owner[i]]["id"] in kept])
    for x0, y0, x1, y1 in (b for l in lines for b in (l.get("ink") or {}).get("ornament") or []):
        sl = (slice(y0, y1), slice(x0, x1))
        keep[sl] |= ink[sl]
    keep = _dilate(keep, _disk(2)) if keep.any() else keep
    return _Page(H, W, lab(draft), ink, n_cc, cc, stats, regions, count, hold, owner, index, keep)


def _window(pg: _Page, l: dict) -> _Window | None:
    """Cắt theo vùng tìm của dòng; cao dòng chặn trên bởi HEIGHT_CAP lần cao các mảng nét cao (phân vị 90: thân chữ hoa, không
    phải trung vị -- dấu thanh rời kéo trung vị xuống)."""
    import cv2
    R2 = line_region(l, pg.H, pg.W, GROW_SEARCH)
    ys, xs = np.nonzero(R2)
    if not len(ys):
        return None
    s = (slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1))
    r = pg.regions[l["id"]][s]
    hgt = float(l.get("size") or (l["box"][3] - l["box"][1]))
    tall = [i for i in np.unique(pg.cc[s][pg.ink[s] & r]) if i > 0 and pg.stats[i, cv2.CC_STAT_HEIGHT] >= 4]
    if len(tall) >= 3:
        hgt = min(hgt, HEIGHT_CAP * float(np.percentile(pg.stats[tall, cv2.CC_STAT_HEIGHT], 90)))
    return _Window(s, R2[s], r, line_region(l, pg.H, pg.W, GROW_CORE)[s], hgt)


def _strokes(pg: _Page, l: dict, w: _Window) -> tuple[np.ndarray, str]:
    """Bước 1 + 2: nét của dòng -> (mặt nạ trong cửa sổ, nguồn: "nét" / "màu mực" / "lõi dòng")."""
    import cv2
    ik, c_ = pg.ink[w.s], pg.cc[w.s]
    own_ = pg.owner[c_]
    got = ((own_ == pg.index[l["id"]]) & w.r2) | (ik & w.r0 & (own_ < 0))
    got &= ~pg.keep[w.s]
    core = max(1.0, float(w.r0.sum()))
    if float((got & w.r0).sum()) >= WEAK * core:
        return got, "nét"
    if l.get("color"):
        cand = w.r2 & (np.linalg.norm(pg.lab[w.s] - hex_lab(l["color"]), axis=2) <= INK_SAME)
        k, c2, s2, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
        touch, ins = np.bincount(c2[w.r0], minlength=k), np.bincount(c2[w.r], minlength=k)
        ok = [i for i in range(1, k) if touch[i] and ins[i] >= INK_IN * s2[i, cv2.CC_STAT_AREA]
              and max(s2[i, cv2.CC_STAT_WIDTH], s2[i, cv2.CC_STAT_HEIGHT]) <= GLYPH_MAX * w.hgt]
        if ok:
            got |= np.isin(c2, ok)
            if float((got & w.r0).sum()) >= WEAK * core:
                return got, "màu mực"
    return got | w.r0, "lõi dòng"


def _disk(radius: int) -> np.ndarray:
    import cv2
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _dilate(m: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.dilate(m.astype(np.uint8), kernel).astype(bool)


def _halo(pg: _Page, w: _Window, got: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Bước 3 + 4: (nét đã nới, quầng + dấu câu, lòng hình chứa dòng hoặc None) trong cửa sổ."""
    import cv2
    grown = _dilate(got, _disk(max(3, int(round(MASK_GROW * w.hgt))))) & w.r2
    near = _dilate(got, _disk(max(4, int(round(HALO_REACH * w.hgt))))) & w.r2
    ring = (w.r2 & ~near & ~pg.ink[w.s]).astype(np.float32)
    if float(ring.sum()) <= 50:
        return grown, np.zeros(got.shape, bool), None
    L = pg.lab[w.s]
    # nền của dòng: trung bình có trọng số của vành ngay ngoài dải sát nét, làm mượt theo cao dòng
    sg = max(6.0, 0.6 * w.hgt)
    den = cv2.GaussianBlur(ring, (0, 0), sg)
    bg = cv2.GaussianBlur(L * ring[..., None], (0, 0), sg) / np.maximum(den[..., None], 1e-3)
    d = np.where(den > 0.02, np.linalg.norm(L - bg, axis=2), 0.0)
    halo = near & ~got & (d > HALO_DE)
    halo = cv2.morphologyEx(halo.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    halo &= _touching(halo | grown, grown)   # quầng dính vào nét; icon cách chữ một khe nền thì không
    flat, reach, inside = float(np.median(d[ring > 0])) <= HALO_FLAT, w.r2, None
    if PLATE_FILL < 1 and halo.any():
        pl = _plate(L, ring > 0, got, grown, w)
        if pl is not None:   # lòng hình: đo lại khác nền so với màu lòng hình, quầng không ra khỏi hình; đuôi bóng mờ chỉ sát nét
            d, inside = pl   # (lòng hình có ánh chuyển màu ở mép: lan xa thì đuôi mờ chính là lòng hình)
            halo = near & inside & ~got & (d > PLATE_DE)
            halo = cv2.morphologyEx(halo.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
            halo &= _touching(halo | grown, grown)
            flat, reach = True, inside & _dilate(got, _disk(max(3, int(round(PLATE_TAIL * w.hgt)))))
    if halo.any() and flat:
        lo, hi = (PLATE_LO, PLATE_DE) if inside is not None else (HALO_LO, HALO_DE)
        tail = ((d > lo) & (d <= hi) & reach & ~got) | halo   # nền phẳng: lan theo đuôi mờ (chỉ điểm nhạt)
        halo = _touching(tail, halo)
    halo |= _punctuation(w, d, got, grown)
    others = ((pg.count[w.s] - w.r.astype(np.uint8)) > 0) & ~w.r0   # vùng của dòng khác: dòng kia tự lo phần của nó
    halo = _dilate(halo & ~others, np.ones((5, 5), np.uint8)) & w.r2 & ~others & ~pg.hold[w.s]
    if inside is not None:
        halo &= _dilate(inside, np.ones((3, 3), np.uint8))
    return grown, halo, inside


def _plate(L: np.ndarray, ring: np.ndarray, got: np.ndarray, grown: np.ndarray, w: _Window) -> tuple | None:
    """Chữ nằm trong một hình (nút, ruy băng, thẻ) có lề hẹp hơn dải sát nét: vành ước nền rơi ra NGOÀI hình, cả lòng hình thành
    "quầng" và bị gỡ -> LaMa vẽ lại hình bằng màu bên ngoài (nút bẹp thành vệt, ruy băng khuyết; probe 11/11). Nhận ra: khe giữa
    các nét (trong khung các nét) gần như MỘT màu (>= PLATE_FILL số điểm trong HALO_DE quanh trung vị) và màu đó khác hẳn vành.
    Viền / bóng chữ thật, chữ trên ảnh chụp: khe lẫn màu -> không phải lòng hình.
    -> (khoảng cách màu tới lòng hình, trong hình = mảng lòng hình chạm khe + các lỗ của nó: chữ, bóng chữ), hoặc None."""
    import cv2
    ys, xs = np.nonzero(got)
    if not len(ys):
        return None
    gap = np.zeros_like(got)
    gap[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True   # khe = trong khung các nét (lõi dòng có thể thò ra ngoài nút)
    gap &= w.r0 & ~grown
    if float(gap.sum()) < 30 or float(ring.sum()) < 30:
        return None
    c = np.median(L[gap], axis=0)
    d = np.linalg.norm(L - c, axis=2)
    same = d <= HALO_DE
    if float(same[gap].mean()) < PLATE_FILL:
        return None
    if float(np.linalg.norm(c - np.median(L[ring], axis=0))) <= HALO_DE:   # khe cùng màu vành: chữ trên nền trơn, đường cũ đã đúng
        return None
    plate = _touching(same & w.r2, gap & same)
    _, cc = cv2.connectedComponents((~plate).astype(np.uint8), connectivity=4)
    edge = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
    holes = ~plate & ~np.isin(cc, edge)
    return d, plate | holes


def _touching(m: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Các mảng liên thông (8 hướng) của m có chạm seed."""
    import cv2
    _, c = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    ids = np.unique(c[seed])
    return np.isin(c, ids[ids > 0])


def _punctuation(w: _Window, d: np.ndarray, got: np.ndarray, grown: np.ndarray) -> np.ndarray:
    """Mảng đậm nhỏ (<= cao dòng) nằm gọn trong lõi dòng (không chạm mép lõi), giữa nét đầu và nét cuối của dòng: dấu câu / gạch
    nối Hi-SAM không coi là nét. Thứ đứng trước chữ là icon đầu dòng: để nguyên."""
    import cv2
    n, c, s, _ = cv2.connectedComponentsWithStats(((d > HALO_DE) & w.r0 & ~grown).astype(np.uint8), 8)
    rim = w.r0 & (cv2.erode(w.r0.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
    edge = np.bincount(c[rim], minlength=n)
    gx = np.nonzero(got.any(axis=0))[0]
    lo, hi = (int(gx.min()), int(gx.max())) if len(gx) else (0, -1)
    small = [i for i in range(1, n) if not edge[i] and max(s[i, cv2.CC_STAT_WIDTH], s[i, cv2.CC_STAT_HEIGHT]) <= w.hgt
             and s[i, cv2.CC_STAT_LEFT] >= lo and s[i, cv2.CC_STAT_LEFT] + s[i, cv2.CC_STAT_WIDTH] - 1 <= hi]
    return np.isin(c, small) if small else np.zeros(got.shape, bool)
