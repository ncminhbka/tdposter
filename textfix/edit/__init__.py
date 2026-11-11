"""SỬA POSTER (11/11): poster cũ + lời sửa -> bản sửa cùng khuôn với brief; FLUX sửa ảnh cũ, TextFix chạy lại từ đầu.

  plan.py          LLM sửa: chữ cũ + lời sửa -> danh sách chữ mới + prompt sửa cho FLUX (code kiểm lại)
  read.py          ảnh lạ: VLM đọc chữ trên ảnh -> danh sách chữ cũ
  instructions.py  lời dặn + khuôn JSON của LLM sửa
Poster của mình mang sẵn chữ cũ trong file PNG (textfix/meta.py).
"""

from .plan import make_edit
from .read import read_image

__all__ = ["make_edit", "read_image", "nearest_ratio"]


def nearest_ratio(w: int, h: int) -> str:
    """Tỉ lệ khung API nhận gần nhất với một ảnh (ảnh lạ không mang tỉ lệ)."""
    import math

    from .instructions import RATIOS
    r = math.log(w / h)
    return min(RATIOS, key=lambda x: abs(math.log(int(x.split(":")[0]) / int(x.split(":")[1])) - r))
