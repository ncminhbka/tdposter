"""LoRA cho FLUX.2 klein 4B vẽ chữ tiếng Việt (docs/LORA_PLAN.md).

Dữ liệu = cặp do chính TextFix sinh: cùng prompt, cùng seed, nháp FLUX (chữ sai) và poster TextFix (chữ đúng nguyên văn); chỉ khác
ở vùng chữ. Các bước, mỗi bước một mô-đun chạy bằng `python -m lora.<bước>`:

  requests  LLM sinh yêu cầu poster tiếng Việt (phủ đủ nguyên âm x thanh), ảnh sản phẩm, lời sửa
  drive     gọi API máy chủ TextFix (FLUX 4B tại chỗ) -> các lượt chạy trong TEXTFIX_RUNS (3 kiểu: tạo mới / có ảnh sản phẩm / sửa)
  collect   lượt chạy -> bộ dữ liệu sạch: poster đích, mặt nạ vùng chữ, ảnh tham chiếu đúng thứ tự FLUX đã nhận
  cache     mã hoá trước: chữ (Qwen3) + ảnh (VAE) -> tệp .pt (lúc huấn luyện chỉ còn DiT trên card)
  train     LoRA trên DiT base 4B, loss flow-matching nhân trọng số vùng chữ, đúng đường suy luận của textfix/draw/flux.py
  sheets    bảng so sánh bằng mắt cùng seed: [TextFix | distill | distill+LoRA | base | base+LoRA]

net.py (lớp LoRA, gắn / lưu / gộp) và flow.py (toán huấn luyện) không cần GPU, có test ở tests/test_lora.py.
"""
