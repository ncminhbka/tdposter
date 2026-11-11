# TextFix API v1: sinh poster cho chatbot

Một lời gọi **đồng bộ**: gửi prompt, nhận lại **một** ảnh. Lõi là dây chuyền TextFix (LLM viết prompt và chốt chữ → FLUX vẽ nháp
→ gỡ chữ méo → viết lại đúng chữ tiếng Việt). Máy chủ chỉ có API này (12/11: bỏ API demo có form, giao diện, zip debug).

```
POST /api/v1/images/generations
Header: x-api-key: <khoá>          (do cổng API kiểm)
        Content-Type: application/json
        Accept: application/json   -> JSON có ảnh base64 (mặc định)
                image/png          -> ảnh PNG thô
        X-Request-Id: <tuỳ chọn>   -> dùng lại làm request_id để đối chiếu log (không gửi thì máy chủ tự sinh)
```

## Thân yêu cầu

| Trường          | Kiểu    | Bắt buộc | Mặc định | Ghi chú                                                                                                                                                                                                                                                                                                                                     |
| ----------------- | -------- | ---------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prompt`        | string   | có        |             | Lời người dùng hoặc prompt đã làm giàu, tiếng Việt hoặc tiếng Anh. Chữ cần in trên ảnh nên đặt trong ngoặc kép.                                                                                                                                                                                                        |
| `image_urls`    | string[] | không     | `[]`      | Link ảnh `https://` public. **Bản đầu chỉ dùng ảnh đầu tiên**, coi là ảnh sản phẩm: poster mới có chứa sản phẩm đó, chữ in trên sản phẩm được giữ. Các ảnh sau bị bỏ qua.                                                                                                                           |
| `images_base64` | string[] | không     | `[]`      | Ảnh gửi thẳng trong thân: base64 hoặc data URI (`data:image/png;base64,...`), tối đa 15 MB mỗi ảnh. **Thắng `image_urls`** (có thì không tải link). Dùng khi máy chủ không ra được kho ảnh. Bản đầu chỉ dùng ảnh đầu tiên, như `image_urls`. `image_base64` (một chuỗi) cũng được nhận. |
| `aspect_ratio`  | string   | không     | `"1:1"`   | `1:1`, `9:16`, `16:9`, `3:4`, `4:3`                                                                                                                                                                                                                                                                                                |

Khoá lạ không làm hỏng yêu cầu (bỏ qua, ghi log).

Ví dụ:

```bash
curl -X POST "https://<cổng>/textfix/api/v1/images/generations" \
  -H "x-api-key: $KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "Poster khai trương tiệm trà sữa Mây, giảm 20% tuần đầu, hotline 0901 234 567", "aspect_ratio": "9:16"}'

# nhận PNG thô
curl -X POST "https://<cổng>/textfix/api/v1/images/generations" \
  -H "x-api-key: $KEY" -H "Content-Type: application/json" -H "Accept: image/png" \
  -d '{"prompt": "Poster giới thiệu sản phẩm này", "image_urls": ["https://<cdn>/abc.png"]}' --output poster.png
```

## Trả lời thành công (200)

```json
{"status": "SUCCESS", "code": 200, "message": "Image generated successfully", "request_id": "…",
 "data": {"image_base64": "iVBORw0…", "mime_type": "image/png", "aspect_ratio": "9:16", "width": 768, "height": 1344,
          "tokens": 1.0, "execution_time_ms": 31250}}
```

Với `Accept: image/png`: thân là ảnh PNG, kèm header `X-Request-Id`, `X-Aspect-Ratio`, `X-Token-Cost`, `X-Execution-Time-Ms`.

Mọi trả lời (cả JSON lẫn PNG) có thêm hai header để đo và tra lỗi, không bắt buộc đọc:
- `X-Run-Id`: mã lượt chạy trên máy chủ (cả khi lỗi, nếu đã bắt đầu chạy), báo kèm khi cần tra một ảnh hỏng.
- `X-Timing`: giây từng bước, ví dụ `cho_ve=0;llm=3.5;ve_nhap=2.5;net_chu=0.2;dong=1.2;cho_thiet_ke=0;vlm=5.6;go_chu=1.4;cho_lama=0;ghep=0.1;dung=3.6;luu=0.1;tong=17.2`.

- **Cỡ ảnh** (~1 MP):
  | Tỉ lệ  | 1:1        | 9:16      | 16:9      | 3:4       | 4:3       |
  | -------- | ---------- | --------- | --------- | --------- | --------- |
  | Cỡ (px) | 1024×1024 | 768×1344 | 1344×768 | 864×1152 | 1152×864 |
- **`tokens`**: bản đầu là một mức cố định mỗi ảnh (cấu hình `TEXTFIX_V1_TOKENS`, mặc định 1.0).

## Thời gian

| Loại            | Thời gian                                                                             |
| ---------------- | -------------------------------------------------------------------------------------- |
| Poster có chữ  | thường **18–30 s** (LLM ~4 s, vẽ ~3 s, đo dòng ~1 s, VLM ~6 s (gỡ chữ chạy song song), dựng ~4 s) |
| Ảnh không chữ | ~12–15 s (khi prompt không có chữ nào cần in)                                    |

- Máy chủ chặn ở **55 s** (`TEXTFIX_V1_DEADLINE`) và trả `504`. Client nên đặt timeout ≥ 60 s.
- Khi bận, máy chủ trả ngay `503` kèm `Retry-After`, không để treo kết nối.

## Lỗi

Khuôn chung:

```json
{"status": "ERROR", "code": 400, "error_code": "ERR_EMPTY_PROMPT", "message": "…", "request_id": "…", "data": null}
```

| HTTP | `error_code`                  | Khi nào                                                                                                                                                                               |
| ---- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 400  | `ERR_EMPTY_PROMPT`            | `prompt` rỗng / chỉ khoảng trắng                                                                                                                                                 |
| 400  | `ERR_INVALID_ASPECT_RATIO`    | tỉ lệ ngoài 5 tỉ lệ trên                                                                                                                                                         |
| 400  | `ERR_IMAGE_DOWNLOAD_FAILED`   | không tải được ảnh đầu của `image_urls`: link chết / lỗi HTTP / quá giờ (15 s) / quá 15 MB / không phải ảnh / không phải `https` / host chưa được cho phép |
| 400  | `ERR_INVALID_IMAGE`           | ảnh trong `images_base64` không giải mã được / không phải ảnh / quá 15 MB                                                                                                 |
| 400  | `ERR_INVALID_REQUEST`         | thân không phải JSON hoặc sai kiểu (ví dụ `image_urls` không phải mảng)                                                                                                    |
| 422  | `ERR_IMAGE_SAFETY_VIOLATION`  | máy vẽ ảnh từ chối nội dung                                                                                                                                                      |
| 500  | `ERR_IMAGE_GENERATION_FAILED` | lỗi trong dây chuyền; `message` có mã nội bộ (`llm_failed`, `vlm_failed`, `render_failed`...)                                                                           |
| 503  | `ERR_SERVICE_BUSY`            | hàng đợi đầy (`TEXTFIX_MAX_QUEUE`); thử lại sau `Retry-After` giây                                                                                                         |
| 503  | `ERR_SERVICE_UNAVAILABLE`     | máy chủ đang khởi động / nạp model lỗi                                                                                                                                         |
| 504  | `ERR_UPSTREAM_TIMEOUT`        | quá 55 s                                                                                                                                                                              |

## Giới hạn của bản đầu

- **Chưa sửa được poster cũ.** Gửi lại poster do chính hệ thống tạo trong `image_urls` thì nó bị coi như ảnh sản phẩm, kết
  quả là một poster mới. Sửa thật (đổi chữ, thêm vật, giữ nguyên phần còn lại) là việc của đợt sau.
- **Chỉ dùng ảnh đầu** của `image_urls`.
- **Chưa chống chạy trùng:** client gọi lại sau timeout thì máy chủ sinh thêm một ảnh.
- **Prompt sơ sài** (ví dụ "poster quán cà phê"): hệ thống tự viết thêm vài câu ngắn (tiêu đề, khẩu hiệu, nút), không bao giờ
  bịa số, giá, ngày, liên hệ.

## Cấu hình máy chủ

| Biến                                               | Mặc định       | Ý nghĩa                                                                                                                                      |
| --------------------------------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `TEXTFIX_IMAGE_HOSTS`                             | `amazonaws.com` | Đuôi tên miền được phép tải ảnh, cách nhau bằng phẩy. Phải thêm host CDN của router.                                           |
| `TEXTFIX_IMAGE_MAX_MB`, `TEXTFIX_IMAGE_TIMEOUT` | 15, 15            | Giới hạn tải ảnh                                                                                                                           |
| `TEXTFIX_IMAGE_ALLOW_PRIVATE`                     | (tắt)            | `1`: host trong danh sách được phép phân giải ra IP nội bộ (CDN trong mạng công ty). Mặc định chặn mọi IP không công khai. |
| `TEXTFIX_V1_DEADLINE`                             | 55                | Giây tối đa cho một yêu cầu, tính cả thời gian tải ảnh; quá thì trả 504                                                          |
| `TEXTFIX_MAX_QUEUE`                               | 8                 | Số yêu cầu tối đa đang chờ hoặc đang chạy; quá thì trả 503                                                                        |
| `TEXTFIX_V1_TOKENS`                               | 1.0               | `tokens` mỗi ảnh                                                                                                                           |

Thử nhanh: `python scripts/v1_smoke.py --url <gốc máy chủ> -n 3 --bad` (in mã, giây, cỡ ảnh từng ca; khoá lấy từ `.env`).
