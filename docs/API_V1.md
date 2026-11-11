# TextFix API v1: sinh poster cho chatbot

Một lời gọi **đồng bộ**: gửi lời chat của người dùng (kèm ảnh sản phẩm nếu có), nhận lại **một** poster. Lõi là dây chuyền
TextFix: LLM viết prompt và chốt chữ → FLUX vẽ nháp → gỡ chữ méo → viết lại đúng chữ tiếng Việt.

```
POST /api/v1/images/generations
Header: x-api-key: <khoá>          (do cổng API kiểm)
        Content-Type: application/json
        Accept: application/json   -> JSON có ảnh base64 (mặc định)
                image/png          -> ảnh PNG thô
        X-Request-Id: <tuỳ chọn>   -> dùng lại làm request_id để đối chiếu log (không gửi thì máy chủ tự sinh)
```

## Thân yêu cầu

| Trường | Kiểu | Bắt buộc | Mặc định | Ghi chú |
|---|---|---|---|---|
| `prompt` | string | có | | Lời người dùng hoặc prompt đã làm giàu, tiếng Việt hoặc tiếng Anh. Chữ cần in trên ảnh nên đặt trong ngoặc kép. |
| `images_base64` | string[] | không | `[]` | Ảnh tham chiếu (ảnh sản phẩm). Mỗi phần tử là base64 của tệp ảnh PNG / JPEG / WebP, có thể kèm tiền tố data URI (`data:image/jpeg;base64,...`). Tối đa 15 MB mỗi ảnh. Bản đầu **chỉ dùng ảnh đầu tiên**: poster mới có chứa sản phẩm đó, chữ in trên sản phẩm được giữ nguyên. |
| `aspect_ratio` | string | không | `"1:1"` | `1:1`, `9:16`, `16:9`, `3:4`, `4:3` |

Đây là **cách duy nhất** để gửi ảnh. Các khoá khác (kể cả `image_urls`, `image_base64` của bản nháp trước) bị bỏ qua, không
báo lỗi.

Máy chủ thu ảnh về cạnh dài 1024 px trước khi dùng, nên client nên thu ảnh về khoảng 1024–1536 px (JPEG) trước khi gửi: thân
yêu cầu nhẹ hơn nhiều mà kết quả như nhau.

Ví dụ:

```bash
# chỉ có lời chat
curl -X POST "https://<cổng>/textfix/api/v1/images/generations" \
  -H "x-api-key: $KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "Poster khai trương tiệm trà sữa Mây, giảm 20% tuần đầu, hotline 0901 234 567", "aspect_ratio": "9:16"}'

# có ảnh sản phẩm: base64 của tệp ảnh trong images_base64 (thân ghi ra tệp vì chuỗi base64 dài)
printf '{"prompt": "Poster giới thiệu sản phẩm này, nền sáng", "images_base64": ["%s"]}' "$(base64 -w0 san_pham.jpg)" > body.json
curl -X POST "https://<cổng>/textfix/api/v1/images/generations" \
  -H "x-api-key: $KEY" -H "Content-Type: application/json" -d @body.json

# nhận PNG thô thay vì JSON
curl -X POST "https://<cổng>/textfix/api/v1/images/generations" \
  -H "x-api-key: $KEY" -H "Content-Type: application/json" -H "Accept: image/png" \
  -d '{"prompt": "Poster quán cà phê Mộc, mở cửa 7h-22h"}' --output poster.png
```

Từ code (ảnh người dùng tải lên chatbot, đang ở dạng bytes):

```python
import base64, requests
body = {"prompt": "Poster giới thiệu sản phẩm này", "aspect_ratio": "1:1",
        "images_base64": [base64.b64encode(image_bytes).decode()]}
r = requests.post(URL, json=body, headers={"x-api-key": KEY}, timeout=60)
png = base64.b64decode(r.json()["data"]["image_base64"])
```

```js
const body = { prompt: "Poster giới thiệu sản phẩm này", images_base64: [buffer.toString("base64")] };
const r = await fetch(URL, { method: "POST", headers: { "x-api-key": KEY, "Content-Type": "application/json" },
                             body: JSON.stringify(body), signal: AbortSignal.timeout(60000) });
const png = Buffer.from((await r.json()).data.image_base64, "base64");
```

## Trả lời thành công (200)

```json
{"status": "SUCCESS", "code": 200, "message": "Image generated successfully", "request_id": "…",
 "data": {"image_base64": "iVBORw0…", "mime_type": "image/png", "aspect_ratio": "9:16", "width": 768, "height": 1344,
          "tokens": 1.0, "execution_time_ms": 21250}}
```

Với `Accept: image/png`: thân là ảnh PNG, kèm header `X-Request-Id`, `X-Aspect-Ratio`, `X-Token-Cost`, `X-Execution-Time-Ms`.

Mọi trả lời (cả JSON lẫn PNG) có thêm hai header để đo và tra lỗi, không bắt buộc đọc:
- `X-Run-Id`: mã lượt chạy trên máy chủ (cả khi lỗi, nếu đã bắt đầu chạy). Báo kèm mã này khi cần tra một ảnh hỏng.
- `X-Timing`: giây từng bước, ví dụ `llm=3.5;cho_ve=0;ve_nhap=2.5;net_chu=0.2;dong=1.2;cho_thiet_ke=0;vlm=5.6;go_chu=1.4;cho_lama=0;ghep=0.1;dung=0.6;luu=0.1;tong=15.2`.

Cỡ ảnh (~1 MP):

| Tỉ lệ | 1:1 | 9:16 | 16:9 | 3:4 | 4:3 |
|---|---|---|---|---|---|
| Cỡ (px) | 1024×1024 | 768×1344 | 1344×768 | 864×1152 | 1152×864 |

`tokens`: bản đầu là một mức cố định mỗi ảnh (1.0).

## Thời gian

| Loại | Thời gian |
|---|---|
| Poster có chữ | khoảng **20 s** khi máy rảnh, tới ~35 s khi nhiều yêu cầu cùng lúc |
| Ảnh không chữ (prompt không có chữ nào cần in) | ~10–15 s |

- Máy chủ chặn ở **55 s** và trả `504`. Client nên đặt timeout ≥ 60 s.
- Khi bận, máy chủ trả ngay `503` kèm `Retry-After`, không để treo kết nối.

## Lỗi

Khuôn chung:

```json
{"status": "ERROR", "code": 400, "error_code": "ERR_EMPTY_PROMPT", "message": "…", "request_id": "…", "data": null}
```

| HTTP | `error_code` | Khi nào |
|---|---|---|
| 400 | `ERR_EMPTY_PROMPT` | `prompt` rỗng / chỉ khoảng trắng |
| 400 | `ERR_INVALID_ASPECT_RATIO` | tỉ lệ ngoài 5 tỉ lệ trên |
| 400 | `ERR_INVALID_IMAGE` | một ảnh trong `images_base64` không giải mã được / không phải ảnh / quá 15 MB |
| 400 | `ERR_INVALID_REQUEST` | thân không phải JSON hoặc sai kiểu (ví dụ `images_base64` không phải mảng) |
| 422 | `ERR_IMAGE_SAFETY_VIOLATION` | máy vẽ ảnh từ chối nội dung |
| 500 | `ERR_IMAGE_GENERATION_FAILED` | lỗi trong dây chuyền; `message` có mã nội bộ (`llm_failed`, `vlm_failed`, `render_failed`...) |
| 503 | `ERR_SERVICE_BUSY` | hàng đợi đầy; thử lại sau `Retry-After` giây |
| 503 | `ERR_SERVICE_UNAVAILABLE` | máy chủ đang khởi động / nạp model lỗi |
| 504 | `ERR_UPSTREAM_TIMEOUT` | quá 55 s |

## Giới hạn của bản đầu

- **Chưa sửa được poster cũ.** Gửi lại poster do chính hệ thống tạo làm ảnh tham chiếu thì nó bị coi như ảnh sản phẩm, kết
  quả là một poster mới. Sửa thật (đổi chữ, thêm vật, giữ nguyên phần còn lại) là việc của đợt sau.
- **Chỉ dùng ảnh đầu** của `images_base64`.
- **Chưa chống chạy trùng:** client gọi lại sau timeout thì máy chủ sinh thêm một ảnh.
- **Prompt sơ sài** (ví dụ "poster quán cà phê"): hệ thống tự viết thêm vài câu ngắn (tiêu đề, khẩu hiệu, nút), không bao giờ
  bịa số, giá, ngày, liên hệ.

## Cấu hình máy chủ (đội vận hành)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `TEXTFIX_IMAGE_MAX_MB` | 15 | Dung lượng tối đa mỗi ảnh trong `images_base64` |
| `TEXTFIX_V1_DEADLINE` | 55 | Giây tối đa cho một yêu cầu; quá thì trả 504 |
| `TEXTFIX_MAX_QUEUE` | 8 | Số yêu cầu tối đa đang chờ hoặc đang chạy; quá thì trả 503 |
| `TEXTFIX_V1_TOKENS` | 1.0 | `tokens` mỗi ảnh |
