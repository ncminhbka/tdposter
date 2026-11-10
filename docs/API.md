# TextFix API

Máy chủ `server/app.py` (FastAPI, cổng `TEXTFIX_PORT`, mặc định 8088). Một yêu cầu = một form (6 loại như app) → các poster
tiếng Việt, kèm đủ ảnh và số liệu debug của từng bước.

## Luồng gọi

```
POST /api/jobs            -> {"job_id", "run_id", "queue"}            gửi yêu cầu (trả ngay)
GET  /api/jobs/{job_id}   -> {"status", "progress", "result", "error", "position", "elapsed"}   hỏi mỗi 1-2 s
GET  /api/jobs/{job_id}/poster                                         ảnh poster PNG khi status = "done" (?index=i, ?format=base64)
GET  /api/runs/{run_id}/zip                                           mọi tệp debug của lượt chạy (zip)
GET  /api/health          -> {"ready", "error", "queue", ...}          máy chủ đã nạp model chưa
POST /api/generate        cùng thân như /api/jobs, chờ tới khi xong (đồng bộ; dùng cho thử nhanh)
```

`status`: `queued` → `running` → `done` | `error`. Trong lúc chạy, `result.posters[i]` hiện dần (nháp có trước poster).
Kết quả và tệp giữ trên máy chủ có giới hạn (job 6 giờ, thư mục lượt chạy theo `MAX_RUNS`); cần giữ lâu thì tải zip.

## Thân yêu cầu (JSON)

Mọi trường đều là chuỗi, trừ khi ghi khác. Trường trống / thiếu = không điền.

**Loại form** — `category` (bắt buộc), một trong:

| `category` | Form trong app | Trường riêng (`*` = bắt buộc khi KHÔNG có `prompt`) |
|---|---|---|
| `promo` | Poster khuyến mại / quảng cáo | `title`, `discount`*, `applied_product`, `date_start`, `date_end` |
| `product_intro` | Ảnh giới thiệu sản phẩm | `title`, `product_name`*, `product_desc`*, `highlights`, `price` |
| `opening` | Banner khai trương | `title`, `opening_date`*, `opening_promo`, `booking_contact` |
| `feedback` | Feedback khách hàng | `title`, `feedback_target`* (sản phẩm / dịch vụ), `feedback_quote`* (mô tả ngắn feedback), `feedback_highlights`, `special_offer` |
| `recruitment` | Tuyển dụng | `title`, `job_position`*, `job_desc`, `apply_deadline`*, `apply_method`* |
| `guide` | Quy trình / hướng dẫn | `title`, `guide_steps` (mảng chuỗi, theo thứ tự) — không trường nào bắt buộc |

**Chung mọi loại**
- `prompt` — mô tả tự do của người dùng ("prompt ảnh"). Có `prompt` thì không đòi trường bắt buộc; prompt luôn thắng form khi
  hai bên mâu thuẫn.
- `image_base64` — ảnh tải lên (base64 hoặc data URI). Chữ in trên sản phẩm trong ảnh được giữ nguyên.
- `image_url` — thay cho `image_base64` (chỉ gửi một trong hai): link ảnh public `https://`, máy chủ tự tải. Host phải nằm trong
  `TEXTFIX_IMAGE_HOSTS` (đuôi tên miền, phẩy; mặc định `amazonaws.com` = S3 public) và phân giải ra IP công khai; không theo
  chuyển hướng; tối đa `TEXTFIX_IMAGE_MAX_MB` (15) MB, chờ `TEXTFIX_IMAGE_TIMEOUT` (15) giây. Ảnh được tải khi nhận yêu cầu
  (trước khi xếp hàng), hỏng thì trả 422 ngay.
- Thông tin cửa hàng (không bắt buộc): `store_name`, `phone`, `address`, `website_link`.

**Hiển thị / thiết kế** (không bắt buộc)
- `style_pref`: `auto` (mặc định) | `hien_dai` | `sang_trong` | `nang_dong` | `am_cung` | `le_hoi`.
- `primary_color`: `#rrggbb`, bỏ trống = tự động.
- `aspect_ratio`: `1:1` (mặc định) | `4:5` | `9:16` | `16:9`.
- `num_images`: 1-4 (mặc định 1). Mỗi ảnh một hướng thiết kế.
- `seed`: số nguyên, để chạy lại đúng như cũ.

Trường thuộc loại form khác và khoá lạ **không làm hỏng yêu cầu**: chúng bị bỏ qua và được liệt kê lại trong
`result.ignored_fields`.

Ví dụ:
```json
{"category": "recruitment", "title": "Tuyển nhân viên pha chế", "job_position": "Nhân viên pha chế",
 "apply_deadline": "30/11/2026", "apply_method": "Gửi CV về hr@maycoffee.vn", "store_name": "Mây Coffee",
 "style_pref": "hien_dai", "aspect_ratio": "4:5", "num_images": 2}
```

## Kết quả (`result`)

```
run_id, status ("running" | "success"), mode ("poster" | "image": không chữ nào cần viết -> ảnh FLUX là kết quả),
width, height, final_poster_url, elapsed_seconds, ignored_fields, debug_zip_url,
brief: {prompt_en, texts: [{text, role, source}], variants, log}      bước 1: LLM viết prompt + danh sách chữ
posters: [ {index, seed, direction, status ("drawing" | "designing" | "done" | "error"), stage, error,
            poster_url, plan_url, ... các ảnh debug, timing, missing, errors} ]
```

**Lấy poster**: `GET /api/jobs/{job_id}/poster` -> PNG (poster xong đầu tiên); `?index=i` poster thứ i; `?format=base64` ->
JSON `{job_id, index, width, height, mime, image_base64}`. Khi xong, `GET /api/jobs/{job_id}` có thêm `"poster"` (đường dẫn này),
`result` có `job_id` (cho `/api/generate`). Lỗi: 409 `not_ready` (chưa xong) / `job_failed` / `poster_failed` (kèm `error`),
404 `job_not_found` / `poster_not_found`, 410 `poster_expired` (đã bị dọn), 422 `invalid_format`.

Các trường `*_url` khác (ảnh debug, `final_poster_url`) là đường dẫn tương đối với gốc máy chủ (`outputs/<run_id>/v<i>/...`).

## Debug từng bước (mỗi poster)

| Bước | Tệp / trường | Nội dung |
|---|---|---|
| 1. LLM | `result.brief`, `brief.json` | prompt FLUX, danh sách chữ sẽ vẽ (vai, nguồn: khách / AI viết thêm), nhật ký kiểm |
| 2. FLUX | `draft_url` (`draft.png`) | nháp FLUX vẽ (chữ giun) |
| 3. Hi-SAM | `hisam_url` (`hisam.jpg`), `ink.png` | nét chữ tìm được (hồng) |
| 4a. Dòng | `slots_url` (`slots.jpg`), `plan.json` → `slots.L` | mỗi dòng: khung, kiểu (ngang / nghiêng / cong / dọc), số đo nét (`ink`: cao chữ, màu, đường chân...), hàng nhiều cỡ (`row`) |
| 4b. Gỡ chữ | `ink_url` (`ink.jpg`), `clean_url` (`clean.jpg`), `base_url` (`base.jpg`), `plan.json` → `erase` | mặt nạ gỡ (xanh lá = nét bị gỡ, vàng = quầng, xanh dương = dòng giữ), nền đã gỡ, nền dựng |
| 4c. VLM | `plan.json` → `assign`, `style`, `accent`, `fates` | câu nào vào dòng nào (một lời gọi, khuôn JSON strict: mọi dòng có mục riêng); **số phận từng dòng** (viết / giữ / xoá + lý do). Dòng FLUX vẽ thừa (không câu khách nào hợp) được viết bằng chữ VLM đoán theo ý nháp: trong `ops` là mục `text` không có `client` |
| 4d. Bố cục | `ops_url` (`ops.jpg`), `plan.json` → `ops` | đơn vị chữ, tập cùng vai, cỡ, lề, font |
| 5. Dựng | `poster_url`, `poster_first_url` (khi vòng sửa chạy), `plan.json` → `render`, `errors`, `errors_first` | cỡ / co / nén từng phần tử, lỗi bộ kiểm lượt đầu và lượt cuối |
| Tổng | `steps_url` (`steps.jpg`), `plan.json` → `log`, `timing` | ảnh ghép nháp → dòng → mặt nạ → nền → poster; nhật ký; thời gian từng bước |

`missing`: câu của khách không có trên poster (FLUX không vẽ chỗ cho nó). `errors` (số): lỗi bộ kiểm còn lại sau vòng sửa
(`too_small` chữ phải co nhiều, `overflow` tràn khung, `content_overlap` chồng chữ, `low_contrast` chìm nền) — poster vẫn trả.

## Lỗi

Mọi lỗi có **mã cố định**:

- **HTTP 422** (đầu vào sai, `detail.code`): `invalid_category`, `invalid_aspect_ratio`, `invalid_num_images`,
  `invalid_image` (ảnh không đọc được, hoặc gửi cả `image_base64` lẫn `image_url`), `invalid_image_url` (không phải https, host
  ngoài danh sách cho phép, trỏ vào địa chỉ nội bộ), `image_fetch_failed` (không tải được: lỗi mạng, HTTP lỗi / chuyển hướng,
  quá giờ, quá dung lượng), `missing_required_fields` (`detail.fields` = các trường thiếu).
- **HTTP 503**: `model_not_ready` (máy chủ đang nạp model / nạp lỗi — xem `/api/health`).
- **Job lỗi** (`status: "error"`, `error = {code, stage, message}`); poster lỗi riêng: `posters[i].error` cùng khuôn. Ảnh
  debug của các bước đã chạy vẫn có (xem `debug_zip_url`).

| `code` | Bước hỏng |
|---|---|
| `llm_failed` | LLM viết prompt / danh sách chữ (gọi LLM lỗi, trả sai khuôn sau khi thử lại) |
| `draw_failed` | FLUX vẽ nháp / Hi-SAM tìm nét (`stage` cho biết bước nào) |
| `draw_moderated` | máy FLUX lọc nội dung (bộ lọc của nhà cung cấp FLUX, TextFix không lọc) |
| `lines_failed` | đo dòng chữ |
| `erase_failed` | gỡ chữ (LaMa) |
| `vlm_failed` | gọi VLM ghép câu (lỗi gọi / sai khuôn) |
| `vlm_empty_plan` | VLM không ghép câu khách nào vào dòng nào — không trả nháp thay poster |
| `render_failed` | dựng chữ (Chromium) |
| `design_failed` | lỗi khác trong bước thiết kế |
| `internal_error` | lỗi không thuộc bước nào ở trên |

Job chỉ `error` khi **mọi** poster lỗi; còn ít nhất một poster thì job `done`, poster lỗi mang `error` riêng.
