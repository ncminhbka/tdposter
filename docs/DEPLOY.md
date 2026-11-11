# Deploy TextFix khi FLUX, LLM, VLM ở máy khác

Dành cho đội dựng API. Máy deploy chạy `server/app.py`. Đây đã là API hoàn chỉnh (một endpoint, mã lỗi, hàng đợi: [API_V1.md](API_V1.md)),
nên chỉ cần đặt nó sau cổng của bạn (domain, HTTPS, xác thực, giới hạn lượt gọi), không phải bọc lại logic.
(`docs/SERVER.md` là hướng dẫn cho máy chủ thử nghiệm JupyterLab, FLUX chạy tại chỗ, không ra internet.)

## Cái gì chạy ở đâu

| Thành phần | Ở đâu | Cấu hình |
|---|---|---|
| FLUX.2 [klein] (vẽ nháp) | **máy khác**, API khuôn `api.bfl.ai` | `TEXTFIX_FLUX_URL`, `TEXTFIX_FLUX_API_KEY` |
| LLM (bước 1: prompt FLUX + danh sách chữ) | **máy khác**, OpenAI-compatible | `TEXTFIX_LLM_URL`, `TEXTFIX_MODEL`, `TEXTFIX_LLM_API_KEY` |
| VLM (bước 4: ghép câu vào dòng) | **máy khác**, OpenAI-compatible, nhìn được ảnh | `TEXTFIX_VLM_URL`, `TEXTFIX_VLM_MODEL`, `TEXTFIX_VLM_API_KEY` (bỏ trống = dùng LLM) |
| Hi-SAM (tìm nét chữ) | máy deploy, GPU (CPU được, chậm) | `TEXTFIX_HISAM` (thư mục) |
| LaMa (vẽ lại nền dưới chữ) | máy deploy, GPU (CPU được, chậm) | `TEXTFIX_LAMA` (đường dẫn tệp) |
| OCR (RapidOCR) | máy deploy, CPU | mô hình ONNX nằm trong gói pip, không cần tải |
| Chromium (dựng chữ) | máy deploy, CPU | `playwright install chromium`, hoặc `PLAYWRIGHT_CHROME_PATH` |

Máy deploy cần: Python 3.10+, một GPU khoảng **8–12 GB** (Hi-SAM đỉnh 4,4 GB, LaMa nhỏ; không có GPU thì mỗi poster chậm
thêm vài giây), vài GB RAM cho mỗi luồng Chromium.

## Các bước

```bash
# 1. mã: theo TAG (không lấy main), từ zip mã nguồn hoặc git
unzip textfix-<tag>-src.zip && cd textfix-<tag>
# 2. thư viện (torch bản CUDA nếu có GPU: cài torch trước theo hướng dẫn của PyTorch)
pip install -r requirements.txt
# 3. Chromium cho Playwright
playwright install --with-deps chromium
# 4. Hi-SAM + LaMa (mục bên dưới), rồi đặt biến môi trường (mục bên dưới)
# 5. tự kiểm: phải in "SẴN SÀNG." (gọi thử LLM, VLM và vẽ thử một ảnh FLUX 512x512; bỏ gọi thử: --no-call)
python scripts/check_server.py
# 6. chạy
python server/app.py --port 8088
curl -s localhost:8088/api/health        # {"ready": true, "failed": false, "queue": 0}; chi tiết model ở log
```

Gói `flux2` (mã BFL, không đi kèm repo) và `requirements` phần FLUX (`transformers`, `accelerate`, `safetensors`) không dùng
khi FLUX ở máy khác; cài `requirements` nguyên vẹn cũng không sao. `bench/`, `scripts/` (trừ `check_server.py`), `tests/` không cần để chạy.

### Hi-SAM và LaMa

Đặt dưới một thư mục, ví dụ `/models`:
- Hi-SAM, thư mục `/models/hisam` (`TEXTFIX_HISAM=/models/hisam`):
  - mã nguồn https://github.com/ymy-k/Hi-SAM/archive/refs/heads/main.zip giải nén thành `/models/hisam/Hi-SAM/`
    (KHÔNG cài `requirements.txt` của Hi-SAM: nó ghim numpy / opencv bản cũ);
  - https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -> `/models/hisam/Hi-SAM/pretrained_checkpoint/`;
  - `sam_tss_l_textseg.pth` (README Hi-SAM, Model Zoo, dòng SAM-TS-L / TextSeg; bản sao
    https://huggingface.co/creative-graphic-design/hi-sam-checkpoints) -> `/models/hisam/`.
- LaMa: https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt -> `/models/lama/big-lama.pt`
  (`TEXTFIX_LAMA=/models/lama/big-lama.pt`).

Thiếu tệp nào thì máy chủ báo lỗi ngay lúc khởi động (`/api/health` có `"failed": true`, chi tiết ở log), không chạy nửa vời.

## Biến môi trường

```bash
# FLUX ở máy khác (có biến này thì máy chủ KHÔNG nạp FLUX lên GPU)
export TEXTFIX_FLUX_URL=https://api.bfl.ai/v1/flux-2-klein-9b   # bản cố định; tránh "-preview" (đổi model theo thời gian)
export TEXTFIX_FLUX_API_KEY=<khoá>
export TEXTFIX_FLUX_CONCURRENCY=4     # số nháp gửi sang FLUX cùng lúc (mặc định 2; BFL cho tối đa 24 việc / tài khoản)
# export TEXTFIX_FLUX_TIMEOUT=120     # giây chờ một nháp
# export TEXTFIX_FLUX_SAFETY=2        # safety_tolerance của BFL, 0 (chặt nhất) .. mức tài khoản được duyệt
# LLM / VLM
export TEXTFIX_LLM_URL=https://<host>/v1
export TEXTFIX_MODEL=<model>
export TEXTFIX_LLM_API_KEY=<khoá>
# export TEXTFIX_VLM_URL=... TEXTFIX_VLM_MODEL=... TEXTFIX_VLM_API_KEY=...   # VLM riêng
# export TEXTFIX_BRIEF_TEMPERATURE=0.3   # nhiệt độ LLM đầu luồng + sửa poster (chép chữ khách: thấp); VLM giữ mức chung
# model gỡ chữ, Chromium
export TEXTFIX_HISAM=/models/hisam
export TEXTFIX_LAMA=/models/lama/big-lama.pt
# export PLAYWRIGHT_CHROME_PATH=...   # chỉ khi không dùng bản Chromium của playwright install
# máy chủ
# export TEXTFIX_GPUS=0               # card cho Hi-SAM / LaMa (mặc định mọi card máy thấy; không card nào: CPU)
# export TEXTFIX_DESIGN_WORKERS=8     # luồng thiết kế song song (mỗi luồng một Chromium, gọi VLM)
# export TEXTFIX_PORT=8088
```

Có thể chép `.env.example` thành `.env` ở gốc repo thay cho `export` (biến đã export thắng `.env`). Không commit `.env`.

**VLM** phải nhận **ít nhất 12 ảnh / lời gọi** và ngữ cảnh **≥ 32k token** (vLLM:
`--limit-mm-per-prompt '{"image": 16}' --max-model-len 32768`). `check_server.py` gọi thử đúng 12 ảnh.

## Tốc độ và số luồng

Mỗi phiên bản poster đi qua hai trạm:
- **VẼ**: gọi FLUX (máy khác), rồi Hi-SAM trên card của máy deploy. Số luồng = `TEXTFIX_FLUX_CONCURRENCY`, chia vòng lên các card.
- **THIẾT KẾ**: OCR, LaMa, gọi VLM, dựng Chromium, kiểm. Số luồng = `TEXTFIX_DESIGN_WORKERS`.

**CPU** (OCR, đo dòng, Chromium chạy CPU): container tự đọc hạn mức CPU (Kubernetes `limits.cpu`, `docker --cpus`) và đặt số
luồng thư viện bằng hạn mức (`docker/run.sh`, in ra dòng `[textfix] CPU: ...`). Hạn mức nên >= 8 lõi; quá nhỏ thì bước đo dòng
chậm. Đặt tay: `TEXTFIX_THREADS`.

Đặt `TEXTFIX_FLUX_CONCURRENCY` bằng số yêu cầu máy FLUX chịu được cùng lúc, `TEXTFIX_DESIGN_WORKERS` theo số lời gọi VLM
song song máy VLM chịu được và RAM (mỗi luồng một Chromium).

## Máy FLUX tự host

Lớp nối (`textfix/draw/remote.py`) nói hai khuôn, tự chọn theo URL (hoặc đặt `TEXTFIX_FLUX_API=chat|bfl`).

**Khuôn chat** (máy tự host kiểu vLLM-Omni, OpenAI-compatible) khi `TEXTFIX_FLUX_URL` tận cùng `/v1/chat/completions`:
- `POST`, header `Authorization: Bearer <TEXTFIX_FLUX_API_KEY>`, thân JSON `{model, messages: [{role: "user", content: [{type:
  "text", text: <prompt>}, {type: "image_url", image_url: {url: "data:image/png;base64,..."}} ...]}], extra_body: {width, height,
  seed, num_inference_steps, guidance_scale}}` (ảnh tham chiếu là các phần `image_url`, tối đa 4);
- trả `choices[0].message.content` chứa phần `image_url` (data URL PNG, hoặc URL ảnh). Đồng bộ: đợi vẽ xong mới trả;
- `TEXTFIX_FLUX_MODEL` (mặc định `black-forest-labs/FLUX.2-klein-9B`), `TEXTFIX_FLUX_STEPS` (4), `TEXTFIX_FLUX_GUIDANCE` (1).

**Khuôn BFL** (còn lại):
- `POST <TEXTFIX_FLUX_URL>`, header `x-key`, thân JSON `{prompt, width, height, seed, output_format: "png", safety_tolerance,
  input_image, input_image_2..4}` (ảnh tham chiếu là base64 PNG) -> `{id, polling_url}`;
- `GET polling_url` (không có thì `<gốc>/v1/get_result?id=`) -> `{status, result: {sample: <URL ảnh>}}`; `status` là `Pending` /
  `Generating` / `Ready` / `Error` / `Request Moderated` / `Content Moderated`.

Yêu cầu để nháp đúng như TextFix cần:
- **dùng nguyên prompt** (không tự nâng prompt: LLM bước 1 đã viết prompt theo hướng dẫn của BFL);
- **đúng cỡ** `width x height` gửi sang (các cỡ trong `textfix/draw/flux.py` `SIZES`, bội số 16, ~1 MP); lệch cỡ thì TextFix co về, mất nét;
- **nhận `seed`**: cùng prompt, cỡ, seed -> cùng ảnh (chạy lại được);
- **nhận ảnh tham chiếu** (ảnh sản phẩm khách tải: FLUX giữ sản phẩm, TextFix giữ chữ in trên sản phẩm);
- **trả PNG** (JPEG nén mạnh làm Hi-SAM thấy nét chữ kém).

## Lỗi riêng của chế độ này

- `draw_failed`: gọi FLUX lỗi, hết giờ, máy FLUX trả lỗi; hoặc Hi-SAM lỗi (`error.stage` cho biết bước nào).
- `draw_moderated`: máy FLUX lọc nội dung (`Request Moderated` / `Content Moderated`). TextFix không lọc nội dung; lọc là của
  nhà cung cấp FLUX, chỉnh bằng `TEXTFIX_FLUX_SAFETY` trong mức tài khoản được duyệt.

## Lưu ý: FLUX.2 [klein] 9B

TextFix được tinh chỉnh và nghiệm thu (bộ `final2`) trên **FLUX.2 [klein] 4B distill**. Mọi bước sau FLUX đo trên chính nháp
(nét chữ, dòng, cỡ, màu), không giả định model nào, nên bản 9B chạy được ngay. Nhưng nháp 9B khác nháp 4B (thường vẽ chữ giun
rõ hơn, bố cục khác), nên các con số nghiệm thu chưa được đo lại trên 9B. Nên chạy lại một tập thử (ví dụ các form của tester)
trước khi mở cho đông người.
