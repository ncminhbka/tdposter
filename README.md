# TextFix

Poster tiếng Việt từ FLUX.2-klein. Model vẽ tốt bố cục, hình ảnh, vỏ và icon, nhưng chữ thành "giun". TextFix **chỉ sửa chữ**:
giữ nguyên mọi điểm ảnh khác của nháp, gỡ chữ giun và viết lại đúng câu của khách vào đúng chỗ model đã vẽ.

## Luồng

Mỗi bước là một gói trong `textfix/`; `textfix/engine.py` nối chúng lại.

```
form / prompt
  1.  LLM: prompt FLUX.2 (theo hướng dẫn nâng prompt của BFL) + câu của khách      brief/
  2.  FLUX.2-klein vẽ nháp; Hi-SAM tìm nét chữ (mặt nạ từng điểm ảnh)              draw/
  4a. DÒNG chữ: OCR chia vùng dòng; hướng, cao chữ, màu, nét đo trên nét Hi-SAM     lines/
  4b. gỡ chữ: mảnh nét của từng dòng + quầng, LaMa lấp một lần                      erase/
  4c. VLM đọc nháp: dòng nào là câu nào; số phận mỗi dòng viết / giữ / xoá          compose/ (vlm, fate, assign, rows)
  4d. bố cục: cỡ theo font thật, nới vào nền trống, tập cùng vai, màu               compose/ (layout, size, style)
  5.  dựng chữ trên nền (Chromium), kiểm lỗi, sửa bằng code                         render/
  -> poster
```

Lộ trình và tình trạng hiện tại: [docs/ROADMAP.md](docs/ROADMAP.md). Nhật ký tune, số đo, hướng đã thử và bỏ: [docs/TUNING.md](docs/TUNING.md). Thiết kế cũ (ô vỏ / ô chi tiết, bản xoá FLUX) nằm ở các nhánh `legacy-slots-si`, `legacy-flux-erase`.

## Chạy

Máy chủ GPU (2x A30, JupyterLab): [docs/SERVER.md](docs/SERVER.md).

```bash
pip install -r requirements.txt
export TEXTFIX_LLM_URL=http://<host>:<port>/v1 TEXTFIX_MODEL=<model nhìn được ảnh> TEXTFIX_LLM_API_KEY=<khoá>
ruff check . && python -m pytest            # kiểm mã + test (không cần GPU / model)
python scripts/check_server.py --fix        # kiểm môi trường, phải in "SẴN SÀNG."
bash scripts/start_server.sh                # API ở cổng 8088
python scripts/dev_batch.py bench/chat1.json --fresh   # chạy một tập qua máy chủ đang bật, báo cáo + một zip
```

Chạy trên máy chủ bằng ô notebook (`%run scripts/...`): kết quả tự gói thành zip < 24 MB và tự tải về ([docs/SERVER.md](docs/SERVER.md)).

API duy nhất, cho chatbot (một lời gọi đồng bộ: lời chat + ảnh sản phẩm -> một poster): [docs/API_V1.md](docs/API_V1.md); thử nhanh
`scripts/v1_smoke.py`.
Deploy khi FLUX / LLM / VLM ở máy khác: [docs/DEPLOY.md](docs/DEPLOY.md).
Docker demo build ngay trên notebook (FLUX 9B / LLM / VLM ở máy khác): [docs/DOCKER.md](docs/DOCKER.md).

## Cấu trúc

| Đường dẫn | Việc |
|---|---|
| `textfix/engine.py` | Nối toàn bộ luồng: `Engine.brief`, `make` (vẽ + sửa), `fix` (sửa nháp có sẵn); hai trạm `draw` / `design` cho máy chủ |
| `textfix/brief/` | 1. LLM đầu luồng: danh sách chữ + prompt FLUX; code kiểm theo bằng chứng (không số bịa, chữ khách luôn có, prompt vẽ đúng danh sách). Test: `tests/test_brief.py` |
| `textfix/draw/` | 2. `flux.py` FLUX tại chỗ, `remote.py` FLUX ở máy khác ([docs/DEPLOY.md](docs/DEPLOY.md)), `hisam.py` nét chữ của nháp. Test: `tests/test_flux*.py` |
| `textfix/lines/` | 4a. Dòng chữ: `page.build` (điều phối), `detect` (dòng OCR + số đo nét), `split` (tách mục), `curve` (chữ cong), `orphans` (nét OCR không thấy), `order` (đánh số), `fold` (so chữ bỏ dấu), `view` (ảnh / JSON); hình học trên nét Hi-SAM: `ink`, `group`, `angle`, `measure`, `lead`, `runs`; `ocr`, `glyph`. Test: `tests/test_lines.py` |
| `textfix/erase/` | 4b. Gỡ chữ giun: `mask` (nét, dự phòng, quầng, dấu câu), `lama` (một lượt LaMa), `view` (ảnh debug). Test: `tests/test_erase.py` |
| `textfix/compose/` | 4c–4d. `call` + `vlm` (MỘT lời gọi VLM, khuôn JSON mỗi dòng một mục), `parse`, `fate` (một bảng số phận dòng), `assign` (câu ↔ dòng, chữ lấp), `rows` (hàng của danh sách / bảng), `layout` (bố cục, tập cùng vai thẳng lề) với `geometry`, `size` (cỡ theo font, nới khung, bề rộng cột chung), `style` (màu), `text` (ngắt dòng). Test: `tests/test_fate.py`, `tests/test_rows.py`, `tests/test_layout.py` |
| `textfix/render/` | 5. `page` (dựng trong Chromium; mã trang ở `js`), `check` (bộ kiểm, tương phản), `fix` (vòng sửa), `geometry`, `style`, `view`; `fonts` (bộ font OFL đủ dấu tiếng Việt), `browser` |
| `textfix/tools/` | Không thuộc dây chuyền: `multigpu` (chạy hàng loạt trên mọi card), `ship` (zip < 24 MB tự tải về, JupyterLab), `audit` (cờ nghi hỏng) |
| `textfix/config.py`, `llm.py`, `color.py` | Biến môi trường, gọi LLM / VLM (OpenAI-compatible), màu CIELAB |
| `server/` | Máy chủ FastAPI ([docs/API_V1.md](docs/API_V1.md)): `app` (dây chuyền, HTTP), `contract` (khuôn yêu cầu / trả lời, mã lỗi, tải ảnh theo link). Test: `tests/test_api.py` |
| `scripts/dev_batch.py`, `batch_report.py` | Chạy một tập (lời chat) qua API của máy chủ tại chỗ, báo cáo + một zip |
| `scripts/replay.py`, `erase_bench.py`, `lines_bench.py` | Máy cá nhân: dựng lại poster / đo bước gỡ chữ / đo bước đo dòng (`bench/lines_gt`) trên nháp đã lưu |
| `docker/` | Image demo: `build.sh` (gom code + trọng số Hi-SAM / LaMa + Chromium, build), `Dockerfile`, `run.sh`, `textfix.env.example` ([docs/DOCKER.md](docs/DOCKER.md)) |
| `scripts/golden.py` | Dấu vân tay trạm thiết kế: tái cấu trúc mã thì chạy `record` trước, `check` sau, phải "khớp hết" |
| `assets/` | Font (OFL) |

Tên cũ (trước 15/10, còn gặp trong `docs/TUNING.md`, `docs/ROADMAP.md`): `slots.py` → `lines/` (+ `compose/call.py`),
`fill.py` → `compose/` (`assign`, `layout`, `geometry`, `size`, `style`, `text`, `parse`), `geo.py` → `lines/curve.py`,
`glyph.py` / `ocr.py` → `lines/`, `flux.py` / `flux_api.py` / `hisam.py` → `draw/` (`flux_api` thành `remote`),
`erase.py` → `erase/`, `render.py` / `fonts.py` / `browser.py` → `render/`, `ship.py` / `multigpu.py` / `audit.py` → `tools/`.

## Giấy phép tài nguyên

- FLUX.2-klein tại chỗ (`textfix/draw/flux.py`) gọi mã suy luận FLUX.2 của Black Forest Labs (gói `flux2`, Apache-2.0,
  [github.com/black-forest-labs/flux2](https://github.com/black-forest-labs/flux2)). Mã đó **không** nằm trong repo này: cài
  riêng ([docs/SERVER.md](docs/SERVER.md)); FLUX ở máy khác thì không cần.
- Font trong `assets/fonts/`: SIL Open Font License.
