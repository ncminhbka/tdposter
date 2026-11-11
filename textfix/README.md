# TextFix

Gói Python tạo **poster tiếng Việt** từ yêu cầu của khách (form và/hoặc prompt tự do). Nguyên tắc cốt lõi: **chỉ sửa chữ**.

- Nháp poster do FLUX.2-klein (hoặc Qwen-Image) vẽ được **giữ nguyên hình ảnh**.
- Chữ "giun" (chữ do model vẽ sai, méo, không đọc được) được **tìm, gỡ và thay** bằng đúng câu chữ của khách.
- Mọi chữ trên poster là **nguyên văn câu khách** (hoặc chữ phụ trợ do LLM viết, được kiểm soát chặt). Code không để model tự bịa số, liên kết, lời hứa.

Kết quả là một poster PNG cùng bộ ảnh trung gian và `plan.json` để debug.

> Repo này là **gói Python** `textfix`. Thư mục chứa gói (repo) cũng là thư mục gốc mà `config.ROOT` trỏ tới: `.env` và `output/kho/` đặt ở thư mục **cha** của gói.

---

## Mục lục

1. [Luồng xử lý](#luồng-xử-lý)
2. [Cấu trúc mã nguồn](#cấu-trúc-mã-nguồn)
3. [Cách dùng](#cách-dùng)
4. [Cấu hình (biến môi trường)](#cấu-hình-biến-môi-trường)
5. [Phụ thuộc và tệp ngoài repo](#phụ-thuộc-và-tệp-ngoài-repo)
6. [Kho ảnh tham chiếu](#kho-ảnh-tham-chiếu)
7. [Công cụ vận hành](#công-cụ-vận-hành)
8. [Tình trạng repo](#tình-trạng-repo)

---

## Luồng xử lý

```
form / prompt ──LLM──► brief (prompt FLUX + danh sách chữ khách)        brief/
        │
        ▼
FLUX.2-klein / Qwen-Image vẽ nháp   ──►  Hi-SAM tìm nét chữ (mặt nạ)     draw/
        │
        ▼
đo dòng chữ: OCR + chữ cong + cụm nét OCR không đọc ra                  lines/
        │
        ▼
gỡ chữ: mặt nạ nét + quầng, LaMa vẽ nền một lần cho cả poster          erase/
        │
        ▼
VLM đọc nháp, ghép câu khách <-> dòng, chọn kiểu theo vai              compose/
code chép nguyên văn, chọn cỡ / lề / tập theo nháp
        │
        ▼
dựng HTML/CSS trong Chromium, kiểm lỗi, sửa bằng code (tối đa 2 lượt)  render/
        │
        ▼
poster + nhật ký (plan.json)
```

Hai nguyên tắc chi phối thiết kế:

- **Hình học do Hi-SAM và code quyết, không do model ngôn ngữ.** VLM chỉ hiểu *nghĩa* (dòng nào là câu nào, vai nào trông ra sao), và chỉ trỏ tới id dòng `L1`, `L2`… Toạ độ, cỡ, màu, khung đều do code đo trên nét chữ thật.
- **Không hỏi lại model.** Lời gọi VLM có một khuôn JSON bắt buộc (mỗi dòng một mục). Lỗi còn lại được sửa bằng code, không gọi lại VLM.

### Các bước trong `engine.py`

| Bước | Trạm | Việc làm | Module |
|---|---|---|---|
| 1 | Brief (LLM) | Chuyển form/prompt thành `B`: `prompt_en` cho FLUX, danh sách `texts` (chữ khách, đã phân loại user / ai / bỏ), `kho` (ảnh mẫu dùng thật) | `brief/` |
| 2 | Vẽ (GPU) | FLUX/Qwen vẽ nháp; Hi-SAM trả `ink` (mặt nạ điểm ảnh của nét chữ) | `draw/` |
| 3 | Đo dòng | OCR (rapidocr, CPU) + nét Hi-SAM → dòng `L#` có khung, hướng, cỡ, màu, đường chân; xử lý chữ cong và cụm nét không có OCR | `lines/` |
| 4 | Gỡ chữ | Mặt nạ từ nét + quầng; LaMa vẽ lại nền một lần cho cả poster | `erase/` |
| 5 | Ghép | VLM ghép câu khách vào dòng; `fate` lập bảng số phận từng dòng (`text` / `keep` / `skip`) | `compose/` |
| 6 | Bố cục | Cỡ, co, lề, căn hàng theo nháp; tập cùng vai dùng chung cỡ | `compose/layout.py` |
| 7 | Dựng & kiểm | Dựng trong Chromium; kiểm chồng nội dung, tràn, co quá nhỏ, chìm nền (WCAG); sửa bằng code, giữ bản nào bớt lỗi | `render/` |

Trường hợp đặc biệt: nếu `B["mode"] == "image"` (ảnh thường, không có chữ khách) thì chỉ có bước 2, poster là nháp.

---

## Cấu trúc mã nguồn

```
__init__.py        mô tả dây chuyền, không chứa logic
config.py          đọc .env, chọn model vẽ (TEXTFIX_DRAW), địa chỉ LLM/VLM
engine.py          Engine (dùng một lần end-to-end) + hai trạm draw() / design() cho máy chủ
llm.py             json_call(): gọi LLM/VLM qua API OpenAI-compatible, có cache theo nội dung
kho.py             kho ảnh tham chiếu: khớp từ khoá, ứng viên và độ chắc
kho_sync.py        đồng bộ kho từ bucket S3/MinIO về thư mục đệm

brief/             LLM đầu luồng
  __init__.py        make_brief(): form/prompt -> B
  instructions.py    lời dặn LLM, khuôn JSON trả lời
  prompt.py          chuẩn hoá prompt FLUX: chữ trong ngoặc kép khớp đúng danh sách đã chốt
  texts.py           chốt danh sách chữ: user / ai / bỏ, không cho LLM thêm sự thật
  evidence.py        đối chiếu chữ LLM viết với lời khách (số, liên kết, email)

draw/              vẽ nháp và tìm nét chữ
  flux.py            FLUX.2-klein tại chỗ (distill 4 bước hoặc base 50 bước)
  remote.py          FLUX ở máy khác (khuôn BFL hoặc chat/completions)
  qwen.py            Qwen-Image (diffusers) tại chỗ, thay FLUX khi TEXTFIX_DRAW=qwen
  hisam.py           Hi-SAM: strokes(draft) -> mặt nạ nét chữ

lines/             đo dòng chữ từ nét Hi-SAM + OCR
  page.py            build(): M = {"L": [dòng], "size", "by", "geo_log"}
  detect.py          dò dòng từ OCR, đo số liệu trên nét
  ocr.py             rapidocr, mỗi luồng một bản
  curve.py           chữ cong (vòm, vòng con dấu)
  orphans.py         cụm nét Hi-SAM mà OCR không đọc ra
  split.py / order.py / lead.py / angle.py / glyph.py / measure.py / ink.py / group.py
                     tách mục trong dòng, thứ tự đọc, đầu dòng, hướng, đo nét và màu
  view.py            ảnh đánh dấu L#, JSON cho plan.json, ghi chú chữ khó

erase/             gỡ chữ
  __init__.py        erase() (LaMa một lượt), settle() (giữ điểm ảnh nháp cho dòng "keep")
  mask.py            mặt nạ theo từng dòng: mảnh nét, dự phòng màu mực, quầng, dấu câu
  lama.py            LaMa (TorchScript big-lama.pt), mỗi thiết bị một bản
  view.py            ảnh debug của bước gỡ chữ

compose/           ghép câu khách vào dòng
  vlm.py             ask(): một lời gọi VLM, khuôn JSON theo từng poster
  call.py            lời dặn + ảnh (nháp, nháp đánh dấu, nền đã gỡ) gửi cho VLM
  parse.py           trả lời VLM -> đơn vị thô, bỏ mục sai khuôn
  assign.py          resolve(): ghép câu <-> dòng thành đơn vị (một câu nhiều dòng, câu ghép hai lần...)
  fate.py            bảng số phận từng dòng, mỗi dòng một lý do
  layout.py          layout(): khung, cỡ, màu, font, tập cùng vai
  size.py / geometry.py / rows.py / style.py / text.py
                     cỡ theo font thật, hình học, hàng của danh sách / bảng, màu và hiệu ứng, ngắt dòng

render/            dựng và kiểm
  browser.py         Chromium (Playwright, bản đồng bộ)
  page.py            render_units(): mỗi phần tử một div (hoặc SVG textPath)
  js.py              mã JavaScript chạy trong trang: đặt, đo, nén, dựng theo đường cong
  fonts.py           bộ font OFL có dấu tiếng Việt, nhúng @font-face base64
  style.py           khoá font của VLM -> khoá trong danh mục
  geometry.py        khung, góc, đường cong, chồng lấn
  check.py           score() và check_units(): chồng, tràn, co quá nhỏ, chìm nền
  fix.py             autofix_units(): sửa bằng code, tối đa FIX_ROUNDS = 2 lượt
  view.py            ảnh debug của đơn vị chữ

color.py           đổi màu sang CIELAB, đo ΔE

tools/             công cụ vận hành, KHÔNG thuộc dây chuyền poster
  multigpu.py        chạy song song, mỗi GPU một tiến trình con giữ một bản FLUX
  audit.py           cờ nghi hỏng của một poster từ plan.json (không phải điểm chất lượng)
  ship.py            gói kết quả thành nhiều zip độc lập, mỗi zip < 24 MB
```

### Hai trạm cho máy chủ nhiều phiên bản

`engine.py` có thêm hai hàm để chạy theo dây chuyền song song:

- `draw(flux, B, W, H, seed, product=None)` là **trạm vẽ**, chạy trên luồng của từng GPU: FLUX/Qwen vẽ nháp rồi Hi-SAM tìm nét. Trả về `{draft, ink, timing, device}`.
- `design(d, B, vlm, page, product=...)` là **trạm thiết kế**, chạy trên nhóm luồng riêng: OCR, LaMa, VLM, dựng và kiểm. Mỗi luồng có một trình duyệt Chromium riêng.

Hàm `Engine.make` và `Engine.fix` gọi lần lượt hai trạm này trong cùng một tiến trình.

---

## Cách dùng

Repo không có CLI. Dùng bằng Python. Thư mục cha của repo phải nằm trên `sys.path` và repo được import dưới tên `textfix`.

```python
from textfix.draw import flux
from textfix.engine import Engine

F = flux.Flux("distill").load()           # FLUX.2-klein tại chỗ (cần GPU và trọng số)
with Engine(F, progress=print) as E:
    B = E.brief(prompt="Khuyến mãi trà sữa, giảm 30%, hotline 0901 234 567", aspect="4:5")
    r = E.make(B, W=1080, H=1350, seed=42)
    r["poster"]        # ảnh RGB (numpy uint8): poster cuối
    r["plan"]          # JSON: dòng, số phận, lệnh, lỗi, nhật ký, câu khách thiếu
    r["steps"]         # ảnh 5 cột: nháp | dòng | nét + mặt nạ | nền đã gỡ | poster
    r["timing"]        # số giây từng bước
```

Ghi chú:

- `B = E.brief(...)` gọi LLM. Với `product_image=True`, người dùng đã tải ảnh sản phẩm: ảnh đó là `image 1` cho FLUX, và chữ in trên sản phẩm được giữ nguyên (`keep`).
- Dùng lại nháp có sẵn: `E.fix(draft, B, seed, ink=None, product=False)` bỏ qua bước vẽ.
- Chạy `Engine` không cần FLUX: `Engine(None)` chỉ dùng được cho `fix()` với nháp có sẵn.
- `LLM` và `VLM` có thể truyền vào `Engine(llm=..., vlm=...)` (hàm nhận `(system, parts)`), tiện cho thử nghiệm.
- Kết quả `make()` ở chế độ `image` chỉ có `draft`, `poster`, `plan = {"mode": "image"}`, `timing`.

Lời gọi LLM/VLM được cache theo nội dung trong `~/.cache/textfix/<tag>/` (tag: `brief`, `designer`). Lời gọi có cùng đầu vào sẽ không tốn thêm tiền hay thời gian; dùng `call(..., fresh=True)` để lấy câu trả lời khác.

---

## Cấu hình (biến môi trường)

Đặt bằng `export` trong terminal, hoặc trong tệp `.env` ở thư mục cha của gói. Biến đã `export` thắng `.env`. Không có giá trị nào được in ra log.

### LLM và VLM (API OpenAI-compatible, ví dụ vLLM)

| Biến | Ý nghĩa |
|---|---|
| `TEXTFIX_LLM_URL`, `TEXTFIX_MODEL` | địa chỉ và tên model LLM (bắt buộc) |
| `TEXTFIX_LLM_API_KEY` | khoá API (không có thì gửi `none`) |
| `TEXTFIX_VLM_URL`, `TEXTFIX_VLM_MODEL`, `TEXTFIX_VLM_API_KEY` | VLM riêng; không đặt thì dùng LLM. VLM phải nhìn được ảnh |
| `TEXTFIX_LLM_TIMEOUT` | giây chờ mỗi lời gọi (mặc định 300). Lỗi mạng hoặc JSON hỏng được gọi lại tối đa 2 lần |

### Vẽ nháp

| Biến | Ý nghĩa |
|---|---|
| `TEXTFIX_DRAW` | `flux` (mặc định) hoặc `qwen`. `qwen` bỏ `TEXTFIX_FLUX_URL` của tiến trình đó |
| `TEXTFIX_WEIGHTS` | thư mục trọng số FLUX.2-klein (xem `draw/flux.py`, `find_weights`) |
| `TEXTFIX_DEVICE_DIT`, `TEXTFIX_DEVICE_TE`, `TEXTFIX_DEVICE_AE` | thiết bị cho DiT, bộ mã hoá chữ, VAE (mặc định `cuda:0`) |
| `TEXTFIX_FLUX_URL`, `TEXTFIX_FLUX_API_KEY` | FLUX ở máy khác (`draw/remote.py`) |
| `TEXTFIX_FLUX_API` | `bfl` hoặc `chat`; mặc định suy từ URL |
| `TEXTFIX_FLUX_MODEL`, `TEXTFIX_FLUX_STEPS`, `TEXTFIX_FLUX_CONCURRENCY`, `TEXTFIX_FLUX_TIMEOUT`, `TEXTFIX_FLUX_GUIDANCE`, `TEXTFIX_FLUX_SAFETY` | tham số của FLUX từ xa |
| `TEXTFIX_QWEN_DIR`, `TEXTFIX_QWEN_MEM`, `TEXTFIX_QWEN_STEPS`, `TEXTFIX_QWEN_CFG`, `TEXTFIX_QWEN_SCALE` | Qwen-Image tại chỗ (`draw/qwen.py`) |
| `TEXTFIX_HISAM`, `TEXTFIX_HISAM_HALF` | thư mục Hi-SAM; `off` để tắt fp16 |

### Gỡ chữ, OCR, kho, font

| Biến | Ý nghĩa |
|---|---|
| `TEXTFIX_LAMA` | đường dẫn `big-lama.pt` (mặc định `~/persistent-data/lama/big-lama.pt`) |
| `TEXTFIX_MASK_GROW` | độ nới mặt nạ quanh nét (mặc định 0,12) |
| `TEXTFIX_THREADS` | số luồng cho OCR |
| `TEXTFIX_FONTS` | thư mục font (mặc định `assets/fonts/`) |
| `PLAYWRIGHT_CHROME_PATH` | Chromium có sẵn trên máy (khi không tải được bản của Playwright) |
| `TEXTFIX_KHO_DIR` | thư mục đệm kho ảnh (mặc định `output/kho/`) |
| `TEXTFIX_KHO_ENDPOINT`, `TEXTFIX_KHO_BUCKET`, `TEXTFIX_KHO_PREFIX`, `TEXTFIX_KHO_ACCESS_KEY`, `TEXTFIX_KHO_SECRET_KEY` | đồng bộ kho từ S3/MinIO. Không đặt endpoint thì tắt đồng bộ |
| `TEXTFIX_ZIP_MB` | giới hạn mỗi zip khi đóng gói kết quả (mặc định 22) |

---

## Phụ thuộc và tệp ngoài repo

Repo chỉ chứa mã nguồn. Các thứ sau phải có trên máy chạy:

**Thư viện Python:** `numpy`, `opencv-python` (`cv2`), `Pillow`, `openai`, `rapidocr_onnxruntime`, `playwright` (kèm Chromium), `torch`, `transformers` (tokenizer cho brief), `diffusers` (chỉ khi dùng Qwen), `minio` và `urllib3` (chỉ khi đồng bộ kho).

**Gói `flux2`** (mã suy luận FLUX.2 của Black Forest Labs, Apache-2.0) không đi kèm repo. Chỉ cần khi vẽ FLUX tại chỗ; FLUX từ xa không cần.

**Trọng số và tệp mô hình:**

| Thứ | Vị trí mặc định | Ghi chú |
|---|---|---|
| FLUX.2-klein 4B (distill, base) | `~/persistent-data/FLUX.2-klein-4B/` (hoặc đường dẫn trong `draw/flux.py`) | DiT, VAE, bộ mã hoá chữ Qwen3-4B |
| Hi-SAM | `~/persistent-data/hisam/` | cần `Hi-SAM/` (mã nguồn có `hi_sam/modeling/build.py`), `Hi-SAM/pretrained_checkpoint/sam_vit_l_0b3195.pth` và `sam_tss_l_textseg.pth`. Thiếu tệp thì báo lỗi và dừng |
| LaMa | `~/persistent-data/lama/big-lama.pt` | TorchScript |
| Qwen-Image | `~/persistent-data/Qwen-Image-2.1-Turbo/` | chỉ khi `TEXTFIX_DRAW=qwen` |
| Font | `assets/fonts/` | chỉ giấy phép OFL, đủ dấu tiếng Việt |
| RapidOCR | tự tải theo gói `rapidocr_onnxruntime` | chạy trên CPU |

**Chromium:** dùng cho bước dựng. Trên máy không ra internet, đặt `PLAYWRIGHT_CHROME_PATH` hoặc để `chromium` / `google-chrome` trong `PATH`.

**Model ngôn ngữ:** LLM và VLM được truy cập qua API OpenAI-compatible (vLLM). Bước gọi VLM yêu cầu model nhìn được ảnh và hỗ trợ `response_format` JSON schema.

---

## Kho ảnh tham chiếu

Kho chứa ảnh mẫu cho món, trang phục, vật Tết đặc trưng Việt Nam (bún thang, áo dài, mâm ngũ quả…) để FLUX vẽ đúng món. Không có ảnh thì "bún thang" dễ thành phở bò.

```
<kho>/index.json     {"entries": [{id, file, tags, aliases, source, sha1?}, ...]}
<kho>/<file>         ảnh JPG hoặc PNG, một ảnh cho mỗi khái niệm
```

- `tags`: tên tiếng Anh của món. `aliases`: cụm tiếng Việt khách có thể gõ, có dấu hoặc không dấu.
- `source` bắt buộc: chỉ đưa ảnh có giấy phép rõ ràng vào kho.
- Khớp **đúng từ khoá, nguyên từ**, không dùng embedding. Khớp **chắc** khi cụm từ có ít nhất hai từ và khách gõ có dấu. Các trường hợp còn lại (từ đơn, gõ không dấu) được LLM quyết ở đầu luồng.
- Tối đa 4 ảnh tham chiếu cho một lượt vẽ (FLUX.2 klein nhận tối đa 4), kể cả ảnh sản phẩm khách tải lên (luôn là `image 1`).
- `kho.problems(entries)` kiểm mục lục trước khi đóng gói: thiếu trường, file không có, trùng id, một cụm trỏ vào hai mục.
- Đồng bộ từ bucket S3: `kho_sync.sync_from_env()` tải mục lục và ảnh còn thiếu hoặc đã đổi (kiểm bằng `sha1`) vào thư mục đệm. Lỗi ở đâu thì giữ bản đệm cũ, không bao giờ làm hỏng luồng chính.

---

## Công cụ vận hành

Thư mục `tools/` không thuộc dây chuyền poster:

- `tools/multigpu.py`: chạy hàng loạt trên mọi GPU. Mỗi card một tiến trình con nạp một bản FLUX riêng (khoảng 17 GB, vừa một A30 24 GB). Tiến trình cha gom kết quả.
- `tools/audit.py`: `flags(plan)` tính cờ nghi hỏng của một poster từ `plan.json`, không cần ảnh và không cần AI. Đây là công cụ chọn poster để xem, **không phải điểm chất lượng**.
- `tools/ship.py`: đóng gói kết quả thành nhiều zip **độc lập**, mỗi zip dưới 24 MB, và tự tải về trình duyệt khi chạy trong notebook.

Các script gọi công cụ này (thư mục `scripts/`) không có trong repo.

---

## Tình trạng repo

- **Không có README, `docs/`, `scripts/`, `server/`, `assets/`, `hi_sam/`, `requirements.txt` hay `.env` mẫu** trong repo này, dù mã còn tham chiếu tới chúng (ví dụ `docs/ROADMAP.md`, `docs/SERVER.md`, `docs/DEPLOY.md`, `docs/KHO.md`, `flux_api.py`, `start_server.sh`). Muốn chạy trên máy mới cần tự cài các phụ thuộc và tệp nêu ở trên.
- **Không có bộ kiểm tự động (test)** trong repo.
- **Không có CLI.** Mọi thứ là API Python: `Engine`, hoặc các hàm trong `draw/`, `lines/`, `erase/`, `compose/`, `render/`.
- Một số mốc thời gian và số đo trong docstring (ví dụ "14/10", "15/10") là ghi chú quyết định của dự án, không phải trạng thái hiện tại. Đọc docstring của từng module để biết lý do.
- Nhánh git `legacy-flux-erase` giữ đường xoá chữ bằng FLUX cũ (đã bỏ khỏi `main`).
