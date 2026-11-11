# Chạy TextFix trên máy chủ GPU (2× A30, JupyterLab)

Deploy khi FLUX / LLM / VLM ở máy khác (máy ra được internet): [DEPLOY.md](DEPLOY.md).

Máy chủ **chỉ cài được bằng pip**: không ra HuggingFace, không apt, không tải được Chromium của Playwright. Mọi thứ khác phải có
sẵn trong repo hoặc `~/persistent-data`. Mọi lệnh gõ trong Terminal của JupyterLab.

## Tóm tắt

```bash
# 1. mã: tải zip repo, giải nén vào ~/work
mkdir -p ~/work && cd ~/work && unzip -q <repo>.zip && cd <thư mục repo>
# 2. thư viện
pip install -r requirements.txt
# 3. biến môi trường (mục 3), lưu vào ~/textfix_env.sh
source ~/textfix_env.sh
# 4. kiểm: phải in "SẴN SÀNG."
python scripts/check_server.py --fix
# 5. chạy (cổng 8088), log: tail -f server.log
bash scripts/start_server.sh
# 6. mở https://<địa-chỉ-jupyter>/user/<tên>/proxy/8088/   (nhớ dấu / cuối)
```

## Cập nhật mã (không git pull được)

Tải zip mới của repo (GitHub: Code -> Download ZIP), tải lên `~/work`, rồi **chép đè** lên thư mục repo đang dùng. Zip không có
`output/` (đã gitignore) nên kết quả cũ, `.env` giữ nguyên:

```bash
cd ~/work && rm -rf _new && mkdir _new && unzip -q <tên zip>.zip -d _new
cp -a _new/*/. ~/work/<thư mục repo>/     # đè mã, giữ output/ và .env
rm -rf _new <tên zip>.zip
```

Không xoá thư mục repo cũ rồi giải nén lại: mất `output/`.

## 1. Cần có trong `~/persistent-data`

| Thứ | Đường dẫn | Ghi chú |
|---|---|---|
| FLUX.2-klein 4B distill: DiT, `vae/`, `text_encoder/` (Qwen3-4B), `tokenizer/` | `~/persistent-data/FLUX.2-klein-4B/` | Tự dò. Chỗ khác: `TEXTFIX_WEIGHTS` |
| FLUX.2-klein 4B base (tuỳ chọn, `--model base`) | `~/persistent-data/FLUX.2-klein-base-4B/` | |
| Chromium (Chrome for Testing, linux64) | `~/persistent-data/chrome-linux64.zip` | `check_server.py --fix` giải nén + ghi đường dẫn |

OCR (RapidOCR) có mô hình ONNX trong gói pip, không cần tải thêm.

Vẽ FLUX tại chỗ cần **mã suy luận FLUX.2 của Black Forest Labs** (gói `flux2`, Apache-2.0, giữ nguyên `flux2/LICENSE.md`):
từ 12/11 đi kèm repo ở thư mục `flux2/` (chép từ `src/flux2` của
[github.com/black-forest-labs/flux2](https://github.com/black-forest-labs/flux2)), vì máy chủ không ra được GitHub.
`check_server.py` báo nếu thiếu. FLUX ở máy khác (`TEXTFIX_FLUX_URL`) thì bỏ qua bước này.

## 2. Thư viện

`pip install -r requirements.txt`. Cảnh báo kiểu `notebook ... requires jupyterlab ...` là xung đột có sẵn của image notebook,
không liên quan, **đừng nâng jupyterlab** (đang chạy chính giao diện notebook).

## 3. Biến môi trường

Lưu vào `~/textfix_env.sh` (ngoài repo, không bao giờ commit khoá), rồi `echo 'source ~/textfix_env.sh' >> ~/.bashrc`:

```bash
export TEXTFIX_LLM_URL=http://<host>:<port>/v1     # LLM / VLM nội bộ (OpenAI-compatible)
export TEXTFIX_MODEL=<tên model nhìn được ảnh>     # một model làm cả prompt FLUX lẫn designer
export TEXTFIX_LLM_API_KEY=<khoá>
# export TEXTFIX_GPUS=0,1                          # máy chủ: card dùng để VẼ (mặc định mọi card, mỗi card một FLUX)
# export TEXTFIX_DESIGN_WORKERS=8                  # máy chủ: số luồng THIẾT KẾ song song (mỗi luồng một Chromium)
export PLAYWRIGHT_CHROME_PATH=/home/jovyan/persistent-data/chrome-linux64/chrome
# kho ảnh tham chiếu món / trang phục Việt (docs/KHO.md); không đặt thì chạy không kho
# export TEXTFIX_KHO_ENDPOINT=http://<host>:<cổng API S3>  TEXTFIX_KHO_BUCKET=<bucket>
# export TEXTFIX_KHO_ACCESS_KEY=<khoá>  TEXTFIX_KHO_SECRET_KEY=<khoá>
```

Máy chủ VLM phải nhận **ít nhất 12 ảnh / lời gọi** (designer gửi 6, vòng sửa tới 12) và ngữ cảnh **≥ 32k token**
(vLLM: `--limit-mm-per-prompt '{"image": 16}' --max-model-len 32768`). `check_server.py` gọi thử đúng 12 ảnh.

Chạy từ ô notebook thay vì terminal: `%env TEXTFIX_MODEL=...` cho từng biến.

## 4. Tự kiểm tra

```bash
python scripts/check_server.py --fix 2>&1 | tee check.log
```

Kiểm: thư mục, gói pip (in một lệnh pip cho gói thiếu, không đụng torch), GPU, trọng số FLUX, font, Chromium (giải nén
zip, `ldd` báo thư viện hệ thống thiếu, mở thử), OCR, LLM / VLM (danh sách model, JSON, 1 và 12 ảnh), cổng.
Mục nào `[LỖI]` thì làm theo dòng `SỬA:` ngay dưới.

## 5. Chạy

```bash
bash scripts/start_server.sh               # distill, 4 bước
bash scripts/start_server.sh --model base  # base, 50 bước, CFG 4 (chậm khoảng 25 lần)
tail -f server.log
curl -s localhost:8088/api/health          # phải thấy "ready": true
```

Khởi động nạp **một FLUX và một Hi-SAM trên mỗi card** và OCR: khoảng 1 phút. Thiếu tệp Hi-SAM hoặc LaMa (hai mục cuối tệp này)
thì máy chủ báo lỗi và dừng ngay lúc khởi động; Chromium mở khi luồng thiết kế dùng lần đầu.

Dây chuyền (docs/ROADMAP.md): mỗi phiên bản qua trạm VẼ (GPU: nháp ~2 s, Hi-SAM tìm nét chữ ~0,2 s) rồi trạm THIẾT KẾ (đo dòng
~2 s, gỡ chữ bằng LaMa một lần, VLM ~11 s, dựng ~2 s, sửa bằng code chỉ khi có lỗi nặng). Các phiên bản và các yêu cầu gối đầu nhau: 2 card vẽ 2 phiên bản cùng lúc,
VLM gọi song song. Chọn nhiều ảnh: LLM viết thêm các **hướng thiết kế khác nhau** (cùng chữ khách), mỗi ảnh một hướng; ảnh xong
trước hiện trước trên UI. 4 ảnh ~35-45 s (một ảnh ~25-30 s).

UI có ô **Debug** (góc trên kết quả): nháp FLUX, **nét chữ + mặt nạ gỡ** (xanh lá = nét bị gỡ, vàng = phần nới / quầng, xanh
dương = nét của dòng giữ, hồng = nét không thuộc dòng nào), **nền đã gỡ chữ**, **Dòng + OCR** (khung L vẽ đè lên nháp, di chuột xem
chữ OCR đọc và lệnh designer), lệnh designer, lượt đầu trước vòng sửa, bảng OCR -> dòng -> lệnh, lỗi, nhật ký.

Kết quả: `output/runs/<run_id>/` gồm `request.json`, `brief.json` (kèm các hướng thiết kế), và mỗi ảnh `v<i>/`: `draft.png`,
`ink.png` (mặt nạ nét chữ Hi-SAM, 1 bit), `ink.jpg` (nét + mặt nạ gỡ chữ, để xem), `base.jpg` (nền đã gỡ chữ), `slots.jpg`,
`poster.png`, `poster_first.png` (khi vòng sửa đổi), `steps.jpg` (nháp | dòng | nét chữ | nền | poster), `ops.jpg` (lệnh
designer), `plan.json` (dòng, lệnh, lỗi còn lại, nhật ký, câu thiếu, thời gian, mục `erase`).

## 6. Thử engine không qua giao diện

Mọi script chạy trên máy chủ tự gói kết quả thành các zip **độc lập, mỗi zip < 24 MB** (giải nén từng cái, không cần ghép) và
**tự tải về** khi chạy bằng ô notebook (`%run`). Chạy từ terminal thì zip vẫn được gói, tải tay từ cây thư mục JupyterLab.
Lần đầu trình duyệt có thể hỏi "cho phép tải nhiều tệp": chọn cho phép.

```python
# ô notebook, thư mục làm việc = thư mục repo (%cd ~/work/<thư mục repo>)
# máy chủ (bash scripts/start_server.sh) phải đang chạy:
%run scripts/dev_batch.py                      # tập dev bench/dev2.json (40 yêu cầu như người dùng điền UI, 73 ảnh) qua máy chủ
                                               # -> dev2_<giờ>_NN.zip ĐẦY ĐỦ (nháp, nét chữ, lượt đầu, poster, plan); đứt giữa
                                               # chừng: chạy lại lệnh này = chờ tiếp job cũ, không gửi lại
%run scripts/dev_batch.py bench/dev3.json      # TẬP DEV 3 (40 yêu cầu, 62 ảnh): typography khó (cong / vòng con dấu / chéo /
                                               # lượn sóng / nhiều cỡ), 11 prompt dài thật (prompt_test), form đủ + prompt sơ sài,
                                               # ảnh thường -> dev3_<giờ>_NN.zip
%run scripts/runs_report.py --last 5           # các lượt người dùng chạy trên giao diện (mặc định đầy đủ; --light chỉ poster)
```

Biến môi trường trong notebook: `%env` từng biến, hoặc chạy `source ~/textfix_env.sh` trước khi mở JupyterLab.

**Dùng cả 2 card:** máy chủ nạp mỗi A30 một FLUX (≈ 17 GB) + một Hi-SAM (1,3 GB, đỉnh ~4 GB khi chạy); LaMa nằm trên card đầu.


## Chạy một tập dev từ terminal và xem kết quả tại chỗ

```bash
source ~/textfix_env.sh
python scripts/dev_batch.py bench/accept1.json --fresh    # chạy lại cả tập (seed cố định), giữ manifest cũ để so
```

Chạy xong, terminal in bảng từng ca và dòng tổng "so lượt trước: tốt lên / xấu đi / như cũ". Xem ảnh: mở
`https://<địa-chỉ-jupyter>/user/<tên>/proxy/8088/outputs/_reports/accept1.html` (nháp cạnh poster, cờ nghi hỏng, poster lượt trước).
Cần gửi kết quả về máy cá nhân: tải MỘT tệp `output/runs_zip/accept1_report_<giờ>.zip` (vài MB) từ cây thư mục JupyterLab.
Thêm `--full` khi cần nháp PNG cỡ gốc (các zip < 24 MB như trước). Lập lại báo cáo mà không chạy lại:
`python scripts/batch_report.py bench/accept1.json`.

## Sự cố

| Hiện tượng | Xử lý |
|---|---|
| `/api/health` có `error` | Đọc `server.log`: thường thiếu trọng số (FLUX, Hi-SAM, LaMa), chưa export biến LLM, hoặc hết VRAM |
| `CUDA out of memory` | `nvidia-smi`: card có bị người khác chiếm không (bộ nhớ đầy mà không có tiến trình trong container = dịch vụ ngoài, báo quản trị). Đặt `TEXTFIX_DEVICE_TE=cuda:1` |
| Lỗi 803 / `cuInit` | `start_server.sh` và `check_server.py` tự bỏ `.../compat` khỏi `LD_LIBRARY_PATH`; chạy tay thì làm tương tự |
| VLM `400` / `too many images` / `maximum context length` | Tăng `--limit-mm-per-prompt` / `--max-model-len` phía máy chủ VLM |
| Poster thiếu câu (giao diện báo "Không vẽ") | Model không chừa chỗ cho câu đó, hoặc designer rút gọn cho vừa: đổi seed, thêm ảnh |

## Hi-SAM (BẮT BUỘC từ 14/10: tìm nét chữ, thay bản xoá FLUX)

Máy chủ không ra internet: tải ở máy khác rồi đưa lên `~/persistent-data/hisam/` (đổi chỗ: `export TEXTFIX_HISAM=<thư mục>`).
- mã nguồn: https://github.com/ymy-k/Hi-SAM/archive/refs/heads/main.zip -> giải nén thành `hisam/Hi-SAM/`
- nền SAM ViT-L (1.249.524.607 byte): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
  -> `hisam/Hi-SAM/pretrained_checkpoint/sam_vit_l_0b3195.pth`
- phần nét chữ, bản TextSeg (122.756.163 byte): `sam_tss_l_textseg.pth` (README của Hi-SAM, mục Model Zoo, dòng SAM-TS-L /
  TextSeg; bản sao: https://huggingface.co/creative-graphic-design/hi-sam-checkpoints) -> `hisam/sam_tss_l_textseg.pth`
- thư viện: `torch` + `einops`. KHÔNG cài `requirements.txt` của Hi-SAM (ghim numpy / opencv bản cũ).

Kiểm tệp + đo thời gian, VRAM riêng Hi-SAM: `python scripts/hisam_test.py --check` rồi `python scripts/hisam_test.py accept2`.
Đo 14/10 trên 2x A30: fp16 0,17 s / ảnh, đỉnh VRAM 4,4 GB. Tắt fp16: `export TEXTFIX_HISAM_HALF=off`.

## LaMa (BẮT BUỘC: vẽ lại nền dưới chữ)

Tệp `big-lama.pt` (196 MB, https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt) đặt ở
`~/persistent-data/lama/big-lama.pt`; đổi chỗ thì `export TEXTFIX_LAMA=<đường dẫn>`. Thiếu tệp: lỗi, không còn đường dự phòng.
plan.json ghi dòng log "nền = nháp trừ chữ (Hi-SAM + LaMa): ..." và mục `erase`.

## Từ 14/10: TextFix chỉ sửa chữ, không còn bản xoá FLUX

Nhánh `main`: nền = nháp; Hi-SAM tìm nét chữ, LaMa lấp nền dưới chữ một lần cho cả poster, VLM chỉ viết dòng chữ. Vỏ, nút, icon
(kể cả icon đầu dòng) là điểm ảnh của nháp. Lộ trình và tình trạng: `docs/ROADMAP.md`.
Đường cũ: nhánh `legacy-flux-erase` (bản xoá FLUX + dọn chữ sót), nhánh `legacy-slots-si` (ô vỏ / ô chi tiết).
