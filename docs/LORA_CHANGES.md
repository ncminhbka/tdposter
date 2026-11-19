# Thay đổi 18-19/11 và runbook pilot LoRA 2 giờ (cho agent chạy trên server notebook)

Đọc hết tệp này trước khi chạy. Kế hoạch dài hạn: `docs/LORA_PLAN.md`. Mục tiêu người dùng chốt 19/11: pilot là để BIẾT hướng
fine-tune có tiềm năng không -- **chỉ nghiên cứu, prod vẫn vẽ bằng FLUX 9B**; không đổi prod, không build image vì việc này.

---

## 1. Thay đổi mã SẢN PHẨM (ảnh hưởng máy chủ)

| Commit | Thay đổi | Lý do |
|---|---|---|
| `0472946` (đã lên prod v0.0.9) | Chromium của luồng thiết kế chết / treo -> bỏ, lượt sau mở lại (`server/app.py` `page`, `drop_browser`, `browser_broken`; `textfix/render/browser.py` `alive`) | Trước: tab sập thì mọi lượt sau của luồng đó lỗi 500 tới khi khởi động lại pod |
| `0472946` | Quá hạn 55 s: 504 VÀ lượt dừng ở điểm kiểm kế tiếp (`server/contract.py` `alive`, `Expired`) | Lượt "ma" giữ GPU / VLM, chatbot gửi lại thì tải dồn thêm |
| `0472946` | Giải mã ảnh base64 ở nhóm luồng riêng (`decode_images`) | Không chặn vòng sự kiện (/api/health, yêu cầu khác) |
| `153cc9a` | `scripts/try_api.py`: thử tay API (gõ prompt, `/sua`, `/anh`, `/khung`) | Người dùng thử nhanh |
| `dba901a` | Brief + LLM sửa: KHÔNG tự bịa font / cỡ / vị trí chữ, nhưng mong muốn chữ của KHÁCH ("chữ viết tay", "tiêu đề thật to", "giá ở góc dưới") được chép vào `prompt_en`; tên font lạ -> tả bằng lời | Trước: lệnh cấm làm mất im lặng mong muốn của khách (router chatbot hay gửi "phông sans-serif trắng nhỏ ở góc trên bên trái"). FLUX vẽ nháp theo -> TextFix vốn theo nháp |
| `ff0a2e5` | Lời dặn bám bản chuẩn BFL (`flux2/system_messages.py`): brief trả lại "kích thước, quan hệ không gian giữa VẬT"; LLM sửa nói phần giữ bằng câu khẳng định + không bịa giá / ưu đãi / ngày / SĐT | So sánh với SYSTEM_MESSAGE_UPSAMPLING_T2I / I2I |
| `25bc40a` | `textfix/draw/flux.py`: `TEXTFIX_FLUX_LORA=<tệp>` gộp LoRA vào DiT lúc nạp (`TEXTFIX_FLUX_LORA_SCALE`) | Thử LoRA trong máy chủ; không đặt -> như cũ |
| (lần này) | `server/app.py`: `TEXTFIX_DATA_MODE=1` -> lưu thêm `v0/base.png` (nền dựng KHÔNG nén mất) và ghi `flux` (biến thể, model, số bước, LoRA) vào `request.json` | Tách chữ TextFix dựng khỏi vùng LaMa lấp (mục 2.2); collect chỉ nhận lượt của đúng model đang học. Mặc định tắt: prod / golden không đổi |

Hai thay đổi lời dặn LLM (`dba901a`, `ff0a2e5`) CHƯA chạy thật trên notebook -> runbook bước 2 thử trước khi sinh dữ liệu.

---

## 2. Pipeline huấn luyện (`lora/`, test `tests/test_lora.py`) -- thiết kế và lý do

### 2.1 Dữ liệu = cặp TextFix
Mỗi lượt máy chủ có nháp FLUX (chữ sai) và poster TextFix (chữ đúng nguyên văn, chép từ câu khách). Caption = `prompt_en` của
LLM brief / LLM sửa = ĐÚNG chuỗi FLUX nhận (kiểm trong mã: `engine.draw` dùng `B["prompt_en"]`, plan.json ghi y chuỗi đó).
Ảnh tham chiếu dựng lại đúng thứ tự FLUX nhận: sửa = [`source.png` (poster cũ), `product.png`?]; tạo mới = [`product.png`?, ảnh kho].

### 2.2 Ba vùng trọng số (rà luồng TextFix 19/11 -- `lora/collect.py` `regions`)
Poster khác nháp không chỉ ở chữ dựng: TextFix còn gỡ chữ nháp bằng LaMa, xoá dòng FLUX vẽ lặp ("skip"), xoá cả nút / thẻ không
còn chữ ("gone"), LaMa vẽ lại vùng dòng "restored", thêm bóng đổ (finish.py). plan.json KHÔNG lưu khung chữ đã dựng; nền dựng
trước đây chỉ có JPEG. Nếu gộp tất cả vào một mặt nạ x6 thì LoRA học MẠNH cả kết cấu mờ của LaMa. Nay (cần `base.png`):
| Vùng | Cách tính | Trọng số | Vì sao |
|---|---|---|---|
| T chữ dựng | poster != base (nới 2 px) | `--text-weight` 6 | chữ đúng nguyên văn: thứ cần học |
| E LaMa lấp | base != draft, trừ T | `--erase-weight` 1 | kết cấu LaMa mờ, không đáng học mạnh |
| U chữ chưa kiểm | dòng `keep` (nhãn sản phẩm, con dấu, biển phụ) + nét Hi-SAM `v0/ink.png` sót, trừ T, E | `--keep-weight` 0,3 | chữ FLUX tự vẽ, thường méo -- đừng củng cố |
| còn lại | | 1 | ảnh model tự vẽ -> neo, chống trôi |
Thiếu `base.png` (lượt cũ / máy chủ thường) -> lùi: T = poster != draft (nới 8 px), E rỗng, `mask_mode: "diff"` trong meta.

### 2.3 Bộ lọc (`collect.reject`, không cần OCR vì chữ TextFix đúng từ cách dựng)
Bỏ: ảnh lạ, ảnh không chữ; còn lỗi dựng; thiếu câu khách; dòng nháp giữ nguyên mang câu khách (có thể sai dấu); prompt > 500 token
(Qwen3 cắt đuôi im lặng -> caption mất câu mà ảnh vẫn có); model vẽ khác 4B distill không LoRA (request.json `flux`); mặt nạ quá
nhỏ / quá lớn; không có chữ dựng. Tách train / eval theo yêu cầu gốc của chuỗi sửa.

### 2.4 Học (`lora/train.py`, `lora/flow.py`, `lora/net.py`)
- LoRA tự viết (không peft) bọc 10 lớp Linear mỗi cặp khối của DiT `flux2/` (qkv, proj, 2 lớp MLP của luồng ảnh và chữ; linear1/2
  khối đơn). Học trên **base 4B**, chạy trên **distill 4B** (quy trình BFL khuyên).
- Mỗi bước: latent poster x0, nhiễu eps MỚI, t; x_t = (1-t)·x0 + t·eps; đích eps - x0 (quy ước `flux2/sampling.py`: t 1->0,
  `img += (t_prev - t_curr)·pred`). Token ảnh tham chiếu nối SAU ảnh đích, đi qua `Flux2.forward` thường (đúng đường máy chủ;
  KHÔNG phải `forward_kv_extract`).
- t: logit-normal dịch theo lịch base + 25% đúng 4 mức t của distill (`--p-distill`).
- 10% bỏ prompt (nhánh không prompt của CFG bản base), 15% caption rút gọn = câu đầu + câu có ngoặc kép + câu nhắc ảnh tham chiếu
  (`--short-caption`: LoRA bám chữ trong ngoặc kép, không bám khuôn câu của brief).
- LoRA float32, DiT bf16 đóng băng, checkpointing từng khối; `--max-minutes` dừng + lưu đúng giờ.

### 2.5 Sinh dữ liệu
- `lora/serve_flux4b.sh`: 4B distill tại chỗ, `TEXTFIX_DATA_MODE=1`, lượt vào `data/lora/runs` không bị xoá, **hạn 1800 s, hàng
  đợi 32** (từ v0.0.9 lượt quá 55 s bị DỪNG -> sinh hàng loạt với hạn 55 s sẽ mất lượt), `--gpus 0` để chừa card.
- `lora/drive.py --bench "bench/*.json"`: 376 yêu cầu thật + 73 lời sửa có sẵn (6 lời sửa ảnh lạ bị bỏ); tỉ lệ 4:5 -> 3:4; ảnh
  sản phẩm bench/refs. `--requests` (LLM sinh, `lora/requests.py`) để dành cho pilot đầy đủ: 30% yêu cầu sơ sài (brief tự thêm
  câu AI như prod), mật độ chữ ít / vừa / dày (dày = 7-10 dòng có câu dài), 15% có mong muốn chữ, một nửa ảnh sản phẩm tự vẽ có
  nhãn; ảnh sản phẩm THẬT: `lora.requests --products <thư mục>` viết yêu cầu gắn với TỪNG ảnh (trường `image_file`), drive gửi đúng ảnh đó.
- `lora/collect.py` in: lý do bỏ, số mẫu theo kiểu, trung bình T / E / U, tỉ lệ câu AI + câu lặp nhiều, tỉ lệ đạt theo số dòng.
- `lora/cache.py --shard i/n`: mã hoá trước (Qwen3 + VAE) song song nhiều card.
- `lora/sheets.py`: bảng so sánh cùng seed [TextFix | distill gốc | distill + LoRA A | + LoRA B], ô phóng to vùng chữ lớn nhất;
  `--no-lora` vẽ trước ảnh gốc, `--split train` xem mẫu đã học.
- `lora/smoke.sh`: chạy trọn chuỗi với trọng số thật trên vài lượt có sẵn (~10 phút) -- lộ lỗi trước khi sinh dữ liệu.

---

## 3. Runbook pilot 2 giờ (2 card A30, người dùng đã đồng ý dùng cả hai)

Từ gốc repo trên notebook (bản mới nhất), `source ~/textfix_env.sh`. Không còn máy chủ TextFix nào khác chạy (các script dùng
`pkill -f server/app.py`). Mọi lệnh dài chạy `nohup ... &` và theo dõi log.

**0:00 -- smoke (card 0, ~10 phút).** `bash lora/smoke.sh` (lấy 6 lượt có sẵn trong `output/runs`). Phải in "SMOKE XANH". Lỗi ->
sửa trong `lora/`, chạy `pytest tests/test_lora.py`, chạy lại. Không có lượt sẵn -> bỏ qua, chạy smoke ở bước 3 trên
`data/lora/runs` sau khi có ~6 lượt (máy chủ khi đó `--gpus 0`, smoke `DEV=cuda:1`).

**0:10 -- máy chủ dữ liệu (2 card).**
```
bash lora/serve_flux4b.sh --llm-port 8887 --llm-model Qwen/Qwen3.5-35B-A3B
until curl -s localhost:8088/api/health | grep -q '"ready": *true'; do sleep 10; done
```
**Thử lời dặn brief mới (6 lượt):** `python -m lora.drive --bench "bench/quick1.json" --limit 6 -j 6 --log data/lora/try.jsonl`,
rồi in `prompt_en`:
`python -c "import json,glob; [print(json.load(open(f))['prompt_en'],'\n') for f in sorted(glob.glob('data/lora/runs/*/brief.json'))[-6:]]"`.
Cần thấy: mỗi câu chữ trong ngoặc kép đúng một lần, 80-130 từ, không tự bịa font / vị trí. Hỏng -> dừng, báo người dùng.

**0:15 -- sinh dữ liệu (~45 phút).**
`nohup python -m lora.drive --bench "bench/*.json" -j 8 > data/lora/drive.log 2>&1 &`
Lúc 0:45 xem nhanh (CPU, không ảnh hưởng sinh): `python -m lora.collect --runs data/lora/runs --out data/lora/ds_peek` -> tỉ lệ
đạt, T / E / U; mở 10 mẫu `data/lora/ds_peek/train/*/{target,text,erase,unver}.png` xem mặt nạ đúng chỗ.

**1:00 -- dừng, gom.** drive chưa xong thì `pkill -f lora.drive`; **chờ 60 s** (draft / ink / base ghi nền); `pkill -f server/app.py`.
`python -m lora.collect --runs data/lora/runs --out data/lora/ds` (dự kiến ~300-450 mẫu, eval 8%).

**1:05 -- mã hoá (2 card).**
```
python -m lora.cache --ds data/lora/ds --device cuda:0 --shard 0/2 & python -m lora.cache --ds data/lora/ds --device cuda:1 --shard 1/2 & wait
```
**1:12 -- học A (card 0), ảnh gốc rồi học B (card 1).**
```
nohup python -m lora.train --ds data/lora/ds --out data/lora/A --rank 32 --lr 1e-4 --text-weight 6 --accum 2 --warmup 20 \
  --steps 100000 --max-minutes 30 --save-every 100 --device cuda:0 > data/lora/A.log 2>&1 &
python -m lora.sheets --ds data/lora/ds --no-lora --out data/lora/sheets --limit 40 --notext 10 --device cuda:1
python -m lora.sheets --ds data/lora/ds --no-lora --split train --out data/lora/sheets_train --limit 10 --notext 0 --device cuda:1
nohup python -m lora.train --ds data/lora/ds --out data/lora/B --rank 32 --lr 2e-4 --text-weight 10 --accum 2 --warmup 20 \
  --steps 100000 --max-minutes 25 --save-every 100 --device cuda:1 > data/lora/B.log 2>&1 &
```
Ghi lại từ `A.log` dòng đầu: VRAM, giây / bước. VRAM tràn -> `--accum 1`; vẫn tràn -> báo.

**1:45 -- bảng so sánh với LoRA (2 card).**
```
LA=$(ls -t data/lora/A/lora_step*.safetensors | head -1); LB=$(ls -t data/lora/B/lora_step*.safetensors | head -1)
python -m lora.sheets --ds data/lora/ds --lora $LA,$LB --out data/lora/sheets --limit 40 --notext 10 --device cuda:0 &
python -m lora.sheets --ds data/lora/ds --lora $LA,$LB --split train --out data/lora/sheets_train --limit 10 --notext 0 --device cuda:1 &
wait
```
**1:55 -- gửi người dùng** (zip < 24 MB; quá thì bớt trang):
`cd data/lora && zip -q ~/work/lora_pilot_sheets.zip sheets/sheet_*.jpg sheets_train/sheet_*.jpg A/log.jsonl B/log.jsonl drive.log`
+ dán: 4 dòng thống kê cuối của collect, dòng cuối A.log / B.log.

### Đọc kết quả (người dùng xem bằng mắt)
Cột: TextFix (chữ đúng) | distill gốc | distill + A | distill + B; dưới mỗi ảnh là ô phóng to vùng chữ lớn nhất.
- **Có tiềm năng**: trên ~40 prompt CHƯA học, tiêu đề / giá của + LoRA đúng chữ / đúng dấu rõ hơn gốc (đếm tay số câu đúng
  trước / sau); ảnh không chữ không sinh chữ lạ; ảnh không xấu đi.
- Mẫu train đúng hẳn mà eval không đổi -> học được nhưng thiếu dữ liệu -> pilot đầy đủ (sinh thêm bằng `lora.requests`).
- Cả mẫu train cũng không đổi -> xem lại cách học (rank / lr / text-weight / số bước) trước khi đầu tư thêm.

---

## 4. Cần từ người dùng / đội chatbot (cho pilot đầy đủ, không cần cho 2 giờ)
- **~100-150 ảnh sản phẩm THẬT khác nhau có chữ** (mỗi ảnh dùng 2-3 lần -> ~300 mẫu kiểu 2 trong ~1.500): chai / lon / hộp /
  túi / mỹ phẩm / đồ ăn đóng gói / điện tử; ưu tiên nhãn TIẾNG VIỆT; sản phẩm rõ, chiếm phần lớn khung, cạnh >= 512 px; jpg / png;
  quyền dùng gồm cả huấn luyện. Chép vào một thư mục rồi `lora.requests --products <thư mục> --out data/lora/requests_products.jsonl` -> `lora.drive --requests data/lora/requests_products.jsonl`. Bộ 140 ảnh tải sẵn (Open Food / Beauty / Products Facts, CC BY-SA): `lora.fetch_products`.
- Dạy FLUX chép đúng nhãn từ ảnh tham chiếu: chưa có cách (không có ảnh đích có nhãn đúng); pilot chỉ giữ không làm tệ thêm (U x0,3).
