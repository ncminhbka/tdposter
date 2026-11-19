# Kế hoạch: LoRA cho FLUX.2 klein 4B vẽ chữ tiếng Việt (19/11)

Mục tiêu: nháp FLUX vẽ đúng chữ tiếng Việt (dấu thanh, ơ ư ă â ê ô đ) nhiều hơn. TextFix vẫn sửa phần còn sai; nháp đúng hơn
thì nhiều dòng được giữ nguyên nét (đẹp, tự nhiên hơn chữ dựng), VLM ghép dễ hơn, ít ca nháp thiếu dòng.

Nguyên tắc: **mỗi chặng có một probe và một cửa qua**. Không qua cửa thì dừng, báo người dùng, không làm chặng sau.
Mọi kết quả ghi vào bảng ở cuối tệp (mục "Nhật ký").

**Người dùng chốt 19/11: đi đường thực dụng.** Chưa dựng thước đo OCR (chặng 0 hoãn); làm thẳng dữ liệu (chặng 2) -> smoke test
(chặng 3) -> pilot (chặng 4), đánh giá bằng MẮT trên bảng so sánh cùng seed. Ổn mới tính tiếp (thước đo, mở rộng...). Thứ tự làm:
**1 (chỉ sinh ảnh mốc) -> 2 -> 3 -> 4**.

**Cập nhật 19/11 (sau): CẢ 3 KIỂU ĐẦU VÀO NGAY TỪ PILOT.** Kiểu 1 tốt mà kiểu 2-3 (ảnh sản phẩm, sửa poster) không được kiểm thì
không đưa lên prod được. Trong klein, ảnh tham chiếu chỉ là token NỐI THÊM vào chuỗi token ảnh, đi qua cùng transformer
(`textfix/draw/flux.py` `_sample`: `torch.cat([x, refs[0]], 1)`, id thời gian 10 / 20, lấy `pred[:, :n]`) -> huấn luyện kiểu 2-3
chỉ khác kiểu 1 ở bước nối token. Vì vậy:
- **Trainer chính = tự viết (~300 dòng, peft LoRA) trên chính mã `flux2/` + `textfix/draw/flux.py`** (cùng cách đóng gói token,
  cùng id ảnh tham chiếu như lúc chạy prod -> không lệch huấn luyện / suy luận); loss flow-matching chỉ trên token ảnh đích,
  nhân trọng số mặt nạ vùng chữ. ai-toolkit chỉ còn là kiểm chéo cho kiểu 1 (nó chưa có klein ở mục edit).
- Pilot trộn ~50% kiểu 1 / 20% kiểu 2 / 30% kiểu 3; bảng so sánh tách theo kiểu.
- Hướng dự phòng nếu LoRA thuần không đủ cho câu dài: dựng sẵn ảnh GLYPH các câu chữ, đưa làm ảnh tham chiếu thêm (klein nhận
  nhiều ảnh tham chiếu) + LoRA học "chép glyph" -- người dùng từng cho thấy truyền glyph vẽ được cả bài thơ dài.

---

## Phương án đã chọn (sau khi tra cứu)

| Phương án | Khả thi trên 2x A30 | Công cụ | Vị trí trong kế hoạch |
|---|---|---|---|
| LoRA trên **base 4B**, loss có mặt nạ vùng chữ, chạy LoRA trên **distill 4B**, cả 3 kiểu đầu vào | Có (BFL: ~12 GB VRAM, <1 giờ/lượt) | trainer tự viết trên mã `flux2/` (chính); ai-toolkit (`mask_path`...) kiểm chéo kiểu 1 | Chặng 3-4, phương án chính |
| Huấn luyện thẳng distill 4 bước theo cặp (nhiễu từ seed -> poster sửa) | Có, nhưng phải tự viết | script riêng (diffusers) | Chặng 5B, chỉ khi LoRA base không chuyển sang distill |
| DPO theo vùng chữ (cặp nháp thua / poster thắng) | Có, quy mô nhỏ | script riêng; tham khảo Diffusion-SDPO, GlyphPrinter | Chặng 6, tuỳ chọn |
| Flow-GRPO, phần thưởng = OCR | Khó (ví dụ tham khảo 4x H800, ~7 phút/bước) | flow_grpo / verl-omni | Ngoài phạm vi pilot |
| Distill lại toàn bộ | Không | -- | Loại |

Lý do chọn: quy trình base -> distill là cách BFL khuyên chính thức; ai-toolkit có sẵn loss theo mặt nạ nên không phải viết
trainer; dữ liệu TextFix là cặp "cùng prompt, cùng seed, chỉ khác chữ" nên mặt nạ vùng chữ có sẵn (so điểm ảnh nháp / poster:
trên 8 ca golden, > 95% điểm ảnh khác nằm trong vùng chữ).

Giới hạn đã biết:
- ai-toolkit chưa có tệp cấu hình mẫu cho klein, và klein chỉ có trong danh sách text-to-image (không có trong mục edit) ->
  chặng 3 phải kiểm tên arch và đường nạp model trong mã. Huấn luyện kiểu SỬA ẢNH (có ảnh tham chiếu) cần công cụ khác, xét ở chặng 5.
- RapidOCR mà pipeline đang dùng (mô hình Trung/Anh) đọc chữ Latin nhưng BỎ DẤU -> không đo được tiếng Việt. Thước đo phải dựng
  mới ở chặng 0.
- Chữ nhỏ (< ~20 px), icon, ngôi sao: giới hạn kiến trúc (mỗi token latent ~16x16 px), không kỳ vọng cải thiện.
- Giấy phép: 4B Apache 2.0 (dùng được cho sản phẩm); 9B phi thương mại. Prod hiện vẽ bằng 9B -> nếu LoRA 4B tốt, phải so 4B+LoRA
  với 9B rồi người dùng quyết có đổi không (chặng 7).

---

## Mã đã viết (`lora/`, test `tests/test_lora.py`) và lệnh chạy từng bước

Đã kiểm tại máy (CPU, DiT FLUX.2 thật nhưng tí hon): LoRA gắn đủ lớp và ra y hệt model gốc khi mới gắn; tắt LoRA = model gốc;
gộp = bản bọc; lưu / nạp khứ hồi; lịch t khớp `flux2.sampling`; một bước học chỉ đổi tham số LoRA và loss giảm; checkpointing
cho cùng gradient; chạy trọn `collect`, `cache` (bộ mã hoá giả), `train` (DiT tí hon) trên dữ liệu giả. CHƯA chạy với trọng số
thật (cần GPU notebook) -- chặng 3 là lần đầu.

**Điều đọc được trong `flux2/` quyết định thiết kế**
- Quy ước lấy mẫu: t chạy 1 -> 0, `img += (t_prev - t_curr) * pred` -> đích huấn luyện = eps - x0 với x_t = (1-t)·x0 + t·eps.
  Bước đầu của bản distill là đúng t = 1.
- Ảnh tham chiếu có HAI đường: `Flux2.forward` thường (token tham chiếu nối SAU ảnh đích, cùng mức t, attention đầy đủ -- đường
  `textfix/draw/flux.py` đang dùng) và `forward_kv_extract` / `forward_kv_cached` (tham chiếu đứng trước, t cố định = 0, chỉ tự
  nhìn nhau, có cache KV). LoRA học trên đường THỨ NHẤT; đổi máy chủ sang đường cache thì phải học lại.
- Bộ mã hoá chữ Qwen3-4B: luôn đệm tới 512 token, ghép 3 tầng [9, 18, 27] -> 7680 chiều. VAE trả latent đã chuẩn hoá, gộp 2x2
  -> 128 kênh ở 1/16 cỡ ảnh (mean, tất định). Mỗi token = ô 16x16 điểm ảnh.
- Bản base: 50 bước, CFG 4.0 với nhánh prompt rỗng -> khi học bỏ prompt 10% để không hỏng nhánh đó. Bản distill: 4 bước, không CFG.
- Mọi lớp Linear của DiT được gọi như module -> LoRA bọc module là đủ (10 lớp mỗi cặp khối: qkv / proj / 2 lớp MLP cho luồng ảnh
  và luồng chữ ở khối đôi; linear1 / linear2 ở khối đơn). 4B: 5 khối đôi + 20 khối đơn.
- `flux2.util.load_flow_model(debug_mode=True)` sửa cấu hình TOÀN CỤC (depth = 1) -> không dùng; test tự dựng model tí hon.

**Lệnh** (trên notebook, từ gốc repo; `~/textfix_env.sh` đã có khoá / địa chỉ; trọng số base + distill trong `~/persistent-data`)

| Chặng | Lệnh | Ghi chú |
|---|---|---|
| 2 máy chủ | `bash lora/serve_flux4b.sh --llm-port 8887 --llm-model Qwen/Qwen3.5-35B-A3B` | 4B tại chỗ mọi card; lượt chạy vào `data/lora/runs`, KHÔNG bị xoá (máy chủ thường chỉ giữ 300 lượt) |
| 2 yêu cầu | `python -m lora.requests --n 400 --out data/lora/requests.jsonl` | LLM sinh, phủ nguyên âm x thanh; 25% kiểu 2 |
| 2 gọi API | `python -m lora.drive --requests data/lora/requests.jsonl -j 6` | kiểu 1 / 2 + sửa nối tiếp (kiểu 3, xác suất 0,35, tối đa 2 lần -> ~32% lượt là sửa); chạy lại được |
| 2 lọc | `python -m lora.collect --runs data/lora/runs --out data/lora/ds` | in lý do bỏ + số mẫu theo kiểu; `ds/eval` để riêng cho bảng so sánh |
| 2 xem | mở 50 thư mục `data/lora/ds/train/*`: `target.png` / `draft.png` / `mask.png` | cửa qua: >= 50% lượt đạt, mặt nạ đúng chỗ chữ |
| 3 mã hoá | dừng máy chủ (`pkill -f server/app.py`); `python -m lora.cache --ds data/lora/ds --device cuda:0` | ~8 MB / mẫu (embedding chữ 512 x 7680) |
| 3 smoke | `python -m lora.train --ds data/lora/ds --out data/lora/smoke --overfit 50 --steps 500 --rank 16` | xem `log.jsonl`: loss giảm, VRAM, giây / bước |
| 3 xem | `python -m lora.sheets --ds data/lora/ds --lora data/lora/smoke/lora_step00500.safetensors --out data/lora/sheets_smoke --limit 8 --notext 4` | (học thuộc: bảng dùng tập eval nên chỉ kiểm đường chạy; xem thêm ảnh trên mẫu train nếu cần) |
| 4 pilot | `python -m lora.train --ds data/lora/ds --out data/lora/r32 --rank 32 --steps 3000` | lưu mỗi 500 bước |
| 4 bảng | `python -m lora.sheets --ds data/lora/ds --lora data/lora/r32/lora_step03000.safetensors --out data/lora/sheets_r32` | gửi người dùng vài trang `sheet_*.jpg` (zip < 24 MB) |
| dùng thử | `TEXTFIX_FLUX_LORA=data/lora/r32/lora_step03000.safetensors bash scripts/start_server.sh --draw flux4b` | máy chủ gộp LoRA vào 4B lúc nạp; `TEXTFIX_FLUX_LORA_SCALE` chỉnh độ mạnh |

Dữ liệu (`data/`) không commit. Ước lượng: ~15 s / lượt trên 2 card -> 1.000 lượt ~ 2-3 giờ (LLM / VLM chịu tải thì nhanh hơn).

## Chặng 0 -- Thước đo chữ tiếng Việt (HOÃN theo quyết định 19/11; làm khi pilot cho thấy đáng đi tiếp)

Không có thước đo đúng thì không có con số đáng tin để so các lượt huấn luyện về sau. GlyphPrinter còn cảnh báo OCR hay "bỏ qua" lỗi nét chữ.

**Việc**
1. Ứng viên: (a) PaddleOCR PP-OCRv5 `latin_PP-OCRv5_mobile_rec` (danh sách ngôn ngữ có tiếng Việt, chưa ai công bố số cho dấu);
   (b) VietOCR (bộ nhận dạng chuyên tiếng Việt) + khung dòng lấy từ RapidOCR / Hi-SAM có sẵn; (c) VLM Qwen3.8-27B đọc chữ (đọc tốt
   nhưng có thể tự sửa lỗi chính tả -> nguy hiểm cho việc chấm).
2. Tập kiểm: poster TextFix (chữ dựng bằng font, BIẾT CHÍNH XÁC nội dung từ khối thông tin PNG) -- 200 ảnh. Thêm 200 bản "cố ý
   sai": dựng lại cùng poster nhưng đổi dấu thanh / bỏ dấu ở 1-2 âm tiết mỗi câu (dùng chính bộ dựng HTML của TextFix).
3. Chấm theo dòng: khớp nguyên văn, CER, và **tỉ lệ bắt được lỗi dấu** trên tập cố ý sai.

**Probe -> cửa qua**
- Trên poster đúng: >= 97% dòng đọc khớp nguyên văn (lỗi còn lại phải xem tay là do OCR, không do ảnh).
- Trên tập cố ý sai: bắt được >= 95% lỗi dấu (không tự "sửa" về đúng).
- Chọn ứng viên tốt nhất; nếu không ứng viên nào qua: thử ghép (VietOCR + VLM kiểm chéo), hoặc fine-tune nhận dạng bằng chính
  dữ liệu dựng (dựng chữ thì có nhãn miễn phí). Vẫn không qua -> dừng, báo người dùng.

**Ra:** `scripts/vi_ocr_eval.py` (hàm `score(image, texts) -> {exact, cer, tone_err}`), dùng chung cho mọi chặng sau.

---

## Chặng 1 -- Mốc: 4B hiện vẽ chữ Việt đúng bao nhiêu

**Việc**
1. Tập giữ riêng 200 prompt (không bao giờ dùng để huấn luyện): lấy từ các bench (quick1, chat1, hard1, sme1, final2...) + LLM
   sinh thêm; chạy qua LLM brief để có `prompt_en` (đúng thứ FLUX thấy). Chia nhóm: số câu, độ dài câu, chữ HOA có dấu.
2. Vẽ nháp bằng **4B distill** (4 bước, đúng bản prod chạy), cùng seed, 1024 px, cả 5 tỉ lệ. KHÔNG chạy base ở mốc (50 bước CFG
   ~46 s / ảnh / A30): base chỉ dùng để chẩn đoán ở chặng 4 khi distill + LoRA không tốt lên.
3. (Đường thực dụng) Chưa chấm số. LƯU nháp + seed + prompt_en của cả 200 prompt: đây là mốc để so bằng mắt ở chặng 4.
4. Xem nhanh 50 nháp, ghi nhận xét: lỗi chủ yếu là sai dấu (chữ cái đúng) hay sai cả chữ cái / chữ vô nghĩa.

**Probe -> cửa qua**
- Có đủ ảnh mốc distill (4 bước) cho 200 prompt, cùng seed cố định (thực ra `lora.sheets` vẽ lại mốc cùng seed, nên mốc chỉ là
  ảnh để xem trước loại lỗi).
- Nếu nháp distill nhìn đã đúng gần hết ở cỡ tiêu đề -> lợi ích còn lại nhỏ, báo người dùng trước khi làm tiếp.

---

## Chặng 2 -- Dữ liệu cặp từ TextFix

**Việc**
1. **Prompt:** LLM sinh yêu cầu poster tiếng Việt đa dạng (ngành hàng SME, giá, giờ, khuyến mãi, câu dài / ngắn, chữ HOA), có
   kiểm độ phủ: đủ 12 nguyên âm x 6 thanh + đ, chữ HOA có dấu (Ặ Ẫ Ỡ...), ~2.000 âm tiết hay gặp. Ghi thống kê phủ.
2. **Sinh:** server TextFix trên notebook với `--draw flux4b` (4B tại chỗ, mỗi card một bản) -> mỗi lượt lưu `brief.json` (seed,
   prompt_en), `draft.png`, `poster.png`, `plan.json`. Cần thêm: lưu **mặt nạ** (so điểm ảnh draft/poster, ngưỡng 3, nới ~8 px)
   và **ảnh mặt nạ PNG trắng/đen** cùng tên cho ai-toolkit.
3. **Lọc tự động (không cần OCR -- chữ TextFix chép nguyên văn nên đúng từ cách dựng):** plan không còn lỗi render
   (`errors` rỗng sau vòng sửa), không có câu khách bị thiếu (`missing` rỗng), không có dòng nháp "keep" chứa chữ khách (dòng
   giữ nét nháp có thể sai dấu -> poster không sạch), điểm ảnh khác ngoài mặt nạ < 1%. Không qua -> bỏ. Xem tay 50 cặp để bắt
   loại lỗi lọc không thấy (vệt LaMa, chữ sót).
3a. **Mật độ chữ** (người dùng 19/11: yêu cầu thật thường nhiều chữ, có câu dài 15-30 chữ; nhiều chữ = nhiều âm tiết mỗi mẫu):
   `lora.requests --density 0.35,0.40,0.25` = lô ít (<= 3 dòng) / vừa (4-6) / dày (7-10 dòng, có câu khách hàng nói, câu tính
   năng, câu ưu đãi). Chữ dưới ~20 px học kém (giới hạn latent) -> trộn, không đẩy hết sang dày. `lora.collect` in tỉ lệ đạt theo
   số dòng (poster dày hay bị nháp vẽ thiếu dòng -> bị loại "thiếu câu khách"); tỉ lệ đạt của nhóm dày quá thấp thì giảm phần dày.
   Đây là việc của TẬP TRAIN; brief trên prod vẫn giữ giới hạn câu AI vì thẩm mỹ (quyết định riêng của người dùng).
3c. **Caption = `prompt_en` của LLM brief** (lời dặn: một đoạn 80-130 từ theo hướng dẫn nâng prompt của BFL, chữ trong ngoặc
   kép đúng một lần, cấm nói font / cỡ / vị trí chữ). Caption vì thế rất đồng dạng -- KHÔNG kéo phần mô tả dài ra (lệch prod);
   độ dài tự tăng theo lượng chữ (lô dày ~350-400 token). Đa dạng hoá nhẹ: `train --short-caption 0.15` dùng caption rút gọn
   (câu đầu + câu có ngoặc kép + câu nhắc ảnh tham chiếu) để LoRA bám vào chữ, không vào khuôn câu. LoRA GẮN với khuôn
   `prompt_en`: muốn sửa lời dặn brief thì sửa TRƯỚC khi sinh dữ liệu; sửa sau -> sinh lại + học lại.
3d. **Chữ in trên ảnh sản phẩm** (nhãn Dove, logo Coca...): TextFix KHÔNG viết lại -- VLM đánh dấu `keep`, giữ nét FLUX vẽ (thường
   méo), caption không trích nhãn (brief: "nhãn / logo giữ như ảnh 1"). Hai việc: (a) `collect` lưu `keep.png` = mọi dòng `keep`
   (nhãn sản phẩm, con dấu, biển phụ: chữ không ai kiểm), `train --keep-weight 0.3` hạ trọng số để LoRA không củng cố chữ méo;
   (b) ảnh tham chiếu phải có nhãn như ảnh khách thật: `requests` cho một nửa ảnh sản phẩm tự vẽ có nhãn / logo, và
   `drive --products <thư mục>` dùng ẢNH SẢN PHẨM THẬT (nguồn tốt nhất: ảnh khách đã gửi mà đội chatbot lưu, đã cho phép dùng).
   Dạy FLUX chép đúng nhãn từ ảnh tham chiếu là bài toán khác (không có đích đúng) -- chưa làm.
3b. **Câu LLM brief tự viết (source "ai")**: yêu cầu sơ sài thì brief thêm tới 6 câu (tiêu đề, nút, nhãn mục, lợi ích chung) --
   có trong `prompt_en` (trong ngoặc kép) và TextFix dựng nguyên văn, nên caption vẫn khớp ảnh. Nhưng: (a) `lora.requests` mặc
   định 30% lô viết sơ sài như người dùng thật, để tỉ lệ câu AI gần prod; (b) `lora.collect` in tỉ lệ câu AI và 20 câu lặp nhiều
   nhất -- 10 câu đầu chiếm > 30% số dòng thì LoRA dồn sức học vào vài câu chung chung -> hạ trọng số các mẫu đó (chưa làm sẵn).
4. **Tách tập:** theo prompt (không để cùng prompt ở cả train và eval); tập eval chặng 1 không bao giờ vào train.
5. **Kiểu 2 (ảnh sản phẩm):** bench chỉ có 5 ảnh sản phẩm -> 4B vẽ vài trăm ảnh sản phẩm KHÔNG chữ (chai, hộp, lon, món ăn, mỹ
   phẩm...) làm ảnh tham chiếu, rồi chạy TextFix có ảnh sản phẩm. Cặp: (prompt_en, ảnh sản phẩm) -> poster TextFix.
6. **Kiểu 3 (sửa poster):** poster v0 (đã qua TextFix) + lời sửa do LLM sinh (đổi giá, thêm / bỏ dòng, đổi nền, đổi ảnh) -> luồng
   sửa (FLUX sửa poster cũ + TextFix) -> poster v1. Cặp: (prompt_en của lần sửa, poster v0 làm ảnh tham chiếu [+ ảnh sản phẩm
   mới]) -> poster v1. Dạy đúng kỹ năng FLUX đang thiếu: giữ chữ đúng của poster cũ, chỉ đổi chỗ được yêu cầu.
7. Mỗi cặp ghi `mode` (1/2/3) và đường dẫn ảnh tham chiếu theo đúng thứ tự FLUX nhận lúc chạy (`refs` trong brief / server).

**Probe nhỏ trước (300 lượt) -> cửa qua**
- Tỉ lệ cặp sạch >= 50% (dưới mức này: xem lý do, sửa pipeline trước khi sinh nhiều).
- Xem tay 50 cặp sạch: poster đẹp, không vệt LaMa rõ, chữ đúng.
- Độ phủ ký tự không lệch quá (mọi thanh trên mỗi nguyên âm có >= 20 lần xuất hiện ở bản 1.000 cặp).
- Qua -> sinh tiếp tới ~1.000-2.000 cặp sạch (2 card, ~15 s/lượt -> vài nghìn lượt/đêm nếu LLM/VLM chịu tải).

**Ra:** thư mục dataset theo khuôn ai-toolkit: `img/xxx.png` (poster), `img/xxx.txt` (caption = prompt_en), `mask/xxx.png`.

---

## Chặng 3 -- Chạy thử trainer (smoke test, không đo chất lượng)

**Trainer chính (tự viết, xem cập nhật đầu tệp):** nạp DiT base 4B qua mã `flux2/`, gắn LoRA (peft) vào các lớp linear của
attention / MLP; mỗi bước: mã hoá poster đích -> token x0; nhiễu ε, t theo lịch của flux2 (`get_schedule` có dịch theo độ dài
chuỗi -- lấy t ngẫu nhiên theo cùng phân bố); x_t = (1-t)·x0 + t·ε; nối token ảnh tham chiếu (kiểu 2-3, `_encode_refs`); dự
đoán `pred[:, :n]`; loss = mean(w · (pred - (ε - x0))²) với w = trọng số mặt nạ thu về lưới token (vùng chữ ~5-10, ngoài 1);
cache trước embedding chữ (Qwen3) và latent ảnh để tiết kiệm VRAM. Smoke: học thuộc 50 cặp trộn 3 kiểu, 500 bước. Lưu LoRA dạng
safetensors; nạp lại vào distill 4B bằng đúng `textfix/draw/flux.py` (thêm tham số nạp LoRA).

**Việc (ai-toolkit -- kiểm chéo kiểu 1, tuỳ chọn)**
1. Cài ai-toolkit trong venv riêng (`--system-site-packages`, không đụng môi trường server TextFix).
2. Tìm trong mã ai-toolkit tên arch / đường nạp cho `FLUX.2-klein-base-4B` (README có liệt kê; examples chưa có cấu hình klein):
   `grep -rn -i "klein\|flux2" ~/work/ai-toolkit --include=*.py | head`. Trỏ model tới trọng số base trong `~/persistent-data`.
3. Đọc đoạn mã áp `mask_path` vào loss để chắc nghĩa của `mask_min_value` (theo chú thích: trắng loss cao hơn đen). Kiểm
   `inverted_mask_prior` (dự đoán của model không LoRA làm đích cho vùng ngoài mặt nạ) có chạy với klein không.
4. Học thuộc 50 ảnh, 500 bước, rank 16: loss phải giảm; ảnh mẫu trên chính prompt huấn luyện phải hiện chữ đúng hơn (học thuộc).

**Probe -> cửa qua**
- Chạy hết không lỗi, VRAM vừa một A30 (ghi lại GB và giây/bước).
- Học thuộc thấy rõ trên ảnh mẫu (chứng minh dữ liệu + caption + mặt nạ đi đúng đường).
- LoRA nạp được vào **distill 4B** trong mã vẽ của TextFix (`textfix/draw/flux.py` dùng mã BFL `flux2/`; cần thêm đường nạp LoRA)
  hoặc bằng diffusers `Flux2KleinPipeline` -- chọn một đường dùng chung cho đo và cho prod sau này.
- Không chạy được klein trên ai-toolkit -> phương án dự phòng: script LoRA tối giản bằng diffusers (theo hướng dẫn của BFL / HF),
  tự cộng trọng số mặt nạ vào loss.

---

## Chặng 4 -- Pilot: LoRA có học được chữ Việt không

**Việc**
1. Huấn luyện trên 1.000-2.000 cặp sạch: rank 32 và 64, lr 1e-4 (BFL gợi ý 8e-5 đến 1e-4), 1024 px (bucket theo tỉ lệ), mặt nạ
   chữ trắng / nền đen với `mask_min_value` sao cho vùng chữ nặng gấp ~5-10 lần; bật `inverted_mask_prior` ở một lượt để so.
   Lưu checkpoint mỗi 500 bước, tới ~3.000 bước.
2. Đo trên tập eval: **distill + LoRA (4 bước)** so với distill, cùng seed. BẢNG SO SÁNH mỗi prompt [TextFix (chữ đúng) | distill
   | distill + LoRA], danh sách câu đúng dưới mỗi hàng; vài trang JPG để người dùng xem bằng mắt. Thêm 20 prompt KHÔNG chữ.
   Base chỉ chạy để CHẨN ĐOÁN khi distill + LoRA không tốt lên: `--variants base --base-steps 28 --limit 16` (2 card, ~10-15 phút).
3. Đo ảnh chung: so cặp ảnh cùng seed trước / sau ở phần ngoài chữ (xem tay 50 cặp phóng to; thêm điểm thẩm mỹ tự động nếu có);
   kiểm prompt KHÔNG chữ (ảnh sản phẩm, phong cảnh) có tự sinh chữ lạ không.
4. Đo đầu-cuối: chạy TextFix (4B distill + LoRA) trên bench quick1: tỉ lệ dòng nháp được GIỮ (đúng sẵn), số ca nháp thiếu dòng,
   thời gian (LoRA gộp vào trọng số thì không chậm thêm).

**Probe -> cửa qua**
- distill + LoRA: nhìn bảng so sánh thấy chữ đúng hơn rõ so với mốc (đếm tay trên 50 prompt: số câu cỡ tiêu đề / giá viết đúng
  hoàn toàn, trước vs sau), lỗi dấu ít đi. Người dùng xem và quyết.
- Ảnh chung không xấu đi thấy được; không sinh chữ lạ trên prompt không chữ.
- Rẽ nhánh:
  - distill + LoRA tốt -> chặng 5 (không cần chạy base).
  - distill + LoRA không tốt -> chạy base chẩn đoán: base + LoRA tốt -> chặng 5B (LoRA học được, không chuyển sang distill).
  - base + LoRA cũng không tốt (LoRA không học được) -> thử một lượt: rank cao hơn, nhiều bước hơn, dữ liệu gấp đôi. Vẫn không -> dừng, báo người dùng
    (kết quả "không khả thi với LoRA" cũng là kết quả có giá trị).

---

## Chặng 5 -- Mở rộng dữ liệu và các kiểu đầu vào

FLUX chỉ thấy 3 kiểu đầu vào: (1) prompt_en -- tạo mới; (2) + ảnh sản phẩm; (3) + poster cũ -- sửa. Pilot chỉ có kiểu 1.

**Việc**
1. Sinh 5.000-10.000 cặp, tỉ lệ gần prod (~50 / 20 / 30, chỉnh theo thống kê thật).
2. Kiểu 2-3 cần huấn luyện có ảnh tham chiếu. ai-toolkit có `control_path` nhưng KHÔNG liệt kê klein trong mục edit -> kiểm mã;
   nếu không được: tìm trainer khác cho klein edit (hướng dẫn huấn luyện của BFL, diffusers, các trainer cộng đồng) -- KHÔNG dùng
   dịch vụ huấn luyện bên ngoài (dữ liệu không ra khỏi hạ tầng nội bộ).
3. Cặp kiểu 3: đầu vào = poster cũ (chữ đúng) + lời sửa, đích = poster mới qua TextFix -> dạy "chép chữ đúng từ ảnh cũ, chỉ đổi
   chỗ được yêu cầu" (hiện FLUX sửa ảnh luôn phá chữ Việt).

**Probe -> cửa qua:** đo TÁCH RIÊNG từng kiểu; LoRA không được làm kiểu nào xấu đi so với mốc (đặc biệt bám ảnh tham chiếu khi sửa).

## Chặng 5B -- (chỉ khi LoRA base không chuyển sang distill) Huấn luyện thẳng distill 4 bước

- Tái tạo đúng nhiễu từ seed (`textfix/draw/flux.py` `_sample`: `torch.Generator(device).manual_seed(seed)`,
  `randn(1, 128, h//16, w//16, bf16)` -> phụ thuộc thiết bị: dữ liệu phải sinh trên cùng loại card). **Probe trước:** tái tạo
  nhiễu rồi chạy lại 4 bước phải ra đúng nháp cũ (sai khác 0).
- Ở mỗi bước k của 4 bước, trạng thái x_t theo đúng lịch suy luận; đích v* = (x_t - x_sửa) / t (quy ước x_t = (1-t)x0 + t·ε);
  loss MSE có mặt nạ; LoRA trên distill. Cửa qua như chặng 4.

## Chặng 6 -- (tuỳ chọn) Học theo cặp ưa thích

- Diffusion-DPO bản flow-matching: cặp cùng prompt, cùng nhiễu và t; thắng = poster TextFix, thua = nháp. Dùng biến thể có chốt
  an toàn (Diffusion-SDPO: DPO thường dễ kéo model rectified-flow lớn ra xa model gốc) và tính theo VÙNG CHỮ (GlyphPrinter).
- Chỉ làm khi chặng 4 đã có LoRA tốt; đo như chặng 4.
- Flow-GRPO (phần thưởng OCR) ghi nhận là hướng mạnh nhất về số (SD3.5-M: chữ đúng 59% -> 92%) nhưng vượt sức 2x A30; chỉ cân
  nhắc nếu có thêm GPU.

## Chặng 7 -- Đưa vào sản phẩm (người dùng quyết)

- So **4B distill + LoRA** với **9B (prod hiện tại)** trên bench chuẩn: chữ đúng, thẩm mỹ, thời gian. 4B là giấy phép Apache 2.0.
- Gộp LoRA vào trọng số (không chậm thêm). Máy FLUX ở xa (khuôn API BFL) phải hỗ trợ trọng số mới -> phối hợp đội vận hành.
- Golden + quick1 + panels1 phải qua như mọi bản phát hành.

---

## Ước lượng thời gian (pilot tới hết chặng 4)

| Chặng | Công | Máy |
|---|---|---|
| 0 Thước đo | 1-2 ngày | CPU/GPU nhẹ |
| 1 Mốc | 0,5 ngày | 1 card |
| 2 Dữ liệu (probe 300 + 1-2k) | 1-2 ngày (chạy đêm) | 2 card + LLM/VLM |
| 3 Smoke test | 0,5-1 ngày | 1 card |
| 4 Pilot | 1-2 ngày | 1 card |

Tổng ~1 tuần làm việc. Trong lúc huấn luyện, server TextFix có thể phải nhường một card -- hỏi người dùng trước.

## Nguồn

- BFL / HF, huấn luyện LoRA klein (base -> distill): https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora ,
  https://docs.bfl.ai/flux_2/flux2_klein_training
- ai-toolkit (mask_path, mask_min_value, inverted_mask_prior, control_path; klein base 4B/9B trong README):
  https://github.com/ostris/ai-toolkit
- Flow-GRPO (OCR 59% -> 92% trên SD3.5-M): https://arxiv.org/html/2505.05470v5 ; tài nguyên tham khảo verl-omni:
  https://verl-omni.readthedocs.io/en/latest/algo/flowgrpo.html
- Diffusion-SDPO (DPO có chốt an toàn, FLUX): https://arxiv.org/pdf/2511.03317
- GlyphPrinter (DPO theo vùng chữ; OCR không nhạy lỗi nét): https://arxiv.org/pdf/2603.15616
- LeX-Art (fine-tune FLUX bằng 10k ảnh chữ tổng hợp): https://arxiv.org/pdf/2503.21749
- EasyText (LoRA đa ngôn ngữ; cảnh báo giảm chất lượng ảnh chung): https://arxiv.org/html/2505.24417v1
- PP-OCRv5 đa ngôn ngữ (latin có tiếng Việt): https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html
- Bộ mã hoá chữ klein 4B = Qwen3-4B: https://docs.api.nvidia.com/nim/reference/black-forest-labs-flux_2-klein-4b

## Nhật ký (ghi số đo từng chặng)

| Ngày | Chặng | Số đo | Qua cửa? | Ghi chú |
|---|---|---|---|---|
