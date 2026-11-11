# Lộ trình: TextFix chỉ sửa chữ, Hi-SAM + LaMa thay bản xoá FLUX

> Tên mô-đun trong tệp này là tên cũ (trước khi xếp lại theo bước, 15/10); bảng tên cũ → chỗ mới ở cuối README.

Bắt đầu 14/10/2026. Tệp này là nguồn duy nhất cho câu hỏi "đang ở đâu, còn gì". Mỗi việc xong thì tick, kèm một dòng số đo
nếu có. Đường cũ (bản xoá FLUX) nằm ở nhánh git `legacy-flux-erase`, không so sánh, không phát triển.

## Tích hợp chatbot -- API v1 (11/11)

Đặc tả của router: một POST đồng bộ `/api/v1/images/generations` ({prompt, images_base64, aspect_ratio} -> ảnh
base64 / PNG), khuôn lỗi riêng. Tài liệu: docs/API_V1.md. Đã lên prod bản `v0.0.5` (11/11 chiều).
- [x] Endpoint v1 + mã lỗi theo đặc tả; hạn 55 s tính cả tải ảnh; 503 khi bận / chưa sẵn sàng; tokens cố định.
- [x] Ảnh: chỉ `images_base64` trong thân (12/11 đội chatbot chốt; đã gỡ `image_base64`, `image_urls` và mã tải ảnh theo link).
- [x] Lời dặn LLM bản chỉ-prompt (bỏ luật form / variants; 1362 -> 1093 từ). chat1 trên notebook: lỗi bộ kiểm 12 -> 10, xem mắt
  ~13 -> ~17 / 24 đạt, câu thiếu 0 -> 5 (đều do FLUX không vẽ chỗ / chữ trên nhãn sản phẩm).
- [x] Máy chủ làm nóng trước khi báo sẵn sàng; chỉ nhận việc khi làm nóng xong (ba yêu cầu đầu sau khi bật từng quá hạn).
- [x] LLM / VLM: chờ 40 s mỗi lần (trước 300), trần 4096 token, log đoạn cuối khi sinh lặp (một lời gọi VLM từng mất 306 s).
- [x] Docker: playwright ghim theo Chromium trong image; build.sh thử mở Chromium; PLAYWRIGHT_CHROME_PATH của notebook không đè.
- Đo trên prod (11/11, 4 ca cùng lúc): 22-35 s mỗi poster; thân 13 MB qua cổng được.
Việc đợt 2 (chưa làm):
- [~] Sửa poster (mục "Sửa poster" bên dưới).
- [~] Khuôn JSON VLM: đã bỏ `why` / `notes` (12/11); còn trần độ dài chuỗi, màu dạng #hex -- chờ log "chạm trần" từ prod
  và kiểm vLLM hỗ trợ maxLength / pattern.
- [~] Chữ AI viết thêm: nút / câu sai ngành, Viết Hoa Mọi Từ, mất tên quán khi khách gõ không dấu. 12/11: đã dặn LLM + code
  sửa Viết Hoa Từng Chữ của câu phụ trợ; chưa đo lại trên chat1.

### Sửa poster (11/11, chưa đo)
Chatbot gửi lại đúng file ảnh đã nhận (lưu S3 không nén) + lời sửa, không thêm trường. Người dùng chốt: mọi yêu cầu sửa chạy
lại cả dây chuyền (FLUX sửa chính poster cũ -> nháp -> TextFix), không đường tắt sửa HTML; ảnh lạ thì VLM đọc chữ rồi sửa.
- [x] Thông tin poster giấu trong PNG (textfix/meta.py: iTXt "textfix" -- chữ, prompt, khung, có ảnh sản phẩm, chuỗi sửa).
  Máy chủ không lưu gì thêm.
- [x] Nhận diện: ảnh có khối thông tin = poster cần sửa (ở vị trí nào cũng được), ảnh còn lại = ảnh sản phẩm mới.
- [x] LLM sửa (textfix/edit/plan.py): danh sách chữ đầy đủ sau sửa + prompt sửa cho FLUX; code kiểm: câu đổi mang số lạ thì
  trả lại câu cũ, câu LLM chép sót thì trả lại đúng chỗ, câu mới phải có số trong lời sửa.
- [x] Ảnh lạ: LLM brief nhận intent "edit" khi có ảnh kèm; Hi-SAM + dòng + VLM chép chữ (textfix/edit/read.py).
- [x] Khung: giữ khung cũ, bỏ qua `aspect_ratio` của yêu cầu; đổi khổ khi lời sửa nói.
- [x] Lượt đo 1 (Qwen3.5-35B + FLUX 9B): 49/49 lượt 200, sửa trung vị 21 s (ảnh lạ 27-30 s); nhận đúng poster của mình 21/21,
  ảnh lạ 4/4 poster (2 ảnh chụp "thêm chữ vào ảnh" thành poster mới -- nhẹ, chữ đúng); câu không sửa giữ nguyên văn mọi ca;
  ảnh lạ: ký tự + số đọc đúng, nhưng VLM gộp hotline / web / địa chỉ thành một câu.
  Sửa sau lượt 1: VLM đọc chỉ gộp câu bị ngắt giữa chừng; "split" mà mẩu nào cũng là nguyên câu = viết lặp (e12, không cắt câu
  qua hai chỗ); mẩu "dòng OCR sót" nằm trong dòng có sẵn không thành dòng mới (e14); khối thông tin ghi số chỗ câu xuất hiện
  (shown), LLM sửa tả "cả hai nút" để FLUX không vẽ nút trống (e03). Golden 24/24 không đổi.
  Còn: sửa nối tiếp 3 lần xuống cấp dần (e13); "chữ to hơn" gần như không đổi (cỡ theo nháp); chữ chìm nền sau đổi bối cảnh (e07).
- [ ] Đo lại trên notebook: `bash scripts/v1_rounds.sh --edit` (bench/edit1.json, 24 ca: đổi số, sửa tên, bỏ / thêm dòng, màu, vật,
  bối cảnh, đổi khổ, thay sản phẩm, sửa nối tiếp 3 lần, 4 ảnh lạ đã bỏ thông tin, 2 ảnh chụp, ảnh sản phẩm + lời tạo mới).
  Chấm: câu không sửa giữ nguyên văn, bố cục / phong cách giữ, không xuống cấp qua 3 lần sửa, dưới 55 s.
- [ ] Ảnh lạ: đọc đúng mọi số, >= 95% câu nguyên văn trên 4 ảnh đã bỏ thông tin. Không đạt thì báo trước khi bật trên prod.

### Gọn mã + tốc độ (12/11, chưa deploy)
Bản đầu ~20 s mỗi ảnh. Đo 11 lượt prod 10/11 (một yêu cầu một lúc): LLM 3,5 | FLUX 2,5 | Hi-SAM 0,2 | đo dòng 1,2 | LaMa 1,4 |
VLM 5,6 | dựng 3,6 s = 17 s, còn ~3 s là chờ / ảnh debug.
- [x] Chỉ còn API v1: bỏ API demo, UI, form, nhiều hướng thiết kế, zip debug; /api/health không lộ địa chỉ nội bộ.
- [x] Header X-Run-Id, X-Timing (đo từng bước qua đúng đường chatbot gọi, cả trên prod).
- [x] Ảnh ra không đổi (golden 24/24): LaMa song song VLM, ảnh debug sau khi trả, nền dựng mã hoá một lần, PNG nén mức 3,
  làm nóng tokenizer FLUX (~20 s ở yêu cầu đầu).
- [x] Lời dặn LLM chỉ cho lời chat; prompt FLUX 80-130 từ (trước trung vị 143); cấm tả font / cỡ / vị trí.
- [x] LLM / VLM: tham số sinh đặt rõ theo thẻ Qwen3.8-27B (presence_penalty 0: phải chép nguyên văn); số token mỗi lời gọi
  vào `calls` của plan.json.
- [x] VLM: bỏ `why` / `notes` (~20% token sinh ra); dòng có câu khách thì `text` / `role` null; ảnh gửi đi cạnh dài 768
  (TEXTFIX_VLM_MAX_SIDE), nhãn L# vẽ to theo tỉ lệ thu. Ước 5,6 s -> ~4 s, chưa đo.
- [x] Dựng: chụp ảnh qua CDP optimizeForSpeed (cùng điểm ảnh, ~4 lần nhanh hơn); nền dưới chữ = nền dựng (bỏ một lượt chụp;
  hết báo oan chữ dải màu chìm nền); settle dùng lại mặt nạ của lượt gỡ khi không dòng nào giữ. 24 lượt đã lưu: dựng 2,25 s
  -> 0,55 s trung vị; golden 23/24 khớp, ca còn lại chỉ bớt một lỗi báo oan (poster trùng).
- [ ] Đo lại trên notebook + prod (X-Timing, calls); OCR (tranh CPU khi tải).
- [ ] Đội hạ tầng: speculative decoding (MTP) + prefix caching cho Qwen3.8-27B; FLUX.2-klein 4B (9B là giấy phép phi thương mại).

### Tune thẩm mỹ lượt 1 (12/11, chat1 trên notebook: Qwen3.5-35B-A3B + FLUX 4B)
Mốc: 24/24 trả 200, tổng trung vị 15,9 s; lỗi bộ kiểm 7 (too_small 6, content_overlap 1), câu thiếu 0; xem mắt ~18/24 đạt.
- [x] [code] Câu gán cho các dòng XẾP CHỒNG lệch cỡ: tổng chữ OCR / độ dài câu <= 1,5 (assign.SPLIT_OCR; chia dòng 0,7-1,38,
  FLUX vẽ lặp 1,68-2,0 trên mọi tập đã lưu) -> một câu chia dòng, mỗi dòng giữ cỡ nháp. c10: câu khai trương in 3 lần -> tiêu đề
  lớn như nháp; dựng lại 24 ca: chỉ c10 đổi, lỗi 7 -> 6. **đo**
- [ ] [prompt LLM] Nút hành động theo ngành: "Đặt lịch" chỉ cho dịch vụ có hẹn (9/24 poster dùng "Đặt lịch ngay", có cửa hàng
  điện thoại, tiệm hoa). **chờ server**
- Chính sách (người dùng 12/11): FLUX vẽ cùng câu ở hai chỗ xa nhau -> viết lặp theo nháp, không xoá (lộ vết LaMa).
- Theo dõi (một ca, chưa đủ để đặt luật): câu kéo của khách thành chữ in ("Quảng cáo phòng gym", c13); VLM nhồi hai câu AI vào
  một dòng nháp chỉ đủ một câu (c13); Qwen3.5 thêm dấu sai cho lời gõ không dấu (c15 "Tay Trang Răng"); dấu nháy lạc chưa gỡ
  (c15); hai dòng chữ hoa có dấu đứng khít (c11 "TIỆM / CẮT"); câu feedback dài trên tờ giấy nghiêng co 0,54 (c14).
- [ ] Prompt LLM / VLM ra kho có phiên bản (đội chatbot đề nghị), kiểm khi nạp, ghi phiên bản vào plan.json.
- [ ] tokens thật; chống chạy trùng (request_id); dùng nhiều ảnh tham chiếu; đọc lời dặn font / vị trí trong prompt đã làm giàu.

## Đã chốt (người dùng, 14/10)

- TextFix **chỉ sửa chữ**. Vỏ, nút, icon to, **icon và dấu đầu dòng**, hoạ tiết là điểm ảnh của nháp. Nháp vẽ xấu thì chịu.
- Không bỏ dòng nào của nháp. Nháp vẽ thừa hay lặp thì VLM viết cho đủ.
- Bỏ hẳn bản xoá FLUX trên `main`. Hi-SAM cho biết chữ ở đâu, LaMa vẽ lại nền dưới chữ.
- Cụm nét Hi-SAM thấy mà OCR không trả dòng: thành dòng, **VLM quyết** giữ / viết lại / xoá. VLM không nhắc tới thì giữ.
- Chữ neon quầng rộng và chữ rất nhỏ: cho chạy đã, xấu thì sửa sau.
- Thiếu tệp Hi-SAM hoặc LaMa: báo lỗi rõ và dừng.
- LaMa chạy **một lần** mỗi poster (gỡ mọi dòng). Dòng VLM chọn giữ được trả lại điểm ảnh nháp bằng mặt nạ, không gỡ lại.

## Dây chuyền đích

| # | Bước | Trạm | Ghi chú |
|---|---|---|---|
| 1 | LLM viết đề | điều phối | không đổi |
| 2 | FLUX vẽ nháp | vẽ (GPU) | không đổi |
| 3 | Hi-SAM tìm nét chữ | vẽ (GPU) | thay "FLUX xoá + dọn chữ sót" (6,5 s -> ~0,2 s) |
| 4 | Đo dòng: OCR + dòng cong + cụm nét chưa ai nhận | thiết kế | không còn ô vỏ, ô chi tiết, dấu đầu dòng |
| 5 | Gỡ mọi dòng: mặt nạ nét + quầng, LaMa lấp | thiết kế | một lần LaMa |
| 6 | VLM điền chữ theo dòng: viết / giữ / xoá | thiết kế | |
| 7 | Mã rà lệnh: chặn chữ, lấp dòng bị bỏ, tách lệnh gộp sai | thiết kế | |
| 8 | Trả điểm ảnh nháp cho dòng được giữ | thiết kế | chỉ tính lại mặt nạ |
| 9 | Dựng chữ, kiểm, tự sửa lỗi nặng | thiết kế | |

## Chặng A -- Hi-SAM + LaMa thay bản xoá (đang làm)

Mã
- [x] Nhánh `legacy-flux-erase` giữ đường cũ (đã push).
- [x] `textfix/hisam.py`: nạp một lần mỗi thiết bị, fp16, đọc / ghi mặt nạ. *(chạy CPU tại máy; chưa chạy trong máy chủ)*
- [x] `lonly.erase_mask` + `lonly.erase`: nét Hi-SAM theo dòng, quầng (dải sát nét, lan đuôi nhạt trên nền phẳng), LaMa. Bỏ đường luật dự phòng. *(tại máy)*
- [x] `slots.build(draft, ink)`: không cần bản xoá; cụm nét chưa ai nhận thành dòng `orphan`. *(tại máy)*
- [x] Gỡ dấu đầu dòng khỏi `main`. Icon trước dòng để nguyên, mã gỡ icon khỏi lệnh VLM (`lonly.marks_rule`). Riêng chấm tròn "•" là ký tự: Hi-SAM gỡ nó cùng chữ nên dòng mới viết lại "• " (`slots._lead_dots`).
- [x] `lonly.erase` (LaMa một lần, gỡ mọi dòng) + `lonly.settle` (trả điểm ảnh nháp cho dòng giữ, chỉ tính lại mặt nạ).
- [x] `lonly.fill_gaps`: dòng `orphan` VLM không nhắc -> giữ; VLM ghi xoá -> xoá.
- [x] Rào tách lệnh VLM gộp các dòng không xếp chồng (`lonly.split_far`; thử trên ca bốn thẻ: tách đúng 4 lệnh).
- [x] Lời dặn VLM (`SYSTEM_L`): icon không phải việc của VLM, chấm tròn, dòng `orphan` (keep / text / skip). *(chưa chạy VLM thật)*
- [x] `engine.draw` / `engine.design`: bỏ xoá FLUX và dọn chữ sót, truyền mặt nạ nét. *(chạy trọn trạm thiết kế tại máy với VLM giả)*
- [x] `server/app.py`: nạp Hi-SAM theo card, kiểm tệp lúc khởi động, lưu `ink.png` / `ink.jpg` / `base.jpg`, bỏ `plate*`. **(CHƯA CHẠY LẦN NÀO)**
- [x] `server/ui.html`: bỏ hai ô bản xoá; bốn ô xem ① Hi-SAM nét chữ thô (`hisam.jpg`), ② mặt nạ gỡ (`ink.jpg`), ③ LaMa nền sạch VLM nhìn (`clean.jpg`), ④ nền dựng (`base.jpg`); bảng "Gỡ chữ" (thời gian Hi-SAM / LaMa, % nét, % mặt nạ, cụm nét lạ VLM quyết gì); bỏ lớp S / I; trang không bị trình duyệt giữ bản cũ. *(dựng thử tại máy bằng Chromium với một lượt giả)*
- [x] UI debug "Chữ khó (typography)": mỗi dòng cong / dọc / nghiêng / dựng thẳng / OCR sót / cụm nét lạ / chấm đầu dòng / khung dời khỏi icon / chữ in trên sản phẩm -- mã nhận ra bằng bằng chứng gì (`l["typo"]`), lệnh VLM / mã, dựng thế nào (`plan["render"]`: xoay, đường cong, cỡ, co, nén); khung xem dòng vẽ dòng nghiêng theo đa giác thật.
- [x] `steps.jpg` năm cột (nháp, dòng, nét chữ, nền, poster); zip báo cáo mang `ink.png` thay bản xoá. **(batch_report chưa chạy)**
- [ ] Tờ ảnh ghép và trang HTML của `batch_report`: thêm cột nét chữ / nền.
- [x] `audit.flags`: cờ cho dòng `orphan`, dòng gỡ theo lõi khung, lệnh gộp bị tách.
- [x] `scripts/erase_bench.py` theo đường mới. *(tuỳ chọn `--recheck` cho số vô lý > 100%: chưa dùng được)*
- [x] `scripts/replay.py` theo đường mới (dựng lại tại máy với lệnh VLM đã lưu).
- [x] Xoá khỏi `main`: `cleanup.py`, `overlay.py`, `textmask.py`, `flux.erase`, 12 script chỉ đường cũ dùng.
- [x] Dọn mã trước chặng B (14/10): bỏ khỏi `slots.py` lời dặn cũ `SYSTEM` / `SYSTEM_V1` / `REVIEW` / `JUDGE` (có danh mục icon), `describe`, vòng VLM duyệt (`review`, `judge`, `salvage`, `accept`; nhánh cũ giữ), các luật ô vỏ / ô chi tiết / icon / hàng sao của `validate`; bộ dựng chỉ còn lệnh chữ; M chỉ còn dòng L; bỏ `flux.edit`, `geo.band_arc`, `ship.pack_dir`, `scripts/probe.py`. `slots.py` 2.021 -> ~1.150 dòng. Lời dặn VLM: 0 chữ "icon". Kiểm: 12 poster dựng lại trước / sau giống nhau (11 giống từng điểm ảnh, 1 lệch một dòng chấm, nhìn như nhau); chạy trọn trạm thiết kế với VLM giả; dựng UI.
- [x] Bỏ bộ icon: `textfix/icons.py`, `assets/icons/` (1.544 SVG), `scripts/fetch_icons.py`, phần `<i-icon>` / `<i-stars>` của `render.expand`, mục đếm icon trong `check_server.py`. Bộ dựng gỡ thẻ icon / sao / svg khỏi chữ và ghi log thay vì vẽ.
- [x] `docs/SERVER.md`, `README.md`, mục 14/10 trong `docs/TUNING.md`, `scripts/check_server.py` kiểm tệp Hi-SAM / LaMa.
- [x] Xoá `docs/DESIGN.md`, `PLAN_BLOCKS.md`, `SLOTS_GT.md`, `tools/annotate.html` (tả / phục vụ đường cũ; còn trong git và các nhánh legacy). Dữ liệu `bench/slots_gt/` giữ nguyên.

Đo tại máy (trước khi xin chạy máy chủ)
- [x] Bước 4 trên 30 nháp accept2: 241 / 242 dòng cũ còn, 3 dòng cong còn đủ, thêm 4 cụm nét (chữ vòng trên huy hiệu trà).
- [x] Bước 4 trên cả 65 nháp (accept2 + mix1): 531 / 532 dòng cũ còn, 7 / 7 dòng cong, 11 cụm nét ở 4 poster, 4 dòng có chấm.
- [ ] Bước 5: mới xem mắt 10 nháp khó với luật quầng cuối (OCR đọc lại nền: 0 ký tự sót ở cả 10); chưa chạy đủ hai tập.
- [ ] Dựng lại với lệnh VLM đã lưu: 30 poster accept2 với luật quầng đầu (xem 8, không vỡ); chưa dựng đủ 65 với luật cuối -- người dùng chọn chạy thẳng máy chủ.

Máy chủ
- [x] Lượt `accept2` trên main 3ac2285 (zip 1014_040431, giải nén ở `output/accept2h/`): đủ 30, không poster nào lỗi. Trung vị
  18,5 s / poster không tính LLM (nét chữ 0,2 s, gỡ chữ 1,7 s; trước ~29 s). 248 dòng (trước 242), 3 / 3 dòng cong. VLM không viết
  icon nào. Không hết VRAM.
- [x] Nghiệm thu, xem 30 poster: gỡ chữ sạch, vỏ / nút / con dấu / huy hiệu còn nguyên. BA LỖI DO CHẶNG A, đã vá (tại máy):
  (1) q03 MẤT LOGO Coca-Cola trên lon: OCR đọc logo thành dòng "eo.mg", VLM không nhắc, mã xoá trống -> poster có ảnh sản phẩm thì
  dòng không ai viết được GIỮ; (2) q02 ba chiếc lá trước dòng bị gặm: quầng nay phải dính vào nét, cỡ dòng không vượt 2,2 lần
  cao nét, dấu câu chỉ gỡ ở giữa chữ, khung dòng dời khỏi icon đứng trước; (3) h04 "NGÀY • HỘI": chấm đầu dòng cần cả OCR
  đọc ra ký tự không phải chữ. Thêm: ngưỡng nền phẳng 4 -> 8 (bóng chữ 3D e04 lúc hết lúc không).
- [x] ~~Một lượt accept2 nữa với bản vá~~ -- người dùng bỏ (14/10); bản vá kiểm tại máy trên đúng các ca lỗi.
- [ ] VRAM đỉnh khi Hi-SAM + LaMa + FLUX chung card (lượt này không lỗi, chưa có số).

## Chặng B -- VLM ghép câu, code chép chữ và dựng theo nháp (đang làm)

Chốt 14/10 (thay hướng "khối, bậc cỡ, chỗ trống thật"): nhóm lấy từ VAI câu khách trong brief, không đoán từ ảnh, không để VLM
khai; không cắt câu khách, chữ dài thì co; nháp thiếu câu thì chịu; các bước luôn theo thứ tự khách. Cơ sở: probe 30 poster
(`output/probe_role/`): chữ đúng câu khách 30/30, 19 tập cùng vai dựng đều. Đường cũ (VLM viết html, group): nhánh `legacy-ops-html`.
- [x] VLM chỉ trả `assign` (câu <-> dòng, `split`, dòng lấp, keep / skip, `bullet`), `styles` theo vai, `accent`, `missing` (`lonly.SYSTEM_L`, `fill.parse`).
- [x] `textfix/fill.py`: chép nguyên văn, một câu nhiều chỗ, dòng chưa ai ghép, chặn số bịa ở dòng lấp, thứ tự các bước (cờ `ordered` của brief).
- [x] Hình theo nháp: khung = nét Hi-SAM, cỡ = trung vị cỡ đo, màu theo dòng, lề theo mép nét, ngắt dòng theo tỉ lệ ký tự OCR; dòng khác cỡ / mẩu cùng hàng.
- [x] Tập cùng vai (cỡ nháp đôi một lệch <= 1,25): một cỡ, một màu, một lề, một mức nén theo mục chật nhất (`render.render_units`).
- [x] Bỏ: html VLM, `_guard`, `validate`, `marks_rule`, `split_far`, `fix_ops`, nhóm / cân cỡ / căn lề của render, `_pack_lists`, `_list_order`.
- [x] UI debug: đơn vị chữ (màu = tập), bảng câu ↔ dòng, cỡ / co từng phần tử; `scripts/replay.py` dựng lại plan cũ qua `fill.from_ops`.
- [x] Dựng lại tại máy 30 accept2 + 35 mix1c (lệnh VLM cũ qua `fill.from_ops`): 0 tập lệch cỡ, chữ đúng nguyên văn; câu thiếu đều đã thiếu trong lệnh cũ.
- [x] VLM không nhận chữ OCR (người dùng 14/10); số bước đặt lại theo thứ tự khách; `engine.design` chạy trơn với VLM giả.
- [x] Sau mix1 / dev2 (14/10 tối): ngắt dòng theo bề rộng nét + font thật, không ép câu thiếu, giữ màu nháp, brief không làm mất
  câu khách (ghép chữ khách / chép sai vài chữ), số trước hàng viết lại, dòng xếp chồng một câu gộp khối, chữ OCR gửi lại VLM làm neo.
- [ ] Hi-SAM phân cấp (từ / dòng) chia dòng thay OCR: tick sát chữ, hai cột gộp, một khung hai hàng (probe 65 nháp: khớp OCR ~96%,
  thắng khung hai hàng + nhóm đoạn, thua chữ chạy vòng) -- thay cụm luật vá OCR của `slots.py` (`_split_gaps`, `_trim_lead`,
  `_lead_dots`, `TILT_*`, `_stacked`, `_orphans`, `fill.TALL`). Sau deadline.
- [ ] CHỮ TRÊN SẢN PHẨM KHÁCH TẢI bị VLM viết đè (dev2 d13: "Chai 650ml" đè dòng "For Damaged Hair" trên nhãn Dove; luật dò theo
  chữ OCR không bắt vì OCR không đọc ra chữ trùng ảnh khách). Hướng: định vị sản phẩm trên nháp bằng HÌNH HỌC (SIFT + RANSAC
  với ảnh khách) -> dòng trong vùng sản phẩm không bao giờ gỡ / viết. Probe 14 nháp (14/10): FLUX chép sát thì chữ nhãn khớp
  15-76 điểm, chữ poster ~0; FLUX vẽ lại sản phẩm (d24v1, d31) thì 0-3; một tiêu đề báo nhầm 16 điểm -> CHƯA có ngưỡng tách sạch,
  cần vùng sản phẩm chắc + tập gán nhãn trước khi dùng. Tạm thời: dò theo chữ OCR + VLM keep; VLM ghép câu khách thì VLM thắng.
- [ ] Ngưỡng chặng B chọn theo vài ca, chưa đo trên tập luyện: `DIFF_RGB` (ac10), `OUTLINE_HEAD / OUTLINE_ABS` (dev3 g12, phá hiệu
  ứng neon dev2 d06), `brief.SNAP` (d03), `PIECE_SIM`. `SET_RATIO 1,25` đã đo lại trên mix1 + dev2: 129 / 150 cặp cùng vai <= 1,2,
  khe thưa 1,2-1,4 -> giữ.

## Thiết kế lại bước 3-4-5 -- kế hoạch DUYỆT 15/10 (đang làm)

Lý do: bước 4 dồn luật vá theo kiểu sai của OCR (OCR quyết hình học dòng); nhiều lỗi cũ có gốc ở bước 1-2 (dòng lặp do LLM, prompt
bị cắt 512 token) đã chặn từ gốc. Nguyên tắc: nét Hi-SAM = hình học, vùng DBNet (`ocr.regions`) = dòng nào, bộ đọc OCR chỉ đọc,
VLM = nghĩa, brief = nội dung; một biểu diễn dòng {path, h_cap, h_x, ink, rows} cho mọi hình dạng; ngưỡng theo cao chữ.
Người dùng chốt: không định vị vùng sản phẩm; chữ FLUX vẽ trên vật GIỮ (kể cả giun); TIN VLM khi xoá dòng, dòng bỏ sót hỏi lại
riêng rồi GIỮ; CHO NỚI chữ vào nền trống cùng màu, không vượt vỏ / dòng khác.
- [ ] Chặng 0: lượt máy chủ dev2 + mix1 với bước 1-2 mới (nháp mới + mốc); bộ đáp án chia dòng `bench/lines_gt` (Claude gán);
  `scripts/lines_bench.py` (chỉ số dòng + dựng lại bằng câu trả lời VLM đã lưu, đổi id theo chồng lấn nét).
- [~] Chặng 1 -- ĐỔI HƯỚNG 15/10 (người dùng chọn "lai"): chia dòng thuần bằng nét (vùng DBNet + mảnh Hi-SAM, tách hàng / mục)
  đạt F1 0,92 < 0,969 của OCR; mỗi luật thêm để gỡ một nhóm ca lại hỏng nhóm khác (chữ dọc bao bì / tiêu đề xếp chồng / cụm
  cùng hàng) -> dừng. Thiết kế lại: GIỮ cách OCR chia vùng dòng (`read_lines`, `_split_gaps`, dòng sót / vòm của `_geo_lines`,
  `_orphans`); mọi đại lượng HÌNH HỌC và SỐ ĐO lấy từ mảnh nét của chính dòng (`textfix/lines/`: ink, group, angle, measure);
  gỡ dần từng luật vá chia dòng, mỗi lần có số đo. Bản thuần nét cất ngoài repo.
  - [x] Hướng dòng hai nguồn (`lines.angle.orient`): nghiêng khi nét VÀ OCR cùng thấy >= 2 độ cùng chiều, góc theo nét; góc nét
    = trục chính + nắn thẳng theo MÉP CHỮ (đạo hàm hình chiếu: chân thò g / y không kéo lệch, "200g" / "Đặt ngay" 5 -> 0 độ);
    nét không đủ mảnh ("-50") -> OCR; dọc khi nét dốc >= 60 độ và khung cao hơn rộng (đọc hai chiều chọn chiều). Bỏ
    `TILT_MIN / SURE / PAIR / LONG`, `_line_angle`. Đòi hai nguồn lệch <= 5 / 8 độ thì tệ hơn (bỏ).
  - [x] Số đo từ nét kèm mỗi dòng (`l["ink"]`): h_cap, h_x, đường chân, màu lõi nét, độ dày, tương phản, số mảnh trang trí;
    bên dùng chuyển ở chặng 4 (cỡ theo font + câu khách, màu một nguồn) rồi bỏ `glyph`.
  - [x] Đầu dòng theo mảnh nét (`lines.lead`): mảnh đầu nhỏ, giữa chiều cao, cách chữ = trang trí (không thuộc chữ); khung bắt đầu
    ở nét chữ đầu khi thừa >= 0,6 cao chữ. Thay `_trim_lead`. `_lead_dots` + cờ `dot` + `bullet` của VLM gỡ ở chặng 2-3 (cần gỡ
    chữ theo mảnh nét để chấm ở lại là điểm ảnh nháp).
  - [x] Chữ nhiều cỡ (`lines.runs`): `segments` trong một dòng (từ lệch cỡ >= 1,6 lần; hoa / thường từng đoạn), `row` giữa các
    dòng (chung đường chân, sát nhau, lệch cỡ >= 1,6: d03 "Giảm | 30% | tất"). Chặng 4 rải câu theo đoạn / hàng, bỏ đoán `accent`.
  - [x] Bench: `lines_bench.py --source build` (chạy lại toàn bộ `slots.build`, đệm OCR ra đĩa), `--kind poster`; kiểu đường
    `bench/lines_gt` gán lại mù. Số (OCR chạy lại trên nháp JPG, cùng điều kiện với mã cũ ở HEAD):
    chữ poster F1 0,968 -> 0,966, tách 6 -> 6, gộp 9 -> 8, sai kiểu 8 -> 5; mọi chữ sai kiểu 19 -> 13. Hai dòng dư đều là mảnh
    cung trên vành con dấu (bánh pía, con dấu cà phê), số dòng khớp không đổi.
  - Giới hạn đã biết: từ 3 chữ có chân ("gẽl") cả hai nguồn cùng lệch; chữ vòng con dấu.
  - Giữ (đo, chưa thay được bằng nét): `_split_gaps`, `_stacked`, nhánh bảng của `_xy_order` (đo lại ở chặng 3 với VLM thật).
- [x] Chặng 2: `erase` dùng đúng mảnh nét của dòng.
  - [x] Bước 1 (nét): mảnh nét SỞ HỮU (`lines.ink.own`, cùng luật bước đo dòng: mỗi mảnh một dòng) + nét không ai sở hữu trong lõi
    dòng; bỏ `WHOLE`. Mảnh trang trí đầu dòng (`lines.lead`) và mảnh của dòng giữ không bao giờ vào mặt nạ.
  - [x] Gỡ `_lead_dots`, cờ `dot`, `bullet` của VLM + mục 6 lời dặn: chấm đầu dòng ở lại là điểm ảnh nháp.
  - [x] `erase_bench` (accept2 + mix1c, 65 nháp): ký tự sót 54 -> 49, poster còn sót 3 -> 3, mặt nạ TB 14,92 -> 14,74%, đổi
    ngoài vùng chữ 0 -> 0. Mặt nạ to hơn ở đúng một nháp (f02: chữ vòm vàng gỡ trọn mảnh + quầng; nền sạch).
  - [ ] Xuất quầng theo dòng làm hiệu ứng: làm ở chặng 4 (nơi dùng).
- [x] Chặng 3: VLM + số phận dòng (`textfix/compose/`; `lonly.py` -> `compose/vlm.py`).
  - [x] `compose/fate.py`: MỘT bảng số phận (`plan.json` "fates": text / keep / skip + lý do). Luật: VLM ghép câu -> text; chữ in
    trên sản phẩm -> keep (trừ khi ghép câu); VLM keep / skip -> theo VLM (TIN VLM khi xoá, bỏ giới hạn "chỉ xoá cụm nét lạ");
    VLM có nhắc mà không còn chữ -> skip; VLM không nhắc, hỏi lại vẫn không -> keep.
  - [x] `compose/vlm.reask`: hỏi lại RIÊNG các dòng bỏ sót (`fresh=True`), thay gọi lại cả bài khi phủ < 60% (`COVER_MIN`);
    gọi lại khi ghép rỗng cũng `fresh=True` (sửa A1: bộ đệm trả lại đúng câu cũ).
  - [x] Lời dặn: bỏ 3b ("cụm từ lấy từ câu khách khác" -> lặp chữ), bỏ "màu phổ biến nhất", mục 4 định nghĩa ba số phận.
  - [x] `tests/test_fate.py` (VLM giả). [ ] Kiểm thật: lượt máy chủ (máy cá nhân không có khoá VLM).
- [x] Chặng 4: bố cục (đang trong `fill.py`; tách `compose/layout.py` ở chặng 6).
  - [x] Cỡ theo FONT THẬT: `fonts.heights` (cao chữ hoa / thường đo trên hình "H" / "x"; danh mục chênh 0,59-0,86 / 0,30-0,73
    em, hằng chung cũ 0,72 / 0,53); `fill.letter_em` chọn hoa / thường theo CÂU KHÁCH; cỡ = `ink.h_cap` / letter_em.
  - [x] Nới khung vào nền trống (`fill.room`): theo lề, cột cùng màu nền trong khung (ΔE <= 15, >= 97% điểm), không vào vùng dòng
    khác, trần 1,25.
  - [x] Sàn tập (`render.SET_FLOOR` 0,8): mục phải co cỡ dưới 0,8 rời tập, tự co riêng (sửa A5). Gỡ 16/10 (chặng 7).
  - [x] Hàng nhiều cỡ: `_rows` không gộp dòng cùng hàng lệch cỡ >= 1,4 (d03 "Giảm" | "30%" | "tất" mỗi dòng cỡ của nó);
    `accent` của VLM bỏ "scale" (chỉ màu).
  - [x] Màu: hai trong ba nguồn đồng ý (`fill.pick_color`: đo khung / lõi nét / VLM theo vai); soát 14 dòng ba nguồn lệch: VLM sai
    6, lõi nét 7, đo khung 3, bỏ phiếu 1. Bỏ `DIFF_RGB` + màu VLM ghi đè.
  - [x] Dựng lại (`scripts/replay.py`, accept2h + mix1c, 65 nháp, câu trả lời VLM đã lưu; mốc = mã chặng 1-3):
    | | mốc | chặng 4-5 |
    |---|---|---|
    | cỡ chữ dựng / chữ nháp, trung vị | 0,86 | 0,95 |
    | trong +-15% cỡ nháp | 53% | 79% |
    | nhỏ hơn 0,8 cỡ nháp | 30% | 14% |
    | phần tử co < 0,6 | 13 | 13 |
    | poster còn lỗi bộ kiểm | 15 | 15 |
    Lỗi theo loại: too_small 17 -> 19 (tính so với cỡ khởi điểm, nay to và đúng nháp hơn), tràn 1 -> 3 (v01 hai nhãn trong vòng
    tròn tràn dọc 2 px -- còn sửa; v03 có từ trước), chồng 1 -> 3 (chữ đúng cỡ nháp, dòng phụ sát tiêu đề thư pháp t03, t02).
  - [ ] Hiệu ứng đối chiếu quầng đo được (B2), độ đậm / có chân đối chiếu nét (B1).
- [x] Chặng 5: render.
  - [x] Tương phản: đúng màu nháp -> giữ màu + quầng ở mọi cỡ; bỏ `OUTLINE_HEAD / OUTLINE_ABS` và nhánh viền tiêu đề.
  - [x] Gỡ `geo.runs / RUNS` (thay bằng `lines.rows`), nhãn bề rộng chữ trong bảng dòng gửi VLM (`slots._look`).
  - [x] Tập: sức chứa theo cả bề cao của khối nhiều dòng (cỡ đúng nháp làm chữ hai dòng trong vòng tròn tràn dọc, v01);
    nén ngang tính ở cỡ chung. `room` chừa khe 0,2 cao dòng khi chạm vật.
- [x] Chặng 6: tài liệu (README, ROADMAP, docs/API.md), UI debug (bảng số phận dòng), `scripts/replay.py`; máy chủ dev2 + mix1
  (15/10, mã mới: VLM bỏ sót 1 dòng / 104 poster, hỏi lại được; sửa 3 lỗi soát thấy: chữ khách trên biển / huy hiệu bị giữ,
  pill bị xoá trống, "50% Giảm" sai thứ tự); `final2` (32 ảnh, 32/32 xong, không lỗi chạy, VLM hỏi lại xin chữ 4 lần đều được).
  Soát 32 poster: đa số đạt. Sửa tổng quát cuối: dòng VLM để trống trong `split` mà đứng cùng hàng cạnh dòng có chữ -> giữ trong
  đơn vị (g18 "Giảm 50%" co 0,32). Còn (giới hạn đã biết, không vá theo ca): bảng nhiều cột (g08 lịch bơi), VLM coi phần nhỏ của
  giá là rác (g14 "đ/thùng"), VLM bỏ cột giá của bảng (dev2 d05).

- [x] Chặng 7 (16/10, người dùng chốt): KHÔNG gọi lại VLM; dòng FLUX vẽ thừa -> VLM đoán ý và viết theo; cùng danh sách cùng cỡ.
  Lý do (final2 15/10): REFILL chạy 4 / 32 ảnh, +8-15 s mỗi lần, 2 / 4 có hại (giá bịa "3.000đ" cho món bịa, dòng lịch trùng);
  REASK / ghép rỗng 0 / 32; tập list lệch cỡ do `SET_FLOOR` (Coca, cơm tấm).
  - [x] `compose/vlm.schema`: khuôn JSON strict dựng theo từng poster -- "lines" có khoá BẮT BUỘC cho mọi L#, câu khách chỉ trỏ
    T# có thật, kiểu theo đúng các vai, font trong danh mục; `llm.json_call` nhận schema theo từng lời gọi. Câu chia nhiều dòng:
    mỗi dòng ghi câu + "part" (`compose.parse.from_lines` gộp; khối cùng cỡ xếp chồng bỏ split, code tự ngắt). Gỡ `reask`,
    `REASK`, `REFILL`, gọi lại khi ghép rỗng (`empty_plan` vẫn là lỗi nặng), `fate.omitted / no_words / merge`.
  - [x] `compose/rows.py`: danh sách / bảng nhận bằng hình học (cột thẳng lề, cỡ gần nhau, đều nhịp; cột cùng hàng -> bảng).
    Câu ghép hai lần: hàng bảng có câu liền kề (tên món của giá) thắng; không gộp dòng ở hai hàng của một danh sách.
  - [x] Dòng FLUX vẽ thừa (lời dặn mục 3): câu cùng vai chưa có chỗ, không còn thì VLM ĐOÁN FLUX định viết gì (đọc chữ giun +
    chữ OCR, hợp các hàng quanh nó) và viết theo, kể cả hàng danh sách / bảng, số / giá (người dùng: "FLUX bịa thì bịa theo").
    `compose.assign.filler`: chữ lấp là câu khách chưa có chỗ -> viết câu đó; không chặn chữ lấp nào nữa (bỏ luật chặn số).
    Câu VLM ghép cho hai dòng (FLUX vẽ lặp), không còn câu cùng vai, VLM không đoán chữ khác -> viết lặp theo nháp (trước:
    gỡ trống). Rà tinh thần "bám nháp" 16/10: mã chỉ còn gỡ trống dòng VLM nói skip, mảnh split VLM nói không mang chữ, mẩu
    nằm trong khung dòng đang viết lại. Giữ (người dùng chốt 14/10): các bước theo thứ tự khách, số trước hàng viết lại.
  - [~] Đã thử rồi BỎ (người dùng 16/10): gỡ cả hàng thừa + dàn đều các hàng còn lại bằng điểm ảnh (bỏ trọn chu kỳ hàng, nhân
    bản nền ở chỗ cắt, chỉ làm khi đường nối không lộ). Replay final2 + dev2: dàn được 1 / 5 danh sách có hàng thừa (cơm tấm
    g01); 4 / 5 bị từ chối (vệt nước, quạt sát mép, icon hai bên hàng, bảng kẻ ô) để lại chấm / dấu tick / icon trơ trọi.
  - [x] `compose.assign.inside`: dòng không mang câu khách nằm >= 80% trong khung một dòng lớn hơn theo số phận dòng đó (mẩu
    logo trên lon Coca -> giữ).
  - [x] Tập cùng vai = cùng vai + cỡ gần + THẲNG LỀ / CÙNG HÀNG (`layout.aligned_groups`); gỡ `render.SET_FLOOR`: cả tập một cỡ.
    Bề rộng cột chung (`size.column_want`); vùng cấm khi nới khung = khung dòng thẳng nới ngang (vùng gỡ chữ nới dọc lấn hàng
    liền kề), bỏ dòng đã gỡ, nới lần lượt (tên món và giá không nới vào chung một khe).
  - [x] Test: `tests/test_rows.py` (bảng, hàng thừa viết chữ đoán, giá về đúng hàng), `tests/test_fate.py` (khuôn, mẩu câu,
    chữ lấp, mẩu logo), `tests/test_layout.py` (tập thẳng lề, cột chung).
  - Replay trên câu trả lời VLM đã lưu (final2 + dev2, 69 poster, bản có gỡ hàng): phần tử co < 0,6 23 -> 16, câu khách thiếu
    26 -> 22, cỡ trong +-15% nháp 68 -> 71%. (too_small 45 -> 24 KHÔNG tin được: render.check_units không tính too_small cho
    mục nào của tập -- `evened` gắn cho cả mục chật nhất -- mà bỏ SET_FLOOR đưa thêm chữ co mạnh vào tập.)
  - [x] Máy chủ `bench/final3.json` (mã d32b906): VLM 0 dòng bỏ sót / 0 mục sai khuôn, một lời gọi, trung vị 9,7 s; 15 / 32 đạt.
    Chấm theo nguồn: docs/TUNING.md mục "Bộ thi final3". Sửa sau chấm: gộp dòng chỉ bị chặn giữa hai hàng đủ ô của bảng (h02 /
    h22, lỗi chặng 7), mẩu VLM chia xếp theo câu (h13), `brief.ORDINAL` không ăn giờ "6:00" (h08), dòng có từ lạ không phải
    chữ sản phẩm (h17), tập chỉ một màu khi nháp tô cùng màu (h31).

## Bước 3: gỡ chữ giun (Hi-SAM + LaMa) -- chuẩn hoá 15/10

- [x] Tách khỏi `lonly.py` thành `textfix/erase.py` (một việc): `erase_mask` 120 dòng -> các bước có tên `_window` (cao dòng
  chặn 2,2 lần nét cao), `_strokes` (nét + dự phòng màu mực / lõi dòng), `_halo` (quầng, đuôi mờ, không lấn dòng khác),
  `_punctuation`; `color.py` (Lab dùng chung, trước là hàm riêng của slots). Chú thích nói nguyên tắc, số đo ở TUNING.
- [x] LaMa: một bản + khoá MỖI CARD, chạy trên card đã vẽ nháp (`draw` trả `device`) -- hết dồn card 0, hết nạp tranh nhau.
- [x] Kiểm hành vi không đổi: mặt nạ bản mới so bản cũ trên 135 nháp thật (dev2 / mix1 / accept2), gỡ hết + giữ theo kế hoạch: 270 / 270 lần giống hệt từng điểm ảnh, cả nhật ký; thời gian như cũ.
- [x] `tests/test_erase.py` 8 test trên ảnh tổng hợp: phủ nét + bóng đổ, icon cách khe không bị gỡ, dòng giữ không vào mặt nạ,
  settle trả điểm ảnh nháp, ngoài mặt nạ không đổi, thiếu LaMa báo lỗi.
- Người dùng chấp nhận: còn vài quầng mờ / nét nhỏ, chữ dựng mới đè lên.

## Bước 2: FLUX vẽ nháp -- chuẩn hoá 15/10

- [x] Đối chiếu mã tham chiếu BFL (`flux2/sampling.py`): lịch, Euler, CFG, ảnh tham chiếu (t = 10, 20...; trần 2024^2 / 1024^2),
  số bước mặc định -- khớp. `guidance=None` như mã tham chiếu (klein 4B không có nhúng guidance: ảnh không đổi).
- [x] `flux.SIZES` là nguồn DUY NHẤT các tỉ lệ (thêm 3:4); `tests/test_flux.py`: ~1 MP, bội 16, đúng tỉ lệ, khớp `brief.ORIENT`,
  tìm trọng số. Ép bf16 trong `Flux.load` (bỏ `check_dtypes` vá ở server); bỏ tham số chết `neg`; `generate` từ chối cỡ lệch 16.
- Số đo (dev2 14/10, 73 nháp, 2x A30): vẽ trung vị 2,3 s, max 4,3 s; ~17 GB / card (docs/SERVER.md).
- [x] Tỉ lệ lạ ở API -> 422 `invalid_aspect_ratio` (4 tỉ lệ của app). Chốt 15/10, docs/API.md.
- [x] Lỗi lúc vẽ -> phiên bản lỗi mã `draw_failed` (không thử lại). Chốt 15/10 cùng D1.
- [x] Không lọc nội dung (người dùng chốt 15/10: không cần).

## Chặng C -- Thi

- [x] `bench/final2.json` (32 ca, chạy 15/10; soát ở chặng 6 / 7).
- [ ] `bench/final3.json` (32 ca mới 16/10, seed 16001-16032): menu / bảng giá nhiều hàng, pill xếp dọc, chấm đầu dòng, bảng kẻ ô,
  thẻ ngang, ba ảnh sản phẩm mới (dầu gội, lon bia, hũ cà phê), đủ 6 loại form, prompt + form. Một lượt, chấm theo nguồn lỗi,
  không sửa mã theo ca của bộ này.
- [ ] (cũ) `bench/final1.json`, một lượt, chấm. Ngưỡng: hỏng <= 10%, không mất chữ sản phẩm, không số / giá / liên hệ bịa.

## Bước LLM đầu luồng (brief) -- LÀM LẠI 15/10, chờ đo trên máy chủ

Người dùng chốt 15/10: lời khách BẨN (không ngoặc kép, viết thường, lỗi gõ, không dấu) là bình thường; LLM chuẩn hoá NHẸ (bỏ từ
thừa, hoa / thường, dấu câu, thêm dấu, lỗi gõ rõ ràng, tách câu dài -- không đổi số / tên, không diễn đạt lại); bịa PHONG PHÚ hơn
(tối đa 6 câu: tiêu đề, khẩu hiệu, nút, nhãn + tối đa 3 dòng lợi ích chung, vẫn cấm số / ưu đãi / tên / cam kết); chữ trong
prompt theo chuẩn BFL (ngoặc kép lồng trong câu tả vật mang chữ) -- KHÔNG đổi sang "code nối danh sách" như đề xuất 14/10.
Đã làm (`textfix/brief/` -- gói 4 tệp + `tests/test_brief.py`, `textfix/llm.py`):
- [x] LLM viết `texts` TRƯỚC `prompt_en` (thứ tự khoá trong khuôn JSON), `source` bắt buộc.
- [x] Kiểm theo SỰ THẬT thay nguyên văn: số / liên kết của câu phải có trong lời khách (bỏ câu bịa SĐT / giá); câu "user" >= 50%
  chữ có trong lời khách (so bỏ dấu + gần đúng), phần LLM thêm không có chữ ưu đãi / cam kết; còn lại xét như chữ phụ trợ. Bỏ
  `_strip_values` / luật "ghép chữ khách"; `_snap` chỉ sửa khi LLM đổi CHỮ CÁI (đánh đổi: lỗi gõ chữ cái của khách giữ nguyên).
- [x] Chuỗi ngoặc kép của khách / ô form "đã có" khi >= 80% chữ + mọi số nằm trong texts (không đòi khớp từng ký tự).
- [x] Chữ thiếu trong prompt chèn SAU CÂU ĐẦU (không nối cuối); chuỗi lạ chỉ bỏ cả câu khi câu không còn chữ đúng; prompt cắt
  câu tả ở cuối cho vừa 512 token (đếm bằng tokenizer Qwen3 ở máy, không có thì ước 0,33 token / byte).
- [x] `llm.json_call`: gọi lại tối đa 2 lần khi JSON hỏng / lỗi mạng (áp cho cả VLM -- một phần A2, D2); tham số `fresh`.
- [ ] Đo: `%run scripts/brief_probe.py` (209 ca dev / accept, chỉ bước LLM) -> so texts / prompt với brief lượt 14/10.
- [ ] Rồi 5 ca vẽ thật: `dev_batch.py bench/dev2.json --only d03,d05,d20,d31,d39 --fresh`.
- [x] Chuẩn hoá (người dùng 15/10: "chuẩn open-source, không vá chằng chịt"): gói `textfix/brief/` -- `instructions.py` (lời dặn,
  khuôn), `evidence.py` (đối chiếu lời khách), `texts.py` (chốt chữ: user / ai / bỏ + bảo đảm), `prompt.py` (hợp đồng prompt, chỗ
  DUY NHẤT sửa prompt); `tests/test_brief.py` 26 test LLM giả. API `make_brief` giữ nguyên.
- [x] Lỗi cũ: bước của form hướng dẫn (API gửi `step_1`, `step_2`...) bị thêm lại với vai `info` -> không có thứ tự / không viết
  hoa; nay vai "step".
- [ ] Ngưỡng `texts.USER_COVERAGE` 0,5 / `PRESENT` 0,8 / `SNAP` 0,85 / `evidence.FUZZY` 0,8 chọn theo suy luận -> chỉnh theo số
  đo brief_probe.
- [x] API cho đội test (chốt 15/10, docs/API.md, tests/test_api.py): (a) trường ngoài loại form bỏ qua + báo `ignored_fields`;
  (b) loại lạ -> 422 `invalid_category`; (d) LLM hỏng -> `llm_failed`; (e) bắt buộc như app, ghi trong docs/API.md. Cũ:
  (a) trường không thuộc loại form bị bỏ âm thầm (`server.app.form_of`); (b) `category` lạ âm thầm thành promo; (c) [ĐÃ SỬA 15/10 ở bước 2: `flux.SIZES` thêm 3:4 = 864x1152, test khớp với `ORIENT`]; (d) LLM hỏng hẳn sau 3 lần -> job "error" (= D1);
  (e) thiếu trường bắt buộc khi không có prompt -> chặn, cần ghi trong tài liệu API.

Ghi nhận 14/10 tối (trước khi làm lại):

Việc của LLM (người dùng chốt): bịa chữ phụ khi đề sơ sài; phân loại có chữ / không chữ; viết trung thực chữ khách; KHAI TRUNG
THỰC những chữ đã bảo FLUX vẽ; prompt_en theo đúng chuẩn làm giàu prompt của BFL (không dặn font / cỡ / vị trí -- đã sửa).
Lỗi đo trên 263 brief đã lưu:
- 45 / 263 viết chữ trong NGOẶC ĐƠN -> lọt phép kiểm prompt_en <-> texts: FLUX vẽ chữ không có trong texts (giá bị đè accept1
  "185.000đ", menu dev2 d05) và câu khách bị nối lặp ở cuối (dev2 d31 phụ đề vẽ hai lần).
- 41 / 263 LLM tự đặt MỘT câu HAI LẦN trong ngoặc -> FLUX vẽ lặp (gốc của nhiều ca "dòng lặp" chặng B phải xử lý hộ).
- LLM ghép chữ khách ("Bún bò tái 40.000đ") / chép sai vài chữ ("Khánh trương") -> câu bị bỏ mà FLUX vẫn vẽ -> chặng B xoá trắng.
- 13 / 263 nhắc câu khách TRẦN (không ngoặc) rồi câu được nối trong ngoặc ở cuối -> có thể vẽ hai lần (chưa xử lý).
- Vai (nhóm của chặng B) không ai kiểm: t02 tên khoá học + giờ học cùng "info" -> một tập.
Hiện đã vá (commit 266fd31 / 223033c / 0aab432): đổi mọi ngoặc về ngoặc kép, mỗi chữ một lần, ghép chữ khách / chép sai -> đoạn
khách viết, vai "step", từ tả tiếng Anh chỉ bỏ ngoặc, câu nhắc ảnh sản phẩm không bị xoá. Người dùng: "rối quá, user không rảnh
viết dấu ngoặc" -> HƯỚNG ĐỀ XUẤT (chưa duyệt): LLM viết prompt_en CHỈ TẢ CẢNH, không chữ; CODE nối "The poster shows these texts:
"A", "B"" đúng một lần (đúng ý 3 của BFL) -> chữ FLUX vẽ = texts theo cấu trúc, bỏ toàn bộ phần đối chiếu ngoặc; phòng thủ còn 3
luật: câu nguồn khách gồm chữ khách viết, câu AI ngắn không số / lời hứa, trường form bị sót thêm lại + tách trường danh sách.
Câu hỏi mở: có ghi vai trước mỗi câu ("headline "A"") không; cần đo trên máy chủ (5 ca dev2 so lượt cũ).

## Bước đo dòng (OCR) -- ghi nhận 14/10 khuya

OCR quyết móng của chặng B (mọi khung, cỡ, lề lấy từ dòng L#) và hậu kiểm của nó là CỤM LUẬT VÁ theo kiểu sai (`slots._split_gaps`,
`_stacked`, `_trim_lead`, `_lead_dots`, `TILT_*`, `_orphans`, `fill.TALL`).
- Tên + giá gộp một dòng: thường ổn (khe >= 1 cao dòng tách trước VLM; VLM ghép hai câu -> cắt tại khe nét). Hỏng khi VLM không
  nhận ra hai câu, hoặc không có khe rõ.
- Hai hàng gộp một khung: YẾU NHẤT -- `TALL` đoán số hàng theo chiều cao, chia câu đều; cỡ chữ đo từ chiều cao khung có thể to
  gấp đôi (e03 huy chương, i02 dấu chồng, d25v1 chữ xếp dọc).
- Tỉ trọng: OCR / Hi-SAM chia dòng khác nhau ~4% số dòng (18 / 546, probe 65 nháp, có cả ca Hi-SAM sai); ~5-10% poster -- ít hơn
  FLUX (chỗ ngắn / thiếu câu ~15-20%), VLM (~15%), LLM (~15-20% brief) nhưng là nguồn lỗi chính còn lại PHÍA CODE.
- Hướng: dòng thẳng theo Hi-SAM phân cấp, OCR chỉ bù chữ cong và đọc chữ trên sản phẩm; bỏ gần hết cụm luật vá. Sau deadline.

## AUDIT 14/10 khuya: VLM ghép + dựng + kiểm + engine -- SỬA NGÀY 15/10 (người dùng: "phải vá hết trước khi demo cho sếp")

P0 -- có thể làm hỏng cả poster khi demo:
- [ ] A1. CACHE LLM / VLM (`llm.json_call`, ~/.cache/textfix): mọi lời gọi lưu theo nội dung đầu vào -> "gọi lại VLM" khi ghép
  rỗng / cụt (`engine.design`) gửi y hệt đầu vào, LẤY LẠI ĐÚNG câu trả lời cũ: hai luật gọi lại KHÔNG BAO GIỜ có tác dụng.
- [ ] A2. VLM trả JSON hỏng / quá giờ: `json.loads` ném lỗi -> `design` lỗi -> phiên bản không có poster (server ghi "error").
  Không có đường lui.
- [ ] A3. Dòng VLM bỏ sót (hoặc ghi skip cho dòng chữ thường) bị XOÁ TRẮNG, kể cả dòng trên pill / một hàng của list -- trái tinh
  thần TextFix ("đứng trên pill hoặc cùng cấp chữ khác mà xoá thì khoảng trống rất xấu"). Hướng: hỏi lại VLM riêng các dòng đó.
- [ ] A4. Chữ lấp VLM viết cho dòng thừa chỉ chặn SỐ (`fill.resolve`), không chặn lời hứa / ưu đãi ("Miễn phí", "Giảm giá"), tên,
  độ dài -- brief có `AI_BAN_WORDS` / `AI_MAX_CHARS` nhưng chặng B không dùng. Ví dụ chữ lấp vô nghĩa: dev2 f05 "Ngân hàng".
- [x] A5. Chữ co tới sàn 12 px (1,2% cạnh ngắn) vẫn xuất: `too_small` không phải lỗi nặng, vòng sửa không đụng (dev2 d26v0 tập liên
  hệ co 0,13; d14v1 0,22). Chính sách "không cắt chữ" -> cần đường khác: thêm dòng khi khung đủ cao / nới khung trong vùng trống.
  -> 15/10 chặng 4: sàn tập `render.SET_FLOOR` (mục phải co cỡ dưới 0,8 rời tập) + nới khung vào nền trống (`fill.room`).
- [ ] A6. Chữ trên sản phẩm khách tải bị VLM viết đè (dev2 d13) -- xem mục "CHỮ TRÊN SẢN PHẨM" ở chặng B.

P1 -- xấu thấy rõ, gặp thường:
- [ ] B1. KIỂU CHỮ VLM tả sai (~8 / 35 poster mix1: font hẹp / đậm hơn nháp, serif thành không chân). Một nguồn: `slots._look` đo
  "letters ~0.40em wide (condensed)" từ SỐ KÝ TỰ OCR của chữ giun (chữ giun hẹp hơn chữ thật ~11%) -> nhãn "condensed" đẩy VLM
  chọn font hẹp. Code không đối chiếu lớp font VLM chọn (serif / sans) với độ tương phản nét đo được (`stroke_contrast`).
  -> 15/10: bỏ nhãn bề rộng khỏi bảng dòng. Còn: đối chiếu serif / sans với tương phản nét (chờ lượt máy chủ mới).
- [ ] B2. HIỆU ỨNG CSS VLM tả (viền / bóng / quầng) không đối chiếu với quầng đo được trên nháp (`l["effect"]`): dev2 d06 neon
  thành chữ mảnh, d38 viền tối.
- [x] B3. `render.OUTLINE_HEAD / OUTLINE_ABS`: tiêu đề chìm nền -> thêm viền đen, phá hiệu ứng (d06). Nên theo luật chung "đúng
  màu nháp thì giữ màu + quầng".
  -> 15/10 chặng 5: bỏ `OUTLINE_*`; đúng màu nháp -> giữ màu + quầng ở mọi cỡ.
- [ ] B4. Lời dặn VLM THỪA so với code: mục 2 bắt VLM chia "split" nhưng code đã bỏ split trong một đơn vị (ngắt theo bề rộng
  nét), chỉ còn dùng mảnh rỗng + câu ghi nhiều mục. Mục 3(b) "lấy cụm từ câu khách khác" sinh lặp nội dung (mix1 m03 tiêu đề hai
  lần). Màu: lời dặn "màu phổ biến nhất" nhưng code đổi sang màu đo từng dòng khi lệch > 120 -- hai nguồn sự thật.
  -> 15/10 chặng 3-4: bỏ 3(b) và "màu phổ biến nhất"; màu = hai trong ba nguồn (`fill.pick_color`). Còn: `split` mục 2.
- [ ] B5. Chồng lấn giữa hai đơn vị còn lại sau vòng sửa (dev2 d37v0, d14v1); tràn khung còn sau nén 0,75 (mix1 v03): chữ thò khỏi
  pill.

P2 -- nhỏ / hiếm:
- [x] C1. `accent` (từ nhấn to / khác màu) không kiểm tỉ lệ, có thể phóng quá khung.
  -> 15/10 chặng 4: `accent` bỏ `scale` (chỉ màu); nhấn bằng cỡ = dòng riêng có cỡ đo (`lines.rows`, `fill._rows`).
- [ ] C2. Luật theo ký tự OCR: thêm ngoặc kép cho câu trích, đặt lại số bước trong dòng ("1. ...").
- [ ] C3. Kiểu theo VAI ép một kiểu cho cả vai dù nháp vẽ hai kiểu (khi LLM gộp hai loại vào một vai -- mix1 t02).
- [ ] C4. "missing" của VLM và của code khác nhau chỉ ghi nhật ký.

## AUDIT 14/10 khuya: bước vẽ + máy chủ

- [x] D1. (P0) Một lỗi bất kỳ trong trạm thiết kế (VLM JSON hỏng / quá giờ, Hi-SAM, LaMa, Chromium) -> phiên bản "error", không
  có poster; yêu cầu MỘT ảnh thì cả job lỗi. Không có đường lui -- cần chọn: thử lại không cache, rồi trả gì (nháp? báo lỗi rõ?).
- [x] D2. (P0) Brief (LLM) hỏng JSON / quá giờ -> cả job lỗi, không thử lại (cache + `max_retries=1` chỉ cho lỗi mạng).
  15/10: `llm.json_call` gọi lại tối đa 2 lần; hết lượt vẫn lỗi job. CHỐT 15/10: lỗi có mã theo bước (`llm_failed`,
  `draw_failed`, `vlm_failed`, `vlm_empty_plan`...), không trả nháp thay poster; ảnh debug vẫn tải được (docs/API.md).
- [ ] D3. (P1) Cache brief theo nội dung: cùng yêu cầu -> cùng brief mãi mãi (kể cả brief lỗi đã lưu); muốn "thử lại" phải xoá
  ~/.cache/textfix. Khi demo: bấm tạo lại cùng yêu cầu ra y hệt brief.
- [ ] D4. (P2) Không có bước kiểm FLUX đã vẽ đủ câu / có vẽ sản phẩm khách tải (giới hạn model; chỉ ghi "missing").
- Chặng A (Hi-SAM + LaMa): người dùng chốt, không audit lại; giới hạn đã biết ở mục cuối.

## AUDIT TỪNG TỆP 14/10 khuya (21 tệp textfix/ + server/; scripts/ là công cụ dev, chưa audit) -- phát hiện MỚI

- [x] (15/10: brief cắt câu tả cho vừa, chữ thiếu chèn sau câu đầu) `flux2/text_encoder.py` (+ `brief.py`): FLUX đọc prompt tối đa 512 TOKEN (cắt bớt). 4 / 272 prompt_en đã lưu dài hơn
  (tới 642): câu khách nằm sau mốc đó FLUX không thấy, không vẽ ("GIẤC NGỦ NGON MỖI ĐÊM", "An Yoga"...). (P1) Hướng "code nối
  chữ vào cuối prompt" sẽ làm CHỮ bị cắt trước -- chữ phải đặt đầu, hoặc giới hạn độ dài phần tả cảnh.
- [ ] `textfix/glyph.py`: màu / độ dày nét / hiệu ứng của dòng đo bằng k-means màu trong khung OCR, KHÔNG dùng mặt nạ nét
  Hi-SAM (đã có, đáng tin hơn) -> chọn nhầm cụm thì ra màu viền thay màu chữ, sai độ đậm (bảng VLM + màu code dùng). 4 hằng số
  ghi "TẠM" chọn theo ca (q01, s08, p13). (P1)
- [ ] `textfix/slots.py` `lines`: cỡ chữ = cao nét / (cao chữ hoa hay chữ thường) theo việc CHỮ OCR có viết hoa không -- chữ giun
  OCR đọc sai hoa / thường -> cỡ lệch ~36%. Sau khi VLM ghép đã biết câu khách thật -> lấy hoa / thường từ câu khách. (P1)
- [ ] `textfix/render.py` `render_units`: tập cùng vai co theo mục chật nhất KHÔNG có sàn (phần tử tập dựng `fit:"none"`, bỏ qua
  sàn 12 px) -- một mục ghép sai / chỗ quá ngắn kéo cả tập co (dev2 d26v0 tập liên hệ x0,15). Tràn khung của mục trong tập
  không được sửa (`autofix_units` bỏ qua tập). (P0, cùng nhóm A5)
- [x] (15/10: `erase.inpaint` chạy trên card đã vẽ nháp, một bản + khoá mỗi card) `textfix/lonly.py` `_lama`: LaMa luôn chạy trên "cuda" (= card 0) cho mọi luồng thiết kế, chung card với một bản FLUX
  + Hi-SAM; nạp lần đầu không khoá (nhiều luồng cùng nạp). Rủi ro hết VRAM card 0 khi tải cao lúc demo. (P1)
- [x] (15/10) `server/app.py` + `textfix/flux.py`: API nhận tỉ lệ "3:4" (brief có, LLM được dặn bố cục dọc) nhưng `flux.SIZES` không có ->
  âm thầm vẽ vuông 1:1. UI chỉ có 4 tỉ lệ nên chỉ lộ qua API. (P2)
- [ ] `textfix/render.py` `_el`: emoji bị xoá âm thầm (không font nào có hình emoji) -- câu khách có emoji mất ký tự. (P2)
- [ ] `textfix/audit.py`: cờ cũ không còn kích hoạt ("-> câu khách", "không nối lại đúng câu khách") -- chỉ ảnh hưởng báo cáo. (P2)
- [ ] `textfix/geo.py`: ~25 ngưỡng cho chữ cong (một phần có số đo: ARC_SPAN trên 18 vùng); `RUNS` tắt -- mã chết. (P2)
- Không thấy vấn đề đáng kể: `browser.py`, `config.py`, `hisam.py`, `multigpu.py`, `ocr.py`, `ship.py`.

## Giới hạn đã biết, chưa xử lý

- Chữ neon quầng rộng: còn bóng mờ sau khi gỡ.
- Chữ rất nhỏ: Hi-SAM thấy ít nét (chân trang accept2 e02); chế độ cắt mảnh chưa đo trên GPU.
- Chữ chạy vòng trên con dấu / huy hiệu: OCR đọc từng mẩu, phần còn lại thành các cụm nét rời.
- Logo kiểu thư pháp (Coca-Cola, Dove) Hi-SAM phần lớn không đánh dấu: không dùng Hi-SAM để nhận chữ sản phẩm.
