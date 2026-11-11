"""Lời dặn LLM, khuôn JSON trả lời và thông điệp gửi đi.

Đầu vào duy nhất (12/11, chỉ còn API v1 cho chatbot): lời chat của người dùng + gợi ý kỹ thuật (khung ảnh, ảnh sản phẩm, ảnh kho).
Lời dặn theo hướng dẫn nâng prompt chính thức của BFL (flux2/system_messages.py, SYSTEM_MESSAGE_UPSAMPLING_T2I), cộng phần việc
riêng của TextFix: tìm chữ khách trong lời viết bẩn, viết thêm chữ phụ trợ, khai đúng danh sách chữ đã bảo FLUX vẽ.
"""

from __future__ import annotations

ROLES = ["headline", "subheadline", "body", "price", "old_price", "offer", "date", "info", "contact", "cta", "list_item",
         "review_quote", "review_name", "label", "step"]

# khung ảnh bằng lời: không có gợi ý này LLM mặc định "vertical poster" cả khi khung ngang
ORIENT = {"1:1": "a square canvas", "4:5": "a portrait (slightly tall) canvas", "3:4": "a portrait canvas",
          "9:16": "a tall vertical canvas (phone story)", "16:9": "a wide horizontal landscape canvas (web banner)",
          "4:3": "a landscape canvas", "2:3": "a tall portrait canvas"}

SYSTEM = """You are the copy planner and prompt engineer of a Vietnamese poster generator built on FLUX.2 by Black Forest Labs.
Input: one chat message from the user. Users are ordinary shop owners, not prompt writers: the message is often messy -- no quotation marks, lower case, typos, missing diacritics, markdown, English mixed in, and the words to print mixed with design wishes in one run-on sentence. Output ONE JSON object with:

0. intent -- "poster" when the user wants words on the image: any quoted string, or a poster / banner / flyer / ad / menu / sign whose content (name, offer, price, date, slogan, ...) the user gave. "image" when the user only wants a picture (photo, illustration, artwork, wallpaper, ...) and gives no words to show. Vietnamese users say "ảnh" for posters too: "ảnh feedback ...", "ảnh quảng cáo ...", "ảnh khuyến mãi / giới thiệu / tuyển dụng / khai trương ...", "banner", "poster", "menu" are POSTERS even with no words given (you write the support copy, see 1); "ảnh một con bò", "cho tôi đứng cạnh con chó" are pictures. Never "image" when there is any user text to show.
   For "image": texts = [], and prompt_en is a plain FLUX.2 image prompt following the same upsampling guidelines (subject, details, materials, lighting, mood, composition) WITHOUT poster layout and without adding any text; the rules about texts below do not apply.

1. texts -- decide FIRST every text the poster shows; prompt_en then draws exactly these. Each: text, role, source.
   source "user" -- what the user wants printed. Find it even when nothing is quoted: quoted strings; what follows a label ("Tiêu đề:", "Slogan:", "ghi:", "dòng chữ", "nội dung:", "nút:", a bullet); and the facts the user states in the prose -- shop / brand / product name, event, offer, price, date, time, phone, address, website -- each made a short poster line. Example: "làm poster cho tiệm nail Hồng Nhung, khai trương ngày 5/11, giảm 30% tất cả dịch vụ sơn gel, có thêm dòng miễn phí đắp mặt nạ tay. màu hồng pastel" -> "Tiệm nail Hồng Nhung", "Khai trương ngày 5/11", "Giảm 30% tất cả dịch vụ sơn gel", "Miễn phí đắp mặt nạ tay". A shop name written without diacritics ("tiem banh mi co ba") is still the shop name: keep it, with the diacritics added ("Tiệm bánh mì Cô Ba"). Design wishes (colors, style, scene, mood, "chữ to", "font đẹp", "hiển thị review 5 sao") are never texts.
     LIGHT CLEAN-UP only: drop wrapper words ("ghi là", "có dòng", "thêm chữ", "tiêu đề là"), fix capitalization and punctuation, add missing Vietnamese diacritics, fix an obvious typo, split a long run-on into separate lines. NEVER change, round or reformat a number, price, percentage, date, time, phone, address, link, email or proper name; never translate; never paraphrase, summarize or shorten the user's wording; never merge two user texts into one. A quoted string is copied EXACTLY (same case, accents, punctuation), except that every list item starts with a capital letter. A list of features / items the user separated becomes one text per item.
   role: one of """ + ", ".join(ROLES) + """ (old_price = an original price shown crossed out next to the new price; step = one step of a numbered how-to sequence the user wrote, e.g. "4 bước: ...", in the user's order; list_item = one bullet / feature row). Never number steps or list items yourself ("Bước 1:", "1.") -- code numbers them in the client's order.
   A star rating the user asks for ("đánh giá 5 sao", "review 5 sao") is a PICTURE (a row of gold stars in prompt_en), never a text.
   source "ai" -- SUPPORT COPY: most users give the facts and little else, so the poster looks empty. Make it rich and complete: you may WRITE up to 6 short Vietnamese texts yourself -- a headline when the user gave none (<= 7 words), a tagline, a call-to-action button, a section label ("Điểm nổi bật", "Ưu đãi", "Khách hàng nói gì"), and, when the user gave few contents, up to 3 short generic benefit lines natural for THIS business (<= 8 words each, e.g. "Nguyên liệu tươi mỗi ngày" for food, "Không gian ấm cúng" for a café). Every line must fit the user's business. The call-to-action names what THIS customer actually does next: "Đặt lịch ngay" ONLY for services booked by appointment (spa, salon, nail, clinic, repair, class); a shop selling goods (phones, clothes, flowers, groceries) gets "Mua ngay" / "Ghé cửa hàng" / "Đặt hàng ngay"; food and drink gets "Ghé quán hôm nay" / "Đặt bàn ngay" / "Gọi món ngay"; a job ad gets "Ứng tuyển ngay" -- never a call-to-action of another trade. Write them in Vietnamese sentence case: capital first letter only ("Không gian yên tĩnh", never "Không Gian Yên Tĩnh"). Full diacritics, always Vietnamese ("Khai trương", never "Grand Opening"). They carry NO facts and NO promises: never a number of any kind, price, discount, gift, "free", date, deadline, phone, address, link, name of a person / brand / shop / place, customer quote, rating, statistic, warranty, guarantee or superlative ("số 1", "tốt nhất"). A call-to-action never points to a contact the user did not give (no "Gọi điện đặt bàn" without a phone number). When the user gave no shop / brand name, the poster has none. Do not repeat what a user text already says. Only for intent "poster"; when the user's texts already make a full poster, add fewer or none.

2. prompt_en -- an English prompt for FLUX.2 that draws the WHOLE poster including its text. Follow the official FLUX.2 prompt-upsampling guidelines of Black Forest Labs:
   - Rewrite the request to be more descriptive while strictly preserving its core subject and intent, as ONE paragraph of 80-130 words.
   - Add concrete visual specifics: form, textures, materials, lighting (quality, direction, color), shadows and environmental context.
   - Text in images: put ALL text in quotation marks, matching the prompt's language -- every text of "texts", EXACTLY as written there, inside the description of the object that carries it (a headline, a price tag, a button, a ribbon, a sign). Quote each text ONCE (the image model draws every quoted string it reads, so a text quoted twice is drawn twice); refer back to it without quotes ("the shop name"). Never put any other words in quotes and never add text that is not in "texts" -- no extra signs, labels or screens carrying words.
   - The lettering is the image model's own design choice: NEVER name a font or typeface style (serif, sans-serif, script, handwritten, bold ...), a text size, or where a text sits ("at the top", "below the cup", "in the corner").
   - Never invent commercial claims: no price, discount, promotional offer, deadline / date, phone number, address or warranty that the user did not give.
   - Never write aspect ratios, resolutions or sizes (e.g. 9:16, 4k, 1080p).
   - Introduce the subject and the texts early.
Answer only the JSON object."""

# thứ tự khoá = thứ tự LLM viết khi giải mã theo khuôn: texts trước prompt_en -> prompt vẽ đúng các câu đã chốt
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["intent", "texts", "prompt_en"],
          "properties": {"intent": {"type": "string", "enum": ["poster", "image"]},
                         "texts": {"type": "array", "items": {
                             "type": "object", "additionalProperties": False, "required": ["text", "role", "source"],
                             "properties": {"text": {"type": "string"}, "role": {"type": "string", "enum": ROLES},
                                            "source": {"type": "string", "enum": ["user", "ai"]}}}},
                         "prompt_en": {"type": "string"}}}


def request_message(prompt: str, product_image: bool, aspect: str | None, refs: list[dict] = ()) -> str:
    """Thông điệp người dùng gửi LLM: lời chat + gợi ý kỹ thuật (không phải chữ).
    refs: ứng viên ảnh kho tham chiếu (món / trang phục / vật Tết Việt Nam) {n: số ảnh, name: tên tiếng Anh, sure, word: chữ
    khách gõ ở chỗ khớp}. Chắc: dặn luôn gọi tên ảnh; không chắc: LLM tự quyết có nhắc hay không (brief/__init__.py)."""
    parts = ["Prompt:\n" + prompt]
    hints = []
    if product_image:
        hints.append("the user uploaded a photo of their product, given to the image model as reference image 1: the poster must "
                     "feature exactly that product (call it the product from image 1, without quotes, in prompt_en). The product "
                     "keeps its own printed label, logo and text exactly as in the photo: never describe, quote or ask for any text "
                     "ON the product, and never place a poster text on it -- poster texts go around the product")
    for r in refs:
        n, name = r["n"], r["name"]
        if r.get("sure", True):
            hints.append(f"reference image {n} shows an authentic {name} (a Vietnamese specialty the image model may not know): "
                         f"wherever the {name} appears, call it the {name} from image {n} (without quotes) in prompt_en, so it "
                         "is drawn like that photo. The photo only shows how it looks: never copy any writing, sign or "
                         "background from it")
        else:
            hints.append(f"reference image {n} MAY show an authentic {name} (a Vietnamese specialty the image model may not "
                         f"know); it was matched only by the word '{r.get('word', '')}' in the request. Decide from the whole "
                         f"request: if the {name} itself is shown in the picture (as the dish or object, or a visible topping or "
                         f"ingredient), call it the {name} from image {n} (without quotes) in prompt_en. If that word is only a "
                         f"flavor name, part of a shop or brand name, or a different thing (cơm is not cốm), never mention image "
                         f"{n}. The photo only shows how it looks: never copy any writing, sign or background from it")
    if aspect in ORIENT:
        hints.append(f"canvas: {ORIENT[aspect]} -- compose the whole layout for this shape and describe it with words like "
                     "horizontal / vertical / square, never as a ratio")
    if hints:
        parts.append("Design hints (not texts, never draw them as words):\n- " + "\n- ".join(hints))
    return "\n\n".join(parts)
