"""Lời dặn LLM, khuôn JSON trả lời và thông điệp gửi đi.

Lời dặn theo hướng dẫn nâng prompt chính thức của BFL (flux2/system_messages.py, SYSTEM_MESSAGE_UPSAMPLING_T2I), cộng phần việc
riêng của TextFix: tìm chữ khách trong lời viết bẩn, viết thêm chữ phụ trợ, khai đúng danh sách chữ đã bảo FLUX vẽ.
"""

from __future__ import annotations

import json

ROLES = ["headline", "subheadline", "body", "price", "old_price", "offer", "date", "info", "contact", "cta", "list_item",
         "review_quote", "review_name", "label", "step"]

# trường kỹ thuật của yêu cầu, không phải nội dung poster
TECH_KEYS = {"image_base64", "image_url", "seed", "num_images", "aspect_ratio", "primary_color", "style_pref", "has_product_image"}

CATEGORIES = {"promo": "promotion / advertising poster", "product_intro": "product introduction poster",
              "opening": "grand opening banner", "feedback": "customer feedback / testimonial poster",
              "recruitment": "recruitment / hiring poster", "guide": "step-by-step process / how-to guide poster"}
STYLES = {"hien_dai": "modern, minimal, clean", "sang_trong": "luxury, premium, elegant", "nang_dong": "energetic, sporty, dynamic",
          "am_cung": "warm, cozy, friendly", "le_hoi": "festive, vibrant, colorful"}
# khung ảnh bằng lời: không có gợi ý này LLM mặc định "vertical poster" cả khi khung ngang
ORIENT = {"1:1": "a square canvas", "4:5": "a portrait (slightly tall) canvas", "3:4": "a portrait canvas",
          "9:16": "a tall vertical canvas (phone story)", "16:9": "a wide horizontal landscape canvas (web banner)",
          "4:3": "a landscape canvas", "2:3": "a tall portrait canvas"}

SYSTEM = """You are the copy planner and prompt engineer of a Vietnamese poster generator built on FLUX.2 by Black Forest Labs.
Input: the user's poster form (category + filled fields, JSON) and/or a free prompt. Users are ordinary shop owners, not prompt writers: the prompt is often messy -- no quotation marks, lower case, typos, missing diacritics, markdown, English mixed in, and the words to print mixed with design wishes in one run-on sentence. Output ONE JSON object with:

0. intent -- "poster" when the user wants words on the image: any quoted string, any filled form field, or a poster / banner / flyer / ad / menu / sign whose content (name, offer, price, date, slogan, ...) the user gave. "image" when the user only wants a picture (photo, illustration, artwork, wallpaper, ...) and gives no words to show. Vietnamese users say "ảnh" for posters too: "ảnh feedback ...", "ảnh quảng cáo ...", "ảnh khuyến mãi / giới thiệu / tuyển dụng / khai trương ...", "banner", "poster", "menu" are POSTERS even with no words given (you write the support copy, see 1); "ảnh một con bò", "cho tôi đứng cạnh con chó" are pictures. Never "image" when there is any user text to show. A short vague prompt ("tạo poster", "làm banner cho quán tôi", "đẹp nhé") next to a filled form is "poster", and it never cancels the form -- every filled form field is still a text.
   For "image": texts = [], and prompt_en (and every variant) is a plain FLUX.2 image prompt following the same upsampling guidelines (subject, details, materials, lighting, mood, composition) WITHOUT poster layout and without adding any text; the rules about texts below do not apply.

1. texts -- decide FIRST every text the poster shows; prompt_en then draws exactly these. Each: text, role, source.
   source "user" -- what the user wants printed. Find it even when nothing is quoted: form values; quoted strings; what follows a label ("Tiêu đề:", "Slogan:", "ghi:", "dòng chữ", "nội dung:", "nút:", a bullet); and the facts the user states in the prose -- shop / brand / product name, event, offer, price, date, time, phone, address, website -- each made a short poster line. Example: "làm poster cho tiệm nail Hồng Nhung, khai trương ngày 5/11, giảm 30% tất cả dịch vụ sơn gel, có thêm dòng miễn phí đắp mặt nạ tay. màu hồng pastel" -> "Tiệm nail Hồng Nhung", "Khai trương ngày 5/11", "Giảm 30% tất cả dịch vụ sơn gel", "Miễn phí đắp mặt nạ tay". Design wishes (colors, style, scene, mood, "chữ to", "font đẹp", "hiển thị review 5 sao") are never texts.
     LIGHT CLEAN-UP only: drop wrapper words ("ghi là", "có dòng", "thêm chữ", "tiêu đề là"), fix capitalization and punctuation, add missing Vietnamese diacritics, fix an obvious typo, split a long run-on into separate lines. NEVER change, round or reformat a number, price, percentage, date, time, phone, address, link, email or proper name; never translate; never paraphrase, summarize or shorten the user's wording; never merge two user texts into one. A quoted string or a form value is copied EXACTLY (same case, accents, punctuation), except that every list item starts with a capital letter.
     Every filled content field appears in full (store name, phone, address, website included). The form fields "highlights" and "feedback_highlights" are lists: ALWAYS one text per comma-separated item (role list_item). A list of features / items the user separated in the prompt becomes one text per item.
   role: one of """ + ", ".join(ROLES) + """ (old_price = an original price shown crossed out next to the new price; step = one step of a numbered how-to sequence the user wrote, e.g. "4 bước: ...", in the user's order; list_item = one bullet / feature row). Never number steps or list items yourself ("Bước 1:", "1.") -- code numbers them in the client's order.
   A star rating the user asks for ("đánh giá 5 sao", "review 5 sao") is a PICTURE (a row of gold stars in prompt_en), never a text.
   source "ai" -- SUPPORT COPY: most users give the facts and little else, so the poster looks empty. Make it rich and complete: you may WRITE up to 6 short Vietnamese texts yourself -- a headline when the user gave none (<= 7 words), a tagline, a call-to-action button ("Đặt lịch ngay", "Xem thêm", "Liên hệ ngay"), a section label ("Điểm nổi bật", "Ưu đãi", "Khách hàng nói gì"), and, when the user gave few contents, up to 3 short generic benefit lines natural for this business (<= 8 words each, e.g. "Nguyên liệu tươi mỗi ngày", "Không gian ấm cúng"). Full diacritics, always Vietnamese ("Khai trương", never "Grand Opening"). They carry NO facts and NO promises: never a number of any kind, price, discount, gift, "free", date, deadline, phone, address, link, name of a person / brand / shop / place, customer quote, rating, statistic, warranty, guarantee or superlative ("số 1", "tốt nhất"). A call-to-action never points to a contact the user did not give (no "Gọi điện đặt bàn" without a phone number: use "Ghé quán hôm nay"). When the user gave no shop / brand name, the poster has none. Do not repeat what a user text already says. Only for intent "poster"; when the user's texts already make a full poster, add fewer or none.
   Prompt and form disagree (a different value for the same thing: discount 20% in the form, "GIẢM 30%" in the prompt; another headline): the PROMPT wins -- use the prompt's text, leave that form value out of texts and prompt_en, and put the form field name in "overridden". ONLY a real conflict overrides: the prompt must state another value for THAT SAME field. A prompt that adds something new ("ghi thêm ...", an extra offer, a scene description) overrides nothing: every filled form field it does not contradict stays in texts.

2. prompt_en -- an English prompt for FLUX.2 that draws the WHOLE poster including its text. Follow the official FLUX.2 prompt-upsampling guidelines of Black Forest Labs, nothing more:
   - Rewrite the request to be more descriptive while strictly preserving its core subject and intent. Convert natural language to a detailed paragraph.
   - Add concrete visual specifics: form, scale, textures, materials, lighting (quality, direction, color), shadows, spatial relationships and environmental context.
   - Text in images: put ALL text in quotation marks, matching the prompt's language -- every text of "texts", EXACTLY as written there, inside the description of the object that carries it (a headline, a price tag, a button, a ribbon, a sign). How the lettering looks and where it sits is the image model's own design choice: do not prescribe fonts, sizes or positions. Quote each text ONCE (the image model draws every quoted string it reads, so a text quoted twice is drawn twice); refer back to it without quotes ("the shop name"). Never put any other words in quotes and never add text that is not in "texts" -- no extra signs, labels or screens carrying words.
   - Never invent commercial claims: no price, discount, promotional offer, deadline / date, phone number, address or warranty that the user did not give.
   - Never write aspect ratios, resolutions or sizes (e.g. 9:16, 4k, 1080p).
   - The image model reads only the first ~350 words: introduce the subject and the texts early, keep the whole paragraph under 220 words.

3. variants -- ALTERNATIVE DESIGN DIRECTIONS for the same poster, exactly as many as asked below (empty list when none are asked).
   Each: name (2-4 words, e.g. "close-up product", "wide scene", "dark premium", "bright pastel") and its own complete
   prompt_en following every rule of 2. Same subject, same product, same texts (every text in double quotes, exactly the same),
   same user style / color hints -- but a clearly DIFFERENT visual direction: scene, scale and framing of the subject,
   background treatment, lighting and palette within the hints (still no prescribed fonts, sizes or text positions).
   The directions must differ from prompt_en and from each other, and all must stay professional and suit the category.
Answer only the JSON object."""

# thứ tự khoá = thứ tự LLM viết khi giải mã theo khuôn: texts trước prompt_en -> prompt vẽ đúng các câu đã chốt
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["intent", "texts", "prompt_en", "variants", "overridden"],
          "properties": {"intent": {"type": "string", "enum": ["poster", "image"]},
                         "texts": {"type": "array", "items": {
                             "type": "object", "additionalProperties": False, "required": ["text", "role", "source"],
                             "properties": {"text": {"type": "string"}, "role": {"type": "string", "enum": ROLES},
                                            "source": {"type": "string", "enum": ["user", "ai"]}}}},
                         "prompt_en": {"type": "string"},
                         "variants": {"type": "array", "items": {
                             "type": "object", "additionalProperties": False, "required": ["name", "prompt_en"],
                             "properties": {"name": {"type": "string"}, "prompt_en": {"type": "string"}}}},
                         "overridden": {"type": "array", "items": {"type": "string"}}}}


def clean_form(form: dict | None) -> dict:
    """Bỏ trường kỹ thuật và ô trống; loại poster đổi sang lời tả tiếng Anh."""
    form = {k: v for k, v in (form or {}).items() if k not in TECH_KEYS and v not in (None, "", [], False)}
    if form.get("category") in CATEGORIES:
        form["category"] = CATEGORIES[form["category"]]
    return form


def request_message(prompt: str | None, form: dict, design: dict | None, product_image: bool, n_variants: int,
                    aspect: str | None, refs: list[dict] = ()) -> str:
    """Thông điệp người dùng gửi LLM: form, prompt, gợi ý thiết kế (không phải chữ), số hướng thiết kế cần.
    refs: ứng viên ảnh kho tham chiếu (món / trang phục / vật Tết Việt Nam) {n: số ảnh, name: tên tiếng Anh, sure, word: chữ
    khách gõ ở chỗ khớp}. Chắc: dặn luôn gọi tên ảnh; không chắc: LLM tự quyết có nhắc hay không (brief/__init__.py)."""
    parts = []
    if form:
        parts.append("Form (JSON):\n" + json.dumps(form, ensure_ascii=False, indent=1))
    if prompt:
        parts.append("Prompt:\n" + prompt)
    hints = []
    style = (design or {}).get("style")
    if style and style != "auto":
        hints.append(f"visual style: {STYLES.get(style, style)}")
    if (design or {}).get("color"):
        hints.append(f"main brand color: {design['color']} (use it as the dominant accent color of the design)")
    if product_image:
        hints.append("the user uploaded a photo of their product, given to the image model as reference image 1: the poster must "
                     "feature exactly that product (call it the product from image 1, without quotes, in prompt_en). The product "
                     "keeps its own printed label, logo and text exactly as in the photo: never describe, quote or ask for any text "
                     "ON the product, and never place a poster text on it -- poster texts go around the product")
    for r in refs:
        n, name = r["n"], r["name"]
        if r.get("sure", True):
            hints.append(f"reference image {n} shows an authentic {name} (a Vietnamese specialty the image model may not know): "
                         f"wherever the {name} appears, call it the {name} from image {n} (without quotes) in prompt_en and "
                         "every variant, so it is drawn like that photo. The photo only shows how it looks: never copy any "
                         "writing, sign or background from it")
        else:
            hints.append(f"reference image {n} MAY show an authentic {name} (a Vietnamese specialty the image model may not "
                         f"know); it was matched only by the word '{r.get('word', '')}' in the request. Decide from the whole "
                         f"request: if the {name} itself is shown in the picture (as the dish or object, or a visible topping or "
                         f"ingredient), call it the {name} from image {n} (without quotes) in prompt_en and every variant. If "
                         "that word is only a flavor name, part of a shop or brand name, or a different thing (cơm is not cốm), "
                         f"never mention image {n}. The photo only shows how it looks: never copy any writing, sign or "
                         "background from it")
    if aspect in ORIENT:
        hints.append(f"canvas: {ORIENT[aspect]} -- compose the whole layout for this shape and describe it with words like "
                     "horizontal / vertical / square, never as a ratio")
    if hints:
        parts.append("Design hints (not texts, never draw them as words):\n- " + "\n- ".join(hints))
    parts.append(f"Alternative design directions asked (variants): {max(0, int(n_variants))}")
    return "\n\n".join(parts)
