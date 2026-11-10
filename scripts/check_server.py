#!/usr/bin/env python3
"""TỰ KIỂM TRA máy chủ trước khi chạy textfix (docs/SERVER.md bước 4). Không nạp FLUX lên GPU, không sinh ảnh.

  cd ~/work/textfix
  python scripts/check_server.py            # chỉ kiểm, không đổi gì
  python scripts/check_server.py --fix      # + giải nén ~/persistent-data/chrome-linux64.zip, ghi PLAYWRIGHT_CHROME_PATH vào .env
  python scripts/check_server.py --no-call  # không gọi thử LLM / VLM

Máy chủ CHỈ pip install được (không HuggingFace, không apt, không tải Chromium của Playwright, không OpenAI): mọi thứ còn lại
phải có sẵn trong repo hoặc ~/persistent-data. Kiểm:
  0. thư mục (repo, ~/persistent-data có gì)                4. tài nguyên (font, icon Lucide; gói flux2 khi vẽ tại chỗ)
  1. gói pip so với requirements.txt -> in MỘT lệnh pip       5. Chromium (zip trong persistent-data, thư viện hệ thống thiếu)
  2. GPU                                                       6. OCR (RapidOCR, mô hình ONNX nằm trong gói pip)
  3. FLUX: trọng số tại chỗ, HOẶC máy FLUX ở xa (TEXTFIX_FLUX_URL)  7. LLM / VLM nội bộ: danh sách model, JSON, NHIỀU ẢNH; 8. cổng
Mã thoát 0 = đủ chạy; 1 = có mục LỖI (đọc dòng "SỬA:").
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HOME = Path(os.path.expanduser("~"))
fails: list[str] = []
pip_need: list[str] = []
MM_IMAGES = 12   # designer gửi 6 ảnh / lời gọi, vòng sửa lỗi tới 12: vLLM cần --limit-mm-per-prompt >= số này


def ok(msg: str) -> None:
    print(f"  [ĐẠT] {msg}")


def bad(msg: str, fix: str) -> None:
    print(f"  [LỖI] {msg}\n        SỬA: {fix}")
    fails.append(msg)


def warn(msg: str) -> None:
    print(f"  [CHÚ Ý] {msg}")


def gb(n: float) -> str:
    return f"{n / 2**30:.1f}GB"


def dir_size(p: Path, depth: int = 3, budget: int = 400) -> str:
    """Dung lượng thư mục, duyệt NÔNG (ổ persistent-data là ổ mạng: duyệt sâu hàng chục nghìn file làm treo); quá ngân sách
    thì in '>= ...'."""
    n, seen, full = 0, 0, True
    stack = [(p, 0)]
    while stack:
        d, k = stack.pop()
        try:
            it = list(os.scandir(d))
        except OSError:
            continue
        for e in it:
            seen += 1
            if seen > budget:
                return ">= " + gb(n)
            try:
                if e.is_file(follow_symlinks=False):
                    n += e.stat(follow_symlinks=False).st_size
                elif e.is_dir(follow_symlinks=False):
                    if k + 1 < depth:
                        stack.append((Path(e.path), k + 1))
                    else:
                        full = False
            except OSError:
                pass
    return gb(n) if full else ">= " + gb(n)


def persistent() -> Path | None:
    for p in (os.environ.get("TEXTFIX_DATA"), HOME / "persistent-data", "/home/jovyan/persistent-data", "/persistent-data"):
        if p and Path(p).is_dir():
            return Path(p)
    return None


# ---------------------------------------------------------------------------------------------------- 1. pip
def check_requirements() -> None:
    from importlib import metadata
    try:
        from packaging.requirements import Requirement
    except ImportError:
        bad("thiếu gói packaging (để so phiên bản)", "pip install packaging")
        return
    alias = {"opencv-python-headless": ("opencv-python-headless", "opencv-python", "opencv-contrib-python",
                                        "opencv-contrib-python-headless")}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        r = Requirement(line)
        got = None
        for name in alias.get(r.name, (r.name,)):
            try:
                got = metadata.version(name)
                break
            except metadata.PackageNotFoundError:
                continue
        if got is None:
            if r.name == "torch":
                bad("không có torch", "cài torch bản CUDA hợp driver (đừng để pip kéo bản CPU); môi trường repo v3 đã có")
            else:
                bad(f"thiếu {r.name}", "xem lệnh pip ở cuối mục")
                pip_need.append(line)
        elif not r.specifier.contains(got, prereleases=True):
            if r.name in ("torch", "torchvision"):
                bad(f"{r.name} {got} không thoả {r.specifier}", "nâng torch bản CUDA cùng bộ với torchvision (không nâng riêng lẻ)")
            else:
                bad(f"{r.name} {got} không thoả {r.specifier}", "xem lệnh pip ở cuối mục")
                pip_need.append(line)
        else:
            ok(f"{r.name} {got}")
    if pip_need:
        print("  -> LỆNH PIP (không đụng torch):\n     pip install " + " ".join(f"'{x}'" for x in pip_need))


# ---------------------------------------------------------------------------------------------------- 5. Chromium
def chrome_candidates(data: Path | None) -> list[tuple[str, Path]]:
    out = []
    if os.environ.get("PLAYWRIGHT_CHROME_PATH"):
        out.append(("PLAYWRIGHT_CHROME_PATH", Path(os.environ["PLAYWRIGHT_CHROME_PATH"])))
    if data:
        for sub in ("chrome-linux64/chrome", "chrome/chrome-linux64/chrome"):
            out.append(("persistent-data", data / sub))
    for b in ("chromium-browser", "chromium", "google-chrome", "google-chrome-stable"):
        if shutil.which(b):
            out.append(("PATH", Path(shutil.which(b))))
    return out


def unzip_chrome(z: Path, dest: Path) -> Path:
    """Giải nén giữ quyền chạy (zipfile bỏ bit x: lấy lại từ external_attr)."""
    with zipfile.ZipFile(z) as zf:
        for info in zf.infolist():
            p = Path(zf.extract(info, dest))
            mode = (info.external_attr >> 16) & 0o777
            if mode and p.is_file():
                p.chmod(mode | stat.S_IRUSR)
        name = next((n for n in zf.namelist() if n.rstrip("/").split("/")[-1] == "chrome" and not n.endswith("/")), None)
    exe = dest / name if name else None
    if exe is None or not exe.is_file():
        raise FileNotFoundError(f"không thấy file chrome trong {z}")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


def set_env_line(key: str, value: str) -> None:
    f = ROOT / ".env"
    lines = f.read_text(encoding="utf-8").splitlines() if f.exists() else []
    lines = [l for l in lines if not l.strip().startswith(f"{key}=")] + [f"{key}={value}"]
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_chrome(data: Path | None, fix: bool) -> None:
    import playwright
    from importlib import metadata
    print(f"  playwright {metadata.version('playwright')} ({Path(playwright.__file__).parent})")
    exe = next((p for _, p in chrome_candidates(data) if p.is_file()), None)
    z = data / "chrome-linux64.zip" if data else None
    if exe is None and z and z.is_file():
        if fix:
            exe = unzip_chrome(z, data)
            ok(f"đã giải nén {z.name} -> {exe}")
        else:
            bad(f"có {z} nhưng chưa giải nén", "python scripts/check_server.py --fix")
            return
    if exe is None:
        try:   # máy ra được internet (docs/DEPLOY.md): Chromium của `playwright install chromium`
            from textfix.render.browser import Browser as Reader
            with Reader() as R:
                R.page.set_content("<p style='font-size:40px'>Ệ Ợ Ữ</p>")
                R.page.screenshot()
            ok("Playwright mở được Chromium của chính nó (playwright install chromium)")
            return
        except Exception:
            pass
        bad("không thấy Chromium nào (máy chủ không tải được bản của Playwright)",
            "chép chrome-linux64.zip (Chrome for Testing, cùng đời với bản playwright) vào ~/persistent-data rồi chạy --fix")
        return
    if not os.access(exe, os.X_OK):
        bad(f"{exe} không có quyền chạy", f"chmod +x {exe}")
        return
    if os.environ.get("PLAYWRIGHT_CHROME_PATH") != str(exe):
        if fix:
            set_env_line("PLAYWRIGHT_CHROME_PATH", str(exe))
            ok(f"đã ghi PLAYWRIGHT_CHROME_PATH={exe} vào .env")
        else:
            warn(f"dùng {exe} nhưng .env chưa có PLAYWRIGHT_CHROME_PATH -- máy chủ có thể không tìm thấy (chạy --fix)")
        os.environ["PLAYWRIGHT_CHROME_PATH"] = str(exe)
    try:
        v = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=30)
        print(f"  {(v.stdout or v.stderr).strip()[:120]}")
    except Exception as e:
        warn(f"không chạy được '{exe} --version' ({e})")
    # thư viện hệ thống thiếu: không apt được -> phải biết trước
    try:
        miss = sorted({l.split()[0] for l in subprocess.run(["ldd", str(exe)], capture_output=True, text=True, timeout=30).stdout
                       .splitlines() if "not found" in l})
        if miss:
            bad(f"Chromium thiếu thư viện hệ thống: {', '.join(miss)}",
                "không apt được: chép các file .so đó (cùng bản Linux) vào một thư mục rồi thêm vào LD_LIBRARY_PATH; "
                "hoặc dùng Chromium repo v3 đang chạy được")
            return
        ok("ldd: đủ thư viện hệ thống")
    except FileNotFoundError:
        warn("không có ldd, bỏ qua kiểm thư viện")
    print("  ... mở Chromium bằng Playwright", flush=True)
    try:
        from textfix.render.browser import Browser as Reader
        t0 = time.time()
        with Reader() as R:
            R.page.set_content("<p style='font-size:40px'>Ệ Ợ Ữ</p>")
            R.page.screenshot()
        ok(f"Playwright mở được Chromium + font + chụp ảnh ({time.time() - t0:.1f}s)")
    except Exception as e:
        bad(f"Playwright không mở được {exe} ({str(e).splitlines()[0][:200]})",
            "Chrome for Testing phải gần đời với playwright (pip install 'playwright==<bản khớp>'); xem dòng lỗi đầy đủ")


# ---------------------------------------------------------------------------------------------------- 8. LLM / VLM
def tiny_png(text: str, w: int = 320, h: int = 120) -> bytes:
    import io
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(im).text((20, 40), text, fill="black")
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def call_json(url: str, model: str, key_env: str, images: list[bytes], ask: str) -> str:
    """Một lời gọi ép khuôn JSON đúng cách textfix gọi (json_schema, tắt suy nghĩ Qwen)."""
    import base64
    from textfix.llm import client as openai_client, extra_body
    schema = {"type": "object", "additionalProperties": False, "required": ["answer"], "properties": {"answer": {"type": "string"}}}
    content: list | str = [{"type": "text", "text": ask + ' Reply as JSON {"answer": "..."}.'}] + [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(p).decode()}} for p in images]
    if not images:
        content = ask + ' Reply as JSON {"answer": "..."}.'
    r = openai_client(url, key_env).chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}],
        response_format={"type": "json_schema", "json_schema": {"name": "t", "schema": schema, "strict": True}},
        extra_body=extra_body())
    return json.loads(r.choices[0].message.content)["answer"]


def check_models(no_call: bool) -> None:
    url, model = os.environ.get("TEXTFIX_LLM_URL"), os.environ.get("TEXTFIX_MODEL")
    vurl = os.environ.get("TEXTFIX_VLM_URL") or url
    vmodel = os.environ.get("TEXTFIX_VLM_MODEL") or model
    dmodel = os.environ.get("TEXTFIX_DESIGNER_MODEL") or vmodel
    print(f"  LLM: {url} / {model}\n  VLM: {vurl} / {vmodel}\n  designer: {dmodel}")
    if not url or not model:
        bad("chưa đặt TEXTFIX_LLM_URL / TEXTFIX_MODEL -- không còn OpenAI, phải dùng model nội bộ",
            "export TEXTFIX_LLM_URL / TEXTFIX_MODEL / TEXTFIX_LLM_API_KEY (docs/SERVER.md mục 3), ví dụ TEXTFIX_LLM_URL=<endpoint>/v1")
        return
    if not vurl or not vmodel:
        bad("chưa có VLM (TEXTFIX_VLM_URL / TEXTFIX_VLM_MODEL)", "đặt VLM nhìn được ảnh, ví dụ Qwen nội bộ")
        return
    if no_call:
        return
    from textfix.llm import client as openai_client
    for name, u, env in (("LLM", url, "TEXTFIX_LLM_API_KEY"), ("VLM", vurl, "TEXTFIX_VLM_API_KEY")):
        print(f"  ... hỏi danh sách model ở {u}", flush=True)
        try:
            ids = [m.id for m in openai_client(u, env).models.list().data]
            print(f"  {name} {u} phục vụ: {ids}")
            want = {"LLM": model, "VLM": vmodel}[name]
            ok(f"có model {want}") if want in ids else bad(f"{u} không có model '{want}'", f"đặt đúng tên trong {ids}")
        except Exception as e:
            bad(f"không lấy được danh sách model ở {u}: {type(e).__name__}: {str(e)[:200]}", "kiểm địa chỉ / khoá (Authorization Bearer)")
    tests = [("LLM chữ", url, model, "TEXTFIX_LLM_API_KEY", [], "Say ok."),
             ("VLM 1 ảnh", vurl, vmodel, "TEXTFIX_VLM_API_KEY", [tiny_png("SALE 50%")], "What text is in the image?"),
             (f"VLM {MM_IMAGES} ảnh", vurl, dmodel, "TEXTFIX_VLM_API_KEY",
              [tiny_png(f"#{i + 1}") for i in range(MM_IMAGES)], "How many images did you receive? Answer with a number.")]
    for name, u, m, env, imgs, ask in tests:
        print(f"  ... gọi {name} (chờ tối đa {os.environ['TEXTFIX_LLM_TIMEOUT']}s)", flush=True)
        try:
            t0 = time.time()
            ans = call_json(u, m, env, imgs, ask)
            ok(f"{name}: JSON đúng khuôn ({time.time() - t0:.1f}s): {ans!r}")
            if len(imgs) > 1 and str(len(imgs)) not in ans:
                warn(f"VLM trả lời '{ans}' (gửi {len(imgs)} ảnh) -- có thể máy chủ cắt bớt ảnh")
        except Exception as e:
            msg = str(e)
            fix = "VLM phải nhìn được ảnh; vLLM bật json_schema"
            if len(imgs) > 1 and any(s in msg.lower() for s in ("at most", "limit", "too many", "multimodal", "max")):
                fix = f"vLLM cần --limit-mm-per-prompt '{{\"image\": 16}}' và --max-model-len >= 32768 (designer gửi tới {MM_IMAGES} ảnh)"
            bad(f"{name} lỗi: {type(e).__name__}: {msg[:300]}", fix)


# ---------------------------------------------------------------------------------------------------- main
def check_local_flux(a, data) -> None:
    print("3. Trọng số FLUX.2-klein (distill + base) -- FLUX chạy tại chỗ")
    from textfix.draw.flux import find_weights
    extra = [Path(os.path.expanduser(a.weights_dir))] if a.weights_dir else []
    if data:
        extra += [d for d in data.iterdir() if d.is_dir() and "klein" in d.name.lower()]
    w = find_weights(extra)
    for k in ("distill", "base"):
        if k in w:
            ok(f"DiT {k}: {w[k]} ({gb(Path(w[k]).stat().st_size)})")
        else:
            bad(f"không thấy DiT {k}", "đặt TEXTFIX_WEIGHTS hoặc KLEIN_4B_MODEL_PATH / KLEIN_4B_BASE_MODEL_PATH tới file .safetensors")
    ok(f"VAE: {w['vae']}") if "vae" in w else bad("không thấy vae/diffusion_pytorch_model.safetensors", "chép thư mục vae/ của FLUX.2-klein")
    if "text_encoder" in w:
        te = Path(w["text_encoder"])
        ok(f"bộ mã hoá chữ: {te} ({dir_size(te)})")
        tok = te.parent / "tokenizer"
        ok(f"tokenizer: {tok}") if (tok / "tokenizer_config.json").exists() or (te / "tokenizer_config.json").exists() else \
            bad(f"không thấy tokenizer cạnh {te}", "chép thư mục tokenizer/ cùng cấp text_encoder/")
        q = json.loads((te / "config.json").read_text(encoding="utf-8")).get("quantization_config")
        if q:
            warn(f"bộ mã hoá chữ lượng tử hoá {q.get('quant_method')} -- A30 không có FP8 phần cứng, lỗi nạp thì cần bản bf16")
    else:
        bad("không thấy text_encoder/config.json", "chép thư mục text_encoder/ (Qwen3-4B) của FLUX.2-klein")


def check_remote_flux(no_call: bool) -> None:
    """FLUX ở máy khác (textfix/draw/remote.py): có địa chỉ + khoá; gọi thử một ảnh nhỏ (tốn một lượt vẽ) trừ khi --no-call."""
    from textfix.draw import remote
    F = remote.RemoteFlux()
    extra = f"model {F.model}, {F.steps} bước" if F.api == "chat" else f"safety_tolerance {F.safety}"
    print(f"3. FLUX ở máy khác, khuôn {F.api}: {F.url} (gửi cùng lúc {remote.concurrency()}, {extra})")
    if F.url.rstrip("/").endswith("-preview"):
        warn("điểm cuối -preview đổi model theo thời gian: cùng seed có thể ra ảnh khác -- nên dùng bản cố định")
    if not F.key:
        warn("chưa đặt TEXTFIX_FLUX_API_KEY (được, nếu máy FLUX không đòi khoá)")
    if no_call:
        return
    try:
        t0 = time.time()
        img = F.generate('A red poster with the words "SALE 50%" in bold white letters', 512, 512, 1)
        ok(f"FLUX vẽ thử được ảnh {img.shape[1]}x{img.shape[0]} trong {time.time() - t0:.1f} s")
    except Exception as e:
        bad(f"gọi FLUX lỗi ({e})", "kiểm TEXTFIX_FLUX_URL / TEXTFIX_FLUX_API_KEY, mạng tới máy FLUX")


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", line_buffering=True)   # in ngay từng dòng (kể cả qua | tee)
    # image notebook đặt libcuda lệch driver (thư mục .../compat) lên đầu LD_LIBRARY_PATH (lỗi 803 ngày 01/10): bộ nạp đọc biến
    # này lúc tiến trình khởi động -> bỏ compat rồi tự chạy lại (giống scripts/start_server.sh)
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if "/compat" in ld:
        os.environ["LD_LIBRARY_PATH"] = ":".join(p for p in ld.split(":") if "/compat" not in p)
        os.execv(sys.executable, [sys.executable, *sys.argv])
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default=os.environ.get("TEXTFIX_WEIGHTS"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("TEXTFIX_PORT", "8088")))
    ap.add_argument("--no-call", action="store_true", help="không gọi thử LLM / VLM / FLUX ở xa")
    ap.add_argument("--fix", action="store_true", help="giải nén Chromium trong persistent-data, ghi PLAYWRIGHT_CHROME_PATH vào .env")
    a = ap.parse_args()
    os.environ.setdefault("TEXTFIX_LLM_TIMEOUT", "300")      # lời gọi LLM / VLM treo thì báo lỗi sau 120 s, không đơ mãi
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # không ra HuggingFace: đừng treo chờ mạng
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from textfix.config import load_env
    load_env()

    print("0. Thư mục")
    print(f"  repo: {ROOT}")
    if HOME / "work" not in ROOT.parents:
        warn(f"repo không nằm trong {HOME / 'work'} (quy ước máy chủ) -- vẫn chạy được")
    ok("có .env (biến đã export vẫn thắng)") if (ROOT / ".env").exists() else print("  (không có .env: dùng biến đã export, docs/SERVER.md mục 3)")
    from textfix.draw import remote
    remote = remote.enabled()
    data = persistent()
    if data is None:
        if remote:   # FLUX ở máy khác (docker): model gỡ chữ chỉ cần TEXTFIX_HISAM / TEXTFIX_LAMA
            warn("không thấy ~/persistent-data -- được, nếu đã đặt TEXTFIX_HISAM / TEXTFIX_LAMA")
        else:
            bad("không thấy ~/persistent-data", "đặt TEXTFIX_DATA=<thư mục chứa model> hoặc tạo ~/persistent-data")
    else:
        print(f"  persistent-data: {data}")
        for p in sorted(data.iterdir()):
            print(f"    {p.name}{'/' if p.is_dir() else ''}  {dir_size(p) if p.is_dir() else gb(p.stat().st_size)}")
        try:
            u = shutil.disk_usage(data)
            print(f"  ổ đĩa còn trống {gb(u.free)} / {gb(u.total)}")
        except OSError:
            pass

    print("1. Gói pip (requirements.txt)")
    print(f"  python {sys.version.split()[0]} ({sys.executable})")
    check_requirements()

    print("2. GPU")
    print("  ... import torch", flush=True)
    try:
        import torch
        print(f"  torch {torch.__version__}, CUDA {torch.version.cuda}")
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n == 0 and remote:
            warn("không thấy GPU: Hi-SAM + LaMa sẽ chạy CPU (mỗi poster chậm thêm vài giây)")
        elif n == 0:
            bad(f"torch {torch.__version__} không thấy GPU", "torch phải là bản CUDA; kiểm tra nvidia-smi, LD_LIBRARY_PATH")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            free, tot = torch.cuda.mem_get_info(i)
            ok(f"cuda:{i} {p.name} {gb(tot)}, đang trống {gb(free)}")
    except Exception as e:
        bad(f"không import được torch ({e})", "dùng môi trường Python đang chạy repo v3 (đã có torch CUDA)")

    if remote:
        check_remote_flux(a.no_call)
    else:
        check_local_flux(a, data)

    print("3b. Model gỡ chữ: Hi-SAM (tìm nét chữ) + LaMa (vẽ lại nền) -- BẮT BUỘC từ 14/10")
    from textfix import erase
    from textfix.draw import hisam
    lost = hisam.missing()
    ok(f"Hi-SAM đủ tệp ở {hisam.HOME}") if not lost else bad("thiếu tệp Hi-SAM: " + "; ".join(lost), "docs/SERVER.md, mục Hi-SAM")
    ok(f"LaMa: {erase.LAMA_PATH}") if erase.lama_ready() else bad(f"thiếu {erase.LAMA_PATH}", "docs/SERVER.md, mục LaMa")
    try:
        import einops  # noqa: F401
        ok("import einops được (Hi-SAM cần)")
    except ImportError:
        bad("thiếu gói einops", "pip install einops (máy không ra internet: chép tệp .whl lên rồi pip install <tệp>)")

    print("4. Tài nguyên trong repo")
    nf = len(list((ROOT / "assets" / "fonts").glob("*")))
    ok(f"{nf} font") if nf >= 40 else bad(f"chỉ có {nf} font trong assets/fonts", "git pull / chép lại assets/fonts")
    if not remote:
        try:
            from flux2 import util  # noqa: F401
            ok("import flux2 được")
        except ModuleNotFoundError as e:
            if e.name != "flux2":
                bad(f"import flux2 lỗi ({e!r})", "xem thông báo; thường do transformers cũ")
            else:
                from textfix.draw.flux import FLUX2_MISSING
                bad("thiếu gói flux2 (mã BFL, không đi kèm repo)", FLUX2_MISSING)
        except BaseException as e:
            bad(f"import flux2 lỗi ({e!r})", "xem thông báo; thường do transformers cũ")
    print("5. Chromium (dựng chữ)")
    try:
        check_chrome(data, a.fix)
    except Exception as e:
        bad(f"kiểm Chromium lỗi ({e})", "pip install playwright; xem thông báo")

    print("6. OCR (RapidOCR, CPU)")
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from textfix.lines.ocr import read_lines
        im = Image.new("RGB", (600, 160), "white")
        f = next(iter(sorted((ROOT / "assets" / "fonts").glob("BeVietnamPro-Black.*"))), None)
        ImageDraw.Draw(im).text((20, 40), "SALE 50%", fill="black", font=ImageFont.truetype(str(f), 64) if f else None)
        ls = read_lines(np.asarray(im), multi=False)
        ok(f"OCR đọc được: {[l['text'] for l in ls]}") if ls else bad("OCR không đọc ra dòng nào", "pip install -U rapidocr-onnxruntime")
    except Exception as e:
        bad(f"OCR lỗi ({e})", "pip install rapidocr-onnxruntime (mô hình ONNX nằm sẵn trong gói)")

    print("7. LLM (prompt FLUX + câu khách) và VLM (designer) -- nội bộ")
    check_models(a.no_call)

    print("8. Cổng")
    with socket.socket() as s:
        busy = s.connect_ex(("127.0.0.1", a.port)) == 0
    warn(f"cổng {a.port} đang có tiến trình dùng") if busy else ok(f"cổng {a.port} còn trống")
    print("\n" + ("SẴN SÀNG." if not fails else f"CÒN {len(fails)} LỖI -- sửa theo dòng 'SỬA:' rồi chạy lại."))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
