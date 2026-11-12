"""Mã JavaScript chạy trong trang Chromium: đặt từng phần tử, đo bề rộng tự nhiên, nén / co, dựng chữ theo đường cong.
"""

from __future__ import annotations

RENDER_JS = r"""
async (o) => {
  const c = document.getElementById('c');
  // nạp HẾT font khai báo trước khi đo: fonts.ready xong ngay khi chưa có chữ nào dùng font (chưa bắt đầu tải) -> lần dựng đầu
  // của trang đo bằng font dự phòng (cỡ / tràn sai), các lần sau đúng nhờ bộ nhớ đệm
  await Promise.all([...document.fonts].map(f => f.load().catch(() => null)));
  await document.fonts.ready;
  const out = [];
  for (const e of o.els) {
    // CHỮ THEO ĐƯỜNG (ô cung / lượn của textfix/lines/curve.py: e.path = các điểm ĐƯỜNG CHÂN, hướng đọc trái -> phải): dựng bằng SVG
    // textPath, chữ căn giữa đường. Kiểu chữ lấy từ chính lệnh của VLM (đo trên một khối ẩn), như mọi lệnh khác. Chữ dài hơn
    // đường: nén ngang tới o.condense rồi mới co cỡ -- cùng thứ tự với dòng thẳng.
    if (e.path && e.path.length >= 2) {
      const c0 = document.getElementById('c'), R0 = c0.getBoundingClientRect();
      const h = document.createElement('div');
      h.style.cssText = `position:absolute;visibility:hidden;white-space:nowrap;line-height:1.15;font-family:${e.font};color:${e.color};` +
                        `font-size:${e.size}px;` + (e.style || '');
      h.innerHTML = e.html;
      c0.appendChild(h);
      let el = h, best = 0;
      const wk = document.createTreeWalker(h, NodeFilter.SHOW_TEXT);
      for (let n = wk.nextNode(); n; n = wk.nextNode()) { const k = n.textContent.trim().length; if (k > best) { best = k; el = n.parentElement; } }
      const cs = getComputedStyle(el);
      let txt = h.textContent.replace(/\s+/g, ' ').trim();
      if (cs.textTransform === 'uppercase') txt = txt.toUpperCase(); else if (cs.textTransform === 'lowercase') txt = txt.toLowerCase();
      // chữ tô dải màu (background-clip:text, ruột chữ trong suốt): SVG không tô được nền vào chữ -> lấy màu chữ của lệnh / màu đo
      const clear = v => !v || v === 'transparent' || /rgba\([^)]*,\s*0\)$/.test(v);
      const grad = /text/.test(cs.webkitBackgroundClip || cs.backgroundClip || '') ? (cs.backgroundImage.match(/rgba?\([^)]*\)/) || [])[0] : null;
      const fill = [cs.webkitTextFillColor, grad, cs.color, e.color].find(v => !clear(v)) || '#111';
      const sty = {ff: cs.fontFamily, fw: cs.fontWeight, fi: cs.fontStyle, fg: fill, ls: cs.letterSpacing,
                   sw: parseFloat(cs.webkitTextStrokeWidth) || 0, sc: cs.webkitTextStrokeColor, sh: cs.textShadow,
                   k: (parseFloat(cs.fontSize) || e.size) / e.size};
      h.remove();
      const NS = 'http://www.w3.org/2000/svg', mk = t => document.createElementNS(NS, t);
      const svg = mk('svg'), p = mk('path'), t = mk('text'), tp = mk('textPath');
      svg.setAttribute('width', R0.width); svg.setAttribute('height', R0.height);
      svg.setAttribute('viewBox', `0 0 ${R0.width} ${R0.height}`);
      svg.style.cssText = 'position:absolute;left:0;top:0;overflow:visible;pointer-events:none';
      if (e.cls) svg.setAttribute('class', e.cls);
      p.setAttribute('id', 'tp_' + e.id); p.setAttribute('fill', 'none');
      p.setAttribute('d', 'M' + e.path.map(q => q[0].toFixed(1) + ',' + q[1].toFixed(1)).join(' L'));
      let s = e.size * sty.k;
      t.style.cssText = `font-family:${sty.ff};font-weight:${sty.fw};font-style:${sty.fi};font-size:${s}px;fill:${sty.fg};` +
                        `letter-spacing:${sty.ls};` + (sty.sh && sty.sh !== 'none' ? `text-shadow:${sty.sh};` : '') +
                        (sty.sw > 0 ? `stroke:${sty.sc};stroke-width:${sty.sw}px;stroke-linejoin:round;` +
                                      'paint-order:stroke fill;' : '');   // viền dưới ruột chữ (xem PAINT ở khung chữ thẳng)
      tp.setAttribute('href', '#tp_' + e.id); tp.setAttribute('startOffset', '50%'); tp.setAttribute('text-anchor', 'middle');
      tp.textContent = txt;
      t.appendChild(tp); svg.appendChild(p); svg.appendChild(t); c0.appendChild(svg);
      const span = p.getTotalLength();
      let w = t.getComputedTextLength(), sx = 1, k = 0;
      const w0 = w;
      if (w > span) {   // nén ngang (giãn chữ + nét) rồi mới co cỡ
        sx = Math.max(e.condense || o.condense, span / w);
        tp.setAttribute('lengthAdjust', 'spacingAndGlyphs'); tp.setAttribute('textLength', (w * sx).toFixed(1));
        while (w * sx > span + 1 && s > o.floor && k < 80) {
          s *= 0.95; t.style.fontSize = s + 'px'; tp.removeAttribute('textLength'); w = t.getComputedTextLength();
          tp.setAttribute('textLength', (w * sx).toFixed(1)); k++;
        }
      }
      const used = w * sx, a0 = Math.max(0, (span - used) / 2), N = 8, polys = [];
      // nét mực = dải chữ dọc đường (từng khúc là một tứ giác lồi: chân chữ -> đỉnh chữ hoa, phía "trên" của hướng đọc)
      for (let i = 0; i < N; i++) {
        const u0 = a0 + used * i / N, u1 = a0 + used * (i + 1) / N;
        const A = p.getPointAtLength(u0), B = p.getPointAtLength(u1);
        const dx = B.x - A.x, dy = B.y - A.y, L = Math.hypot(dx, dy) || 1, nx = dy / L, ny = -dx / L, hh = 0.8 * s;
        polys.push([[A.x, A.y], [B.x, B.y], [B.x + nx * hh, B.y + ny * hh], [A.x + nx * hh, A.y + ny * hh]]);
      }
      const xs = polys.flat().map(q => q[0]), ys = polys.flat().map(q => q[1]);
      const bb = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
      out.push({id: e.id, size: s / sty.k, shrink: e.size ? s / sty.k / e.size : 1, over: Math.max(0, used - span - 1), o0: [Math.max(0, w0 - span), 0],
                fg: sty.fg, fx: sty.sw > 0 || (sty.sh && sty.sh !== 'none'), fs: s, ff: sty.ff, fw: parseInt(sty.fw), fi: sty.fi, ls: sty.ls,
                tt: cs.textTransform, sx: +sx.toFixed(2), ink0: null, angle: 0, runs: [], path: true, polys, ink: bb, box: bb});
      continue;
    }
    const d = document.createElement('div');
    d.style.cssText = `position:absolute;left:${e.box[0]}px;top:${e.box[1]}px;box-sizing:border-box;display:flex;align-items:center;` +
      `justify-content:center;text-align:center;line-height:1.15;font-synthesis:style;font-family:${e.font};color:${e.color};font-size:${e.size}px;` +
      (e.fit === 'grow' ? `min-width:${e.box[2] - e.box[0]}px;min-height:${e.box[3] - e.box[1]}px;width:max-content;` :
                          `width:${e.box[2] - e.box[0]}px;height:${e.box[3] - e.box[1]}px;`) + (e.style || '') +
      // VIỀN DƯỚI RUỘT CHỮ (12/11): font biến thiên (Montserrat, Playfair...) ghép mỗi chữ từ nhiều khối chồng nhau;
      // -webkit-text-stroke vẽ viền quanh TỪNG khối -> đường nối lộ giữa chữ "t", "4", "k", "Đ" (người dùng: "chữ t ghép bởi 2
      // khối"). Vẽ viền trước, ruột phủ lên: chỉ còn viền ngoài
      'paint-order:stroke fill;';
    d.innerHTML = e.html;
    if (e.cls) d.className = e.cls;
    // CHỮ TRẦN LẪN THẺ ngay trong khung (12/10 form1 f09: câu trích 'chữ <span>đậm</span> chữ <span>đậm</span> chữ' không bọc
    // thẻ nào): khung là flex nên mỗi mẩu chữ / mỗi span thành một CỘT xếp ngang -> tràn, co còn 0.32. Có mẩu chữ trần ở cấp
    // ngoài = nội dung chảy theo dòng -> bọc một span (một con flex). Bố cục cột thật (các span, không chữ trần) giữ nguyên.
    if (d.childNodes.length > 1 && [...d.childNodes].some(n => n.nodeType === 3 && n.textContent.trim()) &&
        ![...d.children].some(x => /display\s*:\s*(inline-)?flex/.test(x.getAttribute('style') || ''))) {
      const w = document.createElement('span');
      while (d.firstChild) w.appendChild(d.firstChild);
      d.appendChild(w);
    }
    // CỘT FLEX trong thẻ font: VLM viết <i-font><span style="flex:1">tên</span><span style="flex:0 0 30%">giá</span></i-font> --
    // span bọc font (inline) là con flex duy nhất, tên và giá dính liền "Bún bò tái40.000đ" (09/10 dev2 d05) -> bọc font
    // display:contents, các span flex thành con flex thật của khung (vẫn thừa hưởng font)
    // (icon / sao là SVG có "flex:none" trong style -- không phải cột flex: 13/10 accept1 a04, ba dòng có icon đầu dòng bị trải
    // thành ba cột hẹp)
    for (const x of d.querySelectorAll('[style*="flex"]')) {
      if (x instanceof SVGElement) continue;
      // (con trực tiếp của khung flex bị "blockify": display tính ra là block, không phải inline -- xét theo thẻ span)
      for (let p = x.parentElement; p && p !== d && p.tagName === 'SPAN' && !/flex|grid/.test(getComputedStyle(p).display);
           p = p.parentElement)
        p.style.display = 'contents';
    }
    // dòng nghiêng: renderer tự xoay đúng góc đo (bên dưới) -- transform VLM ghi ở khung (rotate(-14deg)) bỏ trước khi đo,
    // không thì "đo trước khi xoay" là đo khối đã xoay (tràn / nét sai)
    if (e.angle) {
      d.style.transform = 'none';
      // ... và transform xoay / xiên VLM ghi ở các thẻ BÊN TRONG (13/10 dev3 g05, g13: span rotate(31deg) + renderer xoay 31 độ
      // = chữ xoay hai lần, dựng đứng)
      for (const x of d.querySelectorAll('[style]')) if (/rotate|skew|matrix/.test(x.style.transform)) x.style.transform = 'none';
    }
    c.appendChild(d);
    const R0 = c.getBoundingClientRect();
    // TRÀN = nội dung thật (mọi con, kể cả SVG) vượt khung ở BẤT KỲ phía nào (nội dung căn giữa tràn cả trái -- scrollWidth chỉ
    // đếm phía phải: p20 hàng sao thò ra ngoài huy hiệu bên trái)
    // KHUNG Ô ôm NÉT MỰC (cả dấu thanh), còn hộp chữ của trình duyệt = ascent + descent của font (đệm trên / dưới nét): so
    // thẳng thì mọi dòng "tràn dọc" và bị co ~0.6-0.8 (07/10: 80% lỗi too_small, chữ nhỏ / nhạt hơn nháp). Trừ phần đệm đó --
    // đo bằng measureText trên chính chữ của khối, theo font của PHẦN CHỮ CHIẾM NHIỀU NHẤT (i-font / span bên trong có font, cỡ
    // riêng -- đo theo font của khung thì sai: 08/10 h02 khung Plus Jakarta, chữ Baloo dấu cao -> "tràn dọc" oan, co còn 0.6),
    // tính theo đơn vị cỡ của khung (tỉ lệ theo cỡ, đo một lần ở 100px).
    const pad = (() => {
      const t = d.textContent.trim();
      if (!t) return {top: 0, bot: 0};
      let el = d, best = 0;
      const w = document.createTreeWalker(d, NodeFilter.SHOW_TEXT);
      for (let n = w.nextNode(); n; n = w.nextNode()) {
        const k = n.textContent.trim().length;
        if (k > best) { best = k; el = n.parentElement; }
      }
      const cs = getComputedStyle(el), cx = document.createElement('canvas').getContext('2d');
      const r = parseFloat(cs.fontSize) / (parseFloat(getComputedStyle(d).fontSize) || 1);
      cx.font = `${cs.fontStyle} ${cs.fontWeight} 100px ${cs.fontFamily}`;
      const mt = cx.measureText(t);
      return {top: r * Math.max(0, mt.fontBoundingBoxAscent - mt.actualBoundingBoxAscent) / 100,
              bot: r * Math.max(0, mt.fontBoundingBoxDescent - mt.actualBoundingBoxDescent) / 100};
    })();
    let inner = d;   // phần đo nội dung (sau khi nén ngang: bên trong span nén -- hộp span cao theo line-height, đo nó là tràn dọc oan)
    const over = () => {
      const rr = document.createRange(); rr.selectNodeContents(inner);
      const q = rr.getBoundingClientRect(), b = d.getBoundingClientRect();
      const fs = parseFloat(d.style.fontSize) || 0;
      // DÒNG ĐƠN (e.hfit): chỉ kiểm tràn NGANG -- khung ô = khung OCR, không ôm dấu thanh (12/10 bỏ bước nắn khung); nền đã xoá
      // sạch nên dấu / đuôi chữ nhô khỏi khung trên dưới không đè vào đâu, co chữ vì nó là co oan
      const vy = e.hfit ? 0 : Math.max(b.top - q.top - pad.top * fs, q.bottom - b.bottom - pad.bot * fs);
      return {q, v: Math.max(0, b.left - q.left, q.right - b.right, vy)};
    };
    // ô KHÔNG có nội dung (vỏ chỉ có CSS): không đo tràn -- khung nội dung rỗng nằm ở (0,0) cho số tràn ảo (07/10 p11: 429 / 916 px)
    if (!d.textContent.trim() && !d.querySelector('svg')) {
      const b0 = d.getBoundingClientRect();
      out.push({id: e.id, size: e.size, shrink: 1, over: 0, empty: true,
                ink: [b0.left - R0.left, b0.top - R0.top, b0.right - R0.left, b0.bottom - R0.top],
                box: [b0.left - R0.left, b0.top - R0.top, b0.right - R0.left, b0.bottom - R0.top]});
      continue;
    }
    let s = e.size, k = 0, m = over(), sx = 1;
    const m0q = m.q, m0b = d.getBoundingClientRect(), fs0 = parseFloat(d.style.fontSize) || 0;   // tràn lúc CHƯA co: ngang / dọc
    let o0 = [Math.max(0, m0b.left - m0q.left, m0q.right - m0b.right),
              Math.max(0, m0b.top - m0q.top - pad.top * fs0, m0q.bottom - m0b.bottom - pad.bot * fs0)];
    // NGẮT LẠI: khối nhiều dòng (<br> chép theo dòng nháp) tràn ngang -> dòng dài tự gãy thêm, chữ mồ côi ("Khách hàng nói / gì",
    // "Tóc mềm mượt / sau": 09/10 dev3 p01, f02) -> thử bỏ <br>, trình duyệt tự ngắt; GIỮ nếu vừa khung ở nguyên cỡ
    if (o0[0] > 1 && /<br/i.test(d.innerHTML) && !d.querySelector('div,p,ul,ol,li,table,section,[style*="flex"]')) {
      const keep = d.innerHTML, ws = d.style.whiteSpace;
      d.innerHTML = keep.replace(/\s*<br\s*\/?>\s*/gi, ' ');
      d.style.whiteSpace = 'normal'; d.style.textWrap = 'pretty';
      const m2 = over();
      if (m2.v <= 1) { m = m2; o0 = [0, 0]; }
      else { d.innerHTML = keep; d.style.whiteSpace = ws; d.style.textWrap = ''; }
    }
    // NÉN NGANG trước khi co: chữ dài hơn ô (tràn ngang) -> scaleX tới o.condense, giữ cỡ chữ (chiều cao) như nháp -- co cỡ
    // làm chữ bé hẳn so với nháp (08/10 server: too_small 33 / 27 ảnh). Chỉ khối chữ dòng (không div / p / li bên trong: bọc
    // lại sẽ đổi bố cục flex của VLM); tâm nén theo căn lề của khung.
    if (((e.fit !== 'grow' && e.fit !== 'none') || e.even) && o0[0] > 1 && !d.querySelector('div,p,ul,ol,li,table,section,[style*="flex"]')) {
      sx = Math.max(e.condense || o.condense, Math.min(1, m0b.width / Math.max(1, m0q.width)));
      const j = getComputedStyle(d).justifyContent;
      const org = /start|left/.test(j) ? 'left' : /end|right/.test(j) ? 'right' : 'center';
      d.innerHTML = `<span style="display:inline-block;flex:none;white-space:inherit;transform:scaleX(${sx});` +
                    `transform-origin:${org} center">${d.innerHTML}</span>`;
      inner = d.firstElementChild;
      m = over();
    }
    if (e.fit !== 'grow' && e.fit !== 'none') {   // co nội dung cho vừa khung (em theo font-size của khung)
      while (m.v > 1 && s > o.floor && k < 80) { s *= 0.95; d.style.fontSize = s + 'px'; k++; m = over(); }
    }
    const m0 = m.v;   // tràn đo khi CHƯA xoay (sau khi xoay, khung thẳng ôm ngoài lệch so với nội dung)
    let ink0 = null;   // chữ xoay: nét mực CHƯA xoay + tâm xoay -> đa giác nét thật (hộp thẳng ôm chữ xoay chồng oan dòng kề)
    if (e.angle) {
      const q0 = m.q, b0 = d.getBoundingClientRect(), f0 = parseFloat(d.style.fontSize) || 0;
      ink0 = {r: [q0.left - R0.left, q0.top - R0.top + Math.min(pad.top * f0, q0.height / 3),
                  q0.right - R0.left, q0.bottom - R0.top - Math.min(pad.bot * f0, q0.height / 3)],
              c: [(b0.left + b0.right) / 2 - R0.left, (b0.top + b0.bottom) / 2 - R0.top]};
    }
    if (e.angle) { d.style.transformOrigin = '50% 50%'; d.style.transform = `rotate(${e.angle}deg)`; m = over(); m.v = m0; }
    const q = m.q, b = d.getBoundingClientRect();
    // NÉT MỰC = hộp chữ trừ phần đệm trên / dưới của font (dòng chồng sát nhau như FLASH / SALE thì hộp font chồng 0.2-0.3 dù
    // nét không chạm: 08/10 server 15/27 ảnh báo content_overlap oan, vòng sửa chạy vì nó). Chữ xoay: giữ hộp ngoài.
    const fsn = parseFloat(d.style.fontSize) || 0;
    const it = e.angle ? 0 : Math.min(pad.top * fsn, (q.bottom - q.top) / 3), ib = e.angle ? 0 : Math.min(pad.bot * fsn, (q.bottom - q.top) / 3);
    // MÀU CHỮ thật (màu tính được của phần tử chứa nhiều chữ nhất) + có hiệu ứng tách nền không (bóng / viền / chữ tô gradient)
    let fg = null, fx = false, best = 0, fs = null, ff = null, fw = null, fi = null, ls = null, tt = null;
    const runs = [];   // từng đoạn chữ (màu / hiệu ứng riêng): khối hai màu -- "Giảm" trắng + "40%" cam trên nền đỏ (09/10 dev2 d11)
    const tw = document.createTreeWalker(d, NodeFilter.SHOW_TEXT);
    for (let t = tw.nextNode(); t; t = tw.nextNode()) {
      const n = t.textContent.trim().length, cs = getComputedStyle(t.parentElement);
      const tfx = cs.textShadow !== 'none' || parseFloat(cs.webkitTextStrokeWidth) > 0 || cs.backgroundClip === 'text' ||
          cs.webkitBackgroundClip === 'text';
      if (tfx) fx = true;
      if (n > best) { best = n; fg = cs.webkitTextFillColor || cs.color; fs = parseFloat(cs.fontSize); ff = cs.fontFamily; fw = parseInt(cs.fontWeight);
                      fi = cs.fontStyle; ls = cs.letterSpacing; tt = cs.textTransform; }
      if (n >= 2) {
        const rg = document.createRange(); rg.selectNodeContents(t);
        const rc = rg.getBoundingClientRect();
        runs.push({n, fg: cs.webkitTextFillColor || cs.color, fx: tfx,
                   ink: [rc.left - R0.left, rc.top - R0.top, rc.right - R0.left, rc.bottom - R0.top]});
      }
    }
    out.push({id: e.id, size: s, shrink: e.size ? s / e.size : 1, over: m.v, o0, fg, fx, fs, ff, fw, fi, ls, tt, sx: +sx.toFixed(2), ink0, angle: e.angle, runs,
              ink: [q.left - R0.left, q.top - R0.top + it, q.right - R0.left, q.bottom - R0.top - ib],
              box: [b.left - R0.left, b.top - R0.top, b.right - R0.left, b.bottom - R0.top]});
  }
  return out;
}
"""
