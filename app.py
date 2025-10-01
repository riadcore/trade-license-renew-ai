# Trade License Renew (DNCC/DSCC)
# ✅ Robust OCR + Regex
# ✅ Auto-detect Renew Fee + Signboard Charge (per year)
# ✅ Already Renewed logic
# ✅ Restart button
# ✅ No file downloads

import io, os, re, shutil, pathlib
from datetime import date, datetime
from typing import Dict, Any, Optional, Tuple

import gradio as gr
import fitz  # PyMuPDF
from PIL import Image
os.environ["TESSDATA_PREFIX"] = os.path.abspath("tessdata")

# -------- keep storage tiny on Spaces / containers --------
for p in ["~/.cache/huggingface", "~/.cache/pip", "~/.cache/torch", "~/.cache"]:
    pp = pathlib.Path(os.path.expanduser(p))
    if pp.exists():
        shutil.rmtree(pp, ignore_errors=True)
# ---------------------------------------------------------

# -------- OCR (for scanned PDFs / images) --------
try:
    import pytesseract
    # Point to tesseract.exe on Windows
    if os.name == "nt":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

FAST_OCR_CONFIG = r"--oem 1 --psm 6 -l ben+eng"
FALLBACK_OCR_CONFIGS = [
    r"--oem 1 --psm 11 -l ben+eng",  # sparse text
    r"--oem 1 --psm 4 -l ben+eng",   # block/column layout
]

def _prep_image_for_ocr(im: Image.Image) -> Image.Image:
    # 1) grayscale + moderate upscale
    scale = 1.5
    im = im.convert("L")
    im = im.resize((int(im.width * scale), int(im.height * scale)))
    # 2) light threshold normalization (keeps edges & digits)
    im = im.point(lambda x: 255 if x > 200 else (0 if x < 140 else x))
    # 3) clamp width (avoid huge memory)
    max_w = 2200
    if im.width > max_w:
        h = int(im.height * (max_w / im.width))
        im = im.resize((max_w, h))
    return im

def ocr_image_fast(im: Image.Image, config: str = FAST_OCR_CONFIG) -> str:
    if not OCR_AVAILABLE:
        return ""
    im = _prep_image_for_ocr(im)
    txt = pytesseract.image_to_string(im, config=config) or ""
    # If the text is too short or lacks digits, try fallbacks
    if len(txt.strip()) < 10 or not any(ch.isdigit() for ch in txt):
        for cfg in FALLBACK_OCR_CONFIGS:
            t2 = pytesseract.image_to_string(im, config=cfg) or ""
            if len(t2.strip()) > len(txt.strip()):
                txt = t2
    return txt

def extract_text_from_path(path: str, ocr_all_pages: bool = False) -> Tuple[str, str]:
    """Return (text, method). Prefer selectable text, else OCR."""
    p = pathlib.Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        try:
            with fitz.open(path) as doc:
                # 1) selectable text fast path
                txt = []
                for pg in doc:
                    t = pg.get_text("text")
                    if t:
                        txt.append(t)
                if any(len(t) > 30 for t in txt):
                    return "\n".join(txt).strip(), "pymupdf"
                # 2) OCR fallback (higher DPI for sharper numerals)
                pages = range(len(doc)) if ocr_all_pages else range(min(1, len(doc)))
                ocr_chunks = []
                for i in pages:
                    pix = doc[i].get_pixmap(dpi=220, alpha=False)  # try 300 if needed
                    im = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_chunks.append(ocr_image_fast(im))
                return "\n".join(ocr_chunks).strip(), "ocr_pdf"
        except Exception:
            pass

    # Image branch
    try:
        im = Image.open(path)
        return ocr_image_fast(im), "ocr_image"
    except Exception:
        return "", "none"


# =========================
# PARSERS
# =========================

# ---- Business name (Bangla exact labels) ----
BUSINESS_NAME_LABELS = [
    r"১[।.]?\s*ব্যবসা\s*প্রতিষ্ঠানের\s*নাম",
    r"ব্যবসা\s*প্রতিষ্ঠানের\s*নাম",
    r"প্রতিষ্ঠানের\s*নাম",
]

def extract_business_name(text: str) -> str:
    """
    Finds '১। ব্যবসা প্রতিষ্ঠানের নাম' (or variants) and returns the name.
    Works when the name is on the same line after ':' or on the next line(s).
    """
    if not text:
        return ""
    lines = text.splitlines()

    for i, ln in enumerate(lines):
        for lbl in BUSINESS_NAME_LABELS:
            if re.search(lbl, ln):
                # Try SAME LINE first
                tail = re.split(lbl, ln, flags=re.IGNORECASE, maxsplit=1)
                cand = ""
                if len(tail) > 1:
                    cand = tail[1].strip()

                def _clean(s: str) -> str:
                    s = s.strip()
                    s = re.sub(r"^[\s:।–—\-]+", "", s)
                    s = re.sub(r"^[০-৯0-9]+[).:-]*\s*", "", s)
                    return s.strip()

                if cand and not re.fullmatch(r"[:।–—\-]*", cand):
                    name = _clean(cand)
                    if name:
                        return name

                # Otherwise look in next few lines
                for j in range(i + 1, min(i + 4, len(lines))):
                    s = lines[j].strip()
                    if not s or re.fullmatch(r"[:।–—\-]*", s):
                        continue
                    name = _clean(s)
                    if name:
                        return name
    return ""


# --- Corp detection helpers ---
_ZW = re.compile(r"[\u200b\u200c\u200d\u2060]")   # zero-widths
def _clean_bn(s: str) -> str:
    s = normalize_digits(s or "")
    s = _ZW.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# Variants: কর্পোরেশন / করপোরেশন; allow extra spaces/newlines between words
DNCC_PATTERNS = [
    r"ঢাকা\s*উত্তর\s*সিটি\s*ক(?:র্?প)?পোরেশন",
    r"\bdncc\b", r"dncc\.gov\.bd", r"www\.dncc\.gov\.bd",
    r"\bDhaka\s*North\s*City\s*Corporation\b",
]
DSCC_PATTERNS = [
    r"ঢাকা\s*দক্ষিণ\s*সিটি\s*ক(?:র্?প)?পোরেশন",
    r"\bdscc\b", r"dscc\.gov\.bd", r"www\.dscc\.gov\.bd",
    r"\bDhaka\s*South\s*City\s*Corporation\b",
]


def detect_corporation(text: str) -> str:
    t = _clean_bn(text).lower()
    score = {"DNCC": 0, "DSCC": 0}

    # Strong signal: license number prefix
    m = re.search(r"\btrad\/(dscc|dncc)\/\d{3,}\/\d{4}\b", t, flags=re.I)
    if m:
        score[m.group(1).upper()] += 5

    # Headers / body text / URLs
    for pat in DNCC_PATTERNS:
        if re.search(pat, t, flags=re.I):
            score["DNCC"] += 2
    for pat in DSCC_PATTERNS:
        if re.search(pat, t, flags=re.I):
            score["DSCC"] += 2

    # Extra nudge if the lone words appear near 'সিটি'
    if re.search(r"উত্তর\s*সিটি", t): score["DNCC"] += 1
    if re.search(r"দক্ষিণ\s*সিটি", t): score["DSCC"] += 1

    if score["DNCC"] > score["DSCC"]:
        return "DNCC"
    if score["DSCC"] > score["DNCC"]:
        return "DSCC"
    return "UNKNOWN"


def extract_license_number(text: str) -> Optional[str]:
    m = re.search(r"(TRAD\/(?:DSCC|DNCC)\/\d{3,}\/\d{4})", text, flags=re.IGNORECASE)
    return m.group(1) if m else None

def extract_last_renew_year(text: str) -> Optional[str]:
    m = re.search(r"(20\d{2})\s*[-–]\s*(20\d{2})", text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b == a + 1:
            return f"{a}-{b}"
    return None

def extract_valid_until(text: str) -> Optional[str]:
    pats = [
        r"মেয়াদ.*?(?:৩০|30)\s*(?:জুন|June)[,]?\s*(\d{4})",
        r"(?:valid\s*until|validity).*?30\s*(?:June)[,]?\s*(\d{4})",
        r"(?:জুন|June)[,]?\s*(\d{4}).*?(?:পর্যন্ত|until)"
    ]
    for pat in pats:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            year = normalize_digits(m.group(1))   # 👈 force English digits
            return f"30 June {year}"

    years = re.findall(r"(20\d{2})", normalize_digits(text))  # 👈 normalize here too
    if years:
        return f"30 June {max(map(int, years))}"
    return None


def infer_last_renew_from_valid_until(valid_until: Optional[str]) -> Optional[str]:
    if not valid_until:
        return None
    m = re.search(r"(\d{4})$", valid_until.strip())
    if not m:
        return None
    end = int(m.group(1))
    return f"{end-1}-{end}"

# Bangla → Latin digits
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
def normalize_digits(s: Optional[str]) -> Optional[str]:
    return s.translate(BN_DIGITS) if s else s

def _to_float_safe(num_str: Optional[str]) -> Optional[float]:
    """
    Converts a captured money-like string to float.
    Handles Bangla digits, commas, hidden chars, zero-width separators, and digit runs.
    """
    if not num_str:
        return None

    orig = normalize_digits(num_str)

    # 1) Remove common noise
    cleaned = re.sub(r"[^\d.]", "", orig)
    if cleaned:
        try:
            return float(cleaned)
        except Exception:
            pass

    # 2) Join ALL digits, even across broken runs (handles 5​000 → 5000)
    digits_only = "".join(re.findall(r"\d", orig))
    if digits_only:
        try:
            return float(digits_only)
        except Exception:
            return None

    return None



def extract_amount_by_labels(text: str, label_patterns) -> Optional[float]:
    """
    Capture number near label (same/next line). Handles Bangla digits, currency signs,
    and separators like :, -, =, '/=' between label and number. Picks the FIRST plausible match.
    """
    t = normalize_digits(text or "")

    # allow 3,000 / 3 000 / 3000 (with optional decimals)
    # allow 3,000 / 3 000 / 3\u200B000 / 3000
    money_core = r"(?:\d[\d,，\s\u200b\u200c\u200d\u2060]*\d(?:\.\d+)?|\d+)"


    money = rf"(?:৳|Tk\.?|Taka|টাকা)?\s*{money_core}"
    sep = r"(?:\s*[:=–—\-\/=]\s*)?"

    def _plausible(v: Optional[float]) -> bool:
        # Typical per-year fees are in the hundreds to tens of thousands
        return v is not None and 200 <= v <= 50000

    lines = t.splitlines()

    for i, ln in enumerate(lines):
        for lbl in label_patterns:
            if re.search(lbl, ln, flags=re.IGNORECASE):

                # ---- SAME LINE (keep window tight so we don't grab years/VAT later on the line)
                m = re.search(lbl + sep + r".{0,32}?" + money, ln, flags=re.IGNORECASE)
                if m:
                    nums = re.findall(money_core, m.group(0))
                    for cand in nums:  # pick FIRST plausible
                        val = _to_float_safe(cand)
                        if _plausible(val):
                            return val

                # ---- NEXT NON-EMPTY LINE
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                if j < len(lines):
                    m2 = re.search(money, lines[j], flags=re.IGNORECASE)
                    if m2:
                        nums2 = re.findall(money_core, m2.group(0))
                        for cand in nums2:
                            val2 = _to_float_safe(cand)
                            if _plausible(val2):
                                return val2

    # ---- LAST RESORT: within 80 chars after label (across lines)
    for lbl in label_patterns:
        m3 = re.search(lbl + r".{0,80}?" + money, t, flags=re.IGNORECASE | re.DOTALL)
        if m3:
            nums3 = re.findall(money_core, m3.group(0))
            for cand in nums3:
                val3 = _to_float_safe(cand)
                if _plausible(val3):
                    return val3

    return None




def extract_per_year_fees(text: str) -> Tuple[Optional[float], Optional[float]]:
    renew_labels = [
        r"লাইসেন্স\s*/\s*নবায়ন\s*ফি",
        r"লাইসেন্স\s*ফি",
        r"নবায়ন\s*ফি",
        r"লাইসেন্স\s*নবায়ন\s*ফি",
        r"বার্ষিক\s*লাইসেন্স\s*(?:নবায়ন\s*)?ফি",
        r"বাৎসরিক\s*লাইসেন্স\s*(?:নবায়ন\s*)?ফি",
        r"Renew(?:al)?\s*Fee(?:s)?(?:\s*\(per\s*year\))?",
        r"License\s*Fee(?:s)?(?:\s*\(per\s*year\))?",
        r"Renew\s*Fee",
    ]
    sign_labels = [
        r"সাইনবোর্ড\s*কর\s*\(পরিচিতিমূলক\)",   # added priority label
        r"সাইনবোর্ড\s*কর",
        r"সাইনবোর্ড\s*চার্জ",
        r"সাইন\s*বোর্ড\s*কর",
        r"Signboard\s*(?:Charge|Tax)(?:\s*\(per\s*year\))?",
        r"Sign\s*Board\s*(?:Charge|Tax)",
    ]
    return extract_amount_by_labels(text, renew_labels), extract_amount_by_labels(text, sign_labels)


# =========================
# DUE & FINE
# =========================
def current_fy_end_year(today: date) -> int:
    return today.year + 1 if today.month >= 7 else today.year

def parse_last_renew_end_year(lbl: str) -> Optional[int]:
    m = re.match(r"\s*(20\d{2})\s*[-–]\s*(20\d{2})\s*$", lbl)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return b if b == a + 1 else None

def compute_due(last_renew_lbl: str, today: Optional[date] = None) -> int:
    today = today or date.today()
    cur_end = current_fy_end_year(today)
    last_end = parse_last_renew_end_year(last_renew_lbl)
    if last_end is None:
        return 1
    return max(1, cur_end - last_end)

def months_since_july(today: Optional[date] = None) -> int:
    today = today or date.today()
    m = today.month - 7
    return m if m > 0 else 0

def compute_fine_months(due_years: int, today: Optional[date] = None) -> int:
    today = today or date.today()
    msj = months_since_july(today)
    add = msj if msj > 2 else 0
    return max(0, (due_years * 12) - 12 + add)


# =========================
# CALCULATORS
# =========================
def calc_dncc(last_renew_lbl, renew_fee_py, signboard_py, source_tax_py=3000.0,
              service=500.0, others=500.0, book=270.0, bank=50.0) -> Dict[str, float]:
    due = compute_due(last_renew_lbl)
    fine_m = compute_fine_months(due)
    govt_renew = renew_fee_py * due
    signboard  = signboard_py * due
    source_tax = source_tax_py * due
    total_fine = (renew_fee_py * 0.10) * fine_m
    vat = 0.15 * (govt_renew + signboard + total_fine)
    total_govt = govt_renew + signboard + source_tax + total_fine + vat
    grand_total = total_govt + service + others + book + bank
    return {
        "Due (years)": float(due),
        "Fine for Month (auto)": float(fine_m),
        "Govt Renew Fee": float(govt_renew),
        "Signboard Charge": float(signboard),
        "Source TAX": float(source_tax),
        "Total Fine": float(total_fine),
        "VAT (15%)": float(vat),
        "Total Govt. Fees": float(total_govt),
        "Service Charge": float(service),
        "Others": float(others),
        "Book Charge": float(book),
        "Bank Charge": float(bank),
        "Grand Total": float(grand_total),
    }

def calc_dscc(last_renew_lbl, renew_fee_py, signboard_py, source_tax_py=3000.0,
              form_fee=50.0, service=1000.0, bank=50.0) -> Dict[str, float]:
    due = compute_due(last_renew_lbl)
    fine_m = compute_fine_months(due)
    govt_renew = renew_fee_py * due
    signboard  = signboard_py * due
    source_tax = source_tax_py * due
    total_fine = (renew_fee_py * 0.10) * fine_m
    vat = 0.15 * (govt_renew + signboard + total_fine)
    total_govt = govt_renew + signboard + source_tax + total_fine + vat + form_fee
    grand_total = total_govt + service + bank
    return {
        "Due (years)": float(due),
        "Fine for Month (auto)": float(fine_m),
        "Govt Renew Fee": float(govt_renew),
        "Signboard Charge": float(signboard),
        "Source TAX": float(source_tax),
        "Total Fine": float(total_fine),
        "VAT (15%)": float(vat),
        "Form Fee": float(form_fee),
        "Total Govt. Fees": float(total_govt),
        "Service Charge": float(service),
        "Bank Charge": float(bank),
        "Grand Total": float(grand_total),
    }


# =========================
# UI HELPERS
# =========================
def fmt_taka(x: float) -> str:
    try:
        return f"৳{int(x):,}" if float(x).is_integer() else f"৳{x:,.2f}"
    except Exception:
        return str(x)

BN_LABELS = {
    "Due (years)": "বকেয়া বছর",
    "Fine for Month (auto)": "মাসের জরিমানা (স্বয়ংক্রিয়)",
    "Govt Renew Fee": "সরকারি নবায়ন ফি",
    "Signboard Charge": "সাইনবোর্ড চার্জ",
    "Source TAX": "সূত্র কর",
    "Total Fine": "মোট জরিমানা",
    "VAT (15%)": "ভ্যাট (১৫%)",
    "Total Govt. Fees": "মোট সরকারি ফি",
    "Service Charge": "সেবামূল্য",
    "Others": "অন্যান্য",
    "Book Charge": "বইয়ের মূল্য",
    "Bank Charge": "ব্যাংক চার্জ",
    "Form Fee": "ফর্ম ফি",
    "Grand Total": "সর্বমোট"
}

def breakdown_to_html_bn(corp: str, lic: str, biz: str,
                         last_renew: str, due: int, bd: dict,
                         renew_py: Optional[float], sign_py: Optional[float]) -> str:
    rows = []
    for k, v in bd.items():
        if k == "Grand Total":
            continue
        rows.append(
            f"<tr><td>{BN_LABELS.get(k, k)}</td>"
            f"<td style='text-align:right'>{fmt_taka(v)}</td></tr>"
        )
    grand = (
        f"<tr style='font-weight:700;border-top:2px solid #444'>"
        f"<td>{BN_LABELS.get('Grand Total','Grand Total')}</td>"
        f"<td style='text-align:right'>{fmt_taka(bd.get('Grand Total',0))}</td></tr>"
    )
    extra = ""
    if renew_py is not None or sign_py is not None:
        extra = (
            "<div style='margin:.35rem 0 .2rem;color:#666;font-size:13px'>"
            f"Per-year (detected): Renew = {fmt_taka(renew_py or 0)}, "
            f"Signboard = {fmt_taka(sign_py or 0)}</div>"
        )
    hdr = f"""
    <div style="font-family:Inter,system-ui,Segoe UI,Arial; line-height:1.35; max-width:760px;">
      <h3 style="margin:.2rem 0;">ট্রেড লাইসেন্স নবায়ন — {corp}</h3>
      <div style="font-size:14px;color:#888;">
        লাইসেন্স: {lic or '-'} &nbsp;•&nbsp;
        প্রতিষ্ঠানের নাম: {biz or '-'} &nbsp;•&nbsp;
        শেষ নবায়ন বছর: {last_renew or '-'} &nbsp;•&nbsp;
        বকেয়া বছর: {due}
      </div>
      {extra}
      <table style="width:100%; border-collapse:collapse; margin-top:.5rem;">
        <thead>
          <tr style="text-align:left; background:#111; color:#ddd;">
            <th style="padding:8px;">আইটেম</th>
            <th style="padding:8px; text-align:right;">টাকা</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
          {grand}
        </tbody>
      </table>
    </div>
    """
    return hdr

def parse_valid_until_date(valid_until_label: Optional[str]) -> Optional[date]:
    if not valid_until_label:
        return None
    s = valid_until_label.strip().replace(",", "")
    try:
        return datetime.strptime(s, "%d %B %Y").date()
    except Exception:
        m = re.search(r"(20\d{2})", s)
        if m:
            return date(int(m.group(1)), 6, 30)
    return None


# =========================
# PIPELINE
# =========================
def analyze_upload(file_path: str, fast: bool) -> Dict[str, Any]:
    # 1) Try selectable text first (fast path if doc has good text)
    text, method = extract_text_from_path(file_path, ocr_all_pages=not fast)
    corp = detect_corporation(text)
    lic  = extract_license_number(text)
    valid_until = extract_valid_until(text)
    last_renew = extract_last_renew_year(text) or infer_last_renew_from_valid_until(valid_until)
    biz_name = extract_business_name(text)


    # 2) Auto-detect per-year fees from the current text
    renew_py, sign_py = extract_per_year_fees(text)

    # 3) ⛑️ OCR fallback:
    # If BOTH amounts are missing and the method was selectable text ("pymupdf"),
    # force OCR on all pages once and try again.
    if renew_py is None and sign_py is None and method == "pymupdf":
        text_ocr, _ = extract_text_from_path(file_path, ocr_all_pages=True)
        r2, s2 = extract_per_year_fees(text_ocr)
        if r2 is not None or s2 is not None:
            text = text_ocr
            renew_py = r2 if r2 is not None else renew_py
            sign_py  = s2 if s2 is not None else sign_py
            method = "ocr_pdf"

    # 4) Defaults by corporation if still missing
    if corp == "DNCC":
        renew_py = renew_py or 2000.0
        sign_py  = sign_py or 480.0
    elif corp == "DSCC":
        renew_py = renew_py or 500.0
        sign_py  = sign_py or 640.0
    else:
        renew_py = renew_py or 1000.0
        sign_py  = sign_py or 500.0

    # 5) Due/fine summary (same as before)
    due = compute_due(last_renew) if last_renew else 1
    fine_m = compute_fine_months(due)

    return {
        "corporation": corp,
        "license_no": lic or "",
        "valid_until": valid_until or "",
        "last_renew_year": last_renew or "",
        "due_years": due,
        "fine_months": fine_m,
        "renew_fee_py": float(renew_py),
        "signboard_py": float(sign_py),
        "raw_text": text[:3000],
        "method": method,
        "business_name": biz_name or "",   # 👈 NEW
    }



# =========================
# UI
# =========================
def build_ui():
    with gr.Blocks(title="Trade License Renewal Fee AI Software") as demo:
        gr.Markdown("## Trade License Renewal Fee AI Software")

        with gr.Row():
            file_in = gr.File(label="Upload Trade License File", type="filepath")
            fast_ocr = gr.Checkbox(value=True, label="Fast OCR (first page only)")
        analyze_btn = gr.Button("Analyze", variant="primary")

        with gr.Row():
            corp_out = gr.Textbox(label="Detected City Corporation", interactive=True)
            license_out = gr.Textbox(label="License No.", interactive=True)
            biz_in = gr.Textbox(label="Business Name", interactive=True)
        with gr.Row():
            valid_out = gr.Textbox(label="Valid Until", interactive=True)
            last_renew_out = gr.Textbox(label="Last Renew Year (YYYY-YYYY)", interactive=True)
            due_out = gr.Number(label="Due (years)", interactive=False, precision=0)
        fine_months_out = gr.Number(label="Fine for Month (auto)", interactive=False, precision=0)
        method_out = gr.Textbox(label="Extraction Method", interactive=False)



        gr.Markdown("### Detected (auto)")
        with gr.Row():
            renew_py_view = gr.Number(label="Renew Fee (per year)", interactive=True, precision=0)
            sign_py_view = gr.Number(label="Signboard Charge (per year)", interactive=True, precision=0)

        with gr.Accordion("Show extracted text (debug)", open=False):
            raw_text_view = gr.Textbox(lines=8, interactive=False)

        gr.Markdown("### Enter/Adjust Other Fees")
        dncc_group = gr.Group(visible=False)
        with dncc_group:
            gr.Markdown("#### Dhaka North City Corporation (DNCC)")
            with gr.Row():
                dncc_src   = gr.Number(label="Source TAX (per year)", value=3000.0, precision=2)
                dncc_service = gr.Number(label="Service Charge", value=500.0, precision=2)
                dncc_others  = gr.Number(label="Others", value=500.0, precision=2)
            with gr.Row():
                dncc_book    = gr.Number(label="Book Charge", value=270.0, precision=2)
                dncc_bank    = gr.Number(label="Bank Charge", value=50.0, precision=2)

        dscc_group = gr.Group(visible=False)
        with dscc_group:
            gr.Markdown("#### Dhaka South City Corporation (DSCC)")
            with gr.Row():
                dscc_src   = gr.Number(label="Source TAX (per year)", value=3000.0, precision=2)
                dscc_form  = gr.Number(label="Form Fee", value=50.0, precision=2)
                dscc_service= gr.Number(label="Service Charge", value=1000.0, precision=2)
            with gr.Row():
                dscc_bank   = gr.Number(label="Bank Charge", value=50.0, precision=2)

        with gr.Row():
            compute_btn = gr.Button("Compute Grand Total", variant="primary")
            restart_btn = gr.Button("Restart", variant="secondary")

        total_out = gr.Number(label="Grand Total (৳)", precision=2, interactive=False)
        with gr.Accordion("Show Detailed Breakdown (raw)", open=False):
            breakdown_json = gr.Textbox(label="Detailed Breakdown (raw)", lines=10, interactive=False)

        breakdown_html = gr.HTML(label="সারাংশ / Status")

        # Hidden states
        state_corp = gr.State("")
        state_renew_lbl = gr.State("")
        state_license = gr.State("")
        state_valid = gr.State("")
        state_renew_py = gr.State(0.0)
        state_sign_py = gr.State(0.0)

        # ---- Analyze ----
        def on_analyze(file_path: str, fast: bool):
            info = analyze_upload(file_path, fast)
            corp = info["corporation"]
            dncc_vis = (corp == "DNCC")
            dscc_vis = (corp == "DSCC")
            return (
                info["corporation"], info["license_no"],
                info.get("business_name", ""),   # 👈 now auto-fills Business Name
                info["valid_until"], info["last_renew_year"], info["due_years"],
                info["fine_months"], info["method"],
                float(info["renew_fee_py"]), float(info["signboard_py"]),
                info.get("raw_text", ""),
                dncc_vis and gr.update(visible=True) or gr.update(visible=False),
                dscc_vis and gr.update(visible=True) or gr.update(visible=False),
                # states
                corp, info["last_renew_year"], info["license_no"], info["valid_until"],
                float(info["renew_fee_py"]), float(info["signboard_py"]),
                # clear outputs
                gr.update(value=""), gr.update(value=0.0),
            )


        analyze_btn.click(
            on_analyze,
            inputs=[file_in, fast_ocr],
            outputs=[
                corp_out, license_out, biz_in,
                valid_out, last_renew_out, due_out,
                fine_months_out, method_out,
                renew_py_view, sign_py_view,
                raw_text_view,
                dncc_group, dscc_group,
                state_corp, state_renew_lbl, state_license, state_valid,
                state_renew_py, state_sign_py,
                breakdown_html, total_out
            ],
        )

        # ---- Compute ----
        def on_compute(corp: str, last_lbl: str, lic: str, valid_lbl: str, biz_manual: str,
                       renew_py: float, sign_py: float,
                       dncc_src, dncc_serv, dncc_oth, dncc_book, dncc_bank,
                       dscc_src2, dscc_form, dscc_serv, dscc_bank):
            # If valid until is today or in future -> already renewed
            vdt = parse_valid_until_date(valid_lbl)
            today = date.today()
            if vdt is not None and vdt >= today:
                msg_html = (
                    "<div style='font-family:Inter,system-ui; padding:.5rem; "
                    "border:1px solid #444; background:#111; color:#eee; border-radius:8px;'>"
                    "<b>Already Trade License Renewed</b><br>"
                    "Valid Until: " + (valid_lbl or "-") + "</div>"
                )
                return (
                    0.0,
                    {"status": "Already Trade License Renewed", "valid_until": valid_lbl},
                    gr.update(value=msg_html),
                )

            if not corp or corp == "UNKNOWN":
                msg = {"error": "City corporation not detected. Upload a clearer license."}
                return 0.0, msg, gr.update(value="<i>No data</i>")
            if not last_lbl:
                msg = {"error": "Last renew year missing. Make sure it’s visible on the license."}
                return 0.0, msg, gr.update(value="<i>No data</i>")

            if corp == "DNCC":
                bd = calc_dncc(
                    last_lbl,
                    renew_fee_py=renew_py, signboard_py=sign_py,
                    source_tax_py=dncc_src, service=dncc_serv, others=dncc_oth,
                    book=dncc_book, bank=dncc_bank
                )
            else:
                bd = calc_dscc(
                    last_lbl,
                    renew_fee_py=renew_py, signboard_py=sign_py,
                    source_tax_py=dscc_src2, form_fee=dscc_form, service=dscc_serv, bank=dscc_bank
                )

            total = bd["Grand Total"]
            html = breakdown_to_html_bn(
                corp=corp, lic=lic, biz=biz_manual, last_renew=last_lbl,
                due=int(bd.get("Due (years)", 1)), bd=bd,
                renew_py=renew_py, sign_py=sign_py
            )
            return total, bd, gr.update(value=html)

        compute_btn.click(
            on_compute,
            inputs=[
                corp_out, last_renew_out, license_out, valid_out, biz_in,   # 👈 visible + editable
                renew_py_view, sign_py_view,
                # DNCC others:
                dncc_src, dncc_service, dncc_others, dncc_book, dncc_bank,
                # DSCC others:
                dscc_src, dscc_form, dscc_service, dscc_bank,
            ],
            outputs=[total_out, breakdown_json, breakdown_html],
        )

        # ---- Restart ----
        def on_restart():
            return (
                gr.update(value=None),  # file_in
                gr.update(value=""),    # corp_out
                gr.update(value=""),    # license_out
                gr.update(value=""),    # biz_in
                gr.update(value=""),    # valid_out
                gr.update(value=""),    # last_renew_out
                gr.update(value=0),     # due_out
                gr.update(value=0),     # fine_months_out
                gr.update(value=""),    # method_out
                gr.update(value=None),  # renew_py_view
                gr.update(value=None),  # sign_py_view
                gr.update(value=""),    # raw_text_view
                gr.update(visible=False),  # dncc_group
                gr.update(visible=False),  # dscc_group
                "", "", "", "", 0.0, 0.0,   # states
                gr.update(value=""), gr.update(value=0.0),  # breakdown_html, total_out
            )

        restart_btn.click(
            on_restart,
            inputs=[],
            outputs=[
                file_in,
                corp_out, license_out, biz_in,
                valid_out, last_renew_out, due_out,
                fine_months_out, method_out,
                renew_py_view, sign_py_view,
                raw_text_view,
                dncc_group, dscc_group,
                state_corp, state_renew_lbl, state_license, state_valid, state_renew_py, state_sign_py,
                breakdown_html, total_out
            ]
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    demo.launch(server_name="0.0.0.0", server_port=port, share=True, show_error=True)



