# app.py
# B-Kosher Customer Catalog Builder (Streamlit Cloud SAFE)
#
# ✅ Pure Python stack (Streamlit + requests + Pillow + fpdf2) - deploys on Streamlit Cloud
# ✅ Login gate (password from secrets)
# ✅ Default source = WooCommerce API (cached to disk) + optional CSV upload
# ✅ LIVE API import progress bar + live product count + live log
# ✅ Image download progress + retries/backoff + disk cache
# ✅ PDF: A4, 3x3 grid (when columns=3), clickable cards, sale badge, 2dp prices
# ✅ Fixes: HTML entities (&amp; -> &), no "nan", text wrapping, no grey behind images
# ✅ Fix: Unicode-safe PDF text (no FPDFUnicodeEncodingException)
# ✅ Fix: FPDF2 compatibility (NO rounded_rect; uses rect_any fallback)
#
# requirements.txt:
# streamlit==1.31.1
# requests==2.31.0
# Pillow==10.4.0
# fpdf2==2.7.8

import io
import re
import csv
import json
import time
import math
import html
import hashlib
import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st
from PIL import Image

from fpdf import FPDF
from fpdf.enums import XPos, YPos


# =========================
# APP CONFIG + BRANDING
# =========================
st.set_page_config(page_title="B-Kosher Catalog Builder", layout="wide")

BRAND_RED_HEX = "#C8102E"
BRAND_BLUE_HEX = "#004C97"

DEFAULT_TITLE = "B-Kosher Product Catalog"
DEFAULT_SITE = "www.b-kosher.co.uk"
DEFAULT_BASE_URL = "https://www.b-kosher.co.uk"

# Logo file in repo root
LOGO_PNG_PATH = "Bkosher.png"

st.markdown(
    f"""
    <style>
      .stButton > button {{
        background: {BRAND_RED_HEX} !important;
        color: white !important;
        border: 1px solid {BRAND_RED_HEX} !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
      }}
      .stButton > button:hover {{
        background: #a90d26 !important;
        border-color: #a90d26 !important;
      }}
      a {{ color: {BRAND_BLUE_HEX} !important; }}
      section[data-testid="stSidebar"] {{
        border-right: 3px solid {BRAND_BLUE_HEX};
      }}
      div[data-testid="stProgressBar"] > div > div {{
        background-color: {BRAND_BLUE_HEX} !important;
      }}
      .panel {{
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 16px;
        padding: 14px 14px 6px 14px;
        margin: 10px 0 14px 0;
        background: rgba(255,255,255,0.6);
      }}
      .muted {{
        color: #6b7280;
        font-size: 0.92rem;
        margin-top: -4px;
        margin-bottom: 10px;
      }}
      .step {{
        display:flex; gap:10px; align-items:center;
        padding: 10px 12px; border-radius: 14px;
        background: rgba(0, 76, 151, 0.06);
        border: 1px solid rgba(0, 76, 151, 0.15);
        margin: 8px 0 14px 0;
      }}
      .badge {{
        display:inline-block; padding: 2px 8px; border-radius: 999px;
        font-weight: 700; font-size: 0.85rem;
        border: 1px solid rgba(0,0,0,0.10);
        background: white;
      }}
      .badge.blue {{
        color: {BRAND_BLUE_HEX};
        border-color: rgba(0,76,151,0.25);
      }}
      .summary {{
        border-radius: 16px;
        padding: 12px 14px;
        background: rgba(0,0,0,0.03);
        border: 1px solid rgba(0,0,0,0.08);
      }}
      .activity {{
        border-radius: 14px;
        padding: 10px 12px;
        background: rgba(0,0,0,0.03);
        border: 1px solid rgba(0,0,0,0.06);
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        max-height: 320px;
        overflow: auto;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# SECRETS + LOGIN
# =========================
def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


WC_URL = get_secret("WC_URL", "")
WC_CK = get_secret("WC_CK", "")
WC_CS = get_secret("WC_CS", "")
APP_PASSWORD = get_secret("APP_PASSWORD", "")


def login_gate():
    if not APP_PASSWORD:
        st.error("APP_PASSWORD is not set in Streamlit secrets.")
        st.stop()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return

    st.title("B-Kosher Catalog Login")
    st.write("Please enter the password to continue.")
    pw = st.text_input("Password", type="password")

    if st.button("Login"):
        if pw == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


login_gate()


# =========================
# Helpers / sanitizers
# =========================
def pdf_safe(s: str) -> str:
    """
    Make text safe for built-in PDF core fonts (latin-1-ish).
    Converts HTML entities, replaces common unicode punctuation, strips unsupported chars.
    """
    if s is None:
        return ""
    s = str(s)
    s = html.unescape(s)
    s = (
        s.replace("•", " | ")
        .replace("–", "-")
        .replace("—", "-")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("\u00A0", " ")
    )
    try:
        s.encode("latin-1")
        return s
    except Exception:
        return s.encode("latin-1", "ignore").decode("latin-1")


def safe_text(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in ("nan", "none", "null"):
        return ""
    return pdf_safe(s)


def strip_html(s: str) -> str:
    s = safe_text(s)
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_first_image_url(cell) -> str | None:
    s = safe_text(cell)
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        m = re.search(r"https?://\S+", p)
        if m:
            return m.group(0)
    return None


def parse_money(v) -> float | None:
    s = safe_text(v)
    if not s:
        return None
    s = s.replace("£", "").replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def fmt_money(symbol: str, v: float | None) -> str:
    if v is None:
        return ""
    return f"{symbol}{v:.2f}"


def sanitize_url(u: str) -> str:
    # remove consumer_key/consumer_secret from any displayed URL
    try:
        parts = urlsplit(u)
        q = parse_qsl(parts.query, keep_blank_values=True)
        q = [(k, v) for (k, v) in q if k not in ("consumer_key", "consumer_secret")]
        clean_query = urlencode(q)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, clean_query, parts.fragment))
    except Exception:
        return (u or "").replace("consumer_key=", "consumer_key=***").replace("consumer_secret=", "consumer_secret=***")


def primary_category(p: dict) -> str:
    cats = p.get("categories") or []
    if isinstance(cats, list) and cats:
        return safe_text(cats[0]) or "Other"
    return "Other"


def is_in_stock(p: dict) -> bool:
    s = safe_text(p.get("stock_status")).lower()
    return s != "outofstock"


# =========================
# Disk caches
# =========================
IMAGE_CACHE_DIR = Path("./image_cache")
IMAGE_CACHE_DIR.mkdir(exist_ok=True)

API_CACHE_DIR = Path("./api_cache")
API_CACHE_DIR.mkdir(exist_ok=True)
API_CACHE_FILE = API_CACHE_DIR / "products.json"


def cache_path_for_url(url: str) -> Path:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return IMAGE_CACHE_DIR / f"{h}.jpg"


def read_image_cache(url: str) -> bytes | None:
    p = cache_path_for_url(url)
    if p.exists():
        try:
            return p.read_bytes()
        except Exception:
            return None
    return None


def write_image_cache(url: str, b: bytes):
    try:
        cache_path_for_url(url).write_bytes(b)
    except Exception:
        pass


def resize_to_jpeg_bytes(raw: bytes, max_px: int = 900, quality: int = 82) -> bytes | None:
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = im.size
        scale = min(1.0, max_px / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue()
    except Exception:
        return None


def download_with_retries(url: str, *, timeout: int, retries: int, backoff: float, log_cb=None) -> bytes | None:
    if not url or not url.startswith("http"):
        return None

    cached = read_image_cache(url)
    if cached:
        return cached

    for attempt in range(retries + 1):
        if attempt > 0:
            time.sleep(min(backoff * (2 ** (attempt - 1)), 15.0))
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "bkosher-catalog/1.0"})
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                continue
            r.raise_for_status()
            raw = r.content
            cooked = resize_to_jpeg_bytes(raw)
            final = cooked if cooked else raw
            write_image_cache(url, final)
            return final
        except Exception as e:
            if log_cb and attempt == retries:
                log_cb(f"Image failed (giving up): {url[:80]}… ({type(e).__name__})")
            continue
    return None


def api_cache_load():
    if not API_CACHE_FILE.exists():
        return None, None
    try:
        payload = json.loads(API_CACHE_FILE.read_text(encoding="utf-8"))
        products = payload.get("products")
        fetched_at = payload.get("fetched_at", "")
        if isinstance(products, list):
            return products, fetched_at
    except Exception:
        pass
    return None, None


def api_cache_save(products_raw: list[dict]):
    payload = {
        "fetched_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "products": products_raw,
    }
    try:
        API_CACHE_FILE.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


# =========================
# Woo API fetch (LIVE progress)
# =========================
def wc_fetch_products(
    base_url: str,
    ck: str,
    cs: str,
    *,
    per_page: int = 25,
    timeout: int = 30,
    log_cb=None,
    progress_cb=None,
    count_cb=None,
):
    if not (base_url and ck and cs):
        raise RuntimeError("Woo API secrets missing. Set WC_URL, WC_CK, WC_CS in Streamlit Secrets.")

    base_url = base_url.rstrip("/")
    endpoint = f"{base_url}/wp-json/wc/v3/products"
    session = requests.Session()

    page = 1
    out = []
    ssl_retries = 8

    while True:
        params = {
            "consumer_key": ck,
            "consumer_secret": cs,
            "per_page": per_page,
            "page": page,
            "status": "publish",
        }

        if log_cb:
            log_cb(f"API: fetching page {page}…")
        if progress_cb:
            progress_cb(min(0.90, 0.02 + (page * 0.01)))

        last_exc = None
        for attempt in range(ssl_retries + 1):
            if attempt > 0:
                time.sleep(min(0.8 * (2 ** (attempt - 1)), 20.0))
            try:
                r = session.get(endpoint, params=params, timeout=timeout, headers={"User-Agent": "bkosher-catalog/1.0"})

                if r.status_code in (401, 403):
                    try:
                        j = r.json()
                        raise RuntimeError(f"{j.get('code')} ({r.status_code}): {j.get('message')}")
                    except Exception:
                        raise RuntimeError(f"Unauthorized ({r.status_code}). Check Woo REST API permissions.")

                if r.status_code >= 400:
                    raise RuntimeError(f"API request failed ({r.status_code}) at {sanitize_url(r.url)}")

                batch = r.json()
                if not batch:
                    if log_cb:
                        log_cb("API: no more products.")
                    if progress_cb:
                        progress_cb(1.0)
                    return out

                out.extend(batch)
                if count_cb:
                    count_cb(len(out))

                if len(batch) < per_page:
                    if log_cb:
                        log_cb(f"API: completed ({len(out)} products).")
                    if progress_cb:
                        progress_cb(1.0)
                    return out

                page += 1
                if page % 10 == 0:
                    time.sleep(0.25)

                last_exc = None
                break

            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_exc = e
                continue

        if last_exc is not None:
            raise RuntimeError(
                "API fetch failed due to repeated SSL/connection errors while paging products.\n"
                f"Details: {type(last_exc).__name__}: {str(last_exc)[:200]}"
            )


def wc_to_product(p: dict) -> dict:
    cats = []
    for c in (p.get("categories") or []):
        n = safe_text(c.get("name"))
        if n:
            cats.append(n)

    attrs = []
    for a in (p.get("attributes") or []):
        n = safe_text(a.get("name"))
        opts = a.get("options") or []
        opts_s = ", ".join([safe_text(x) for x in opts if safe_text(x)])
        if n and opts_s:
            attrs.append((n, opts_s))

    reg = parse_money(p.get("regular_price"))
    sale = parse_money(p.get("sale_price"))
    on_sale = bool(p.get("on_sale")) or (sale is not None and reg is not None and sale < reg)

    img_url = None
    imgs = p.get("images") or []
    if imgs:
        img_url = safe_text(imgs[0].get("src")) or None

    return {
        "id": p.get("id"),
        "name": safe_text(p.get("name")),
        "sku": safe_text(p.get("sku")),
        "categories": cats,
        "short_desc": strip_html(p.get("short_description") or ""),
        "regular_price": reg,
        "sale_price": sale,
        "on_sale": on_sale,
        "attributes": attrs,
        "url": safe_text(p.get("permalink")) or "",
        "stock_status": safe_text(p.get("stock_status")) or "",
        "_img_url": img_url,
        "_image_path": None,
    }


def get_products_from_api_or_cache_live(
    wc_url: str,
    wc_ck: str,
    wc_cs: str,
    *,
    timeout: int = 30,
    force_refresh: bool = False,
    log_cb=None,
    progress_cb=None,
    count_cb=None,
):
    if not force_refresh:
        cached_raw, fetched_at = api_cache_load()
        if cached_raw is not None:
            if log_cb:
                log_cb(f"Loaded {len(cached_raw)} products from disk cache ({fetched_at}).")
            if progress_cb:
                progress_cb(1.0)
            normalized = [wc_to_product(p) for p in cached_raw]
            return normalized, fetched_at, "disk_cache"

    raw = wc_fetch_products(
        wc_url, wc_ck, wc_cs,
        per_page=25,
        timeout=timeout,
        log_cb=log_cb,
        progress_cb=progress_cb,
        count_cb=count_cb,
    )
    api_cache_save(raw)
    fetched_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    normalized = [wc_to_product(p) for p in raw]
    return normalized, fetched_at, "api"


# =========================
# CSV loader
# =========================
def read_csv_bytes(uploaded_file) -> list[dict]:
    content = uploaded_file.getvalue()
    try:
        text = content.decode("utf-8")
    except Exception:
        text = content.decode("latin-1", errors="replace")

    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def best_key(keys: list[str], candidates: list[str]) -> str | None:
    lower = {k.lower(): k for k in keys}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


# =========================
# PDF helpers (FPDF2 compatibility)
# =========================
def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def rect_any(pdf: FPDF, x: float, y: float, w: float, h: float, r: float, style: str):
    """
    Use rounded_rect if available; otherwise fallback to rect.
    Avoids crashes when rounded_rect is missing.
    """
    if hasattr(pdf, "rounded_rect"):
        # Some versions: rounded_rect(x,y,w,h,r,style=...)
        pdf.rounded_rect(x, y, w, h, r, style=style)
    else:
        pdf.rect(x, y, w, h, style=style)


class CatalogPDF(FPDF):
    def __init__(self, title: str, logo_path: str, brand_site: str, disclaimer: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.catalog_title = pdf_safe(title)
        self.logo_path = logo_path
        self.brand_site = pdf_safe(brand_site)
        self.disclaimer = pdf_safe(disclaimer)
        self.blue = hex_to_rgb(BRAND_BLUE_HEX)
        self.red = hex_to_rgb(BRAND_RED_HEX)
        self.set_auto_page_break(auto=False)

    def header(self):
        left = 12
        top = 10
        try:
            self.image(self.logo_path, x=left, y=top, h=11)
            title_x = left + 38
        except Exception:
            title_x = left

        self.set_text_color(*self.blue)
        self.set_font("Helvetica", "B", 13)
        self.set_xy(title_x, top + 1)
        self.cell(0, 8, pdf_safe(self.catalog_title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 9)
        self.set_xy(-12 - 25, top + 2)
        self.cell(25, 8, pdf_safe(f"Page {self.page_no()}"), align="R")

        self.set_draw_color(*self.blue)
        self.set_line_width(0.6)
        self.line(12, 26, self.w - 12, 26)

    def footer(self):
        y = self.h - 14
        self.set_draw_color(*self.red)
        self.set_line_width(0.5)
        self.line(12, y, self.w - 12, y)

        self.set_text_color(55, 65, 81)
        self.set_font("Helvetica", "", 8)
        self.set_xy(12, y + 2)
        self.cell(0, 8, pdf_safe(f"{self.brand_site} | {self.disclaimer}"))


def truncate_to_fit(pdf: FPDF, text: str, max_w: float) -> str:
    text = pdf_safe(text)
    if pdf.get_string_width(text) <= max_w:
        return text
    ell = "..."
    t = text
    while t and pdf.get_string_width(t + ell) > max_w:
        t = t[:-1]
    return (t + ell) if t else ell


def draw_sale_badge(pdf: FPDF, x: float, y: float, w: float, p: dict, currency: str):
    reg = p.get("regular_price")
    sale = p.get("sale_price")
    if not p.get("on_sale") or sale is None or reg is None or sale >= reg:
        return

    badge_w = 26
    badge_h = 9
    bx = x + w - badge_w - 3
    by = y + 3

    pdf.set_fill_color(*hex_to_rgb(BRAND_RED_HEX))
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    rect_any(pdf, bx, by, badge_w, badge_h, 2, style="F")
    pdf.set_xy(bx, by + 1.7)
    pdf.cell(badge_w, 6, "SALE", align="C")

    save_amt = reg - sale
    save_pct = (save_amt / reg) * 100 if reg else 0
    pdf.set_text_color(17, 24, 39)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_xy(x, by + badge_h + 1)
    msg = pdf_safe(f"Save {currency}{save_amt:.2f} ({save_pct:.0f}%)")
    pdf.cell(w - 2, 4, msg, align="R")


def make_catalog_pdf_bytes(
    products: list[dict],
    *,
    title: str,
    page_size: str,
    columns: int,
    currency_symbol: str,
    show_price: bool,
    show_sku: bool,
    show_desc: bool,
    show_attrs: bool,
    exclude_oos: bool,
):
    today_str = datetime.date.today().strftime("%d %b %Y")
    disclaimer = pdf_safe(f"Prices correct as of {today_str}")

    fmt = "A4" if page_size == "A4" else "Letter"
    pdf = CatalogPDF(
        title=title or DEFAULT_TITLE,
        logo_path=LOGO_PNG_PATH,
        brand_site=DEFAULT_SITE,
        disclaimer=disclaimer,
        orientation="P",
        unit="mm",
        format=fmt,
    )

    margin = 12
    gutter = 6
    header_space = 28
    footer_space = 18
    category_bar_h = 10

    usable_w = pdf.w - 2 * margin
    card_w = (usable_w - (columns - 1) * gutter) / columns

    rows = 3
    usable_h = pdf.h - header_space - footer_space - category_bar_h - 8
    card_h = (usable_h - (rows - 1) * gutter) / rows

    def grouped_products():
        grouped = {}
        for p in products:
            if exclude_oos and not is_in_stock(p):
                continue
            grouped.setdefault(primary_category(p), []).append(p)
        cats = sorted(grouped.keys(), key=lambda s: s.lower())
        return cats, grouped

    cats, grouped = grouped_products()

    # Contents page
    pdf.add_page()
    pdf.set_text_color(*hex_to_rgb(BRAND_BLUE_HEX))
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(margin, 36)
    pdf.cell(0, 10, "Contents", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_text_color(55, 65, 81)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(margin)
    pdf.cell(0, 7, disclaimer, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    per_page = rows * columns
    page_no = 2
    cat_first = {}
    for cat in cats:
        n = len(grouped.get(cat, []))
        if n == 0:
            continue
        cat_first[cat] = page_no
        page_no += math.ceil(n / per_page)

    pdf.set_text_color(17, 24, 39)
    pdf.set_font("Helvetica", "", 11)
    y = 60
    for cat in cats:
        if y > pdf.h - 30:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.set_text_color(*hex_to_rgb(BRAND_BLUE_HEX))
            pdf.set_xy(margin, 36)
            pdf.cell(0, 10, "Contents (cont.)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(17, 24, 39)
            y = 55

        pdf.set_xy(margin, y)
        pdf.cell(0, 7, pdf_safe(cat))
        pdf.set_xy(pdf.w - margin - 25, y)
        pdf.set_text_color(107, 114, 128)
        pdf.cell(25, 7, str(cat_first.get(cat, "")), align="R")
        pdf.set_text_color(17, 24, 39)
        y += 8

    # Content pages
    blue_rgb = hex_to_rgb(BRAND_BLUE_HEX)
    red_rgb = hex_to_rgb(BRAND_RED_HEX)

    for cat in cats:
        items = grouped.get(cat, [])
        if not items:
            continue

        idx = 0
        while idx < len(items):
            pdf.add_page()

            # Category bar
            pdf.set_fill_color(*blue_rgb)
            bar_y = header_space
            rect_any(pdf, margin, bar_y, pdf.w - 2 * margin, category_bar_h, 3, style="F")
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_xy(margin + 4, bar_y + 2.2)
            pdf.cell(0, 6, pdf_safe(cat))

            # Cards grid
            start_y = header_space + category_bar_h + 6
            for r in range(rows):
                yy = start_y + r * (card_h + gutter)
                xx = margin
                for _c in range(columns):
                    if idx >= len(items):
                        break
                    p = items[idx]
                    idx += 1

                    # Card
                    pdf.set_draw_color(*blue_rgb)
                    pdf.set_line_width(0.5)
                    rect_any(pdf, xx, yy, card_w, card_h, 3, style="D")

                    # Clickable link
                    url = safe_text(p.get("url"))
                    if url.startswith("http"):
                        pdf.link(x=xx, y=yy, w=card_w, h=card_h, link=url)

                    # Sale badge
                    draw_sale_badge(pdf, xx, yy, card_w, p, currency_symbol)

                    pad = 3.5
                    img_h = card_h * 0.48
                    img_w = card_w - 2 * pad
                    img_x = xx + pad
                    img_y = yy + pad + 3

                    # Image (no grey background)
                    if p.get("_image_path"):
                        try:
                            pdf.image(p["_image_path"], x=img_x, y=img_y, w=img_w, h=img_h)
                        except Exception:
                            pass
                    else:
                        pdf.set_text_color(120, 120, 120)
                        pdf.set_font("Helvetica", "I", 8)
                        pdf.set_xy(xx, img_y + img_h / 2 - 2)
                        pdf.cell(card_w, 4, "No image", align="C")

                    # Text area
                    text_top = img_y + img_h + 2
                    tx = xx + pad
                    max_w = card_w - 2 * pad

                    # Product name (single line)
                    pdf.set_text_color(0, 0, 0)
                    pdf.set_font("Helvetica", "B", 9.5)
                    name = safe_text(p.get("name")).replace("\n", " ")
                    name = truncate_to_fit(pdf, name, max_w)
                    pdf.set_xy(tx, text_top)
                    pdf.cell(max_w, 4.5, name)

                    line_y = text_top + 5.2

                    # Price
                    if show_price:
                        reg = p.get("regular_price")
                        sale = p.get("sale_price")
                        on_sale = bool(p.get("on_sale")) and sale is not None and reg is not None and sale < reg

                        if on_sale:
                            pdf.set_text_color(*red_rgb)
                            pdf.set_font("Helvetica", "B", 10)
                            pdf.set_xy(tx, line_y)
                            pdf.cell(0, 5, pdf_safe(fmt_money(currency_symbol, sale)))

                            pdf.set_text_color(107, 114, 128)
                            pdf.set_font("Helvetica", "", 8.5)
                            reg_txt = pdf_safe(fmt_money(currency_symbol, reg))
                            rx = tx + 18
                            pdf.set_xy(rx, line_y + 0.6)
                            pdf.cell(0, 4, reg_txt)
                            wtxt = pdf.get_string_width(reg_txt)
                            pdf.set_draw_color(107, 114, 128)
                            pdf.set_line_width(0.3)
                            pdf.line(rx, line_y + 2.5, rx + wtxt, line_y + 2.5)
                            line_y += 6
                        else:
                            if reg is not None:
                                pdf.set_text_color(*red_rgb)
                                pdf.set_font("Helvetica", "B", 10)
                                pdf.set_xy(tx, line_y)
                                pdf.cell(0, 5, pdf_safe(fmt_money(currency_symbol, reg)))
                                line_y += 6

                        pdf.set_text_color(0, 0, 0)

                    # SKU
                    if show_sku:
                        sku = safe_text(p.get("sku"))
                        if sku:
                            pdf.set_text_color(31, 41, 55)
                            pdf.set_font("Helvetica", "", 8.5)
                            pdf.set_xy(tx, line_y)
                            pdf.cell(0, 4, pdf_safe(f"SKU: {sku}"))
                            line_y += 4.5

                    # Attributes (max 2 lines)
                    if show_attrs:
                        attrs = p.get("attributes") or []
                        if attrs:
                            pdf.set_text_color(55, 65, 81)
                            pdf.set_font("Helvetica", "", 7.8)
                            shown = 0
                            for (an, av) in attrs:
                                if shown >= 2:
                                    break
                                an = safe_text(an)
                                av = safe_text(av)
                                if not an or not av:
                                    continue
                                line = truncate_to_fit(pdf, f"{an}: {av}", max_w)
                                pdf.set_xy(tx, line_y)
                                pdf.cell(0, 4, line)
                                line_y += 4.0
                                shown += 1

                    # Description (wrap, max 2 lines)
                    if show_desc:
                        desc = strip_html(p.get("short_desc"))
                        if desc:
                            pdf.set_text_color(75, 85, 99)
                            pdf.set_font("Helvetica", "", 7.6)
                            desc = pdf_safe(desc)
                            for _ in range(2):
                                if not desc:
                                    break
                                words = desc.split(" ")
                                line = ""
                                while words and pdf.get_string_width((line + " " + words[0]).strip()) <= max_w:
                                    line = (line + " " + words.pop(0)).strip()
                                if not line:
                                    line = truncate_to_fit(pdf, desc, max_w)
                                    desc = ""
                                else:
                                    desc = " ".join(words).strip()
                                pdf.set_xy(tx, line_y)
                                pdf.cell(0, 4, line)
                                line_y += 4.0

                    xx += card_w + gutter

    out = pdf.output(dest="S")
    if isinstance(out, str):
        out = out.encode("latin1", "ignore")
    return out


# =========================
# UI state
# =========================
if "step" not in st.session_state:
    st.session_state.step = 1
if "products_raw" not in st.session_state:
    st.session_state.products_raw = []
if "products_filtered" not in st.session_state:
    st.session_state.products_filtered = []
if "last_pdf" not in st.session_state:
    st.session_state.last_pdf = None


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("### B-Kosher")
    try:
        st.image(LOGO_PNG_PATH, width=190)
    except Exception:
        st.caption("Logo missing: upload Bkosher.png to repo root")

    if WC_URL and WC_CK and WC_CS:
        st.success("Woo API configured (secrets)")
    else:
        st.warning("Woo API secrets missing (API mode won't work)")

    cached_raw, cached_at = api_cache_load()
    if cached_raw is not None:
        st.info(f"API cache: {len(cached_raw):,} items\n\nLast fetch: {cached_at}")

    st.markdown("---")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    if st.button("🔄 Reset (start over)"):
        for k in list(st.session_state.keys()):
            if k not in ("_stcore",):
                del st.session_state[k]
        st.rerun()


st.title("Customer Catalog Builder")


def step_indicator(step: int):
    labels = {
        1: "Step 1 — Choose data source",
        2: "Step 2 — Load products",
        3: "Step 3 — Filter & layout",
        4: "Step 4 — Generate & download",
    }
    st.markdown(
        f"""
        <div class="step">
          <span class="badge blue">Step {step}/4</span>
          <div><b>{labels.get(step,"")}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Step 1 — Choose source (DEFAULT = API)
# =========================
step_indicator(st.session_state.step)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown("### Data source")
st.markdown('<div class="muted">Default is WooCommerce API (cached). CSV upload is available as a fallback.</div>', unsafe_allow_html=True)

data_source = st.radio(
    "Source",
    ["WooCommerce API", "CSV Upload"],
    index=0,
    horizontal=True,
    key="data_source",
)

st.markdown("</div>", unsafe_allow_html=True)

if st.button("Continue → Load products"):
    st.session_state.step = 2
    st.rerun()


# =========================
# Step 2 — Load products
# =========================
if st.session_state.step >= 2:
    step_indicator(2)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Load products")
    st.markdown('<div class="muted">API mode uses a disk cache for speed. CSV mode loads instantly.</div>', unsafe_allow_html=True)

    if data_source == "CSV Upload":
        base_url = st.text_input("Base URL (used to build links if CSV has none)", value=DEFAULT_BASE_URL)
        csv_file = st.file_uploader("Upload WooCommerce CSV export", type=["csv"], key="csv_file")
        load_btn = st.button("Load products from CSV", disabled=(csv_file is None))

        if load_btn:
            rows = read_csv_bytes(csv_file)
            if not rows:
                st.error("CSV looks empty.")
                st.stop()

            keys = list(rows[0].keys())
            col_id = best_key(keys, ["ID", "Id"])
            col_name = best_key(keys, ["Name", "Product name", "Title"])
            col_sku = best_key(keys, ["SKU"])
            col_reg = best_key(keys, ["Regular price", "Regular Price", "Price"])
            col_sale = best_key(keys, ["Sale price", "Sale Price"])
            col_cats = best_key(keys, ["Categories", "Category"])
            col_desc = best_key(keys, ["Short description", "Short Description", "Description"])
            col_imgs = best_key(keys, ["Images", "Image", "Image URLs"])
            col_url = best_key(keys, ["Permalink", "Product URL", "URL", "Link"])
            col_stock = best_key(keys, ["Stock status", "Stock Status", "stock_status"])
            col_slug = best_key(keys, ["Slug", "slug"])

            products = []
            for row in rows:
                name = safe_text(row.get(col_name)) if col_name else ""
                if not name:
                    continue

                sku = safe_text(row.get(col_sku)) if col_sku else ""
                cats = []
                if col_cats:
                    cats = [c.strip() for c in safe_text(row.get(col_cats)).split(",") if c.strip()]
                desc = strip_html(row.get(col_desc)) if col_desc else ""

                reg = parse_money(row.get(col_reg)) if col_reg else None
                sale = parse_money(row.get(col_sale)) if col_sale else None
                on_sale = (sale is not None and reg is not None and sale < reg)

                stock_status = safe_text(row.get(col_stock)) if col_stock else ""
                ss = stock_status.lower()
                if "out" in ss:
                    stock_status = "outofstock"
                elif "instock" in ss or "in stock" in ss:
                    stock_status = "instock"

                img_url = parse_first_image_url(row.get(col_imgs)) if col_imgs else None

                url = safe_text(row.get(col_url)) if col_url else ""
                if not url.startswith("http"):
                    pid = safe_text(row.get(col_id)) if col_id else ""
                    if pid.isdigit():
                        url = f"{base_url.rstrip('/')}/?post_type=product&p={pid}"
                    else:
                        slug = safe_text(row.get(col_slug)) if col_slug else ""
                        if slug:
                            url = f"{base_url.rstrip('/')}/{slug.lstrip('/')}"

                attrs = []
                for i in range(1, 21):
                    ncol = f"Attribute {i} name"
                    vcol = f"Attribute {i} value(s)"
                    if ncol in row and vcol in row:
                        n = safe_text(row.get(ncol))
                        v = safe_text(row.get(vcol))
                        if n and v:
                            attrs.append((n, v))

                products.append({
                    "id": safe_text(row.get(col_id)) if col_id else "",
                    "name": name,
                    "sku": sku,
                    "categories": cats,
                    "short_desc": desc,
                    "regular_price": reg,
                    "sale_price": sale,
                    "on_sale": bool(on_sale),
                    "attributes": attrs,
                    "url": url,
                    "stock_status": stock_status,
                    "_img_url": img_url,
                    "_image_path": None,
                })

            st.session_state.products_raw = products
            st.session_state.step = 3
            st.success(f"Loaded {len(products):,} products from CSV.")
            st.rerun()

    else:
        api_timeout = st.slider("API timeout (seconds)", 10, 60, 30)

        c1, c2 = st.columns(2)
        with c1:
            load_cached_btn = st.button("Load products (use cache if available)")
        with c2:
            refresh_btn = st.button("Refresh cache (fetch again)")

        if load_cached_btn or refresh_btn:
            prog = st.progress(0.0)
            status = st.empty()
            loaded_count = st.empty()
            log_box = st.empty()
            logs = []

            def log(msg):
                logs.append(msg)
                log_box.markdown(
                    f"<div class='activity'>{html.escape(chr(10).join(logs[-18:]))}</div>",
                    unsafe_allow_html=True
                )

            def set_progress(v):
                prog.progress(max(0.0, min(1.0, float(v))))

            def set_count(n):
                loaded_count.info(f"Products loaded so far: {n:,}")

            try:
                status.info("Fetching products from WooCommerce API…")
                products, fetched_at, source = get_products_from_api_or_cache_live(
                    WC_URL, WC_CK, WC_CS,
                    timeout=api_timeout,
                    force_refresh=bool(refresh_btn),
                    log_cb=log,
                    progress_cb=set_progress,
                    count_cb=set_count,
                )
                status.success("Done.")
                set_progress(1.0)

                st.session_state.products_raw = products
                st.session_state.step = 3
                st.success(f"Loaded {len(products):,} products • Source: {source} • Cached at: {fetched_at}")
                st.rerun()
            except Exception as e:
                st.error(str(e))
                st.stop()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Step 3 — Filter & layout
# =========================
if st.session_state.step >= 3:
    step_indicator(3)
    products = st.session_state.products_raw

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Filter & layout")
    st.markdown('<div class="muted">Choose what appears in the catalog and how it looks.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        st.session_state.catalog_title = st.text_input("Catalog title", value=st.session_state.get("catalog_title", DEFAULT_TITLE))
    with c2:
        st.session_state.page_size = st.selectbox("Page size", ["A4", "Letter"], index=0)
        st.session_state.columns = st.selectbox("Cards per row", [1, 2, 3], index=2)
    with c3:
        st.session_state.currency_symbol = st.text_input("Currency symbol", value=st.session_state.get("currency_symbol", "£"))
        st.session_state.preset = st.selectbox("Image download preset", ["Reliable", "Normal", "Fast"], index=0)

    all_cats = sorted({c for p in products for c in (p.get("categories") or []) if c}, key=lambda s: s.lower())
    st.session_state.selected_cats = st.multiselect("Categories (optional)", all_cats, default=st.session_state.get("selected_cats", []))
    st.session_state.search = st.text_input("Search (name or SKU)", value=st.session_state.get("search", ""))

    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        st.session_state.show_price = st.checkbox("Price", value=st.session_state.get("show_price", True))
    with t2:
        st.session_state.show_sku = st.checkbox("SKU", value=st.session_state.get("show_sku", True))
    with t3:
        st.session_state.show_desc = st.checkbox("Description", value=st.session_state.get("show_desc", True))
    with t4:
        st.session_state.show_attrs = st.checkbox("Attributes", value=st.session_state.get("show_attrs", True))
    with t5:
        st.session_state.exclude_oos = st.checkbox("Exclude out-of-stock", value=st.session_state.get("exclude_oos", True))

    filtered = products[:]
    if st.session_state.selected_cats:
        sset = set(st.session_state.selected_cats)
        filtered = [p for p in filtered if set(p.get("categories") or []).intersection(sset)]

    q = st.session_state.search.strip().lower()
    if q:
        filtered = [p for p in filtered if (q in safe_text(p.get("name")).lower()) or (q in safe_text(p.get("sku")).lower())]

    if st.session_state.exclude_oos:
        filtered = [p for p in filtered if is_in_stock(p)]

    filtered.sort(key=lambda p: (primary_category(p).lower(), safe_text(p.get("name")).lower()))
    st.session_state.products_filtered = filtered

    sale_count = sum(1 for p in filtered if p.get("on_sale"))
    missing_img = sum(1 for p in filtered if not p.get("_img_url"))
    missing_url = sum(1 for p in filtered if not safe_text(p.get("url")).startswith("http"))

    st.markdown(
        f"""
        <div class="summary">
          <b>Catalog summary</b><br>
          Products: <b>{len(filtered):,}</b> • On sale: <b>{sale_count:,}</b><br>
          Missing image URLs: <b>{missing_img:,}</b> • Missing clickable URLs: <b>{missing_url:,}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Continue → Generate PDF", disabled=(len(filtered) == 0)):
        st.session_state.step = 4
        st.rerun()


# =========================
# Step 4 — Download images + build PDF
# =========================
if st.session_state.step >= 4:
    step_indicator(4)
    filtered = st.session_state.products_filtered
    if not filtered:
        st.error("No products selected.")
        st.stop()

    preset = st.session_state.get("preset", "Reliable")
    if preset == "Reliable":
        dl_workers, dl_retries, dl_timeout, dl_backoff = 4, 10, 25, 0.9
    elif preset == "Normal":
        dl_workers, dl_retries, dl_timeout, dl_backoff = 6, 6, 18, 0.7
    else:
        dl_workers, dl_retries, dl_timeout, dl_backoff = 10, 3, 12, 0.5

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Generate & download")
    st.markdown('<div class="muted">Downloading images (cached + retries), then building the PDF.</div>', unsafe_allow_html=True)

    progress = st.progress(0.0)
    status = st.empty()
    activity_box = st.empty()
    activity = []
    t0 = time.time()

    def log(msg: str):
        activity.append(f"[{time.time()-t0:6.1f}s] {msg}")
        activity_box.markdown(f"<div class='activity'>{html.escape(chr(10).join(activity[-22:]))}</div>", unsafe_allow_html=True)

    items = [p for p in filtered if p.get("_img_url")]
    total = max(1, len(items))
    done = 0
    ok = 0

    status.info("Stage 1/2 — Downloading images…")
    log(f"Preset={preset} workers={dl_workers} retries={dl_retries}")

    def dl_task(p: dict):
        b = download_with_retries(p["_img_url"], timeout=dl_timeout, retries=dl_retries, backoff=dl_backoff, log_cb=log)
        return p, b

    with ThreadPoolExecutor(max_workers=dl_workers) as ex:
        futures = [ex.submit(dl_task, p) for p in items]
        for fut in as_completed(futures):
            p, b = fut.result()
            if b:
                p["_image_path"] = str(cache_path_for_url(p["_img_url"]))
                ok += 1
            else:
                p["_image_path"] = None
            done += 1
            progress.progress(min(0.75, (done / total) * 0.75))
            if done % 50 == 0 or done == total:
                log(f"Images: {done}/{total} ok={ok} missing={done-ok}")

    status.info("Stage 2/2 — Building PDF…")
    progress.progress(0.85)

    pdf_bytes = make_catalog_pdf_bytes(
        filtered,
        title=st.session_state.get("catalog_title", DEFAULT_TITLE),
        page_size=st.session_state.get("page_size", "A4"),
        columns=int(st.session_state.get("columns", 3)),
        currency_symbol=st.session_state.get("currency_symbol", "£"),
        show_price=bool(st.session_state.get("show_price", True)),
        show_sku=bool(st.session_state.get("show_sku", True)),
        show_desc=bool(st.session_state.get("show_desc", True)),
        show_attrs=bool(st.session_state.get("show_attrs", True)),
        exclude_oos=bool(st.session_state.get("exclude_oos", True)),
    )

    st.session_state.last_pdf = pdf_bytes
    progress.progress(1.0)
    status.success("PDF ready.")

    st.download_button("Download PDF", data=pdf_bytes, file_name="bkosher_catalog.pdf", mime="application/pdf")
    st.caption("Clickability note: Some PDF viewers ignore links. Best test: Chrome/Edge or Adobe Acrobat Reader.")
    st.markdown("</div>", unsafe_allow_html=True)