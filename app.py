# app.py
# B-Kosher Customer Catalog Builder (WooCommerce)
#
# ✅ Password login page first (password stored in Streamlit Secrets)
# ✅ WooCommerce API credentials stored in Streamlit Secrets (no re-typing)
# ✅ Data source: CSV upload OR WooCommerce API
# ✅ API caching:
#    - Disk cache (api_cache/products.json) for fast loads
#    - Streamlit cache (st.cache_data) to avoid repeat work per session
#    - Refresh button to force re-fetch
# ✅ API fetch hardened:
#    - requests.Session + HTTP retries + SSL/connection retries + polite paging delay
#    - NEVER prints consumer_key/consumer_secret in any error message
# ✅ PDF: NO FRONT COVER (starts with Contents page)
# ✅ PDF: 3x3 on A4 when columns=3
# ✅ Clickable product cards (opens product URL)
# ✅ Sale badge + Save £X (Y%) + regular price struck-through
# ✅ Optional attributes, description, SKU
# ✅ Reliable image downloading: disk cache + retries/backoff + presets
# ✅ Clean text: converts &amp; -> &, strips HTML, avoids "nan"
# ✅ Preview first N pages if PyMuPDF installed (optional)
#
# -------------------------------
# LOCAL SECRETS SETUP (REQUIRED)
# -------------------------------
# In the SAME folder as app.py create:
#
#   .streamlit/secrets.toml
#
# Contents:
#   WC_URL = "https://b-kosher.co.uk"  # Try without www if SSL errors
#   WC_CK  = "ck_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
#   WC_CS  = "cs_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
#   APP_PASSWORD = "Bkosher1234!"
#
# Install:
#   python -m pip install streamlit pandas requests reportlab pillow svglib
# Optional preview:
#   python -m pip install pymupdf
#
# Run:
#   python -m streamlit run app.py

import io
import re
import time
import math
import json
import html
import base64
import hashlib
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import pandas as pd
import requests
import streamlit as st
from PIL import Image

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader, simpleSplit

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional PDF preview
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ModuleNotFoundError:
    HAS_PYMUPDF = False


# =========================
# APP CONFIG + BRAND
# =========================
st.set_page_config(page_title="B-Kosher Catalog Builder", layout="wide")

BRAND_RED_HEX = "#C8102E"
BRAND_BLUE_HEX = "#004C97"
BRAND_RED = colors.HexColor(BRAND_RED_HEX)
BRAND_BLUE = colors.HexColor(BRAND_BLUE_HEX)

DEFAULT_TITLE = "B-Kosher Product Catalog"
DEFAULT_SITE = "www.b-kosher.co.uk"
DEFAULT_BASE_URL = "https://www.b-kosher.co.uk"

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
      .panel h3 {{
        margin: 0 0 8px 0;
        font-size: 1.05rem;
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
      .badge.red {{
        color: {BRAND_RED_HEX};
        border-color: rgba(200,16,46,0.25);
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
# SECRETS + LOGIN GATE
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

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Login"):
            if pw == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    with c2:
        st.caption("If you don’t have access, please contact B-Kosher.")
    st.stop()


login_gate()


# =========================
# URL sanitizer (NO KEY LEAKS)
# =========================
def sanitize_url(u: str) -> str:
    """Remove consumer_key/consumer_secret from any URL before showing it."""
    try:
        parts = urlsplit(u)
        q = parse_qsl(parts.query, keep_blank_values=True)
        q = [(k, v) for (k, v) in q if k not in ("consumer_key", "consumer_secret")]
        clean_query = urlencode(q)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, clean_query, parts.fragment))
    except Exception:
        return (u or "").replace("consumer_key=", "consumer_key=***").replace("consumer_secret=", "consumer_secret=***")


# =========================
# UI helpers
# =========================
def render_logo_svg_in_app(svg_path: str, width: int = 200):
    b = Path(svg_path).read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    st.markdown(
        f"""<div style="margin:0 0 10px 0;">
            <img src="data:image/svg+xml;base64,{b64}" width="{width}" />
        </div>""",
        unsafe_allow_html=True,
    )


# =========================
# Data helpers
# =========================
def safe_text(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    if s.lower() in ("nan", "none", "null"):
        return ""
    return html.unescape(s)


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
    return f"{symbol}{v:,.2f}"


def wrap_with_ellipsis(text: str, font_name: str, font_size: float, max_width: float, max_lines: int) -> list[str]:
    text = safe_text(text)
    if not text:
        return []
    lines = simpleSplit(text, font_name, font_size, max_width)
    if len(lines) <= max_lines:
        return lines
    kept = lines[:max_lines]
    last = kept[-1]
    ell = "…"
    from reportlab.pdfbase.pdfmetrics import stringWidth
    while last and stringWidth(last + ell, font_name, font_size) > max_width:
        last = last[:-1].rstrip()
    kept[-1] = (last + ell) if last else ell
    return kept


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def primary_category(p: dict) -> str:
    cats = p.get("categories") or []
    if isinstance(cats, list) and cats:
        return safe_text(cats[0]) or "Other"
    return "Other"


def is_in_stock(p: dict) -> bool:
    s = safe_text(p.get("stock_status")).lower()
    if s == "outofstock":
        return False
    return True


def effective_price(p: dict) -> float | None:
    reg = p.get("regular_price")
    sale = p.get("sale_price")
    if p.get("on_sale") and sale is not None:
        return sale
    return reg


# =========================
# Disk caches (images + api)
# =========================
CACHE_DIR = Path("./image_cache")
CACHE_DIR.mkdir(exist_ok=True)

API_CACHE_DIR = Path("./api_cache")
API_CACHE_DIR.mkdir(exist_ok=True)
API_CACHE_FILE = API_CACHE_DIR / "products.json"


def cache_path_for_url(url: str) -> Path:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.jpg"


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


def download_with_retries(url: str, *, timeout: int, retries: int, backoff: float) -> tuple[bytes | None, str]:
    if not url or not url.startswith("http"):
        return None, "permanent"

    cached = read_image_cache(url)
    if cached:
        return cached, "ok"

    for attempt in range(retries + 1):
        if attempt > 0:
            time.sleep(min(backoff * (2 ** (attempt - 1)), 15.0))
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "bkosher-catalog/1.0"})
            if r.status_code == 404:
                return None, "permanent"
            if r.status_code in (429, 500, 502, 503, 504):
                continue
            r.raise_for_status()
            raw = r.content
            cooked = resize_to_jpeg_bytes(raw)
            final = cooked if cooked else raw
            write_image_cache(url, final)
            return final, "ok"
        except Exception:
            continue

    return None, "temporary"


def bytes_to_pil(b: bytes) -> Image.Image | None:
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
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
# SVG logo for PDF
# =========================
@st.cache_resource(show_spinner=False)
def load_svg_drawing(svg_path: str):
    return svg2rlg(svg_path)


def draw_svg_logo(c: canvas.Canvas, drawing, x: float, y: float, target_h: float) -> tuple[float, float]:
    if drawing is None:
        return (0.0, 0.0)
    dw = float(getattr(drawing, "width", 0.0))
    dh = float(getattr(drawing, "height", 0.0))
    if dw <= 0 or dh <= 0:
        return (0.0, 0.0)
    scale = target_h / dh
    placed_w = dw * scale
    placed_h = dh * scale
    c.saveState()
    c.translate(x, y)
    c.scale(scale, scale)
    renderPDF.draw(drawing, c, 0, 0)
    c.restoreState()
    return (placed_w, placed_h)


# =========================
# WooCommerce API fetch (HARDENED)
# =========================
def wc_fetch_products(
    base_url: str,
    ck: str,
    cs: str,
    *,
    per_page: int = 25,          # safer than 50 for big catalogs
    timeout: int = 30,
    status: str = "publish",
    log_cb=None,
    max_pages: int = 10000,
    polite_delay_every: int = 10,
    polite_delay_sec: float = 0.3,
):
    if not (base_url and ck and cs):
        raise RuntimeError("Woo API secrets missing. Set WC_URL, WC_CK, WC_CS in secrets.toml.")

    base_url = base_url.rstrip("/")
    endpoint = f"{base_url}/wp-json/wc/v3/products"

    session = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    page = 1
    out = []
    ssl_retries = 8

    while True:
        if page > max_pages:
            raise RuntimeError(f"Stopped: exceeded max_pages={max_pages}. Check pagination.")

        params = {
            "consumer_key": ck,
            "consumer_secret": cs,
            "per_page": per_page,
            "page": page,
            "status": status,
        }

        if log_cb:
            log_cb(f"Fetching API page {page} (per_page={per_page})…")

        last_exc = None
        for attempt in range(ssl_retries + 1):
            if attempt > 0:
                sleep_s = min(0.8 * (2 ** (attempt - 1)), 20.0)
                if log_cb:
                    log_cb(f"Connection hiccup — retry {attempt}/{ssl_retries} after {sleep_s:.1f}s…")
                time.sleep(sleep_s)

            try:
                r = session.get(endpoint, params=params, timeout=timeout, headers={"User-Agent": "bkosher-catalog/1.0"})

                if r.status_code in (401, 403):
                    try:
                        j = r.json()
                        raise RuntimeError(f"{j.get('code')} ({r.status_code}): {j.get('message')}")
                    except Exception:
                        raise RuntimeError(f"Unauthorized ({r.status_code}). Check Woo REST API key permissions.")

                if r.status_code >= 400:
                    safe_u = sanitize_url(r.url)
                    raise RuntimeError(f"API request failed ({r.status_code}) at {safe_u}")

                batch = r.json()
                if not batch:
                    return out

                out.extend(batch)

                if polite_delay_every and page % polite_delay_every == 0:
                    time.sleep(polite_delay_sec)

                if len(batch) < per_page:
                    return out

                page += 1
                last_exc = None
                break

            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_exc = e
                continue

        if last_exc is not None:
            raise RuntimeError(
                "API fetch failed due to repeated SSL/connection errors while paging products. "
                "Try: (1) set WC_URL without 'www', (2) lower per_page, (3) wait and retry, "
                "or (4) use the disk cache if already created.\n"
                f"Details: {type(last_exc).__name__}: {str(last_exc)[:200]}"
            )


def wc_to_product(p: dict) -> dict:
    cats = [safe_text(c.get("name")) for c in (p.get("categories") or []) if safe_text(c.get("name"))]

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
        "_image_pil": None,
    }


# =========================
# Cached API loader (disk + st.cache_data)
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def get_products_from_api_or_cache(
    wc_url: str,
    wc_ck: str,
    wc_cs: str,
    timeout: int = 30,
    force_refresh: bool = False,
):
    if not force_refresh:
        cached_raw, fetched_at = api_cache_load()
        if cached_raw is not None:
            normalized = [wc_to_product(p) for p in cached_raw]
            return normalized, fetched_at, "disk_cache"

    raw = wc_fetch_products(wc_url, wc_ck, wc_cs, per_page=25, timeout=timeout)
    api_cache_save(raw)
    normalized = [wc_to_product(p) for p in raw]
    fetched_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return normalized, fetched_at, "api"


# =========================
# PDF preview
# =========================
def pdf_preview_images(pdf_bytes: bytes, max_pages: int = 2, zoom: float = 1.35) -> list[bytes]:
    if not HAS_PYMUPDF:
        return []
    imgs = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = min(max_pages, doc.page_count)
    mat = fitz.Matrix(zoom, zoom)
    for i in range(n):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        imgs.append(pix.tobytes("png"))
    doc.close()
    return imgs


# =========================
# PDF generator (NO COVER)
# =========================
def make_catalog_pdf(
    products: list[dict],
    *,
    title: str,
    pagesize_name: str,
    columns: int,
    show_price: bool,
    show_sku: bool,
    show_desc: bool,
    show_attrs: bool,
    exclude_oos: bool,
    new_page_per_category: bool,
    currency_symbol: str,
    brand_site: str,
    logo_drawing=None,
):
    pagesize = A4 if pagesize_name == "A4" else letter
    page_w, page_h = pagesize

    margin = 12 * mm
    gutter = 6 * mm
    header_h = 16 * mm
    footer_h = 14 * mm
    category_bar_h = 18  # points

    pdf = io.BytesIO()
    c = canvas.Canvas(pdf, pagesize=pagesize)

    today_str = datetime.date.today().strftime("%d %b %Y")
    disclaimer = f"Prices correct as of {today_str}"

    def draw_header(page_no: int):
        c.setStrokeColor(BRAND_BLUE)
        c.setLineWidth(1.1)
        c.line(margin, page_h - margin - header_h + 6, page_w - margin, page_h - margin - header_h + 6)

        logo_h = 11 * mm
        logo_x = margin
        logo_y = page_h - margin - (logo_h + 3)
        logo_w = 0.0
        if logo_drawing is not None:
            logo_w, _ = draw_svg_logo(c, logo_drawing, logo_x, logo_y, logo_h)

        tx = margin + (logo_w + 10 if logo_w > 0 else 0)
        c.setFillColor(BRAND_BLUE)
        c.setFont("Helvetica-Bold", 12.8)
        c.drawString(tx, page_h - margin - 13, safe_text(title) or DEFAULT_TITLE)

        c.setFillColor(colors.black)
        c.setFont("Helvetica", 9)
        c.drawRightString(page_w - margin, page_h - margin - 13, f"Page {page_no}")

    def draw_footer():
        c.setStrokeColor(BRAND_RED)
        c.setLineWidth(0.9)
        y_line = margin + footer_h - 8
        c.line(margin, y_line, page_w - margin, y_line)
        c.setFillColor(colors.HexColor("#374151"))
        c.setFont("Helvetica", 8.5)
        c.drawString(margin, margin + 2, f"{brand_site} • {disclaimer}")
        c.setFillColor(colors.black)

    def draw_category_bar(cat_name: str):
        bar_y = page_h - margin - header_h - 8
        c.setFillColor(BRAND_BLUE)
        c.roundRect(margin, bar_y - category_bar_h, page_w - 2 * margin, category_bar_h, 6, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin + 10, bar_y - 13, cat_name)
        c.setFillColor(colors.black)

    def draw_sale_badge(x: float, y: float, w: float, h: float, p: dict):
        reg = p.get("regular_price")
        sale = p.get("sale_price")
        if not p.get("on_sale") or sale is None or reg is None or sale >= reg:
            return
        save_amt = reg - sale
        save_pct = (save_amt / reg) * 100 if reg else 0

        badge_w = 28 * mm
        badge_h = 10 * mm
        bx = x + w - badge_w - 6
        by = y + h - badge_h - 6

        c.setFillColor(BRAND_RED)
        c.roundRect(bx, by, badge_w, badge_h, 5, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 9.5)
        c.drawCentredString(bx + badge_w / 2, by + 2.6, "SALE")

        c.setFillColor(colors.HexColor("#111827"))
        c.setFont("Helvetica", 8.2)
        c.drawRightString(x + w - 6, by - 4, f"Save {currency_symbol}{save_amt:.2f} ({save_pct:.0f}%)")
        c.setFillColor(colors.black)

    def draw_card(x: float, y: float, card_w: float, card_h: float, p: dict):
        c.setLineWidth(1.0)
        c.setStrokeColor(BRAND_BLUE)
        c.roundRect(x, y, card_w, card_h, 8, stroke=1, fill=0)

        url = safe_text(p.get("url"))
        if url.startswith("http"):
            inset = 0.5
            c.linkURL(url, (x + inset, y + inset, x + card_w - inset, y + card_h - inset), relative=0, thickness=0)

        draw_sale_badge(x, y, card_w, card_h, p)

        pad = 7

        img_box_h = card_h * 0.48
        img_box_w = card_w - 2 * pad
        img_x = x + pad
        img_y = y + card_h - pad - img_box_h

        img = p.get("_image_pil")
        if img:
            iw, ih = img.size
            scale = min(img_box_w / iw, img_box_h / ih)
            nw, nh = iw * scale, ih * scale
            ox = img_x + (img_box_w - nw) / 2
            oy = img_y + (img_box_h - nh) / 2
            c.drawImage(ImageReader(img), ox, oy, nw, nh, preserveAspectRatio=True, anchor="c")

        text_x = x + pad
        text_w = card_w - 2 * pad
        line_y = img_y - 8

        c.setFont("Helvetica-Bold", 10.0)
        for ln in wrap_with_ellipsis(p.get("name", ""), "Helvetica-Bold", 10.0, text_w, 2):
            c.drawString(text_x, line_y, ln)
            line_y -= 12

        if show_price:
            reg = p.get("regular_price")
            sale = p.get("sale_price")
            on_sale = bool(p.get("on_sale")) and sale is not None and reg is not None and sale < reg

            if on_sale:
                c.setFillColor(BRAND_RED)
                c.setFont("Helvetica-Bold", 10.6)
                c.drawString(text_x, line_y, fmt_money(currency_symbol, sale))

                c.setFillColor(colors.HexColor("#6b7280"))
                reg_txt = fmt_money(currency_symbol, reg)
                c.setFont("Helvetica", 8.6)
                rx = text_x + 58
                c.drawString(rx, line_y + 1, reg_txt)
                c.setLineWidth(1)
                c.setStrokeColor(colors.HexColor("#6b7280"))
                c.line(rx, line_y + 4.5, rx + c.stringWidth(reg_txt, "Helvetica", 8.6), line_y + 4.5)

                c.setStrokeColor(BRAND_BLUE)
                c.setFillColor(colors.black)
                line_y -= 14
            else:
                if reg is not None:
                    c.setFillColor(BRAND_RED)
                    c.setFont("Helvetica-Bold", 10.6)
                    c.drawString(text_x, line_y, fmt_money(currency_symbol, reg))
                    c.setFillColor(colors.black)
                    line_y -= 14

        if show_sku:
            sku = safe_text(p.get("sku"))
            if sku:
                c.setFont("Helvetica", 8.8)
                c.setFillColor(colors.HexColor("#222222"))
                c.drawString(text_x, line_y, f"SKU: {sku}")
                line_y -= 12
                c.setFillColor(colors.black)

        if show_attrs:
            attrs = p.get("attributes") or []
            if attrs:
                c.setFont("Helvetica", 8.2)
                c.setFillColor(colors.HexColor("#374151"))
                shown = 0
                for (an, av) in attrs:
                    if shown >= 2:
                        break
                    an = safe_text(an)
                    av = safe_text(av)
                    if not an or not av:
                        continue
                    one = wrap_with_ellipsis(f"{an}: {av}", "Helvetica", 8.2, text_w, 1)
                    if one:
                        c.drawString(text_x, line_y, one[0])
                        line_y -= 10
                        shown += 1
                c.setFillColor(colors.black)

        if show_desc:
            desc = strip_html(p.get("short_desc", ""))
            if desc:
                c.setFont("Helvetica", 8.2)
                c.setFillColor(colors.HexColor("#555555"))
                for ln in wrap_with_ellipsis(desc, "Helvetica", 8.2, text_w, 2):
                    c.drawString(text_x, line_y, ln)
                    line_y -= 10
                c.setFillColor(colors.black)

    grouped = {}
    for p in products:
        if exclude_oos and not is_in_stock(p):
            continue
        grouped.setdefault(primary_category(p), []).append(p)
    categories = sorted(grouped.keys(), key=lambda s: s.lower())

    usable_w = page_w - 2 * margin
    card_w = (usable_w - (columns - 1) * gutter) / columns

    reserved_top = header_h + category_bar_h + 18
    reserved_bottom = footer_h + 8
    usable_h_cards = page_h - margin - margin - reserved_top - reserved_bottom

    if pagesize_name == "A4" and columns == 3:
        rows = 3
        card_h = (usable_h_cards - (rows - 1) * gutter) / rows
    else:
        target = 78 * mm
        rows = max(1, int(usable_h_cards // (target + gutter)))
        card_h = (usable_h_cards - (rows - 1) * gutter) / rows

    # Contents page numbers (content starts page 2)
    page_no = 2
    cat_first_page = {}
    per_page = rows * columns
    for cat in categories:
        items = grouped[cat]
        if not items:
            continue
        cat_first_page[cat] = page_no
        pages_for_cat = math.ceil(len(items) / per_page) if per_page else 1
        page_no += pages_for_cat

    c.setFillColor(BRAND_BLUE)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin, page_h - margin - 50, "Contents")

    c.setFillColor(colors.HexColor("#374151"))
    c.setFont("Helvetica", 10)
    c.drawString(margin, page_h - margin - 70, disclaimer)

    y = page_h - margin - 105
    c.setFont("Helvetica", 11)
    c.setFillColor(colors.HexColor("#111827"))

    for cat in categories:
        if y < margin + 80:
            c.showPage()
            c.setFillColor(BRAND_BLUE)
            c.setFont("Helvetica-Bold", 20)
            c.drawString(margin, page_h - margin - 50, "Contents (cont.)")
            y = page_h - margin - 90
            c.setFont("Helvetica", 11)
            c.setFillColor(colors.HexColor("#111827"))

        pg = cat_first_page.get(cat, "")
        c.drawString(margin, y, cat)
        c.setFillColor(colors.HexColor("#6b7280"))
        c.drawRightString(page_w - margin, y, str(pg))
        c.setFillColor(colors.HexColor("#111827"))
        y -= 18

    c.showPage()

    page_no = 2
    for cat in categories:
        items = grouped[cat]
        if not items:
            continue

        idx = 0
        while idx < len(items):
            draw_header(page_no)
            draw_category_bar(cat)

            start_y = page_h - margin - reserved_top - card_h
            y = start_y

            for _r in range(rows):
                x = margin
                for _c in range(columns):
                    if idx >= len(items):
                        break
                    draw_card(x, y, card_w, card_h, items[idx])
                    idx += 1
                    x += card_w + gutter
                y -= card_h + gutter
                if idx >= len(items):
                    break

            draw_footer()
            c.showPage()
            page_no += 1

    c.save()
    pdf.seek(0)
    return pdf.read()


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
if "last_preview_imgs" not in st.session_state:
    st.session_state.last_preview_imgs = []


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("### B-Kosher")
    try:
        render_logo_svg_in_app("bkosher.svg", width=190)
    except Exception:
        st.caption("Put bkosher.svg next to app.py.")

    if WC_URL and WC_CK and WC_CS:
        st.success("Woo API configured (secrets)")
    else:
        st.error("Woo API secrets missing")

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
        4: "Step 4 — Preview & export",
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
# Step 1 — Choose data source
# =========================
step_indicator(st.session_state.step)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown("### Data source")
st.markdown('<div class="muted">Choose: upload a CSV export, or load from WooCommerce API (cached).</div>', unsafe_allow_html=True)
data_source = st.radio("Source", ["CSV Upload", "WooCommerce API"], horizontal=True, key="data_source")
st.markdown("</div>", unsafe_allow_html=True)

if st.button("Continue → Load products"):
    st.session_state.step = 2
    st.rerun()


# =========================
# Step 2 — Load products (CSV or API with caching)
# =========================
if st.session_state.step >= 2:
    step_indicator(2)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Load products")
    st.markdown('<div class="muted">API loads from disk cache instantly, unless you press refresh.</div>', unsafe_allow_html=True)

    activity = []
    activity_box = st.empty()

    def log(msg: str):
        activity.append(msg)
        activity_box.markdown(f"<div class='activity'>{html.escape(chr(10).join(activity[-18:]))}</div>", unsafe_allow_html=True)

    if data_source == "CSV Upload":
        base_url = st.text_input("Base URL (used to build links if CSV has none)", value=DEFAULT_BASE_URL)
        csv_file = st.file_uploader("Upload WooCommerce CSV export", type=["csv"], key="csv_file")
        load_btn = st.button("Load products from CSV", disabled=(csv_file is None))

        if load_btn:
            log("Reading CSV…")
            df = pd.read_csv(csv_file)

            col_id = pick_col(df, ["ID", "Id"])
            col_name = pick_col(df, ["Name", "Product name", "Title"])
            col_sku = pick_col(df, ["SKU"])
            col_reg = pick_col(df, ["Regular price", "Regular Price", "Price"])
            col_sale = pick_col(df, ["Sale price", "Sale Price"])
            col_cats = pick_col(df, ["Categories", "Category"])
            col_desc = pick_col(df, ["Short description", "Short Description", "Description"])
            col_imgs = pick_col(df, ["Images", "Image", "Image URLs"])
            col_url = pick_col(df, ["Permalink", "Product URL", "URL", "Link"])
            col_stock = pick_col(df, ["Stock status", "Stock Status", "stock_status"])
            col_slug = pick_col(df, ["Slug", "slug"])

            log("Normalizing rows…")
            products = []
            for _, row in df.iterrows():
                name = safe_text(row.get(col_name))
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
                    if ncol in df.columns and vcol in df.columns:
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
                    "_image_pil": None,
                })

            st.session_state.products_raw = products
            st.session_state.step = 3
            st.success(f"Loaded {len(products):,} products from CSV.")
            st.rerun()

    else:
        st.write("WooCommerce API uses secrets + caching (no keys shown).")
        api_timeout = st.slider("API timeout (seconds)", 10, 60, 30)

        c1, c2 = st.columns(2)
        with c1:
            load_cached_btn = st.button("Load products (use cache if available)")
        with c2:
            refresh_btn = st.button("Refresh cache (fetch again)")

        if load_cached_btn or refresh_btn:
            try:
                with st.spinner("Loading products…"):
                    products, fetched_at, source = get_products_from_api_or_cache(
                        WC_URL, WC_CK, WC_CS,
                        timeout=api_timeout,
                        force_refresh=bool(refresh_btn),
                    )
                st.session_state.products_raw = products
                st.session_state.step = 3
                st.success(f"Loaded {len(products):,} products • Source: {source} • Cached at: {fetched_at}")
                st.rerun()
            except Exception as e:
                # Important: our exceptions are sanitized; do not show raw request URL
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

    g1, g2, g3 = st.columns([1.2, 1.0, 1.0])
    with g1:
        st.session_state.catalog_title = st.text_input("Catalog title", value=st.session_state.get("catalog_title", DEFAULT_TITLE))
        st.session_state.logo_path = st.text_input("Logo path (SVG)", value=st.session_state.get("logo_path", "bkosher.svg"))
    with g2:
        st.session_state.pagesize_name = st.selectbox("Page size", ["A4", "Letter"], index=0 if st.session_state.get("pagesize_name", "A4") == "A4" else 1)
        st.session_state.columns = st.selectbox("Cards per row", [1, 2, 3], index={1:0,2:1,3:2}.get(int(st.session_state.get("columns", 3)), 2))
    with g3:
        st.session_state.currency_symbol = st.text_input("Currency symbol", value=st.session_state.get("currency_symbol", "£"))
        st.session_state.preview_pages = st.selectbox("Preview pages", [1, 2, 3], index={1:0,2:1,3:2}.get(int(st.session_state.get("preview_pages", 2)), 1))

    try:
        _ = load_svg_drawing(st.session_state.logo_path)
    except Exception:
        st.warning("Logo not loaded (PDF will still generate, but without logo).")

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

    st.session_state.new_page_per_category = st.checkbox("New page per category", value=st.session_state.get("new_page_per_category", True))

    st.session_state.sort_mode = st.selectbox(
        "Sort",
        [
            "Category → Name",
            "Category → Price (low→high)",
            "Category → Price (high→low)",
            "On sale first → Name",
            "In stock first → Name",
        ],
        index=["Category → Name", "Category → Price (low→high)", "Category → Price (high→low)", "On sale first → Name", "In stock first → Name"].index(
            st.session_state.get("sort_mode", "Category → Name")
        ),
    )

    st.session_state.preset = st.selectbox(
        "Image download preset",
        ["Reliable", "Normal", "Fast"],
        index=["Reliable", "Normal", "Fast"].index(st.session_state.get("preset", "Reliable")),
    )

    # Filter
    filtered = products[:]
    if st.session_state.selected_cats:
        sset = set(st.session_state.selected_cats)
        filtered = [p for p in filtered if set(p.get("categories") or []).intersection(sset)]

    q = st.session_state.search.strip().lower()
    if q:
        filtered = [p for p in filtered if (q in safe_text(p.get("name")).lower()) or (q in safe_text(p.get("sku")).lower())]

    if st.session_state.exclude_oos:
        filtered = [p for p in filtered if is_in_stock(p)]

    # Sort
    if st.session_state.sort_mode == "Category → Name":
        filtered.sort(key=lambda p: (primary_category(p).lower(), safe_text(p.get("name")).lower()))
    elif st.session_state.sort_mode == "Category → Price (low→high)":
        filtered.sort(key=lambda p: (primary_category(p).lower(), (effective_price(p) is None), effective_price(p) or 1e18, safe_text(p.get("name")).lower()))
    elif st.session_state.sort_mode == "Category → Price (high→low)":
        filtered.sort(key=lambda p: (primary_category(p).lower(), (effective_price(p) is None), -(effective_price(p) or -1e18), safe_text(p.get("name")).lower()))
    elif st.session_state.sort_mode == "On sale first → Name":
        filtered.sort(key=lambda p: (not bool(p.get("on_sale")), safe_text(p.get("name")).lower()))
    else:
        filtered.sort(key=lambda p: (not is_in_stock(p), safe_text(p.get("name")).lower()))

    st.session_state.products_filtered = filtered

    missing_img_url = sum(1 for p in filtered if not p.get("_img_url"))
    missing_price = sum(1 for p in filtered if p.get("regular_price") is None and p.get("sale_price") is None)
    missing_url = sum(1 for p in filtered if not safe_text(p.get("url")).startswith("http"))
    sale_count = sum(1 for p in filtered if p.get("on_sale"))

    st.markdown(
        f"""
        <div class="summary">
          <b>Catalog summary</b><br>
          Products: <b>{len(filtered):,}</b> • On sale: <b>{sale_count:,}</b><br>
          Missing image URLs: <b>{missing_img_url:,}</b> • Missing prices: <b>{missing_price:,}</b> • Missing clickable URLs: <b>{missing_url:,}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Generate PDF + Preview", disabled=(len(filtered) == 0)):
        st.session_state.step = 4
        st.session_state.last_pdf = None
        st.session_state.last_preview_imgs = []
        st.rerun()


# =========================
# Step 4 — Generate + Preview + Download
# =========================
if st.session_state.step >= 4:
    step_indicator(4)
    filtered = st.session_state.products_filtered
    if not filtered:
        st.error("No products selected.")
        st.stop()

    try:
        logo_drawing = load_svg_drawing(st.session_state.get("logo_path", "bkosher.svg"))
    except Exception:
        logo_drawing = None

    preset = st.session_state.get("preset", "Reliable")
    if preset == "Reliable":
        dl_workers, dl_retries, dl_timeout, dl_backoff = 4, 8, 25, 0.8
    elif preset == "Normal":
        dl_workers, dl_retries, dl_timeout, dl_backoff = 6, 5, 18, 0.6
    else:
        dl_workers, dl_retries, dl_timeout, dl_backoff = 10, 3, 12, 0.5

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Preview & export")
    st.markdown('<div class="muted">Downloads images (cached + retries), builds PDF, renders preview (optional), then download.</div>', unsafe_allow_html=True)

    progress = st.progress(0.0)
    stage = st.empty()
    activity_box = st.empty()
    activity = []
    start_time = time.time()

    def log(msg: str):
        elapsed = time.time() - start_time
        activity.append(f"[{elapsed:6.1f}s] {msg}")
        activity_box.markdown(f"<div class='activity'>{html.escape(chr(10).join(activity[-22:]))}</div>", unsafe_allow_html=True)

    stage.info("Stage 1/3 — Downloading images…")
    log(f"Preset={preset} workers={dl_workers} retries={dl_retries} timeout={dl_timeout}s")

    items = [p for p in filtered if p.get("_img_url")]
    total = max(1, len(items))
    done = 0
    ok = 0
    noimg = 0

    def task(p: dict):
        b, _status = download_with_retries(p["_img_url"], timeout=dl_timeout, retries=dl_retries, backoff=dl_backoff)
        return p, b

    with ThreadPoolExecutor(max_workers=dl_workers) as ex:
        futures = [ex.submit(task, p) for p in items]
        for fut in as_completed(futures):
            p, b = fut.result()
            if b:
                p["_image_pil"] = bytes_to_pil(b)
                ok += 1
            else:
                p["_image_pil"] = None
                noimg += 1

            done += 1
            progress.progress(min(0.70, (done / total) * 0.70))
            if done % 50 == 0 or done == total:
                log(f"Images: {done}/{total} (ok {ok}, missing {noimg})")

    stage.info("Stage 2/3 — Building PDF…")
    progress.progress(0.80)

    pdf_bytes = make_catalog_pdf(
        filtered,
        title=st.session_state.get("catalog_title", DEFAULT_TITLE),
        pagesize_name=st.session_state.get("pagesize_name", "A4"),
        columns=int(st.session_state.get("columns", 3)),
        show_price=bool(st.session_state.get("show_price", True)),
        show_sku=bool(st.session_state.get("show_sku", True)),
        show_desc=bool(st.session_state.get("show_desc", True)),
        show_attrs=bool(st.session_state.get("show_attrs", True)),
        exclude_oos=bool(st.session_state.get("exclude_oos", True)),
        new_page_per_category=bool(st.session_state.get("new_page_per_category", True)),
        currency_symbol=st.session_state.get("currency_symbol", "£"),
        brand_site=DEFAULT_SITE,
        logo_drawing=logo_drawing,
    )
    st.session_state.last_pdf = pdf_bytes
    progress.progress(0.92)

    stage.info("Stage 3/3 — Rendering preview…")
    if HAS_PYMUPDF:
        try:
            imgs = pdf_preview_images(pdf_bytes, max_pages=int(st.session_state.get("preview_pages", 2)), zoom=1.35)
            st.session_state.last_preview_imgs = imgs
            stage.success("Ready.")
        except Exception as e:
            st.session_state.last_preview_imgs = []
            stage.warning(f"Preview failed: {str(e)[:200]} (you can still download the PDF).")
    else:
        st.session_state.last_preview_imgs = []
        stage.warning("Preview is disabled (PyMuPDF not installed). Run: python -m pip install pymupdf")

    progress.progress(1.0)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("PDF preview")
    if st.session_state.last_preview_imgs:
        for i, img_bytes in enumerate(st.session_state.last_preview_imgs, start=1):
            st.image(img_bytes, caption=f"Preview page {i}", use_container_width=True)
    else:
        st.info("Preview not available (but PDF is ready).")

    st.download_button(
        "Download PDF",
        data=st.session_state.last_pdf,
        file_name="bkosher_catalog.pdf",
        mime="application/pdf",
    )

    st.caption(
        "Clickability note: Some built-in PDF viewers ignore link annotations. "
        "Best test: open in Chrome/Edge or Adobe Acrobat Reader."
    )
