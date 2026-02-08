# app.py
# B-Kosher Catalog Builder (Streamlit)
# - WooCommerce API (default) or CSV upload
# - Resume-able API import (continues after timeouts / restarts)
# - Brand styling + logo (shown ONLY after login)
# - Category tree (parent + child + grandchild selectable)
# - Filters: sale only, exclude OOS, include/exclude private, search
# - PDF: Portrait/Landscape, grid density (Standard 3x3, Compact 6x5), clickable product links, sale badge
# - Text always wrapped/contained within product boxes
#
# Notes for Streamlit Cloud:
# - Put secrets in .streamlit/secrets.toml or Streamlit Cloud Secrets:
#     APP_PASSWORD = "Bkosher1234!"
#     WC_BASE_URL = "https://www.b-kosher.co.uk"
#     WC_CONSUMER_KEY = "ck_..."
#     WC_CONSUMER_SECRET = "cs_..."
#
# Local:
# - create .streamlit/secrets.toml with the same keys.

import base64
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps

from fpdf import FPDF  # fpdf2

# ----------------------------
# Branding (B-Kosher)
# ----------------------------
BRAND_RED = (200, 16, 46)     # #C8102E
BRAND_BLUE = (0, 76, 151)     # #004C97
BRAND_GREY = (107, 114, 128)  # #6B7280

DEFAULT_TITLE = "B-Kosher Product Catalog"
BRAND_SITE = "www.b-kosher.co.uk"
LOGO_FILENAME_CANDIDATES = [
    "B-kosher logo high q.png",
    "Bkosher.png",
    "bkosher.png",
    "bkosher.svg",
]

# ----------------------------
# Helpers / constants
# ----------------------------
CACHE_DIR = ".bkcache"
os.makedirs(CACHE_DIR, exist_ok=True)

API_PRODUCTS_CACHE = os.path.join(CACHE_DIR, "products.jsonl")
API_PROGRESS_FILE = os.path.join(CACHE_DIR, "progress.json")
CATS_CACHE_FILE = os.path.join(CACHE_DIR, "categories.json")

IMG_CACHE_DIR = os.path.join(CACHE_DIR, "images")
os.makedirs(IMG_CACHE_DIR, exist_ok=True)

USER_AGENT = "BkosherCatalogBuilder/1.0 (+streamlit)"

# For HTML cleanup
AMP_FIX_RE = re.compile(r"&amp;", re.IGNORECASE)


def now_str() -> str:
    return datetime.now().strftime("%d %b %Y")


def safe_text(s: Any) -> str:
    """Clean text for PDF and UI; avoid 'nan' and decode ampersands."""
    if s is None:
        return ""
    if isinstance(s, float) and pd.isna(s):
        return ""
    if isinstance(s, str):
        s2 = s.strip()
        if s2.lower() == "nan":
            return ""
        s2 = AMP_FIX_RE.sub("&", s2)
        s2 = s2.replace("\u00a0", " ")
        return s2
    return str(s)


def money_fmt(value: Any, currency: str = "£") -> str:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        v = float(value)
        return f"{currency}{v:.2f}"
    except Exception:
        return ""


def redact_url(url: str) -> str:
    """Remove consumer_key/consumer_secret from URL if present."""
    try:
        url = re.sub(r"(consumer_key=)[^&]+", r"\1REDACTED", url)
        url = re.sub(r"(consumer_secret=)[^&]+", r"\1REDACTED", url)
        return url
    except Exception:
        return url


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def inject_css() -> None:
    """Only CSS here. NO header/logo HTML (prevents showing before login)."""
    brand_blue = f"rgb({BRAND_BLUE[0]},{BRAND_BLUE[1]},{BRAND_BLUE[2]})"
    brand_red = f"rgb({BRAND_RED[0]},{BRAND_RED[1]},{BRAND_RED[2]})"

    st.markdown(
        f"""
<style>
/* Buttons */
.stButton > button {{
  background: {brand_red};
  color: white;
  border: 0;
  border-radius: 10px;
  padding: 0.55rem 1rem;
  font-weight: 700;
}}
.stButton > button:hover {{ filter: brightness(0.95); }}

/* Titles */
h1, h2, h3 {{ color: {brand_blue}; }}

/* Cards */
.bk-card {{
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 14px;
  background: white;
}}
.bk-muted {{ color:#6b7280; font-size: 0.92rem; }}

.bk-log {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 12px;
  background: #0b1220;
  color: #d1d5db;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.08);
  max-height: 280px;
  overflow: auto;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    cols = st.columns([0.18, 0.82])
    with cols[0]:
        for fn in LOGO_FILENAME_CANDIDATES:
            if os.path.exists(fn):
                st.image(fn, use_column_width=True)
                break
    with cols[1]:
        st.markdown(
            f"<div style='font-weight:900;font-size:24px;"
            f"color:rgb({BRAND_BLUE[0]},{BRAND_BLUE[1]},{BRAND_BLUE[2]});"
            f"line-height:1.1;'>B-Kosher Catalog Builder</div>",
            unsafe_allow_html=True,
        )
        st.caption("Customer-facing product catalog generator")


def get_secret(key: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(key, default))
    except Exception:
        return default


def require_login() -> None:
    app_pw = get_secret("APP_PASSWORD", "")
    if not app_pw:
        st.warning("APP_PASSWORD is not set. Add it in Streamlit secrets to enable login protection.")
        st.session_state["authed"] = True
        return

    if st.session_state.get("authed"):
        return

    st.markdown("## Login")
    st.write("Enter the password to access the catalog builder.")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if pw == app_pw:
            st.session_state["authed"] = True
            st.success("Logged in.")
            st.rerun()
        else:
            st.error("Incorrect password.")

    # IMPORTANT: Stop rendering the rest of the app if not authed
    st.stop()


# ----------------------------
# Woo API client
# ----------------------------
@dataclass
class WCConfig:
    base_url: str
    consumer_key: str
    consumer_secret: str
    timeout: int = 30


def build_wc_config_from_secrets() -> WCConfig:
    base = get_secret("WC_BASE_URL", "https://www.b-kosher.co.uk").rstrip("/")
    ck = get_secret("WC_CONSUMER_KEY", "")
    cs = get_secret("WC_CONSUMER_SECRET", "")
    return WCConfig(base_url=base, consumer_key=ck, consumer_secret=cs, timeout=30)


def wc_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def wc_get_json(
    sess: requests.Session,
    cfg: WCConfig,
    path: str,
    params: Dict[str, Any],
    timeout: Optional[int] = None,
    retries: int = 5,
    backoff: float = 1.25,
    log_fn=None,
) -> Any:
    """GET JSON with retry; never leak keys in errors."""
    url = f"{cfg.base_url}{path}"
    params = dict(params or {})
    params["consumer_key"] = cfg.consumer_key
    params["consumer_secret"] = cfg.consumer_secret

    t = timeout or cfg.timeout
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = sess.get(url, params=params, timeout=t)
            if r.status_code >= 500:
                # server error: retry
                last_err = f"Server error {r.status_code}"
                if log_fn:
                    log_fn(f"API: {last_err} (attempt {attempt}/{retries})")
                time.sleep(backoff * attempt)
                continue
            if r.status_code == 401:
                raise RuntimeError("Unauthorized (401). Check your WooCommerce REST API key permissions.")
            if r.status_code >= 400:
                raise RuntimeError(f"API request failed ({r.status_code}) at {redact_url(r.url)}")
            return r.json()
        except requests.exceptions.SSLError as e:
            last_err = f"SSL error: {e}"
            if log_fn:
                log_fn(f"API: SSL error (attempt {attempt}/{retries}) — retrying")
            time.sleep(backoff * attempt)
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            if log_fn:
                log_fn(f"API: network error (attempt {attempt}/{retries}) — retrying")
            time.sleep(backoff * attempt)
        except Exception as e:
            last_err = str(e)
            # If it's our explicit error, do not loop forever—still retry some cases.
            if log_fn:
                log_fn(f"API: {last_err} (attempt {attempt}/{retries})")
            time.sleep(backoff * attempt)

    raise RuntimeError(last_err or "Unknown API error")


# ----------------------------
# Categories (tree paths)
# ----------------------------
def load_categories_from_api(cfg: WCConfig, timeout: int, log_fn) -> List[dict]:
    sess = wc_session()
    all_cats: List[dict] = []
    page = 1
    per_page = 100
    while True:
        log_fn(f"API: fetching categories page {page}...")
        data = wc_get_json(
            sess,
            cfg,
            "/wp-json/wc/v3/products/categories",
            params={"per_page": per_page, "page": page, "hide_empty": False},
            timeout=timeout,
            retries=5,
            log_fn=log_fn,
        )
        if not isinstance(data, list) or len(data) == 0:
            break
        all_cats.extend(data)
        if len(data) < per_page:
            break
        page += 1

    # cache
    try:
        with open(CATS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(all_cats, f)
    except Exception:
        pass
    return all_cats


def load_categories_cached() -> List[dict]:
    if os.path.exists(CATS_CACHE_FILE):
        try:
            with open(CATS_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def build_cat_maps(cats: List[dict]) -> Tuple[Dict[int, dict], Dict[int, List[int]], Dict[int, str]]:
    by_id = {}
    children = {}
    for c in cats:
        try:
            cid = int(c.get("id"))
        except Exception:
            continue
        by_id[cid] = c
    for cid, c in by_id.items():
        parent = int(c.get("parent") or 0)
        children.setdefault(parent, []).append(cid)

    def path_for(cid: int) -> str:
        parts = []
        cur = cid
        seen = set()
        while cur and cur in by_id and cur not in seen:
            seen.add(cur)
            parts.append(safe_text(by_id[cur].get("name")))
            cur = int(by_id[cur].get("parent") or 0)
        parts.reverse()
        return " > ".join([p for p in parts if p])

    path_map = {cid: path_for(cid) for cid in by_id.keys()}
    return by_id, children, path_map


# ----------------------------
# Products load / resume cache
# ----------------------------
def read_progress() -> dict:
    if os.path.exists(API_PROGRESS_FILE):
        try:
            with open(API_PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def write_progress(p: dict) -> None:
    try:
        with open(API_PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(p, f)
    except Exception:
        pass


def clear_api_cache() -> None:
    for fn in [API_PRODUCTS_CACHE, API_PROGRESS_FILE]:
        try:
            if os.path.exists(fn):
                os.remove(fn)
        except Exception:
            pass


def append_products_jsonl(products: List[dict]) -> None:
    if not products:
        return
    with open(API_PRODUCTS_CACHE, "a", encoding="utf-8") as f:
        for p in products:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def read_products_jsonl() -> List[dict]:
    if not os.path.exists(API_PRODUCTS_CACHE):
        return []
    out = []
    with open(API_PRODUCTS_CACHE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def fetch_products_resumeable(
    cfg: WCConfig,
    timeout: int,
    include_private: bool,
    log_fn,
    progress_bar,
    status_box,
) -> List[dict]:
    """
    Fetch publish products always.
    If include_private=True, also fetch status=private and merge.
    Resume is implemented by storing last completed page per status.
    """
    sess = wc_session()
    per_page = 100

    prog = read_progress()
    if "statuses" not in prog:
        prog["statuses"] = {}
    statuses = ["publish"] + (["private"] if include_private else [])

    # Load already cached products first
    cached = read_products_jsonl()
    cached_ids = set()
    for p in cached:
        try:
            cached_ids.add(int(p.get("id")))
        except Exception:
            pass

    all_products = cached[:]  # will extend, dedupe later

    # Try to estimate total pages: unknown reliably; we just show progress as "pages fetched".
    total_pages_guess = 1
    fetched_pages = 0

    for st_status in statuses:
        last_done = int(prog["statuses"].get(st_status, 0))
        page = max(1, last_done + 1)

        while True:
            status_box.info(f"Fetching products from API… status={st_status} page={page}")
            log_fn(f"API: fetching products page {page} (status={st_status})…")

            data = wc_get_json(
                sess,
                cfg,
                "/wp-json/wc/v3/products",
                params={"per_page": per_page, "page": page, "status": st_status},
                timeout=timeout,
                retries=6,
                backoff=1.35,
                log_fn=log_fn,
            )

            if not isinstance(data, list) or len(data) == 0:
                # finished this status
                prog["statuses"][st_status] = page - 1
                write_progress(prog)
                break

            # append new products (avoid duplicates)
            new = []
            for p in data:
                try:
                    pid = int(p.get("id"))
                except Exception:
                    continue
                if pid in cached_ids:
                    continue
                cached_ids.add(pid)
                new.append(p)

            if new:
                append_products_jsonl(new)
                all_products.extend(new)

            prog["statuses"][st_status] = page
            write_progress(prog)

            fetched_pages += 1
            total_pages_guess = max(total_pages_guess, fetched_pages + 1)
            progress_bar.progress(min(0.98, fetched_pages / float(total_pages_guess)))

            if len(data) < per_page:
                break
            page += 1

    progress_bar.progress(1.0)
    status_box.success(f"Loaded {len(all_products):,} products (cached + fetched).")

    # Final dedupe (by id)
    seen = set()
    uniq = []
    for p in all_products:
        try:
            pid = int(p.get("id"))
        except Exception:
            continue
        if pid in seen:
            continue
        seen.add(pid)
        uniq.append(p)
    return uniq


# ----------------------------
# CSV parsing
# ----------------------------
def parse_wc_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(file_bytes), dtype=str, keep_default_na=False)
    # Normalize common columns
    # Woo export often has:
    # ID, Type, SKU, Name, Published, Is featured?, Visibility in catalog, Short description, Description,
    # Regular price, Sale price, Categories, Images, Stock, Stock quantity, In stock?, etc.
    col_map = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in col_map:
                return col_map[n]
        return None

    name_c = pick("name")
    sku_c = pick("sku")
    reg_c = pick("regular price", "regular_price")
    sale_c = pick("sale price", "sale_price")
    cats_c = pick("categories", "category")
    img_c = pick("images", "image", "image src", "image_src")
    stock_c = pick("in stock?", "in_stock", "stock status", "stock_status")
    url_c = pick("permalink", "url", "product url", "product_url", "link")
    pub_c = pick("published", "status", "visibility in catalog", "catalog visibility", "visibility")

    out = pd.DataFrame()
    out["id"] = df[pick("id")] if pick("id") else ""
    out["name"] = df[name_c] if name_c else ""
    out["sku"] = df[sku_c] if sku_c else ""
    out["regular_price"] = df[reg_c] if reg_c else ""
    out["sale_price"] = df[sale_c] if sale_c else ""
    out["categories_raw"] = df[cats_c] if cats_c else ""
    out["image"] = df[img_c] if img_c else ""
    out["in_stock"] = df[stock_c] if stock_c else ""
    out["permalink"] = df[url_c] if url_c else ""
    out["status"] = df[pub_c] if pub_c else ""

    # Clean
    for c in out.columns:
        out[c] = out[c].map(safe_text)

    return out


# ----------------------------
# Products normalization
# ----------------------------
def normalize_products_from_api(products: List[dict]) -> pd.DataFrame:
    rows = []
    for p in products:
        pid = p.get("id")
        name = safe_text(p.get("name"))
        sku = safe_text(p.get("sku"))
        status = safe_text(p.get("status"))  # publish/private/draft
        permalink = safe_text(p.get("permalink"))

        # prices (Woo returns as strings)
        reg = safe_text(p.get("regular_price"))
        sale = safe_text(p.get("sale_price"))
        price = safe_text(p.get("price"))

        # stock
        stock_status = safe_text(p.get("stock_status"))  # instock/outofstock/onbackorder
        in_stock = "instock" if stock_status == "instock" else stock_status

        # images
        img_url = ""
        imgs = p.get("images") or []
        if isinstance(imgs, list) and imgs:
            img_url = safe_text(imgs[0].get("src"))

        # categories (ids + names)
        cat_paths = []
        cat_ids = []
        cats = p.get("categories") or []
        if isinstance(cats, list):
            for c in cats:
                try:
                    cid = int(c.get("id"))
                    cat_ids.append(cid)
                except Exception:
                    continue
                nm = safe_text(c.get("name"))
                if nm:
                    cat_paths.append(nm)

        # attributes
        attrs_out = []
        attrs = p.get("attributes") or []
        if isinstance(attrs, list):
            for a in attrs:
                aname = safe_text(a.get("name"))
                opts = a.get("options") or []
                if not aname or not isinstance(opts, list) or not opts:
                    continue
                v = ", ".join([safe_text(x) for x in opts if safe_text(x)])
                if v:
                    attrs_out.append(f"{aname}: {v}")

        # meta / brand / kashrut often in custom attributes or in tags; keep placeholders from attributes for now
        # If you want specific fields, you can map them here.

        rows.append(
            {
                "id": pid,
                "name": name,
                "sku": sku,
                "status": status,
                "permalink": permalink,
                "regular_price": reg,
                "sale_price": sale,
                "price": price,
                "stock_status": stock_status,
                "in_stock": in_stock,
                "image": img_url,
                "cat_ids": cat_ids,
                "cat_names": cat_paths,
                "attributes": "\n".join(attrs_out),
            }
        )
    df = pd.DataFrame(rows)
    # Clean up
    for c in df.columns:
        if c in ("cat_ids", "cat_names"):
            continue
        df[c] = df[c].map(safe_text)
    return df


# ----------------------------
# Image caching / download
# ----------------------------
def img_cache_path(url: str) -> str:
    return os.path.join(IMG_CACHE_DIR, sha1(url) + ".jpg")


def download_image_bytes(url: str, timeout: int = 25, retries: int = 5, backoff: float = 1.25, log_fn=None) -> Optional[bytes]:
    if not url:
        return None

    # local cache
    path = img_cache_path(url)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            pass

    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})

    last = None
    for attempt in range(1, retries + 1):
        try:
            r = s.get(url, timeout=timeout)
            if r.status_code >= 500:
                last = f"{r.status_code}"
                if log_fn:
                    log_fn(f"IMG: {last} for {url} (attempt {attempt}/{retries})")
                time.sleep(backoff * attempt)
                continue
            if r.status_code >= 400:
                return None
            b = r.content
            if not b:
                return None
            # normalize to RGB JPG for PDF stability
            try:
                im = Image.open(BytesIO(b))
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGB")
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=85)
                b2 = buf.getvalue()
            except Exception:
                b2 = b
            try:
                with open(path, "wb") as f:
                    f.write(b2)
            except Exception:
                pass
            return b2
        except Exception as e:
            last = str(e)
            if log_fn:
                log_fn(f"IMG: retry {attempt}/{retries} ({last})")
            time.sleep(backoff * attempt)
    return None


# ----------------------------
# PDF generation (fpdf2)
# ----------------------------
def _ensure_pdf_text(s: str) -> str:
    """
    FPDF core fonts are Latin-1; avoid unicode crash by replacing unsupported chars.
    If you use a Unicode font, you can remove this.
    """
    if not isinstance(s, str):
        s = safe_text(s)
    s = safe_text(s)
    return s.encode("latin-1", "replace").decode("latin-1")


def wrap_lines(pdf: FPDF, text: str, max_w: float, max_lines: int) -> List[str]:
    text = safe_text(text)
    if not text:
        return []
    text = _ensure_pdf_text(text)

    words = re.split(r"(\s+)", text)
    lines = []
    cur = ""
    for w in words:
        test = (cur + w) if cur else w
        if pdf.get_string_width(test) <= max_w:
            cur = test
        else:
            if cur.strip():
                lines.append(cur.strip())
            cur = w.strip()
            if len(lines) >= max_lines:
                break
    if len(lines) < max_lines and cur.strip():
        lines.append(cur.strip())

    # Ellipsis if overflow
    if len(lines) == max_lines:
        # Ensure last line fits with …
        last = lines[-1]
        ell = "…"
        while pdf.get_string_width(last + ell) > max_w and len(last) > 0:
            last = last[:-1]
        lines[-1] = (last.strip() + ell) if last.strip() else ell
    return lines


class CatalogPDF(FPDF):
    def __init__(self, title: str, brand_site: str, disclaimer: str, orientation: str, fmt: str):
        super().__init__(orientation=orientation, unit="mm", format=fmt)
        self.title_text = _ensure_pdf_text(title)
        self.brand_site = _ensure_pdf_text(brand_site)
        self.disclaimer = _ensure_pdf_text(disclaimer)
        self.set_auto_page_break(auto=False)
        self.set_creator("B-Kosher Catalog Builder")

    def header(self):
        margin = 10
        y = 8
        self.set_xy(margin, y)

        # logo (if exists)
        logo_path = None
        for fn in LOGO_FILENAME_CANDIDATES:
            if os.path.exists(fn):
                logo_path = fn
                break
        if logo_path and logo_path.lower().endswith(".png"):
            try:
                self.image(logo_path, x=margin, y=6, w=22)
            except Exception:
                pass

        # Title
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*BRAND_BLUE)
        self.set_xy(margin + 26, 9)
        self.cell(0, 6, self.title_text, ln=0)

        # Page number
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)
        self.set_xy(self.w - margin - 25, 9)
        self.cell(25, 6, f"Page {self.page_no()}", ln=0, align="R")

        # Separator line
        self.set_draw_color(*BRAND_BLUE)
        self.set_line_width(0.6)
        self.line(margin, 18, self.w - margin, 18)

    def footer(self):
        margin = 10
        self.set_y(self.h - 14)

        # Red line
        self.set_draw_color(*BRAND_RED)
        self.set_line_width(0.4)
        self.line(margin, self.h - 16, self.w - margin, self.h - 16)

        # Footer text
        self.set_text_color(*BRAND_GREY)
        self.set_font("Helvetica", "", 8)
        left = f"{self.brand_site} | {self.disclaimer}"
        self.set_x(margin)
        self.cell(0, 6, left, ln=0, align="L")


def make_catalog_pdf_bytes(
    df: pd.DataFrame,
    title: str,
    orientation: str,       # "P" or "L"
    page_format: str,       # "A4"
    grid: str,              # "Standard" or "Compact"
    currency_symbol: str,
    show_price: bool,
    show_sku: bool,
    show_desc: bool,
    show_attributes: bool,
    exclude_oos: bool,
    only_sale: bool,
    include_private_in_pdf: bool,
    category_paths: Dict[int, str],
    group_by_hierarchy: bool,
    log_fn=None,
) -> bytes:
    # Filter
    d = df.copy()

    if exclude_oos:
        # normalize
        d = d[~d["stock_status"].isin(["outofstock"])]

    if only_sale:
        d = d[(d["sale_price"].astype(str).str.strip() != "") & (d["sale_price"].astype(str).str.strip() != "0")]

    if not include_private_in_pdf:
        d = d[d["status"].isin(["publish", "published", "public", ""])]

    # Sort by hierarchy paths
    def best_path_for_row(row) -> str:
        cids = row.get("cat_ids", [])
        if not isinstance(cids, list):
            return ""
        paths = []
        for cid in cids:
            if cid in category_paths:
                paths.append(category_paths[cid])
        paths = [p for p in paths if p]
        if not paths:
            return ""
        return sorted(paths)[0]

    if group_by_hierarchy:
        d["_group_path"] = d.apply(best_path_for_row, axis=1)
    else:
        d["_group_path"] = ""

    d["_name_sort"] = d["name"].map(lambda x: safe_text(x).lower())
    d = d.sort_values(by=["_group_path", "_name_sort"], ascending=True)

    # Grid settings
    # We intentionally use Compact = 6x5 (NOT 6x6) to prevent text overflow.
    if orientation == "P":
        if grid == "Standard":
            cols, rows = 3, 3
            box_h = 60
        else:  # Compact
            cols, rows = 6, 5
            box_h = 32
    else:  # Landscape
        if grid == "Standard":
            cols, rows = 4, 3
            box_h = 55
        else:
            cols, rows = 7, 4
            box_h = 34

    pdf = CatalogPDF(
        title=title,
        brand_site=BRAND_SITE,
        disclaimer=f"Prices correct as of {now_str()}",
        orientation=orientation,
        fmt=page_format,
    )
    pdf.set_margins(10, 10, 10)
    pdf.add_page()

    margin = 10
    top_y = 22
    gap_x = 4
    gap_y = 6
    usable_w = pdf.w - 2 * margin
    box_w = (usable_w - gap_x * (cols - 1)) / cols

    # Category divider
    cat_bar_h = 10

    def draw_category_bar(text: str):
        pdf.set_fill_color(*BRAND_BLUE)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 12)
        pdf.rounded_rect(margin, pdf.get_y(), pdf.w - 2 * margin, cat_bar_h, 2.5, style="F")
        pdf.set_xy(margin + 4, pdf.get_y() + 2.5)
        pdf.cell(0, 5, _ensure_pdf_text(text), ln=1)
        pdf.ln(4)
        pdf.set_text_color(0, 0, 0)

    # Start at top content area
    pdf.set_xy(margin, top_y)

    current_group = None
    i_on_page = 0
    n_per_page = cols * rows

    for _, row in d.iterrows():
        group = safe_text(row.get("_group_path", ""))

        if group_by_hierarchy:
            if group != current_group:
                # new group -> new page if not enough space for bar + at least 1 row
                if i_on_page != 0:
                    # move to next row start
                    pass
                # If we're mid page and there's no room, add new page
                if pdf.get_y() + cat_bar_h + box_h > pdf.h - 18:
                    pdf.add_page()
                    pdf.set_xy(margin, top_y)
                    i_on_page = 0
                draw_category_bar(group if group else "Uncategorised")
                current_group = group
                i_on_page = 0  # restart grid after each bar
                # reset grid origin after bar
                grid_origin_y = pdf.get_y()
            else:
                grid_origin_y = pdf.get_y() if i_on_page == 0 else grid_origin_y
        else:
            grid_origin_y = top_y

        # If page is full -> add page
        if i_on_page >= n_per_page:
            pdf.add_page()
            pdf.set_xy(margin, top_y)
            i_on_page = 0
            grid_origin_y = pdf.get_y()

        # compute cell position within current page grid
        r = i_on_page // cols
        c = i_on_page % cols
        x = margin + c * (box_w + gap_x)
        y = grid_origin_y + r * (box_h + gap_y)

        # If y would exceed page, new page
        if y + box_h > pdf.h - 18:
            pdf.add_page()
            pdf.set_xy(margin, top_y)
            i_on_page = 0
            r, c = 0, 0
            x = margin
            y = top_y
            grid_origin_y = y

        # Box border
        pdf.set_draw_color(*BRAND_BLUE)
        pdf.set_line_width(0.35)
        pdf.rect(x, y, box_w, box_h)

        pad = 2.0
        img_h = box_h * 0.55
        text_y = y + pad + img_h + 1

        # Image
        img_url = safe_text(row.get("image"))
        if img_url:
            b = None
            try:
                b = download_image_bytes(img_url, timeout=25, retries=4, backoff=1.1, log_fn=log_fn)
            except Exception:
                b = None

            if b:
                # Save to temp for fpdf
                tmp_path = os.path.join(CACHE_DIR, f"tmp_{sha1(img_url)}.jpg")
                try:
                    with open(tmp_path, "wb") as f:
                        f.write(b)
                    # Fit image inside area without grey background
                    pdf.image(tmp_path, x=x + pad, y=y + pad, w=box_w - 2 * pad, h=img_h)
                except Exception:
                    pdf.set_font("Helvetica", "I", 7)
                    pdf.set_text_color(*BRAND_GREY)
                    pdf.set_xy(x + pad, y + pad + img_h / 2)
                    pdf.cell(box_w - 2 * pad, 5, "No image", align="C")
                finally:
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
            else:
                pdf.set_font("Helvetica", "I", 7)
                pdf.set_text_color(*BRAND_GREY)
                pdf.set_xy(x + pad, y + pad + img_h / 2)
                pdf.cell(box_w - 2 * pad, 5, "No image", align="C")
        else:
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_text_color(*BRAND_GREY)
            pdf.set_xy(x + pad, y + pad + img_h / 2)
            pdf.cell(box_w - 2 * pad, 5, "No image", align="C")

        # Sale badge
        sale_price = safe_text(row.get("sale_price"))
        is_sale = bool(sale_price) and sale_price not in ("0", "0.0", "0.00")
        if is_sale:
            badge_w, badge_h = 12, 6
            pdf.set_fill_color(*BRAND_RED)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 8)
            pdf.rect(x + box_w - badge_w - 1.5, y + 1.5, badge_w, badge_h, style="F")
            pdf.set_xy(x + box_w - badge_w - 1.5, y + 2.5)
            pdf.cell(badge_w, 4, "SALE", align="C")
            pdf.set_text_color(0, 0, 0)

        # Text area
        pdf.set_text_color(0, 0, 0)

        # Choose smaller font for compact layouts
        if grid == "Compact":
            name_font = 7
            meta_font = 6
            max_name_lines = 2
            max_meta_lines = 2
        else:
            name_font = 9
            meta_font = 7
            max_name_lines = 2
            max_meta_lines = 3

        # Name (wrapped)
        pdf.set_font("Helvetica", "B", name_font)
        name = safe_text(row.get("name"))
        max_w = box_w - 2 * pad
        name_lines = wrap_lines(pdf, name, max_w, max_name_lines)

        # Link entire box to product page if exists
        link = safe_text(row.get("permalink"))
        if link:
            # Transparent link rectangle
            try:
                pdf.link(x, y, box_w, box_h, link)
            except Exception:
                pass

        yy = text_y
        for line in name_lines:
            pdf.set_xy(x + pad, yy)
            pdf.cell(max_w, 3.7 if grid == "Compact" else 4.2, line)
            yy += 3.8 if grid == "Compact" else 4.3

        # Price lines
        if show_price:
            pdf.set_font("Helvetica", "B", meta_font + 1)
            pdf.set_text_color(*BRAND_RED)

            reg_price = safe_text(row.get("regular_price"))
            # if API provides "price" only
            if not reg_price:
                reg_price = safe_text(row.get("price"))

            if is_sale:
                sp = money_fmt(sale_price, currency_symbol)
                rp = money_fmt(reg_price, currency_symbol)
                pdf.set_xy(x + pad, yy)
                pdf.cell(max_w, 3.6, _ensure_pdf_text(sp))
                # strike-through regular price
                if rp:
                    pdf.set_text_color(*BRAND_GREY)
                    pdf.set_font("Helvetica", "", meta_font)
                    pdf.set_xy(x + pad + 10, yy)
                    pdf.cell(max_w, 3.6, _ensure_pdf_text(rp))
                yy += 4.2
            else:
                rp = money_fmt(reg_price, currency_symbol)
                pdf.set_xy(x + pad, yy)
                pdf.cell(max_w, 3.6, _ensure_pdf_text(rp))
                yy += 4.2

            pdf.set_text_color(0, 0, 0)

        # SKU
        if show_sku:
            pdf.set_font("Helvetica", "", meta_font)
            sku = safe_text(row.get("sku"))
            if sku:
                pdf.set_xy(x + pad, yy)
                pdf.cell(max_w, 3.4, _ensure_pdf_text(f"SKU: {sku}"))
                yy += 3.8

        # Description (not required, and must never overflow)
        if show_desc:
            pdf.set_font("Helvetica", "", meta_font)
            desc = safe_text(row.get("description", "")) or ""
            if desc:
                lines = wrap_lines(pdf, desc, max_w, max_meta_lines)
                for line in lines:
                    pdf.set_xy(x + pad, yy)
                    pdf.cell(max_w, 3.2, line)
                    yy += 3.5

        # Attributes
        if show_attributes:
            pdf.set_font("Helvetica", "", meta_font)
            attrs = safe_text(row.get("attributes", ""))
            if attrs:
                # Attributes already newline-joined
                attr_lines = []
                for chunk in attrs.split("\n"):
                    chunk = safe_text(chunk)
                    if not chunk:
                        continue
                    attr_lines.extend(wrap_lines(pdf, chunk, max_w, 1))
                # Fit whatever remaining vertical space allows
                # (we still cap to max_meta_lines)
                attr_lines = attr_lines[:max_meta_lines]
                for line in attr_lines:
                    pdf.set_xy(x + pad, yy)
                    pdf.cell(max_w, 3.2, _ensure_pdf_text(line))
                    yy += 3.5

        i_on_page += 1

    # Output as real bytes (not bytearray)
    out = pdf.output()
    if isinstance(out, bytearray):
        out = bytes(out)
    elif isinstance(out, str):
        out = out.encode("latin-1", "ignore")
    return out


# ----------------------------
# UI / app
# ----------------------------
def main():
    st.set_page_config(page_title="B-Kosher Catalog Builder", page_icon="🧾", layout="wide")
    inject_css()
    require_login()
    render_header()

    # Logger
    log_lines = st.session_state.get("log_lines", [])
    def log(msg: str):
        msg = safe_text(msg)
        log_lines.append(f"{datetime.now().strftime('%H:%M:%S')}  {msg}")
        st.session_state["log_lines"] = log_lines[-350:]

    # Sidebar / steps
    st.markdown("---")
    st.markdown("### Step 1 — Choose data source")
    source = st.radio("Source", ["WooCommerce API", "CSV Upload"], horizontal=True, index=0)

    cfg = build_wc_config_from_secrets()

    st.markdown("### Step 2 — Load products")
    api_timeout = st.slider("API timeout (seconds)", 10, 60, int(st.session_state.get("api_timeout", 30)))
    st.session_state["api_timeout"] = api_timeout

    include_private_fetch = st.checkbox(
        "Include private/unpublished products (requires API user permission)",
        value=bool(st.session_state.get("include_private_fetch", False)),
    )
    st.session_state["include_private_fetch"] = include_private_fetch

    status_box = st.empty()
    progress_bar = st.progress(0.0)

    colA, colB = st.columns([0.5, 0.5])
    with colA:
        if st.button("Load (use cache if available)"):
            st.session_state["loaded_df"] = None

            if source == "WooCommerce API":
                if not cfg.consumer_key or not cfg.consumer_secret:
                    st.error("WooCommerce API secrets missing. Add WC_CONSUMER_KEY and WC_CONSUMER_SECRET to secrets.")
                else:
                    try:
                        # categories (cached)
                        cats_cached = load_categories_cached()
                        if not cats_cached:
                            log("Loading category hierarchy…")
                            cats_cached = load_categories_from_api(cfg, api_timeout, log)
                        st.session_state["categories"] = cats_cached

                        log("Starting resume-able product import…")
                        prods = fetch_products_resumeable(
                            cfg=cfg,
                            timeout=api_timeout,
                            include_private=include_private_fetch,
                            log_fn=log,
                            progress_bar=progress_bar,
                            status_box=status_box,
                        )
                        df = normalize_products_from_api(prods)
                        st.session_state["loaded_df"] = df
                    except Exception as e:
                        st.error(safe_text(e))
            else:
                st.info("Upload a CSV below, then click Load again.")

    with colB:
        if st.button("Refresh cache (fetch again)"):
            clear_api_cache()
            st.session_state["loaded_df"] = None
            progress_bar.progress(0.0)
            status_box.info("Cache cleared. Click Load to fetch again.")
            log("Cache cleared.")

    if source == "CSV Upload":
        up = st.file_uploader("Upload WooCommerce product export CSV", type=["csv"])
        if up is not None:
            try:
                df_csv = parse_wc_csv(up.getvalue())
                # For CSV mode, keep minimal mapping into the same columns used by PDF code
                df = pd.DataFrame()
                df["id"] = df_csv.get("id", "")
                df["name"] = df_csv.get("name", "")
                df["sku"] = df_csv.get("sku", "")
                df["regular_price"] = df_csv.get("regular_price", "")
                df["sale_price"] = df_csv.get("sale_price", "")
                df["price"] = df_csv.get("regular_price", "")
                df["status"] = df_csv.get("status", "")
                df["stock_status"] = df_csv.get("in_stock", "").map(lambda x: "instock" if "yes" in safe_text(x).lower() else safe_text(x).lower())
                df["permalink"] = df_csv.get("permalink", "")
                df["image"] = df_csv.get("image", "")
                df["attributes"] = ""
                df["cat_ids"] = [[] for _ in range(len(df))]
                df["cat_names"] = df_csv.get("categories_raw", "").map(lambda s: [x.strip() for x in safe_text(s).split(",") if x.strip()])
                st.session_state["loaded_df"] = df
                status_box.success(f"Loaded {len(df):,} products from CSV.")
            except Exception as e:
                st.error(f"CSV error: {safe_text(e)}")

    st.markdown("### Live logs")
    st.markdown(f"<div class='bk-log'>{'<br/>'.join(log_lines[-120:])}</div>", unsafe_allow_html=True)

    df_loaded = st.session_state.get("loaded_df")
    if df_loaded is None or not isinstance(df_loaded, pd.DataFrame) or df_loaded.empty:
        st.info("Load products to continue.")
        return

    st.markdown("---")
    st.markdown("### Step 3 — Configure & filter")

    # Category path maps (API mode)
    cats = st.session_state.get("categories", [])
    cat_by_id, cat_children, cat_path_map = build_cat_maps(cats) if cats else ({}, {}, {})

    # Defaults requested:
    show_sku = st.checkbox("Show SKU", value=False)
    show_desc = st.checkbox("Show description", value=False)
    show_attributes = st.checkbox("Show attributes", value=True)
    show_price = st.checkbox("Show price", value=True)

    exclude_oos = st.checkbox("Exclude out-of-stock", value=bool(st.session_state.get("exclude_oos", False)))
    st.session_state["exclude_oos"] = exclude_oos

    only_sale = st.checkbox("Only sale items", value=False)

    # IMPORTANT: Toggle must exist here (PDF stage), not only in fetch stage
    include_private_in_pdf = st.checkbox(
        "Include private/unpublished products in PDF",
        value=False,
        help="If unchecked, PDF includes only published products (recommended for customers).",
    )

    currency_symbol = st.text_input("Currency symbol", value="£")

    orientation_label = st.selectbox("Page orientation", ["Portrait", "Landscape"], index=0)
    orientation = "P" if orientation_label == "Portrait" else "L"

    grid = st.selectbox("Grid density", ["Standard", "Compact"], index=0, help="Standard=3×3 portrait. Compact uses a denser grid (safe for text).")

    # Category selection (tree paths, parents included)
    st.markdown("#### Categories (tree)")
    if cat_path_map:
        # show parent categories too: selecting parent should include all descendants
        # We'll let user select any node by path.
        all_paths = sorted([p for p in cat_path_map.values() if p])
        selected_paths = st.multiselect("Choose categories (parent/child/grandchild)", all_paths)
    else:
        selected_paths = st.multiselect("Choose categories (from data)", sorted(set(sum(df_loaded["cat_names"].tolist(), []))) if "cat_names" in df_loaded else [])

    search = st.text_input("Search (name or SKU)")

    # Apply filters
    filtered = df_loaded.copy()

    # Category filter
    if selected_paths:
        if cat_path_map:
            sel = set(selected_paths)

            def row_matches_paths(row) -> bool:
                cids = row.get("cat_ids", [])
                if not isinstance(cids, list):
                    return False
                # match if any category path equals or is under selected parent path
                paths = []
                for cid in cids:
                    p = cat_path_map.get(cid, "")
                    if p:
                        paths.append(p)
                for p in paths:
                    for s in sel:
                        if p == s or p.startswith(s + " > "):
                            return True
                return False

            filtered = filtered[filtered.apply(row_matches_paths, axis=1)]
        else:
            # CSV mode using names
            sel = set([safe_text(x) for x in selected_paths])

            def row_has_name(row) -> bool:
                names = row.get("cat_names", [])
                if not isinstance(names, list):
                    return False
                for n in names:
                    if safe_text(n) in sel:
                        return True
                return False

            filtered = filtered[filtered.apply(row_has_name, axis=1)]

    if search.strip():
        q = search.strip().lower()
        filtered = filtered[
            filtered["name"].map(lambda x: q in safe_text(x).lower())
            | filtered["sku"].map(lambda x: q in safe_text(x).lower())
        ]

    # Exclude private here for preview selection count too
    if not include_private_in_pdf:
        filtered = filtered[filtered["status"].isin(["publish", "published", "public", ""])]

    if exclude_oos:
        filtered = filtered[~filtered["stock_status"].isin(["outofstock"])]

    if only_sale:
        filtered = filtered[(filtered["sale_price"].astype(str).str.strip() != "") & (filtered["sale_price"].astype(str).str.strip() != "0")]

    st.info(f"Selected products: {len(filtered):,}")

    with st.expander("Preview (first 9 products)"):
        prev = filtered.head(9)
        for _, r in prev.iterrows():
            st.write(f"**{safe_text(r.get('name'))}** — {money_fmt(r.get('sale_price') or r.get('regular_price') or r.get('price'), currency_symbol)}")

    st.markdown("---")
    st.markdown("### Step 4 — Generate PDF")

    title = st.text_input("Catalog title", value=DEFAULT_TITLE)

    gen = st.button("Generate PDF")
    if gen:
        # progress UI during image warming is optional; PDF does on-demand per item
        progress = st.progress(0.0)
        status = st.empty()
        status.info("Generating PDF…")

        try:
            pdf_bytes = make_catalog_pdf_bytes(
                df=filtered,
                title=title,
                orientation=orientation,
                page_format="A4",
                grid=grid,
                currency_symbol=currency_symbol,
                show_price=show_price,
                show_sku=show_sku,
                show_desc=show_desc,
                show_attributes=show_attributes,
                exclude_oos=exclude_oos,
                only_sale=only_sale,
                include_private_in_pdf=include_private_in_pdf,
                category_paths=cat_path_map,
                group_by_hierarchy=True,
                log_fn=log,
            )
            progress.progress(1.0)
            status.success("PDF ready.")

            # Streamlit requires bytes, not bytearray
            if isinstance(pdf_bytes, bytearray):
                pdf_bytes = bytes(pdf_bytes)
            elif isinstance(pdf_bytes, str):
                pdf_bytes = pdf_bytes.encode("latin-1", "ignore")

            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"bkosher_catalog_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )
            st.caption("Tip: If clickable links don’t work in your PDF viewer, test in Chrome/Edge.")
        except Exception as e:
            status.error(safe_text(e))


if __name__ == "__main__":
    main()