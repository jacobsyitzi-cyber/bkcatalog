# app.py — B-Kosher Catalog Builder (Streamlit)
# - Default source: WooCommerce API (secrets)
# - Backup source: CSV upload
# - Customer-facing login (password stored in Streamlit secrets)
# - Builds printable PDF catalog (FPDF2) with branding + clickable product links
# - Robust API fetch with resume checkpoints + disk cache (prevents restarts on iPhone/background)
# - Grid presets: Standard (3×3) + Compact (6×5) + others
# - Parent category selection supported (select Alcohol and it includes Alcohol > Beer, etc.)
# - Optional: include/exclude private products in the PDF (and in API fetch if API user permits)
#
# REQUIRED SECRETS (Streamlit Cloud -> App -> Settings -> Secrets):
# APP_PASSWORD = "...."
# WC_STORE_URL = "https://www.b-kosher.co.uk"
# WC_CONSUMER_KEY = "ck_...."
# WC_CONSUMER_SECRET = "cs_...."
#
# requirements.txt (recommended):
# streamlit==1.31.1
# requests==2.31.0
# pandas==2.2.2
# pillow==10.4.0
# fpdf2==2.7.9

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from PIL import Image
from requests.auth import HTTPBasicAuth

from fpdf import FPDF  # fpdf2


# ----------------------------
# Branding (B-Kosher)
# ----------------------------
BRAND_BLUE = (0, 76, 151)     # #004C97
BRAND_RED = (200, 16, 46)     # #C8102E
BRAND_SITE = "www.b-kosher.co.uk"
DEFAULT_TITLE = "B-Kosher Product Catalog"

# File in repo (you said your repo has Bkosher.png)
LOGO_FILENAME_CANDIDATES = [
    "Bkosher.png",
    "B-kosher logo high q.png",
    "bkosher.svg",  # if you still have it
]

# Cache locations (Streamlit Cloud allows /tmp)
APP_CACHE_DIR = os.path.join("/tmp", "bkcatalog_cache")
IMG_CACHE_DIR = os.path.join(APP_CACHE_DIR, "images")
CHKPT_DIR = os.path.join(APP_CACHE_DIR, "checkpoints")
os.makedirs(IMG_CACHE_DIR, exist_ok=True)
os.makedirs(CHKPT_DIR, exist_ok=True)


# ----------------------------
# Utility helpers
# ----------------------------
def now_str() -> str:
    return datetime.now().strftime("%d %b %Y")


def safe_unescape(s: Any) -> str:
    """Turn '&amp;' into '&' and drop NaN/None to ''."""
    if s is None:
        return ""
    if isinstance(s, float) and pd.isna(s):
        return ""
    s = str(s)
    if s.strip().lower() == "nan":
        return ""
    return html.unescape(s)


def money_fmt(value: Any, currency_symbol: str = "£") -> str:
    try:
        if value is None:
            return ""
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return ""
        v = float(value)
        return f"{currency_symbol}{v:.2f}"
    except Exception:
        return ""


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def redact_url(url: str) -> str:
    """Remove query params so keys never appear in errors/logs."""
    return url.split("?")[0]


def read_logo_bytes() -> Optional[bytes]:
    for fn in LOGO_FILENAME_CANDIDATES:
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                return f.read()
    return None


def logo_as_data_uri() -> Optional[str]:
    b = read_logo_bytes()
    if not b:
        return None
    # assume png/jpg
    mime = "image/png" if b[:8].startswith(b"\x89PNG") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(b).decode('ascii')}"


# ----------------------------
# Streamlit UI theme CSS (branding + mobile-friendly)
# ----------------------------
def inject_css() -> None:
    logo_uri = logo_as_data_uri()
    brand_blue = f"rgb({BRAND_BLUE[0]},{BRAND_BLUE[1]},{BRAND_BLUE[2]})"
    brand_red = f"rgb({BRAND_RED[0]},{BRAND_RED[1]},{BRAND_RED[2]})"
    header_logo_html = ""
    if logo_uri:
        header_logo_html = f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
          <img src="{logo_uri}" style="height:46px;object-fit:contain;" />
          <div>
            <div style="font-weight:800;font-size:22px;color:{brand_blue};line-height:1.1;">
              B-Kosher Catalog Builder
            </div>
            <div style="font-size:12px;color:#6b7280;">Customer-facing PDF catalog generator</div>
          </div>
        </div>
        """

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

/* Section headers */
h2, h3 {{ color: {brand_blue}; }}

/* Cards */
.bk-card {{
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 14px;
  background: white;
}}
.bk-muted {{ color:#6b7280; font-size: 0.9rem; }}

/* Log box */
.bk-log {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 12px;
  background: #0b1220;
  color: #d1d5db;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.08);
  max-height: 260px;
  overflow: auto;
}}
</style>
{header_logo_html}
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# Login gate
# ----------------------------
def require_login() -> None:
    st.session_state.setdefault("logged_in", False)

    def get_secret(name: str) -> Optional[str]:
        try:
            v = st.secrets.get(name)
            return str(v) if v is not None else None
        except Exception:
            return None

    if st.session_state["logged_in"]:
        return

    st.markdown("## Login")
    st.markdown('<div class="bk-card">', unsafe_allow_html=True)
    st.markdown(
        "Enter the password to access the catalog builder. "
        "The password is stored in Streamlit **secrets** (not visible in code)."
    )
    pw = st.text_input("Password", type="password")
    st.markdown("</div>", unsafe_allow_html=True)

    expected = get_secret("APP_PASSWORD")
    if not expected:
        st.error(
            "APP_PASSWORD is not set in Streamlit secrets.\n\n"
            "On Streamlit Cloud: App → Settings → Secrets\n"
            "Locally: create `.streamlit/secrets.toml` with:\n"
            'APP_PASSWORD="your_password"'
        )
        st.stop()

    if st.button("Login"):
        if pw == expected:
            st.session_state["logged_in"] = True
            st.success("Logged in.")
            st.rerun()
        else:
            st.error("Incorrect password.")
            st.stop()


# ----------------------------
# WooCommerce API
# ----------------------------
@dataclass
class WCConfig:
    store_url: str
    ck: str
    cs: str
    timeout: int = 30

    @property
    def api_base(self) -> str:
        return self.store_url.rstrip("/") + "/wp-json/wc/v3"


def get_wc_config_from_secrets(timeout: int) -> Optional[WCConfig]:
    try:
        store = st.secrets.get("WC_STORE_URL")
        ck = st.secrets.get("WC_CONSUMER_KEY")
        cs = st.secrets.get("WC_CONSUMER_SECRET")
        if not store or not ck or not cs:
            return None
        return WCConfig(str(store), str(ck), str(cs), timeout=timeout)
    except Exception:
        return None


def wc_get_json(cfg: WCConfig, endpoint: str, params: Dict[str, Any], log_fn) -> Tuple[int, Any, Dict[str, str]]:
    """GET JSON with safe retries and WITHOUT putting keys in URL."""
    url = cfg.api_base + endpoint
    auth = HTTPBasicAuth(cfg.ck, cfg.cs)

    last_exc = None
    for attempt in range(1, 6):
        try:
            r = requests.get(url, params=params, auth=auth, timeout=cfg.timeout)
            headers = dict(r.headers)
            if r.status_code >= 400:
                return r.status_code, None, headers
            return r.status_code, r.json(), headers
        except Exception as e:
            last_exc = e
            log_fn(f"API network error (attempt {attempt}/5): {type(e).__name__}")
            time.sleep(min(2 ** attempt, 12))
    raise last_exc  # bubble up after retries


def checkpoint_path(key: str) -> str:
    return os.path.join(CHKPT_DIR, f"{key}.json")


def load_checkpoint(key: str) -> Dict[str, Any]:
    p = checkpoint_path(key)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_checkpoint(key: str, data: Dict[str, Any]) -> None:
    p = checkpoint_path(key)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, p)


def fetch_categories(cfg: WCConfig, log_fn) -> List[Dict[str, Any]]:
    cats: List[Dict[str, Any]] = []
    page = 1
    per_page = 100

    while True:
        log_fn(f"API: fetching categories page {page}…")
        status, data, headers = wc_get_json(cfg, "/products/categories", {"per_page": per_page, "page": page}, log_fn)
        if status >= 400 or not isinstance(data, list):
            log_fn(f"API categories failed ({status}) at {redact_url(cfg.api_base + '/products/categories')}")
            break
        if not data:
            break
        cats.extend(data)
        if len(data) < per_page:
            break
        page += 1
    return cats


def build_category_maps(categories: List[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[int]]]:
    by_id: Dict[int, Dict[str, Any]] = {}
    children: Dict[int, List[int]] = {}
    for c in categories:
        cid = int(c.get("id"))
        by_id[cid] = c
        parent = int(c.get("parent") or 0)
        children.setdefault(parent, []).append(cid)
    return by_id, children


def category_full_path(cat_id: int, by_id: Dict[int, Dict[str, Any]]) -> str:
    parts: List[str] = []
    cur = cat_id
    guard = 0
    while cur and cur in by_id and guard < 50:
        parts.append(safe_unescape(by_id[cur].get("name", "")))
        cur = int(by_id[cur].get("parent") or 0)
        guard += 1
    return " > ".join(reversed([p for p in parts if p]))


def descendants(root_id: int, children: Dict[int, List[int]]) -> List[int]:
    out: List[int] = []
    stack = [root_id]
    seen = set()
    while stack:
        cid = stack.pop()
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
        for ch in children.get(cid, []):
            stack.append(ch)
    return out


def normalize_wc_product(p: Dict[str, Any], store_url: str) -> Dict[str, Any]:
    name = safe_unescape(p.get("name", ""))
    sku = safe_unescape(p.get("sku", ""))
    status = safe_unescape(p.get("status", ""))
    stock_status = safe_unescape(p.get("stock_status", ""))
    permalink = p.get("permalink") or ""
    if not permalink:
        pid = p.get("id")
        if pid:
            permalink = store_url.rstrip("/") + f"/?p={pid}"
    price = p.get("price")
    reg = p.get("regular_price")
    sale = p.get("sale_price")
    on_sale = bool(p.get("on_sale"))

    # images
    img_url = ""
    images = p.get("images") or []
    if images and isinstance(images, list):
        img_url = images[0].get("src") or ""

    # categories list
    cats = p.get("categories") or []
    cat_ids: List[int] = []
    cat_names: List[str] = []
    if isinstance(cats, list):
        for c in cats:
            try:
                cat_ids.append(int(c.get("id")))
            except Exception:
                pass
            nm = safe_unescape(c.get("name", ""))
            if nm:
                cat_names.append(nm)

    # attributes
    attrs: Dict[str, str] = {}
    for a in (p.get("attributes") or []):
        n = safe_unescape(a.get("name", ""))
        opts = a.get("options") or []
        if n and isinstance(opts, list) and opts:
            attrs[n] = ", ".join([safe_unescape(x) for x in opts if safe_unescape(x)])

    # Common “brand/kashrut” keys if present as attributes
    brand = attrs.get("Brand") or attrs.get("brand") or safe_unescape(p.get("brands", ""))  # some plugins
    kashrut = attrs.get("Kashrus") or attrs.get("Kashrut") or attrs.get("kashrut") or ""

    short_desc = safe_unescape(p.get("short_description", ""))
    # strip HTML tags quickly for safety
    short_desc = re.sub(r"<[^>]+>", "", short_desc).strip()

    return {
        "id": p.get("id"),
        "name": name,
        "sku": sku,
        "status": status,  # publish/private/draft
        "stock_status": stock_status,  # instock/outofstock/onbackorder
        "price": price,
        "regular_price": reg,
        "sale_price": sale,
        "on_sale": on_sale,
        "image_url": img_url,
        "permalink": permalink,
        "category_ids": cat_ids,
        "category_names": cat_names,
        "attributes": attrs,
        "brand": brand,
        "kashrut": kashrut,
        "description": short_desc,
    }


def api_fetch_all_products(
    cfg: WCConfig,
    include_private_fetch: bool,
    log_fn,
    progress_fn,
    resume_key: str,
) -> List[Dict[str, Any]]:
    """
    Fetch products with disk checkpoint:
    - saves page number + collected count as it goes
    - resumes after app restart / iPhone background kill
    """
    key = sha1(f"{cfg.store_url}|{cfg.ck[-6:]}|{include_private_fetch}|{resume_key}")
    chk = load_checkpoint(key)

    # We avoid status=any if the server errors. We'll fallback automatically.
    wanted_status = "any" if include_private_fetch else "publish"
    effective_status = chk.get("effective_status") or wanted_status
    page = int(chk.get("page") or 1)

    all_products: List[Dict[str, Any]] = chk.get("products") or []
    if not isinstance(all_products, list):
        all_products = []

    per_page = 25  # smaller pages are more reliable on some hosting

    # Try to detect total pages from headers once
    total_pages: Optional[int] = chk.get("total_pages")

    log_fn(f"Resume: status={effective_status} page={page} already={len(all_products)}")

    while True:
        progress_fn(0.0, f"API: fetching products page {page} (status={effective_status})…")
        params = {"per_page": per_page, "page": page, "status": effective_status}

        status, data, headers = wc_get_json(cfg, "/products", params, log_fn)
        if status == 500 and effective_status == "any":
            # fallback: your server (or a plugin) sometimes 500s on status=any
            log_fn("API returned 500 for status=any. Falling back to status=publish.")
            effective_status = "publish"
            page = 1
            all_products = []
            save_checkpoint(key, {"effective_status": effective_status, "page": page, "products": all_products})
            continue

        if status == 401:
            log_fn("API unauthorized (401). Check the API key permissions.")
            break

        if status >= 400:
            log_fn(f"API request failed ({status}) at {redact_url(cfg.api_base + '/products')}")
            break

        if not isinstance(data, list):
            log_fn("API returned unexpected response type.")
            break

        # Total pages
        try:
            if total_pages is None and headers.get("X-WP-TotalPages"):
                total_pages = int(headers["X-WP-TotalPages"])
        except Exception:
            total_pages = None

        if not data:
            break

        for p in data:
            all_products.append(normalize_wc_product(p, cfg.store_url))

        # Save checkpoint after each page (very important for iPhone/background)
        save_checkpoint(
            key,
            {
                "effective_status": effective_status,
                "page": page + 1,
                "products": all_products,
                "total_pages": total_pages,
            },
        )

        # progress hint
        if total_pages:
            frac = min(page / max(total_pages, 1), 1.0)
            progress_fn(frac, f"Fetched {len(all_products):,} products… ({page}/{total_pages})")
        else:
            progress_fn(0.0, f"Fetched {len(all_products):,} products… (page {page})")

        page += 1

        # Streamlit health-check friendliness: yield time
        time.sleep(0.05)

    # final
    save_checkpoint(
        key,
        {"effective_status": effective_status, "page": page, "products": all_products, "total_pages": total_pages},
    )
    return all_products


# ----------------------------
# CSV parsing (Woo export)
# ----------------------------
def parse_csv_products(csv_bytes: bytes, store_url: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, keep_default_na=False)
    cols = {c.lower(): c for c in df.columns}

    def col(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_name = col("name")
    c_sku = col("sku")
    c_status = col("status")
    c_stock = col("stock status", "stock_status")
    c_price = col("price")
    c_regular = col("regular price", "regular_price")
    c_sale = col("sale price", "sale_price")
    c_images = col("images")
    c_permalink = col("permalink", "product url", "product_url", "url")
    c_categories = col("categories", "category")
    c_short = col("short description", "short_description", "description")

    # Attribute columns in Woo export often like: "Attribute 1 name", "Attribute 1 value(s)"
    attr_name_cols = [c for c in df.columns if re.match(r"attribute\s+\d+\s+name", c, re.I)]
    attr_val_cols = [c for c in df.columns if re.match(r"attribute\s+\d+\s+value", c, re.I)]

    products: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        name = safe_unescape(row.get(c_name, "")) if c_name else ""
        sku = safe_unescape(row.get(c_sku, "")) if c_sku else ""
        status = safe_unescape(row.get(c_status, "publish")) if c_status else "publish"
        stock = safe_unescape(row.get(c_stock, "")) if c_stock else ""
        permalink = safe_unescape(row.get(c_permalink, "")) if c_permalink else ""
        if not permalink:
            permalink = store_url.rstrip("/")  # fallback

        img_url = ""
        if c_images:
            raw = safe_unescape(row.get(c_images, ""))
            # woo export uses comma-separated
            if raw:
                img_url = raw.split(",")[0].strip()

        cat_names: List[str] = []
        if c_categories:
            raw = safe_unescape(row.get(c_categories, ""))
            if raw:
                cat_names = [x.strip() for x in raw.split(",") if x.strip()]

        attrs: Dict[str, str] = {}
        for nc, vc in zip(attr_name_cols, attr_val_cols):
            n = safe_unescape(row.get(nc, "")).strip()
            v = safe_unescape(row.get(vc, "")).strip()
            if n and v:
                attrs[n] = v

        brand = attrs.get("Brand") or attrs.get("brand") or ""
        kashrut = attrs.get("Kashrus") or attrs.get("Kashrut") or attrs.get("kashrut") or ""

        reg = row.get(c_regular, "") if c_regular else ""
        sale = row.get(c_sale, "") if c_sale else ""
        price = row.get(c_price, "") if c_price else (sale or reg)
        on_sale = bool(str(sale).strip()) and str(sale).strip() != "0"

        desc = safe_unescape(row.get(c_short, "")) if c_short else ""

        products.append(
            {
                "id": None,
                "name": name,
                "sku": sku,
                "status": status,
                "stock_status": stock,
                "price": price,
                "regular_price": reg,
                "sale_price": sale,
                "on_sale": on_sale,
                "image_url": img_url,
                "permalink": permalink,
                "category_ids": [],
                "category_names": cat_names,
                "attributes": attrs,
                "brand": brand,
                "kashrut": kashrut,
                "description": desc,
            }
        )

    return products


# ----------------------------
# Image downloading (disk cached)
# ----------------------------
def img_cache_path(url: str) -> str:
    return os.path.join(IMG_CACHE_DIR, sha1(url) + ".jpg")


def download_image(url: str, timeout: int, log_fn, retries: int = 6) -> Optional[str]:
    if not url:
        return None
    p = img_cache_path(url)
    if os.path.exists(p) and os.path.getsize(p) > 2000:
        return p

    last = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            if r.status_code >= 400:
                last = f"HTTP {r.status_code}"
                time.sleep(min(1.5 * attempt, 10))
                continue
            b = r.content
            if not b or len(b) < 2000:
                last = "Empty"
                time.sleep(min(1.5 * attempt, 10))
                continue
            # normalize to JPEG for PDF
            im = Image.open(io.BytesIO(b)).convert("RGB")
            im.thumbnail((900, 900))
            im.save(p, format="JPEG", quality=85, optimize=True)
            return p
        except Exception as e:
            last = type(e).__name__
            log_fn(f"Image download retry {attempt}/{retries}: {last}")
            time.sleep(min(1.5 * attempt, 10))
    log_fn(f"Image failed: {last}")
    return None


# ----------------------------
# PDF building (FPDF2)
# ----------------------------
def wrap_lines(pdf: FPDF, text: str, max_w: float) -> List[str]:
    """Word-wrap text to fit width using pdf.get_string_width()."""
    text = safe_unescape(text).strip()
    if not text:
        return []
    words = re.split(r"\s+", text)
    lines: List[str] = []
    cur = ""
    for w in words:
        test = w if not cur else (cur + " " + w)
        if pdf.get_string_width(test) <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def clip_lines(lines: List[str], max_lines: int) -> List[str]:
    if len(lines) <= max_lines:
        return lines
    out = lines[:max_lines]
    # add ellipsis to last line
    out[-1] = (out[-1][: max(0, len(out[-1]) - 1)] + "…") if out[-1] else "…"
    return out


class CatalogPDF(FPDF):
    def __init__(self, title: str, logo_path: Optional[str], orientation: str, page_size: str):
        super().__init__(orientation=orientation, unit="mm", format=page_size)
        self.title_txt = title
        self.logo_path = logo_path
        self.set_auto_page_break(auto=False)
        self.set_margins(10, 10, 10)

    def header(self):
        # logo
        y0 = 8
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                self.image(self.logo_path, x=10, y=y0, w=22)
            except Exception:
                pass

        # title
        self.set_text_color(*BRAND_BLUE)
        self.set_font("Helvetica", "B", 14)
        self.set_xy(35, y0 + 2)
        self.cell(0, 6, safe_unescape(self.title_txt), ln=0)

        # page #
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 9)
        self.set_xy(-35, y0 + 2)
        self.cell(25, 6, f"Page {self.page_no()}", align="R")

        # divider line
        self.set_draw_color(*BRAND_BLUE)
        self.set_line_width(0.5)
        self.line(10, 32, self.w - 10, 32)

    def footer(self):
        self.set_y(-15)
        # red line
        self.set_draw_color(*BRAND_RED)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(40, 40, 40)
        self.cell(0, 6, f"{BRAND_SITE} | Prices correct as of {now_str()}", align="L")


def build_pdf_bytes(
    products: List[Dict[str, Any]],
    title: str,
    page_orientation: str,
    grid_cols: int,
    grid_rows: int,
    currency_symbol: str,
    show_price: bool,
    show_sku: bool,
    show_description: bool,
    show_attributes: bool,
    exclude_oos: bool,
    only_sale: bool,
    include_private_in_pdf: bool,
    category_by_id: Optional[Dict[int, Dict[str, Any]]],
    category_children: Optional[Dict[int, List[int]]],
) -> bytes:
    # filter status
    filtered: List[Dict[str, Any]] = []
    for p in products:
        stt = (p.get("status") or "").lower().strip()
        if not include_private_in_pdf and stt and stt != "publish":
            continue
        if exclude_oos and (p.get("stock_status") or "").lower() == "outofstock":
            continue
        if only_sale and not bool(p.get("on_sale")):
            continue
        filtered.append(p)

    # sort by category tree path + product name
    def best_cat_path(p: Dict[str, Any]) -> str:
        # API provides ids; CSV provides names. We'll prefer ids if present.
        if category_by_id and p.get("category_ids"):
            paths = [category_full_path(int(cid), category_by_id) for cid in p["category_ids"] if int(cid) in category_by_id]
            paths = [x for x in paths if x]
            if paths:
                return sorted(paths)[0]
        # fallback
        names = p.get("category_names") or []
        if names:
            return sorted([safe_unescape(x) for x in names if safe_unescape(x)])[0]
        return "Uncategorised"

    filtered.sort(key=lambda p: (best_cat_path(p), safe_unescape(p.get("name", "")).lower()))

    # group by best cat path
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for p in filtered:
        g = best_cat_path(p)
        groups.setdefault(g, []).append(p)

    # Prepare logo file for PDF
    logo_path = None
    b = read_logo_bytes()
    if b:
        logo_path = os.path.join(APP_CACHE_DIR, "logo.png")
        with open(logo_path, "wb") as f:
            f.write(b)

    orientation = "L" if page_orientation.lower().startswith("land") else "P"
    pdf = CatalogPDF(title=title, logo_path=logo_path, orientation=orientation, page_size="A4")
    pdf.set_title(title)

    # layout constants
    margin = 10
    header_h = 34
    footer_h = 18
    cat_bar_h = 10
    gap = 4

    # card metrics depend on grid
    usable_w = pdf.w - 2 * margin
    usable_h = pdf.h - header_h - footer_h - margin - cat_bar_h

    card_w = (usable_w - gap * (grid_cols - 1)) / grid_cols
    card_h = (usable_h - gap * (grid_rows - 1)) / grid_rows

    # typography for dense grids
    if grid_cols >= 6:
        name_font = 7.2
        meta_font = 6.2
        price_font = 7.0
        max_name_lines = 2
        max_meta_lines = 1
        max_attr_lines = 1
    else:
        name_font = 8.6
        meta_font = 7.2
        price_font = 8.2
        max_name_lines = 2
        max_meta_lines = 2
        max_attr_lines = 2

    def draw_category_bar(cat_name: str):
        bar_y = header_h + 4
        pdf.set_xy(margin, bar_y)
        pdf.set_fill_color(*BRAND_BLUE)
        pdf.set_draw_color(*BRAND_BLUE)
        pdf.rect(margin, bar_y, pdf.w - 2 * margin, cat_bar_h, style="F")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_xy(margin + 4, bar_y + 2.5)
        # avoid duplicated category text: single bar only
        pdf.cell(0, 5, safe_unescape(cat_name), ln=0)

    def draw_sale_badge(x: float, y: float):
        pdf.set_fill_color(*BRAND_RED)
        pdf.set_draw_color(*BRAND_RED)
        pdf.rect(x, y, 14, 6, style="F")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_xy(x, y + 1.2)
        pdf.cell(14, 4, "SALE", align="C")

    def draw_card(p: Dict[str, Any], x: float, y: float):
        # clickable area
        link_url = safe_unescape(p.get("permalink", "")).strip()

        # outer box
        pdf.set_draw_color(*BRAND_BLUE)
        pdf.set_line_width(0.4)
        pdf.rect(x, y, card_w, card_h)

        # reserve areas
        pad = 2.2
        img_h = card_h * (0.60 if grid_cols >= 6 else 0.62)
        text_top = y + img_h + pad

        # image box
        img_x = x + pad
        img_y = y + pad
        img_w = card_w - 2 * pad
        img_h2 = img_h - 2 * pad

        # image (no grey background)
        img_path = None
        if p.get("image_url"):
            img_path = download_image(str(p.get("image_url")), timeout=20, log_fn=lambda *_: None)

        if img_path and os.path.exists(img_path):
            try:
                pdf.image(img_path, x=img_x, y=img_y, w=img_w, h=img_h2)
            except Exception:
                pass
        else:
            pdf.set_text_color(120, 120, 120)
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_xy(img_x, img_y + img_h2 / 2 - 2)
            pdf.cell(img_w, 4, "No image", align="C")

        # sale badge
        if bool(p.get("on_sale")):
            draw_sale_badge(x + card_w - 16, y + 2)

        # Name (wrapped, clipped)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "B", name_font)
        name_lines = clip_lines(wrap_lines(pdf, p.get("name", ""), card_w - 2 * pad), max_name_lines)
        yy = text_top
        pdf.set_xy(x + pad, yy)
        pdf.multi_cell(card_w - 2 * pad, 3.2 if grid_cols >= 6 else 3.6, "\n".join(name_lines), align="C")
        yy = pdf.get_y() + 0.2

        # Price line
        if show_price:
            sale = p.get("sale_price")
            reg = p.get("regular_price")
            pr = p.get("price")
            # choose displayed price
            disp = money_fmt(sale or pr or reg, currency_symbol=currency_symbol)
            reg_fmt = money_fmt(reg, currency_symbol=currency_symbol) if reg else ""
            pdf.set_font("Helvetica", "B", price_font)
            pdf.set_text_color(*BRAND_RED)
            pdf.set_xy(x + pad, yy)
            pdf.cell(card_w - 2 * pad, 4, disp, align="C")
            # strike-through regular if on sale and reg exists
            if bool(p.get("on_sale")) and reg_fmt:
                pdf.set_text_color(120, 120, 120)
                pdf.set_font("Helvetica", "", max(price_font - 1.2, 6))
                pdf.set_xy(x + pad, yy + 3.4)
                pdf.cell(card_w - 2 * pad, 3.2, reg_fmt, align="C")
            yy = yy + (7 if grid_cols >= 6 else 8)

        # SKU
        if show_sku and p.get("sku"):
            pdf.set_text_color(60, 60, 60)
            pdf.set_font("Helvetica", "", meta_font)
            lines = clip_lines(wrap_lines(pdf, f"SKU: {p.get('sku')}", card_w - 2 * pad), 1)
            pdf.set_xy(x + pad, yy)
            pdf.cell(card_w - 2 * pad, 3.2, lines[0], align="C")
            yy += 3.8

        # Brand / Kashrut
        brand = safe_unescape(p.get("brand", ""))
        kash = safe_unescape(p.get("kashrut", ""))
        meta_parts = []
        if brand:
            meta_parts.append(f"Brand: {brand}")
        if kash:
            meta_parts.append(f"Kashrus: {kash}")
        if meta_parts:
            pdf.set_text_color(60, 60, 60)
            pdf.set_font("Helvetica", "", meta_font)
            meta_text = " | ".join(meta_parts)
            meta_lines = clip_lines(wrap_lines(pdf, meta_text, card_w - 2 * pad), max_meta_lines)
            pdf.set_xy(x + pad, yy)
            pdf.multi_cell(card_w - 2 * pad, 3.0, "\n".join(meta_lines), align="C")
            yy = pdf.get_y() + 0.2

        # Attributes (compact)
        if show_attributes and isinstance(p.get("attributes"), dict) and p["attributes"]:
            # show top few attributes, excluding Brand/Kashrus duplicates
            items = []
            for k, v in p["attributes"].items():
                kk = safe_unescape(k)
                if kk.lower() in ("brand", "kashrus", "kashrut"):
                    continue
                vv = safe_unescape(v)
                if kk and vv:
                    items.append(f"{kk}: {vv}")
            if items:
                pdf.set_text_color(70, 70, 70)
                pdf.set_font("Helvetica", "", meta_font)
                txt = "; ".join(items[:2])
                attr_lines = clip_lines(wrap_lines(pdf, txt, card_w - 2 * pad), max_attr_lines)
                pdf.set_xy(x + pad, yy)
                pdf.multi_cell(card_w - 2 * pad, 2.8, "\n".join(attr_lines), align="C")
                yy = pdf.get_y() + 0.2

        # Description (very short)
        if show_description and p.get("description"):
            pdf.set_text_color(70, 70, 70)
            pdf.set_font("Helvetica", "", meta_font)
            desc_lines = clip_lines(wrap_lines(pdf, p.get("description", ""), card_w - 2 * pad), 2 if grid_cols < 6 else 1)
            pdf.set_xy(x + pad, yy)
            pdf.multi_cell(card_w - 2 * pad, 2.8, "\n".join(desc_lines), align="C")

        # Make the whole card clickable (if link exists)
        if link_url:
            try:
                pdf.link(x, y, card_w, card_h, link_url)
            except Exception:
                pass

    # draw
    for cat_name, plist in groups.items():
        pdf.add_page()
        draw_category_bar(cat_name)

        start_x = margin
        start_y = header_h + 4 + cat_bar_h + 6

        i = 0
        for p in plist:
            row = (i // grid_cols) % grid_rows
            col = i % grid_cols
            page_slot = i // (grid_cols * grid_rows)
            if page_slot > 0 and (i % (grid_cols * grid_rows) == 0):
                pdf.add_page()
                draw_category_bar(cat_name)
                start_y = header_h + 4 + cat_bar_h + 6

            xx = start_x + col * (card_w + gap)
            yy = start_y + row * (card_h + gap)
            draw_card(p, xx, yy)
            i += 1

    out = pdf.output()  # bytes in modern fpdf2, sometimes bytearray
    if isinstance(out, bytearray):
        out = bytes(out)
    if isinstance(out, str):
        out = out.encode("latin-1", "ignore")
    return out


# ----------------------------
# App
# ----------------------------
def main():
    st.set_page_config(page_title="B-Kosher Catalog Builder", page_icon="🧾", layout="wide")
    inject_css()
    require_login()

    # --- Step 1: Source (default API) ---
    st.markdown("## Step 1 — Choose data source")
    st.caption("Default = WooCommerce API. CSV upload is a backup option.")

    source = st.radio("Source", ["WooCommerce API", "CSV Upload"], index=0, horizontal=True)

    # State
    st.session_state.setdefault("products", [])
    st.session_state.setdefault("categories", [])
    st.session_state.setdefault("cat_by_id", None)
    st.session_state.setdefault("cat_children", None)

    # Live log UI
    st.markdown("### Live log")
    log_box = st.empty()
    st.session_state.setdefault("log_lines", [])

    def log(msg: str) -> None:
        st.session_state["log_lines"].append(msg)
        st.session_state["log_lines"] = st.session_state["log_lines"][-300:]
        log_box.markdown('<div class="bk-log">' + "\n".join(st.session_state["log_lines"]) + "</div>", unsafe_allow_html=True)

    # Progress UI
    progress = st.progress(0.0)
    status = st.empty()

    def progress_fn(frac: float, msg: str) -> None:
        if frac and frac > 0:
            progress.progress(min(max(frac, 0.0), 1.0))
        status.info(msg)

    st.markdown("---")

    # --- Step 2: Load products ---
    st.markdown("## Step 2 — Load products")

    if source == "WooCommerce API":
        api_timeout = st.slider("API timeout (seconds)", 10, 60, 30)
        include_private_fetch = st.checkbox(
            "Include private/unpublished products (requires API user permission)",
            value=False,
            help="If your API user lacks permission, the app will automatically fall back to published products.",
        )

        cfg = get_wc_config_from_secrets(api_timeout)
        if not cfg:
            st.error(
                "WooCommerce API secrets not set.\n\nSet these in Streamlit secrets:\n"
                "WC_STORE_URL, WC_CONSUMER_KEY, WC_CONSUMER_SECRET"
            )
            st.stop()

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Load (use cache if available)"):
                # If already checkpointed, it resumes automatically
                st.session_state["log_lines"] = []
                log("Loading category hierarchy…")
                cats = fetch_categories(cfg, log)
                st.session_state["categories"] = cats
                by_id, children = build_category_maps(cats)
                st.session_state["cat_by_id"] = by_id
                st.session_state["cat_children"] = children

                log("Fetching products from API… (resume-safe)")
                products = api_fetch_all_products(
                    cfg,
                    include_private_fetch=include_private_fetch,
                    log_fn=log,
                    progress_fn=progress_fn,
                    resume_key="products",
                )
                st.session_state["products"] = products
                progress.progress(1.0)
                status.success(f"Loaded {len(products):,} products.")
        with c2:
            if st.button("Refresh cache (fetch again)"):
                # Clear checkpoints for both modes
                st.session_state["log_lines"] = []
                for fn in os.listdir(CHKPT_DIR):
                    if fn.endswith(".json"):
                        try:
                            os.remove(os.path.join(CHKPT_DIR, fn))
                        except Exception:
                            pass
                st.success("Cleared checkpoints. Now click Load to fetch again.")

    else:
        up = st.file_uploader("Upload WooCommerce product export CSV", type=["csv"])
        if up is not None:
            st.session_state["log_lines"] = []
            log("Parsing CSV…")
            # Use store URL secret if available, else fallback
            store_url = st.secrets.get("WC_STORE_URL", "https://www.b-kosher.co.uk")
            products = parse_csv_products(up.getvalue(), str(store_url))
            st.session_state["products"] = products
            progress.progress(1.0)
            status.success(f"Loaded {len(products):,} products from CSV.")

    products: List[Dict[str, Any]] = st.session_state.get("products") or []
    if not products:
        st.info("Load products to continue.")
        st.stop()

    st.markdown("---")

    # --- Step 3: Filters + PDF settings ---
    st.markdown("## Step 3 — Configure catalog")

    # Grid density presets
    GRID_PRESETS = {
        "Standard (3×3)": (3, 3),
        "Compact (6×5)": (6, 5),   # chosen to prevent text overflow vs 6×6
        "Medium (4×4)": (4, 4),
        "Large (2×3)": (2, 3),
    }

    colA, colB, colC = st.columns([1, 1, 1])

    with colA:
        page_orientation = st.selectbox("Page orientation", ["Portrait", "Landscape"], index=0)
        grid_label = st.selectbox("Grid density", list(GRID_PRESETS.keys()), index=0)
        grid_cols, grid_rows = GRID_PRESETS[grid_label]
        currency_symbol = st.text_input("Currency symbol", value="£")

    with colB:
        # Defaults requested: SKU + description unticked
        show_price = st.checkbox("Show price", value=True)
        show_sku = st.checkbox("Show SKU", value=False)
        show_description = st.checkbox("Show description", value=False)
        show_attributes = st.checkbox("Show attributes", value=True)

    with colC:
        exclude_oos = st.checkbox("Exclude out-of-stock", value=False)
        only_sale = st.checkbox("Only sale items", value=False)

        # ✅ This is the missing toggle you asked for (in the PDF/filter step)
        include_private_in_pdf = st.checkbox(
            "Include private/unpublished products in the catalog",
            value=False,
            help="Controls what goes into the PDF. (To FETCH private products, tick it in Step 2 as well.)",
        )

    # Category selection (supports parent categories and includes descendants)
    st.markdown("### Categories (tree)")
    cat_by_id = st.session_state.get("cat_by_id")
    cat_children = st.session_state.get("cat_children")

    # Build selectable category options
    cat_options: List[Tuple[str, Any]] = []
    if cat_by_id and isinstance(cat_by_id, dict):
        for cid in sorted(cat_by_id.keys(), key=lambda x: category_full_path(x, cat_by_id)):
            path = category_full_path(cid, cat_by_id)
            if path:
                cat_options.append((path, cid))

        selected_paths = st.multiselect(
            "Choose categories (select a parent like 'Alcohol' to include its children)",
            options=[p for p, _ in cat_options],
            default=[],
        )
        selected_ids = [cid for p, cid in cat_options if p in selected_paths]
        selected_all_ids: List[int] = []
        if selected_ids and cat_children:
            for rid in selected_ids:
                selected_all_ids.extend(descendants(int(rid), cat_children))
            selected_all_ids = sorted(list(set(selected_all_ids)))
    else:
        # CSV fallback: use category names seen in file
        all_names = sorted(
            list(
                {
                    safe_unescape(n)
                    for p in products
                    for n in (p.get("category_names") or [])
                    if safe_unescape(n)
                }
            )
        )
        selected_names = st.multiselect("Choose categories", options=all_names, default=[])
        selected_all_ids = []
        selected_ids = []
        selected_paths = selected_names

    search = st.text_input("Search (name or SKU)", value="")

    # Apply filters
    filtered: List[Dict[str, Any]] = []
    s = search.strip().lower()

    for p in products:
        # status in PDF filter happens in build_pdf too, but also here for preview count
        stt = (p.get("status") or "").lower().strip()
        if not include_private_in_pdf and stt and stt != "publish":
            continue

        if exclude_oos and (p.get("stock_status") or "").lower() == "outofstock":
            continue
        if only_sale and not bool(p.get("on_sale")):
            continue

        # category filter
        if cat_by_id and selected_ids:
            p_ids = p.get("category_ids") or []
            if not any(int(cid) in selected_all_ids for cid in p_ids):
                continue
        elif (not cat_by_id) and selected_paths:
            p_names = [safe_unescape(x) for x in (p.get("category_names") or [])]
            if not any(n in selected_paths for n in p_names):
                continue

        # search filter
        if s:
            nm = safe_unescape(p.get("name", "")).lower()
            sku = safe_unescape(p.get("sku", "")).lower()
            if s not in nm and s not in sku:
                continue

        filtered.append(p)

    st.info(f"Selected products: {len(filtered):,}")

    with st.expander("Preview (first 12 products)", expanded=False):
        cols = st.columns(4)
        for i, p in enumerate(filtered[:12]):
            with cols[i % 4]:
                img_url = p.get("image_url") or ""
                if img_url:
                    st.image(img_url, use_column_width=True)
                st.markdown(f"**{safe_unescape(p.get('name',''))}**")
                if show_price:
                    st.markdown(f"<span style='color:rgb({BRAND_RED[0]},{BRAND_RED[1]},{BRAND_RED[2]});font-weight:800;'>{money_fmt(p.get('sale_price') or p.get('price') or p.get('regular_price'), currency_symbol)}</span>", unsafe_allow_html=True)
                if p.get("permalink"):
                    st.link_button("Open product", p["permalink"])

    st.markdown("---")

    st.markdown("## Step 4 — Generate PDF")
    title = st.text_input("Catalog title", value=DEFAULT_TITLE)

    if st.button("Generate PDF"):
        st.session_state["log_lines"] = []
        log("Preparing PDF…")
        progress.progress(0.05)
        status.info("Building PDF…")

        pdf_bytes = build_pdf_bytes(
            products=filtered,
            title=title,
            page_orientation=page_orientation,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            currency_symbol=currency_symbol.strip() or "£",
            show_price=show_price,
            show_sku=show_sku,
            show_description=show_description,
            show_attributes=show_attributes,
            exclude_oos=exclude_oos,
            only_sale=only_sale,
            include_private_in_pdf=include_private_in_pdf,
            category_by_id=cat_by_id if isinstance(cat_by_id, dict) else None,
            category_children=cat_children if isinstance(cat_children, dict) else None,
        )

        # ✅ Streamlit download_button MUST receive bytes (not bytearray)
        if isinstance(pdf_bytes, bytearray):
            pdf_bytes = bytes(pdf_bytes)
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode("latin-1", "ignore")

        progress.progress(1.0)
        status.success("PDF ready.")

        fname = f"bkosher_catalog_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
        )
        st.caption("Tip: PDF links work best in Chrome/Edge PDF viewer.")

    st.markdown("---")
    st.caption("Brand colours: Pantone 186C (red) + Pantone 2945C (blue).")


if __name__ == "__main__":
    main()