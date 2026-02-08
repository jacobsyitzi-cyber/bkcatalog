# app.py — B-Kosher Catalog Builder (API default)
# - WooCommerce API or CSV upload
# - Parent category selection includes children
# - Optional include private/unpublished (API permission required)
# - Grid densities: Standard (3x3), Compact (6x5) to prevent overflow
# - PDF via fpdf2 (no reportlab), safe bytes output for Streamlit download
# - Live progress + logs, reliable image downloader with retries

from __future__ import annotations

import base64
import concurrent.futures as futures
import dataclasses
import datetime as dt
import hashlib
import html
import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

from fpdf import FPDF

try:
    from PIL import Image
except Exception:
    Image = None  # Pillow should be installed via requirements.txt


# =========================
# Branding
# =========================
BRAND_BLUE = (0, 76, 151)   # #004C97
BRAND_RED = (200, 16, 46)   # #C8102E
BRAND_SITE = "www.b-kosher.co.uk"
DEFAULT_TITLE = "B-Kosher Product Catalog"

LOGO_FILE_CANDIDATES = [
    "Bkosher.png",                    # what you currently have in repo
    "B-kosher logo high q.png",        # what you said your file is called
    "bkosher.svg",
]

SESSION_CACHE_KEY = "products_df_v1"


# =========================
# Helpers
# =========================
def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def now_date_str() -> str:
    return dt.datetime.now().strftime("%d %b %Y")


def safe_text(s: Any) -> str:
    """Cleans text for PDF core fonts (latin-1) and UI.
    - Converts HTML entities (&amp; -> &)
    - Removes 'nan'
    - Replaces unsupported chars
    """
    if s is None:
        return ""
    if isinstance(s, float) and pd.isna(s):
        return ""
    s = str(s)
    if s.lower() == "nan":
        return ""
    s = html.unescape(s)  # handles &amp; etc
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pdf_safe_latin1(s: str) -> str:
    """Ensure text doesn't crash fpdf2 with unicode encoding exceptions."""
    s = safe_text(s)
    # Replace some common offenders
    s = s.replace("•", "-").replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"')
    # Strip remaining non-latin-1 characters
    return s.encode("latin-1", "ignore").decode("latin-1", "ignore")


def fmt_price(x: Any, currency: str = "£") -> str:
    try:
        if x is None:
            return ""
        if isinstance(x, str) and x.strip() == "":
            return ""
        val = float(x)
        return f"{currency}{val:.2f}"
    except Exception:
        return ""


def is_on_sale(row: pd.Series) -> bool:
    # Woo API gives on_sale boolean, sale_price, regular_price etc.
    if "on_sale" in row and bool(row.get("on_sale")):
        return True
    sp = row.get("sale_price", "")
    rp = row.get("regular_price", "")
    try:
        return float(sp) > 0 and float(sp) < float(rp)
    except Exception:
        return False


def pick_logo_bytes() -> Optional[bytes]:
    for fn in LOGO_FILE_CANDIDATES:
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                return f.read()
    return None


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# =========================
# WooCommerce API
# =========================
@dataclasses.dataclass
class WooCreds:
    store_url: str
    consumer_key: str
    consumer_secret: str
    timeout: int = 30


def wc_get(creds: WooCreds, endpoint: str, params: Dict[str, Any]) -> requests.Response:
    url = creds.store_url.rstrip("/") + endpoint
    # DO NOT leak keys in errors: never print full url with query
    params = dict(params)
    params["consumer_key"] = creds.consumer_key
    params["consumer_secret"] = creds.consumer_secret
    r = requests.get(url, params=params, timeout=creds.timeout)
    return r


def fetch_all_categories(creds: WooCreds, log_fn, progress_fn) -> List[Dict[str, Any]]:
    cats: List[Dict[str, Any]] = []
    page = 1
    per_page = 100
    while True:
        log_fn(f"API: fetching categories page {page}…")
        r = wc_get(creds, "/wp-json/wc/v3/products/categories", {
            "per_page": per_page, "page": page, "hide_empty": "false"
        })
        if r.status_code != 200:
            log_fn(f"API categories failed ({r.status_code}).")
            break
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            break
        cats.extend(data)
        page += 1
        progress_fn(min(0.25, 0.05 + len(cats) / 5000 * 0.2))
        if len(data) < per_page:
            break
    return cats


def fetch_products_by_status(
    creds: WooCreds,
    status_value: str,
    log_fn,
    progress_fn,
    resume_state: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    """Fetch products by a specific Woo status ('publish', 'private', etc.)"""
    out: List[Dict[str, Any]] = []
    per_page = 25  # safer
    page = 1

    # Resume support within same session
    if resume_state and resume_state.get("status") == status_value:
        page = int(resume_state.get("page", 1))
        out = list(resume_state.get("items", []))

    while True:
        log_fn(f"API: fetching products page {page} (status={status_value})…")
        r = wc_get(creds, "/wp-json/wc/v3/products", {
            "per_page": per_page,
            "page": page,
            "status": status_value,
        })
        if r.status_code != 200:
            # Do not show keys — show endpoint + status only
            log_fn(f"API request failed ({r.status_code}) for products (status={status_value}) page={page}.")
            try:
                msg = r.json().get("message", "")
                if msg:
                    log_fn(f"API says: {safe_text(msg)}")
            except Exception:
                pass
            break

        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            break

        out.extend(data)
        page += 1

        # Update resume snapshot
        st.session_state["api_resume"] = {"status": status_value, "page": page, "items": out}

        # progress is approximate, but gives confidence
        progress_fn(min(0.85, 0.25 + len(out) / 8000 * 0.6))

        if len(data) < per_page:
            break

    return out


def normalize_products_from_api(products: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for p in products:
        cats = p.get("categories", []) or []
        images = p.get("images", []) or []
        attrs = p.get("attributes", []) or []

        # pick main image
        img_url = ""
        if len(images) > 0:
            img_url = safe_text(images[0].get("src", ""))

        # attributes string
        attr_parts = []
        for a in attrs:
            name = safe_text(a.get("name", ""))
            opts = a.get("options", []) or []
            opts = [safe_text(x) for x in opts if safe_text(x)]
            if name and opts:
                attr_parts.append(f"{name}: {', '.join(opts)}")
        attr_text = " | ".join(attr_parts)

        # category ids + names
        cat_ids = [c.get("id") for c in cats if isinstance(c, dict) and "id" in c]
        cat_names = [safe_text(c.get("name")) for c in cats if isinstance(c, dict)]

        rows.append({
            "id": p.get("id"),
            "name": safe_text(p.get("name", "")),
            "sku": safe_text(p.get("sku", "")),
            "permalink": safe_text(p.get("permalink", "")),
            "status": safe_text(p.get("status", "")),  # publish/private/draft
            "stock_status": safe_text(p.get("stock_status", "")),  # instock/outofstock/onbackorder
            "price": safe_text(p.get("price", "")),
            "regular_price": safe_text(p.get("regular_price", "")),
            "sale_price": safe_text(p.get("sale_price", "")),
            "on_sale": bool(p.get("on_sale", False)),
            "short_description": safe_text(p.get("short_description", "")),
            "description": safe_text(p.get("description", "")),
            "brand": safe_text((p.get("brands") or "")),  # if plugin returns it
            "image_url": img_url,
            "category_ids": cat_ids,
            "category_names": cat_names,
            "attributes": attr_text,
        })
    df = pd.DataFrame(rows)

    # Brand / Kashrus often come through as attributes; we’ll try to detect common names
    # If you have specific attribute names (e.g. "Kashrus") this will help show it.
    # Keep as-is; you can refine later.
    return df


# =========================
# CSV loading
# =========================
def load_products_from_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, keep_default_na=False)
    # Attempt to map Woo export columns
    colmap_candidates = {
        "name": ["Name", "Product name", "Title"],
        "sku": ["SKU", "Sku"],
        "permalink": ["Permalink", "Product URL", "URL"],
        "price": ["Price", "Regular price", "price"],
        "regular_price": ["Regular price", "Regular Price"],
        "sale_price": ["Sale price", "Sale Price"],
        "description": ["Description", "Short description", "Short Description"],
        "image_url": ["Images", "Image", "Image URL", "Images (URL)"],
        "category_names": ["Categories", "Category"],
        "stock_status": ["Stock status", "Stock Status"],
        "status": ["Published", "Status"],
        "attributes": ["Attribute 1 name", "Attributes"],  # very variable
    }

    def pick(colnames):
        for c in colnames:
            if c in df.columns:
                return c
        return None

    out = pd.DataFrame()
    out["name"] = df[pick(colmap_candidates["name"])].map(safe_text) if pick(colmap_candidates["name"]) else ""
    out["sku"] = df[pick(colmap_candidates["sku"])].map(safe_text) if pick(colmap_candidates["sku"]) else ""
    out["permalink"] = df[pick(colmap_candidates["permalink"])].map(safe_text) if pick(colmap_candidates["permalink"]) else ""
    out["price"] = df[pick(colmap_candidates["price"])].map(safe_text) if pick(colmap_candidates["price"]) else ""
    out["regular_price"] = df[pick(colmap_candidates["regular_price"])].map(safe_text) if pick(colmap_candidates["regular_price"]) else out["price"]
    out["sale_price"] = df[pick(colmap_candidates["sale_price"])].map(safe_text) if pick(colmap_candidates["sale_price"]) else ""
    out["description"] = df[pick(colmap_candidates["description"])].map(safe_text) if pick(colmap_candidates["description"]) else ""
    imgcol = pick(colmap_candidates["image_url"])
    if imgcol:
        # Woo export sometimes has multiple separated by commas
        out["image_url"] = df[imgcol].map(lambda x: safe_text(str(x).split(",")[0]) if safe_text(x) else "")
    else:
        out["image_url"] = ""
    catcol = pick(colmap_candidates["category_names"])
    out["category_names"] = df[catcol].map(lambda x: [safe_text(i) for i in str(x).split(",") if safe_text(i)]) if catcol else [[]] * len(df)
    out["category_ids"] = [[]] * len(df)
    out["stock_status"] = df[pick(colmap_candidates["stock_status"])].map(safe_text) if pick(colmap_candidates["stock_status"]) else ""
    out["status"] = df[pick(colmap_candidates["status"])].map(safe_text) if pick(colmap_candidates["status"]) else "publish"
    out["on_sale"] = out.apply(is_on_sale, axis=1)
    out["attributes"] = ""
    out["short_description"] = ""

    return out


# =========================
# Category tree utilities
# =========================
def build_category_tree(categories: List[Dict[str, Any]]):
    by_id = {c["id"]: c for c in categories if isinstance(c, dict) and "id" in c}
    children = {}
    for c in categories:
        pid = c.get("parent", 0)
        children.setdefault(pid, []).append(c["id"])

    def descendants(cat_id: int) -> List[int]:
        res = []
        stack = [cat_id]
        seen = set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            res.append(x)
            for ch in children.get(x, []):
                stack.append(ch)
        return res

    def path_name(cat_id: int) -> str:
        parts = []
        cur = cat_id
        guard = 0
        while cur and cur in by_id and guard < 50:
            parts.append(safe_text(by_id[cur].get("name", "")))
            cur = by_id[cur].get("parent", 0)
            guard += 1
        parts.reverse()
        return " > ".join([p for p in parts if p])

    all_paths = []
    for cid in by_id.keys():
        all_paths.append((path_name(cid), cid))

    all_paths.sort(key=lambda t: t[0].lower())
    return by_id, children, descendants, all_paths, path_name


def category_display_for_product(row: pd.Series, path_name_fn) -> str:
    # choose the shortest top-level category path (best looking)
    ids = row.get("category_ids", []) or []
    if not isinstance(ids, list):
        ids = []
    paths = [path_name_fn(i) for i in ids if isinstance(i, int)]
    paths = [p for p in paths if p]
    if not paths:
        # fallback to names
        names = row.get("category_names", []) or []
        if isinstance(names, list) and names:
            return safe_text(names[0])
        return "Uncategorized"
    # Prefer the deepest path (more specific)
    paths.sort(key=lambda p: (p.count(">"), len(p)))
    return paths[-1]


# =========================
# Image downloading (reliable)
# =========================
def download_with_retries(url: str, timeout: int, retries: int, backoff: float, log_fn) -> Optional[bytes]:
    if not url:
        return None
    headers = {"User-Agent": "BkosherCatalogBot/1.0"}
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            if r.status_code == 200 and r.content:
                return r.content
            log_fn(f"IMG: {r.status_code} on attempt {attempt} for image.")
        except Exception as e:
            log_fn(f"IMG: error attempt {attempt}: {type(e).__name__}")
        time.sleep(backoff * attempt)
    return None


def preprocess_image_bytes(img_bytes: bytes, max_px: int = 900) -> Optional[bytes]:
    if not img_bytes:
        return None
    if Image is None:
        return img_bytes  # fpdf2 can often handle it directly
    try:
        im = Image.open(io.BytesIO(img_bytes))
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_px / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)))
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=85, optimize=True)
        return out.getvalue()
    except Exception:
        return img_bytes


# =========================
# PDF
# =========================
class CatalogPDF(FPDF):
    def __init__(self, title: str, logo_bytes: Optional[bytes], orientation: str, currency_symbol: str):
        super().__init__(orientation=orientation, unit="mm", format="A4")
        self.title_txt = pdf_safe_latin1(title)
        self.logo_bytes = logo_bytes
        self.currency = currency_symbol

        self.set_auto_page_break(auto=True, margin=12)

        # Use core fonts (latin-1 safe); avoid unicode crashes by sanitizing text
        self.set_font("Helvetica", size=10)

    def header(self):
        margin = 10
        self.set_fill_color(255, 255, 255)
        self.set_draw_color(*BRAND_BLUE)
        self.set_line_width(0.3)

        # Logo
        if self.logo_bytes:
            try:
                # write bytes to temp stream for fpdf2
                # fpdf2 accepts file-like via name? safest: temp file
                tmp_name = f"/tmp/bk_logo_{os.getpid()}.png"
                if not os.path.exists(tmp_name):
                    with open(tmp_name, "wb") as f:
                        f.write(self.logo_bytes)
                self.image(tmp_name, x=margin, y=6, w=24)
            except Exception:
                pass

        # Title
        self.set_text_color(*BRAND_BLUE)
        self.set_xy(margin + 28, 9)
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 8, self.title_txt)

        # Page number
        self.set_font("Helvetica", size=10)
        self.set_text_color(0, 0, 0)
        self.set_xy(-30, 9)
        self.cell(20, 8, f"Page {self.page_no()}", align="R")

        # Underline
        self.set_draw_color(*BRAND_BLUE)
        self.line(margin, 18, self.w - margin, 18)
        self.ln(14)

    def footer(self):
        margin = 10
        self.set_y(-12)
        self.set_draw_color(*BRAND_RED)
        self.line(margin, self.get_y(), self.w - margin, self.get_y())
        self.ln(1.5)
        self.set_font("Helvetica", size=8)
        self.set_text_color(60, 60, 60)
        text = f"{BRAND_SITE} | Prices correct as of {now_date_str()}"
        self.cell(0, 6, pdf_safe_latin1(text))

    def draw_category_bar(self, label: str):
        margin = 10
        bar_h = 10
        y = self.get_y() + 1
        self.set_fill_color(*BRAND_BLUE)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 11)
        self.rounded_rect(margin, y, self.w - 2 * margin, bar_h, 2.5, style="F")
        self.set_xy(margin + 3, y + 2.5)
        self.cell(0, 5, pdf_safe_latin1(label))
        self.ln(bar_h + 4)
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", size=10)

    def rounded_rect(self, x, y, w, h, r, style=""):
        # fpdf2 supports rounded_rect in newer versions, but keep compatibility
        try:
            super().rounded_rect(x, y, w, h, r, style=style)
        except Exception:
            # fallback: normal rect
            if style:
                self.rect(x, y, w, h, style=style)
            else:
                self.rect(x, y, w, h)

    def wrap_text_lines(self, text: str, max_width: float, max_lines: int) -> List[str]:
        text = pdf_safe_latin1(text)
        if not text:
            return []
        words = text.split(" ")
        lines = []
        cur = ""
        for w in words:
            candidate = (cur + " " + w).strip()
            if self.get_string_width(candidate) <= max_width:
                cur = candidate
            else:
                if cur:
                    lines.append(cur)
                cur = w
                if len(lines) >= max_lines:
                    break
        if len(lines) < max_lines and cur:
            lines.append(cur)
        # If overflow, ellipsize last line
        if len(lines) == max_lines and len(words) > 0:
            # ensure last line ends with …
            last = lines[-1]
            if self.get_string_width(last) > max_width:
                # trim
                while last and self.get_string_width(last + "...") > max_width:
                    last = last[:-1]
                lines[-1] = last + "..."
            else:
                # check if more words remain (approx)
                pass
        return lines[:max_lines]


def make_catalog_pdf_bytes(
    df: pd.DataFrame,
    title: str,
    orientation: str,
    currency_symbol: str,
    grid: Tuple[int, int],
    show_price: bool,
    show_sku: bool,
    show_description: bool,
    show_attributes: bool,
    include_brand: bool,
    include_kashrut: bool,
    category_mode: str,
    category_path_fn,
    progress_cb=None,
    log_fn=None,
) -> bytes:
    if progress_cb is None:
        progress_cb = lambda x: None
    if log_fn is None:
        log_fn = lambda s: None

    logo_bytes = pick_logo_bytes()

    cols, rows = grid
    pdf = CatalogPDF(title=title, logo_bytes=logo_bytes, orientation=orientation, currency_symbol=currency_symbol)
    pdf.add_page()

    margin = 10
    gutter = 4
    header_space = 6

    usable_w = pdf.w - 2 * margin
    usable_h = pdf.h - 26 - 14  # header+footer safety

    card_w = (usable_w - gutter * (cols - 1)) / cols
    card_h = (usable_h - header_space - gutter * (rows - 1)) / rows

    # Layout areas within card (tuned to prevent overlap)
    pad = 2.0
    img_h = card_h * 0.55
    text_area_h = card_h - img_h - 2 * pad

    # Font sizes adapt
    name_font = 7.5 if cols >= 6 else 9
    small_font = 6.5 if cols >= 6 else 8

    # Grouping
    if category_mode == "path":
        df = df.copy()
        df["_cat"] = df.apply(lambda r: category_display_for_product(r, category_path_fn), axis=1)
    else:
        df = df.copy()
        df["_cat"] = df.get("category_names", "").apply(lambda x: safe_text(x[0]) if isinstance(x, list) and x else "Uncategorized")

    df["_cat"] = df["_cat"].fillna("Uncategorized")
    cats = sorted(df["_cat"].unique(), key=lambda x: str(x).lower())

    total = len(df)
    done = 0

    for ci, cat in enumerate(cats):
        subset = df[df["_cat"] == cat].copy()
        subset = subset.sort_values(by=["name"], key=lambda s: s.str.lower())

        # New page per category if not enough room for bar + at least one row
        pdf.draw_category_bar(cat)

        x0 = margin
        y0 = pdf.get_y()

        col_i = 0
        row_i = 0

        for _, r in subset.iterrows():
            if row_i >= rows:
                pdf.add_page()
                pdf.draw_category_bar(cat)
                x0 = margin
                y0 = pdf.get_y()
                col_i = 0
                row_i = 0

            xx = x0 + col_i * (card_w + gutter)
            yy = y0 + row_i * (card_h + gutter)

            # Card border
            pdf.set_draw_color(*BRAND_BLUE)
            pdf.set_line_width(0.4)
            pdf.rect(xx, yy, card_w, card_h)

            # Clickable link for whole card
            link = safe_text(r.get("permalink", ""))
            if link:
                try:
                    pdf.link(xx, yy, card_w, card_h, link)
                except Exception:
                    pass

            # Sale badge
            if bool(is_on_sale(r)):
                pdf.set_fill_color(*BRAND_RED)
                pdf.set_text_color(255, 255, 255)
                pdf.set_font("Helvetica", "B", 7)
                badge_w = 12
                badge_h = 6
                pdf.rect(xx + card_w - badge_w - 1.2, yy + 1.2, badge_w, badge_h, style="F")
                pdf.set_xy(xx + card_w - badge_w - 1.2, yy + 1.8)
                pdf.cell(badge_w, badge_h - 1, "SALE", align="C")
                pdf.set_text_color(0, 0, 0)

            # Image
            img_x = xx + pad
            img_y = yy + pad
            img_w = card_w - 2 * pad

            img_bytes = r.get("_img_bytes", None)
            if img_bytes:
                try:
                    tmp_name = f"/tmp/bk_img_{sha1(str(r.get('id', '')) + str(r.get('image_url', '')))}.jpg"
                    if not os.path.exists(tmp_name):
                        with open(tmp_name, "wb") as f:
                            f.write(img_bytes)
                    pdf.image(tmp_name, x=img_x, y=img_y, w=img_w, h=img_h)
                except Exception:
                    # draw "No image"
                    pdf.set_font("Helvetica", "I", 7)
                    pdf.set_text_color(120, 120, 120)
                    pdf.set_xy(img_x, img_y + img_h / 2 - 2)
                    pdf.cell(img_w, 4, "No image", align="C")
                    pdf.set_text_color(0, 0, 0)
            else:
                pdf.set_font("Helvetica", "I", 7)
                pdf.set_text_color(120, 120, 120)
                pdf.set_xy(img_x, img_y + img_h / 2 - 2)
                pdf.cell(img_w, 4, "No image", align="C")
                pdf.set_text_color(0, 0, 0)

            # Text area
            text_x = xx + pad
            text_y = yy + pad + img_h + 1.0
            text_w = card_w - 2 * pad

            # Product name (max 2 lines)
            pdf.set_font("Helvetica", "B", name_font)
            pdf.set_xy(text_x, text_y)
            name_lines = pdf.wrap_text_lines(safe_text(r.get("name", "")), text_w, max_lines=2)
            for ln in name_lines:
                pdf.cell(text_w, 3.5, ln, ln=1)
                pdf.set_x(text_x)

            # Price
            pdf.set_font("Helvetica", "B", small_font)
            if show_price:
                price = fmt_price(r.get("price", ""), currency_symbol)
                sale = fmt_price(r.get("sale_price", ""), currency_symbol)
                reg = fmt_price(r.get("regular_price", ""), currency_symbol)

                # If sale: show sale + struck reg (simple)
                pdf.set_text_color(*BRAND_RED)
                if is_on_sale(r) and sale:
                    pdf.cell(text_w, 3.2, sale, ln=1)
                    if reg:
                        pdf.set_text_color(120, 120, 120)
                        pdf.set_font("Helvetica", "", small_font)
                        pdf.cell(text_w, 3.0, reg, ln=1)
                        pdf.set_font("Helvetica", "B", small_font)
                else:
                    pdf.cell(text_w, 3.2, price, ln=1)
                pdf.set_text_color(0, 0, 0)

            # SKU
            if show_sku:
                sku = safe_text(r.get("sku", ""))
                if sku:
                    pdf.set_font("Helvetica", "", small_font)
                    pdf.cell(text_w, 3.0, pdf_safe_latin1(f"SKU: {sku}"), ln=1)

            # Brand & Kashrut — (requested: B. Brand and kashrut)
            # We try to derive:
            # - brand from "brand" column if present
            # - kashrut from attributes text if it contains "Kashrus" or "Kashrut" etc.
            brand = safe_text(r.get("brand", ""))
            attrs = safe_text(r.get("attributes", ""))

            kashrut_val = ""
            # naive extraction
            m = re.search(r"(Kashrut|Kashrus)\s*:\s*([^|]+)", attrs, flags=re.IGNORECASE)
            if m:
                kashrut_val = safe_text(m.group(2))

            pdf.set_font("Helvetica", "", small_font)
            if include_brand and brand:
                pdf.cell(text_w, 3.0, pdf_safe_latin1(f"Brand: {brand}"), ln=1)
            if include_kashrut and kashrut_val:
                pdf.cell(text_w, 3.0, pdf_safe_latin1(f"Kashrus: {kashrut_val}"), ln=1)

            # Description (max 2 lines)
            if show_description:
                desc = safe_text(r.get("short_description", "")) or safe_text(r.get("description", ""))
                if desc:
                    pdf.set_font("Helvetica", "", small_font)
                    dlines = pdf.wrap_text_lines(desc, text_w, max_lines=2)
                    for ln in dlines:
                        pdf.cell(text_w, 3.0, ln, ln=1)
                        pdf.set_x(text_x)

            # Attributes (max 2 lines)
            if show_attributes:
                at = safe_text(r.get("attributes", ""))
                if at:
                    pdf.set_font("Helvetica", "", small_font)
                    alines = pdf.wrap_text_lines(at, text_w, max_lines=2)
                    for ln in alines:
                        pdf.cell(text_w, 3.0, ln, ln=1)
                        pdf.set_x(text_x)

            # advance grid
            col_i += 1
            if col_i >= cols:
                col_i = 0
                row_i += 1

            done += 1
            if done % 25 == 0:
                progress_cb(min(0.99, done / max(1, total)))
                log_fn(f"PDF: placed {done}/{total}")

        # after category, move to new page if space too tight for next bar
        pdf.ln(2)
        if pdf.get_y() > pdf.h - 60:
            pdf.add_page()

    # Important: ensure Streamlit receives bytes (not bytearray)
    out = pdf.output(dest="S")
    if isinstance(out, (bytearray,)):
        out = bytes(out)
    elif isinstance(out, str):
        out = out.encode("latin-1", "ignore")
    elif not isinstance(out, (bytes,)):
        out = bytes(out)

    return out


# =========================
# UI + Flow
# =========================
def inject_brand_css():
    blue = rgb_to_hex(BRAND_BLUE)
    red = rgb_to_hex(BRAND_RED)
    st.markdown(
        f"""
        <style>
            .bk-title {{
                font-size: 1.4rem;
                font-weight: 800;
                color: {blue};
                margin-bottom: 0.25rem;
            }}
            .bk-sub {{
                color: #666;
                margin-top: 0;
            }}
            div.stButton > button {{
                background-color: {red} !important;
                color: white !important;
                border-radius: 10px !important;
                border: 0px !important;
                padding: 0.6rem 1rem !important;
                font-weight: 700 !important;
            }}
            .bk-card {{
                border: 1px solid #eee;
                border-radius: 14px;
                padding: 14px;
                background: #fff;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def require_login():
    # Local: allow env var fallback
    pw = None
    try:
        pw = st.secrets.get("APP_PASSWORD", None)
    except Exception:
        pw = None
    if not pw:
        pw = os.environ.get("APP_PASSWORD", "")

    if not pw:
        st.warning("APP_PASSWORD is not set (Streamlit secrets or environment). Login is disabled.")
        return True

    if st.session_state.get("logged_in"):
        return True

    st.markdown('<div class="bk-title">B-Kosher Catalog Login</div>', unsafe_allow_html=True)
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if p == pw:
            st.session_state["logged_in"] = True
            st.success("Logged in.")
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


def load_api_creds_from_secrets() -> Optional[WooCreds]:
    try:
        store_url = st.secrets.get("WC_STORE_URL", "")
        ck = st.secrets.get("WC_CONSUMER_KEY", "")
        cs = st.secrets.get("WC_CONSUMER_SECRET", "")
    except Exception:
        store_url = ck = cs = ""
    store_url = safe_text(store_url)
    ck = safe_text(ck)
    cs = safe_text(cs)
    if store_url and ck and cs:
        return WooCreds(store_url=store_url, consumer_key=ck, consumer_secret=cs, timeout=30)
    return None


def main():
    st.set_page_config(page_title="B-Kosher Catalog Builder", layout="wide")
    inject_brand_css()
    require_login()

    logo_bytes = pick_logo_bytes()
    col1, col2 = st.columns([1, 8])
    with col1:
        if logo_bytes:
            st.image(logo_bytes, use_column_width=True)
    with col2:
        st.markdown(f'<div class="bk-title">{DEFAULT_TITLE}</div>', unsafe_allow_html=True)
        st.markdown('<div class="bk-sub">Default = WooCommerce API. CSV upload is a backup option.</div>', unsafe_allow_html=True)

    st.divider()

    # -------------------------
    # Step 1: source (API default)
    # -------------------------
    if "source" not in st.session_state:
        st.session_state["source"] = "api"

    st.markdown("### Step 1 — Choose data source")
    src = st.radio("Source", ["WooCommerce API", "CSV Upload"], index=0, horizontal=True)
    st.session_state["source"] = "api" if src == "WooCommerce API" else "csv"

    # -------------------------
    # Step 2: Load products
    # -------------------------
    st.markdown("### Step 2 — Load products")

    api_timeout = st.slider("API timeout (seconds)", 10, 60, 30)
    include_private_fetch = st.checkbox(
        "Include private/unpublished products (requires API user permission)",
        value=False,
        help="This affects API fetching. If enabled, the app fetches publish + private and merges them.",
    )

    # live log box
    log_box = st.empty()
    progress = st.progress(0.0)

    def log(msg: str):
        msg = safe_text(msg)
        lines = st.session_state.get("log_lines", [])
        lines.append(msg)
        lines = lines[-12:]
        st.session_state["log_lines"] = lines
        log_box.code("\n".join(lines))

    def prog(x: float):
        progress.progress(max(0.0, min(1.0, float(x))))

    # Buttons
    c1, c2 = st.columns([1, 1])
    with c1:
        load_btn = st.button("Load (use cache if available)")
    with c2:
        refresh_btn = st.button("Refresh cache (fetch again)")

    # CSV uploader if selected
    uploaded_csv = None
    if st.session_state["source"] == "csv":
        uploaded_csv = st.file_uploader("Upload WooCommerce product export CSV", type=["csv"])

    # Load cache if exists and not refresh
    if SESSION_CACHE_KEY not in st.session_state:
        st.session_state[SESSION_CACHE_KEY] = None

    if refresh_btn:
        st.session_state[SESSION_CACHE_KEY] = None
        st.session_state["api_resume"] = None
        st.session_state["log_lines"] = []
        st.success("Cache cleared. Click Load to fetch again.")

    if load_btn:
        st.session_state["log_lines"] = []
        prog(0.02)

        if st.session_state["source"] == "csv":
            if not uploaded_csv:
                st.error("Please upload a CSV first.")
            else:
                log("CSV: loading…")
                df = load_products_from_csv(uploaded_csv.getvalue())
                st.session_state[SESSION_CACHE_KEY] = df
                prog(1.0)
                st.success(f"Loaded {len(df):,} products from CSV.")
        else:
            creds = load_api_creds_from_secrets()
            if not creds:
                st.error("WooCommerce API secrets are missing. Add WC_STORE_URL, WC_CONSUMER_KEY, WC_CONSUMER_SECRET to Streamlit secrets.")
            else:
                creds.timeout = int(api_timeout)
                log("Loading category hierarchy…")
                cats = fetch_all_categories(creds, log, prog)
                st.session_state["api_categories"] = cats

                log("Fetching products from API…")
                resume_state = st.session_state.get("api_resume", None)

                # Always fetch publish
                publish_items = fetch_products_by_status(creds, "publish", log, prog, resume_state=None)

                items = publish_items

                # Optional: private
                if include_private_fetch:
                    log("Fetching private products…")
                    private_items = fetch_products_by_status(creds, "private", log, prog, resume_state=None)
                    # Some sites also store unpublished as draft; do NOT default fetch (often forbidden)
                    # but we can try softly:
                    log("Attempting draft products (if permitted)…")
                    draft_items = fetch_products_by_status(creds, "draft", log, prog, resume_state=None)

                    # Merge unique by id
                    merged = {}
                    for p in (publish_items + private_items + draft_items):
                        pid = p.get("id")
                        if pid is not None:
                            merged[pid] = p
                    items = list(merged.values())

                df = normalize_products_from_api(items)
                st.session_state[SESSION_CACHE_KEY] = df
                prog(1.0)
                st.success(f"Loaded {len(df):,} products from API.")

    df: Optional[pd.DataFrame] = st.session_state.get(SESSION_CACHE_KEY, None)
    if df is None or len(df) == 0:
        st.info("Load products to continue.")
        return

    st.caption(f"Loaded products in memory: **{len(df):,}**")

    # Build category tree if available
    categories = st.session_state.get("api_categories", []) or []
    by_id = children = descendants = all_paths = path_name_fn = None
    if categories:
        by_id, children, descendants, all_paths, path_name_fn = build_category_tree(categories)

    st.divider()

    # -------------------------
    # Step 3: Filters + Settings
    # -------------------------
    st.markdown("### Step 3 — Filters & PDF settings")

    left, right = st.columns([1.1, 1.2])

    with left:
        st.markdown("#### Filters")

        # IMPORTANT: This is the checkbox you keep asking for — in FILTERS (affects PDF)
        include_private_in_pdf = st.checkbox(
            "Include private/unpublished products in the catalog",
            value=False,
            help="If unchecked, private/draft items will be filtered out even if they were fetched.",
        )

        exclude_oos = st.checkbox("Exclude out-of-stock", value=False)
        only_sale = st.checkbox("Only sale items", value=False)

        search = st.text_input("Search (name or SKU)", value="")

        # Category selection (parent works)
        selected_cat_ids = []
        if all_paths and descendants:
            options = [p for (p, cid) in all_paths]
            selected_paths = st.multiselect("Categories (tree)", options=options)
            if selected_paths:
                # map to ids and include descendants
                name_to_id = {p: cid for (p, cid) in all_paths}
                root_ids = [name_to_id[p] for p in selected_paths if p in name_to_id]
                selected_set = set()
                for rid in root_ids:
                    for d in descendants(rid):
                        selected_set.add(d)
                selected_cat_ids = sorted(list(selected_set))
        else:
            # CSV fallback
            cat_names = []
            for v in df.get("category_names", []):
                if isinstance(v, list):
                    cat_names.extend(v)
            cat_names = sorted(set([safe_text(x) for x in cat_names if safe_text(x)]), key=lambda x: x.lower())
            selected_names = st.multiselect("Categories", options=cat_names)
            selected_cat_ids = selected_names  # name-based

        st.markdown("#### Display options")

        currency_symbol = st.text_input("Currency symbol", value="£")

        # Defaults requested: SKU + Description off
        show_price = st.checkbox("Show price", value=True)
        show_sku = st.checkbox("Show SKU", value=False)
        show_description = st.checkbox("Show description", value=False)
        show_attributes = st.checkbox("Show attributes", value=True)

        # Brand + Kashrut option you selected earlier ("B. Brand and kashrut")
        include_brand = st.checkbox("Show Brand line", value=True)
        include_kashrut = st.checkbox("Show Kashrut line", value=True)

        grid_density = st.selectbox("Grid density", ["Standard (3×3)", "Compact (6×5)"], index=0)

        orientation = st.selectbox("Page orientation", ["Portrait", "Landscape"], index=0)

        title = st.text_input("PDF title", value=DEFAULT_TITLE)

    # Apply filters
    filtered = df.copy()

    # status filtering for PDF
if "status" in filtered.columns:
    if include_private_in_pdf:
        filtered = filtered[filtered["status"].isin(["publish", "private"])]
    else:
        filtered = filtered[filtered["status"].isin(["publish"])]
    if exclude_oos and "stock_status" in filtered.columns:
        filtered = filtered[filtered["stock_status"].astype(str).str.lower().isin(["instock", "onbackorder"])]

    if only_sale:
        filtered = filtered[filtered.apply(is_on_sale, axis=1)]

    if search.strip():
        s = search.strip().lower()
        filtered = filtered[
            filtered["name"].astype(str).str.lower().str.contains(s, na=False)
            | filtered["sku"].astype(str).str.lower().str.contains(s, na=False)
        ]

    # category filtering
    if selected_cat_ids:
        if categories and isinstance(selected_cat_ids[0], int):
            filtered = filtered[filtered["category_ids"].apply(lambda ids: any((i in selected_cat_ids) for i in (ids or [])) if isinstance(ids, list) else False)]
        else:
            # name based
            sel_names = set([safe_text(x) for x in selected_cat_ids])
            filtered = filtered[filtered["category_names"].apply(lambda names: any((safe_text(n) in sel_names) for n in (names or [])) if isinstance(names, list) else False)]

    # Count summary
    with right:
        st.markdown("#### Selection summary")
        st.markdown(f'<div class="bk-card"><b>Selected products:</b> {len(filtered):,}</div>', unsafe_allow_html=True)

        with st.expander("Preview (first 9 products)", expanded=False):
            prev = filtered.head(9)
            if len(prev) == 0:
                st.info("No products match filters.")
            else:
                for _, r in prev.iterrows():
                    st.write(f"• **{safe_text(r.get('name',''))}** ({safe_text(r.get('status',''))}) — {fmt_price(r.get('price',''), currency_symbol)}")

        st.markdown("#### Image download preset")
        preset = st.selectbox("Image download preset", ["Reliable", "Balanced", "Fast"], index=0)

        if preset == "Reliable":
            dl_workers, dl_retries, dl_timeout = 3, 6, 30
        elif preset == "Balanced":
            dl_workers, dl_retries, dl_timeout = 6, 4, 20
        else:
            dl_workers, dl_retries, dl_timeout = 10, 2, 15

        st.caption("Reliable is recommended on mobile / large catalogs.")

    st.divider()

    # -------------------------
    # Step 4: Generate PDF
    # -------------------------
    st.markdown("### Step 4 — Generate PDF")

    generate = st.button("Generate PDF")

    if generate:
        if len(filtered) == 0:
            st.error("No products selected.")
            return

        status = st.empty()
        progress2 = st.progress(0.0)
        log_box2 = st.empty()
        st.session_state["log_lines_pdf"] = []

        def log2(m):
            m = safe_text(m)
            lines = st.session_state.get("log_lines_pdf", [])
            lines.append(m)
            lines = lines[-14:]
            st.session_state["log_lines_pdf"] = lines
            log_box2.code("\n".join(lines))

        def prog2(x):
            progress2.progress(max(0.0, min(1.0, float(x))))

        # Download images (reliable)
        status.info("Stage 1/2 — Downloading images…")
        log2(f"Workers={dl_workers} retries={dl_retries} timeout={dl_timeout}")

        work_df = filtered.copy()

        # Cache images by url in session to avoid re-downloading on rerun
        if "img_cache" not in st.session_state:
            st.session_state["img_cache"] = {}
        img_cache: Dict[str, bytes] = st.session_state["img_cache"]

        urls = work_df["image_url"].fillna("").astype(str).tolist()
        unique_urls = [u for u in sorted(set(urls)) if u]

        total_u = len(unique_urls)
        done_u = 0

        def dl_one(url):
            if url in img_cache:
                return url, img_cache[url]
            b = download_with_retries(url, timeout=dl_timeout, retries=dl_retries, backoff=0.6, log_fn=log2)
            if b:
                b = preprocess_image_bytes(b)
                img_cache[url] = b
            return url, b

        with futures.ThreadPoolExecutor(max_workers=dl_workers) as ex:
            futs = [ex.submit(dl_one, u) for u in unique_urls]
            for f in futures.as_completed(futs):
                url, b = f.result()
                done_u += 1
                if done_u % 10 == 0 or done_u == total_u:
                    prog2(min(0.75, done_u / max(1, total_u) * 0.75))
                    log2(f"Images: {done_u}/{total_u}")

        # Map bytes onto rows
        work_df["_img_bytes"] = work_df["image_url"].map(lambda u: img_cache.get(u, None) if safe_text(u) else None)

        # PDF grid
        if grid_density.startswith("Standard"):
            grid = (3, 3) if orientation == "Portrait" else (4, 3)
        else:
            # Compact = 6×5 (not 6×6) to prevent text loss
            grid = (6, 5) if orientation == "Portrait" else (6, 5)

        status.info("Stage 2/2 — Building PDF…")

        pdf_bytes = make_catalog_pdf_bytes(
            work_df,
            title=title,
            orientation="P" if orientation == "Portrait" else "L",
            currency_symbol=currency_symbol,
            grid=grid,
            show_price=show_price,
            show_sku=show_sku,
            show_description=show_description,
            show_attributes=show_attributes,
            include_brand=include_brand,
            include_kashrut=include_kashrut,
            category_mode="path" if categories else "name",
            category_path_fn=path_name_fn if path_name_fn else (lambda x: ""),
            progress_cb=lambda x: prog2(0.75 + 0.25 * x),
            log_fn=log2,
        )

        # ensure Streamlit gets bytes
        if isinstance(pdf_bytes, bytearray):
            pdf_bytes = bytes(pdf_bytes)
        elif isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode("latin-1", "ignore")
        elif not isinstance(pdf_bytes, (bytes,)):
            pdf_bytes = bytes(pdf_bytes)

        status.success("PDF ready.")

        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"bkosher_catalog_{now_date_str().replace(' ', '_')}.pdf",
            mime="application/pdf",
        )
        st.caption("Tip: PDF links work best in Chrome/Edge PDF viewer.")

    st.divider()
    st.caption("If your iPhone browser sleeps, Streamlit may stop running. The cache makes reloading fast, and the API fetch avoids 'status=any' (which causes 500 on your site).")


if __name__ == "__main__":
    main()