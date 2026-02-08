# app.py — B-Kosher Catalog Builder (Streamlit)
# - WooCommerce API (default) or CSV upload
# - Login gate first
# - Resumable API fetching in batches (prevents Streamlit Cloud timeouts)
# - Category tree selection incl. parents
# - Optional private/unpublished products
# - PDF generation with fpdf2, clickable product tiles, safe rounded_rect fallback
# - Brand colours + logo

from __future__ import annotations

import io
import os
import re
import json
import time
import math
import html
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import requests
import pandas as pd
import streamlit as st
from PIL import Image

from fpdf import FPDF  # fpdf2

# ---------------------------
# Branding
# ---------------------------
BRAND_BLUE = (0, 76, 151)     # #004C97
BRAND_RED = (200, 16, 46)     # #C8102E
BRAND_SITE = "www.b-kosher.co.uk"
DEFAULT_TITLE = "B-Kosher Product Catalog"

# Your repo file (update if needed)
LOGO_FILENAME = "Bkosher.png"  # <-- your repo screenshot shows Bkosher.png


# ---------------------------
# Helpers
# ---------------------------

def rgb01(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return rgb

def safe_unescape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = html.unescape(s)  # converts &amp; to &
    # Also fix common double-encoded patterns:
    s = s.replace("&amp;", "&")
    return s

def sanitize_latin1(s: str) -> str:
    """fpdf2 built-in fonts are latin-1. Keep it safe."""
    if s is None:
        return ""
    s = safe_unescape(str(s))
    # remove control chars
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", s)
    # replace non-latin1 with space
    return s.encode("latin-1", "replace").decode("latin-1")

def money_fmt(v, currency="£") -> str:
    try:
        f = float(v)
    except Exception:
        return ""
    return f"{currency}{f:0.2f}"

def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()

def read_logo_bytes() -> Optional[bytes]:
    if os.path.exists(LOGO_FILENAME):
        with open(LOGO_FILENAME, "rb") as f:
            return f.read()
    return None

def st_log(msg: str) -> None:
    st.session_state.setdefault("live_logs", [])
    st.session_state["live_logs"].append(msg)
    # keep last 200
    st.session_state["live_logs"] = st.session_state["live_logs"][-200:]


# ---------------------------
# Secrets / Config
# ---------------------------

def get_secret(name: str, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

def get_wc_config() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Supports both naming styles:
      - WC_BASE_URL / WC_CONSUMER_KEY / WC_CONSUMER_SECRET
      - WC_URL / WC_CK / WC_CS
    """
    base = get_secret("WC_BASE_URL") or get_secret("WC_URL")
    ck = get_secret("WC_CONSUMER_KEY") or get_secret("WC_CK")
    cs = get_secret("WC_CONSUMER_SECRET") or get_secret("WC_CS")
    if base:
        base = base.rstrip("/")
    return base, ck, cs

def get_app_password() -> Optional[str]:
    return get_secret("APP_PASSWORD")


# ---------------------------
# Login gate
# ---------------------------

def require_login():
    # Nothing else should render before login
    pwd = get_app_password()
    if not pwd:
        st.error("APP_PASSWORD is not set in Streamlit secrets.")
        st.stop()

    if st.session_state.get("authed") is True:
        return

    # Minimal page
    st.set_page_config(page_title="B-Kosher Catalog Builder", layout="wide")

    # Logo + title (no raw HTML leakage)
    logo_bytes = read_logo_bytes()
    col1, col2 = st.columns([1, 6])
    with col1:
        if logo_bytes:
            st.image(logo_bytes, width=90)
    with col2:
        st.markdown("## B-Kosher Catalog Builder")
        st.caption("Login required.")

    st.markdown("### Login")
    pw_in = st.text_input("Password", type="password")

    if st.button("Login", use_container_width=True):
        if pw_in == pwd:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    st.stop()


# ---------------------------
# WooCommerce API client
# ---------------------------

class WCClient:
    def __init__(self, base_url: str, ck: str, cs: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.ck = ck
        self.cs = cs
        self.timeout = timeout

    def get(self, path: str, params: dict) -> requests.Response:
        url = f"{self.base_url}/wp-json/wc/v3/{path.lstrip('/')}"
        # IMPORTANT: do not print keys in errors
        safe_params = dict(params)
        # attach auth
        safe_params["consumer_key"] = self.ck
        safe_params["consumer_secret"] = self.cs

        return requests.get(url, params=safe_params, timeout=self.timeout)

    def fetch_categories_all(self, per_page=100) -> List[dict]:
        cats = []
        page = 1
        while True:
            st_log(f"API: fetching categories page {page}...")
            r = self.get("products/categories", {"per_page": per_page, "page": page, "hide_empty": False})
            if r.status_code != 200:
                raise RuntimeError(f"API categories failed ({r.status_code}).")
            batch = r.json()
            if not batch:
                break
            cats.extend(batch)
            if len(batch) < per_page:
                break
            page += 1
        return cats

    def fetch_products_page(self, page: int, per_page: int, status: str) -> List[dict]:
        st_log(f"API: fetching products page {page} (status={status})...")
        r = self.get("products", {"per_page": per_page, "page": page, "status": status})
        if r.status_code != 200:
            # Do not include URL with keys
            raise RuntimeError(f"API request failed ({r.status_code}) at /products?page={page}&per_page={per_page}&status={status}")
        return r.json()


# ---------------------------
# Category tree utilities
# ---------------------------

@dataclass
class CatNode:
    id: int
    name: str
    parent: int

def build_category_maps(categories: List[dict]):
    nodes: Dict[int, CatNode] = {}
    children: Dict[int, List[int]] = {}
    for c in categories:
        cid = int(c.get("id", 0))
        p = int(c.get("parent", 0) or 0)
        nm = safe_unescape(c.get("name", "")).strip()
        nodes[cid] = CatNode(id=cid, name=nm, parent=p)
        children.setdefault(p, []).append(cid)
        children.setdefault(cid, [])

    # stable sort by name
    for pid in children:
        children[pid] = sorted(children[pid], key=lambda x: nodes.get(x, CatNode(x, "", 0)).name.lower())

    def path(cid: int) -> List[str]:
        parts = []
        cur = cid
        seen = set()
        while cur and cur in nodes and cur not in seen:
            seen.add(cur)
            parts.append(nodes[cur].name)
            cur = nodes[cur].parent
        return list(reversed(parts))

    def path_str(cid: int) -> str:
        return " > ".join(path(cid))

    def descendants(root_id: int) -> List[int]:
        out = []
        stack = [root_id]
        while stack:
            x = stack.pop()
            out.append(x)
            stack.extend(children.get(x, []))
        return out

    return nodes, children, path_str, descendants


# ---------------------------
# CSV loader
# ---------------------------

def load_products_from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, dtype=str, keep_default_na=False)
    # Try to map common WooCommerce export columns
    # We'll normalize to columns used by the app
    colmap = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    name_c = pick("name")
    sku_c = pick("sku")
    price_c = pick("regular price", "regular_price", "regular price (£)", "price")
    sale_c = pick("sale price", "sale_price")
    stock_c = pick("stock status", "stock_status")
    perm_c = pick("permalink", "url")
    cats_c = pick("categories", "category")
    desc_c = pick("short description", "short_description", "description")
    img_c = pick("images", "image", "image 1", "image_1")

    out = pd.DataFrame()
    out["id"] = df.get(pick("id"), "")
    out["name"] = df.get(name_c, "")
    out["sku"] = df.get(sku_c, "")
    out["regular_price"] = df.get(price_c, "")
    out["sale_price"] = df.get(sale_c, "")
    out["stock_status"] = df.get(stock_c, "")
    out["permalink"] = df.get(perm_c, "")
    out["categories_raw"] = df.get(cats_c, "")
    out["short_description"] = df.get(desc_c, "")

    # images may be comma-separated; take first
    imgs = df.get(img_c, "")
    out["image_url"] = imgs.apply(lambda s: (str(s).split(",")[0].strip() if s else ""))

    # CSV export typically lacks private status. Treat all as publish.
    out["status"] = "publish"
    out["attributes"] = ""  # CSV export varies; keep empty
    out["category_ids"] = ""  # not available
    out["category_paths"] = out["categories_raw"].apply(lambda x: safe_unescape(x))
    return out


# ---------------------------
# Normalize WooCommerce product JSON -> DataFrame
# ---------------------------

def wc_products_to_df(products: List[dict], cat_path_str) -> pd.DataFrame:
    rows = []
    for p in products:
        cats = p.get("categories") or []
        cat_ids = [int(c.get("id")) for c in cats if c.get("id")]
        cat_paths = [cat_path_str(cid) for cid in cat_ids] if cat_path_str else []
        images = p.get("images") or []
        img_url = images[0].get("src") if images else ""

        attrs = p.get("attributes") or []
        attr_pairs = []
        for a in attrs:
            nm = safe_unescape(a.get("name", "")).strip()
            opts = a.get("options") or []
            opts = [safe_unescape(o).strip() for o in opts if o]
            if nm and opts:
                attr_pairs.append(f"{nm}: {', '.join(opts)}")

        rows.append({
            "id": p.get("id", ""),
            "name": safe_unescape(p.get("name", "")).strip(),
            "sku": p.get("sku", "") or "",
            "regular_price": p.get("regular_price", "") or "",
            "sale_price": p.get("sale_price", "") or "",
            "price": p.get("price", "") or "",
            "on_sale": bool(p.get("on_sale", False)),
            "stock_status": p.get("stock_status", "") or "",
            "permalink": p.get("permalink", "") or "",
            "short_description": safe_unescape(p.get("short_description", "")).strip(),
            "status": p.get("status", "") or "publish",
            "image_url": img_url or "",
            "category_ids": cat_ids,
            "category_paths": cat_paths,
            "attributes": " | ".join(attr_pairs),
        })

    df = pd.DataFrame(rows)
    # Ensure expected columns exist
    for c in ["name","sku","regular_price","sale_price","stock_status","permalink","short_description","status","image_url","attributes","category_paths","category_ids","on_sale"]:
        if c not in df.columns:
            df[c] = ""
    return df


# ---------------------------
# Image cache + downloader (retries, safe)
# ---------------------------

CACHE_DIR = ".cache"
IMG_DIR = os.path.join(CACHE_DIR, "images")
DATA_DIR = os.path.join(CACHE_DIR, "data")

ensure_dir(IMG_DIR)
ensure_dir(DATA_DIR)

def img_cache_path(url: str) -> str:
    ext = ".jpg"
    m = re.search(r"\.(png|jpg|jpeg|webp)(\?|$)", url.lower())
    if m:
        ext = "." + m.group(1)
        if ext == ".jpeg":
            ext = ".jpg"
    return os.path.join(IMG_DIR, sha1(url) + ext)

def download_with_retries(url: str, timeout: int, retries: int, backoff: float = 1.2) -> Optional[bytes]:
    if not url:
        return None
    for i in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
        except Exception:
            pass
        time.sleep((backoff ** i) * 0.4)
    return None

def get_image_bytes(url: str, timeout: int = 20, retries: int = 4) -> Optional[bytes]:
    if not url:
        return None
    path = img_cache_path(url)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            pass

    b = download_with_retries(url, timeout=timeout, retries=retries)
    if b:
        try:
            with open(path, "wb") as f:
                f.write(b)
        except Exception:
            pass
    return b


# ---------------------------
# PDF generator (fpdf2)
# ---------------------------

class CatalogPDF(FPDF):
    def __init__(self, orientation: str, title: str, currency: str):
        super().__init__(orientation=orientation, unit="mm", format="A4")
        self.title_txt = sanitize_latin1(title)
        self.currency = currency
        self.logo_bytes = read_logo_bytes()
        self.set_auto_page_break(auto=False)
        self.set_margins(10, 10, 10)

    # ---- FIX: rounded_rect compatibility ----
    def rounded_rect(self, x, y, w, h, r=0, style=""):
        """
        Compatibility wrapper:
        - If fpdf2 supports rounded rect, use it
        - Else fallback to normal rect
        """
        try:
            return FPDF.rounded_rect(self, x, y, w, h, r, style=style)
        except Exception:
            pass
        try:
            return FPDF.round_rect(self, x, y, w, h, r, style=style)  # some forks
        except Exception:
            pass
        self.rect(x, y, w, h, style=style)

    def header(self):
        # Top line + logo + title + page number
        self.set_font("Helvetica", "B", 14)
        x0 = 10
        y0 = 8

        if self.logo_bytes:
            # Put logo left
            try:
                self.image(io.BytesIO(self.logo_bytes), x=x0, y=y0, w=24)
            except Exception:
                pass

        self.set_xy(x0 + 28, y0 + 2)
        self.set_text_color(*BRAND_BLUE)
        self.cell(0, 7, self.title_txt, ln=0)

        # page number right
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)
        self.set_xy(10, y0 + 2)
        self.cell(0, 7, f"Page {self.page_no()}", align="R")

        # divider line
        self.set_draw_color(*BRAND_BLUE)
        self.set_line_width(0.8)
        self.line(10, 24, 200, 24)
        self.ln(12)

    def footer(self):
        self.set_draw_color(*BRAND_RED)
        self.set_line_width(0.6)
        self.line(10, 287, 200, 287)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(60, 60, 60)
        self.set_xy(10, 289)
        self.cell(0, 6, f"{BRAND_SITE} | Prices correct as of {time.strftime('%d %b %Y')}", ln=0)

    def category_bar(self, text: str):
        text = sanitize_latin1(text)
        self.set_fill_color(*BRAND_BLUE)
        self.set_text_color(255, 255, 255)
        self.set_draw_color(*BRAND_BLUE)
        self.set_line_width(0.4)
        y = 28
        self.rounded_rect(10, y, 190, 10, 2, style="F")
        self.set_xy(12, y + 2.2)
        self.set_font("Helvetica", "B", 11)
        self.cell(186, 6, text, ln=0)
        self.set_text_color(0, 0, 0)

    def wrap_lines(self, text: str, max_width: float, max_lines: int, font=("Helvetica","",8)) -> List[str]:
        text = sanitize_latin1(text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        self.set_font(*font)
        words = text.split(" ")
        lines = []
        cur = ""
        for w in words:
            trial = (cur + " " + w).strip()
            if self.get_string_width(trial) <= max_width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
                if len(lines) >= max_lines:
                    break
        if len(lines) < max_lines and cur:
            lines.append(cur)

        # Ellipsis if overflow
        if len(lines) == max_lines and words:
            # ensure last line fits with ellipsis
            last = lines[-1]
            ell = "…"
            while self.get_string_width(last + ell) > max_width and len(last) > 0:
                last = last[:-1]
            lines[-1] = (last + ell).strip()
        return lines

    def product_tile(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        product: dict,
        show_price: bool,
        show_sku: bool,
        show_desc: bool,
        show_attrs: bool,
        currency: str,
        img_timeout: int,
        img_retries: int,
        dense: bool,
        add_link: bool,
    ):
        pad = 2.0
        self.set_draw_color(*BRAND_BLUE)
        self.set_line_width(0.35)
        self.rounded_rect(x, y, w, h, 2, style="D")

        # Make whole tile clickable
        if add_link:
            url = product.get("permalink") or ""
            if url:
                try:
                    self.link(x, y, w, h, url)
                except Exception:
                    pass

        # layout
        img_h = h * (0.55 if dense else 0.58)
        text_y = y + img_h + pad
        text_h = h - img_h - (2 * pad)

        # Image area
        img_x = x + pad
        img_y = y + pad
        img_w = w - 2 * pad
        img_h2 = img_h - 2 * pad

        # image bg removed (requested)
        url = product.get("image_url") or ""
        b = get_image_bytes(url, timeout=img_timeout, retries=img_retries) if url else None
        if b:
            try:
                im = Image.open(io.BytesIO(b)).convert("RGB")
                iw, ih = im.size
                if iw and ih:
                    # fit while preserving aspect
                    scale = min(img_w / iw, img_h2 / ih)
                    dw = iw * scale
                    dh = ih * scale
                    dx = img_x + (img_w - dw) / 2
                    dy = img_y + (img_h2 - dh) / 2
                    bio = io.BytesIO()
                    im.save(bio, format="JPEG", quality=85)
                    self.image(io.BytesIO(bio.getvalue()), x=dx, y=dy, w=dw, h=dh)
            except Exception:
                pass
        else:
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(120, 120, 120)
            self.set_xy(x, y + img_h/2 - 2)
            self.cell(w, 4, "No image", align="C")
            self.set_text_color(0, 0, 0)

        # SALE badge
        on_sale = bool(product.get("on_sale")) or (product.get("sale_price") not in ("", None))
        if on_sale:
            badge_w, badge_h = 14, 6
            self.set_fill_color(*BRAND_RED)
            self.set_text_color(255,255,255)
            self.rounded_rect(x + w - badge_w - 1.5, y + 1.5, badge_w, badge_h, 1.5, style="F")
            self.set_font("Helvetica","B",8)
            self.set_xy(x + w - badge_w - 1.5, y + 2.2)
            self.cell(badge_w, 4, "SALE", align="C")
            self.set_text_color(0,0,0)

        # Text block
        max_w = w - 2 * pad
        cursor_y = text_y

        # Name
        self.set_font("Helvetica", "B", 7 if dense else 8)
        name_lines = self.wrap_lines(product.get("name",""), max_w, 2 if dense else 2, font=("Helvetica","B", 7 if dense else 8))
        for ln in name_lines:
            self.set_xy(x + pad, cursor_y)
            self.cell(max_w, 3.7 if dense else 4.1, ln, ln=1)
            cursor_y += (3.7 if dense else 4.1)

        # Price
        if show_price:
            rp = product.get("regular_price") or product.get("price") or ""
            sp = product.get("sale_price") or ""
            self.set_text_color(*BRAND_RED)
            self.set_font("Helvetica", "B", 7 if dense else 8)
            shown = ""
            if sp not in ("", None) and str(sp).strip() != "":
                shown = money_fmt(sp, currency)
            else:
                shown = money_fmt(rp, currency)
            if shown:
                self.set_xy(x + pad, cursor_y)
                self.cell(max_w, 3.7 if dense else 4.1, shown, ln=1)
                cursor_y += (3.7 if dense else 4.1)
            self.set_text_color(0,0,0)

        # SKU
        if show_sku:
            sku = str(product.get("sku","") or "").strip()
            if sku:
                self.set_font("Helvetica","", 6.5 if dense else 7)
                self.set_text_color(80,80,80)
                self.set_xy(x + pad, cursor_y)
                self.cell(max_w, 3.4 if dense else 3.8, f"SKU: {sanitize_latin1(sku)}", ln=1)
                cursor_y += (3.4 if dense else 3.8)
                self.set_text_color(0,0,0)

        # Description
        if show_desc:
            desc = sanitize_latin1(product.get("short_description","") or "")
            desc = re.sub(r"<[^>]+>", "", desc).strip()
            if desc and desc.lower() != "nan":
                self.set_font("Helvetica","", 6.2 if dense else 7)
                lines = self.wrap_lines(desc, max_w, 2 if dense else 3, font=("Helvetica","", 6.2 if dense else 7))
                for ln in lines:
                    if cursor_y > y + h - 6:
                        break
                    self.set_xy(x + pad, cursor_y)
                    self.cell(max_w, 3.2 if dense else 3.6, ln, ln=1)
                    cursor_y += (3.2 if dense else 3.6)

        # Attributes
        if show_attrs:
            attrs = sanitize_latin1(product.get("attributes","") or "")
            if attrs and attrs.lower() != "nan":
                self.set_font("Helvetica","", 6.0 if dense else 6.8)
                lines = self.wrap_lines(attrs.replace(" | ", " • "), max_w, 2 if dense else 2, font=("Helvetica","", 6.0 if dense else 6.8))
                for ln in lines:
                    if cursor_y > y + h - 5:
                        break
                    self.set_xy(x + pad, cursor_y)
                    self.cell(max_w, 3.1 if dense else 3.5, ln, ln=1)
                    cursor_y += (3.1 if dense else 3.5)


def pdf_bytes(pdf: CatalogPDF) -> bytes:
    out = pdf.output()  # fpdf2 returns bytes
    # Some envs return bytearray
    if isinstance(out, bytearray):
        return bytes(out)
    if isinstance(out, str):
        return out.encode("latin-1", "ignore")
    return out


# ---------------------------
# Filtering / grouping
# ---------------------------

def product_is_oos(p: pd.Series) -> bool:
    return str(p.get("stock_status","")).lower() in ("outofstock","out_of_stock","out-of-stock","0","false")

def product_is_private(p: pd.Series) -> bool:
    return str(p.get("status","")).lower() in ("private","draft","pending")

def product_is_on_sale(p: pd.Series) -> bool:
    # best-effort
    sp = str(p.get("sale_price","") or "").strip()
    return (sp != "" and sp.lower() != "nan") or bool(p.get("on_sale", False))

def get_primary_category_path(row: pd.Series) -> str:
    paths = row.get("category_paths", [])
    if isinstance(paths, str):
        # CSV path
        return paths.split(",")[0].strip()
    if isinstance(paths, list) and paths:
        return sorted(paths, key=lambda s: (len(s.split(">")), s.lower()))[0]
    return "Uncategorized"

def group_products(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # sort by category path then name
    df = df.copy()
    df["__cat"] = df.apply(get_primary_category_path, axis=1)
    df["__name"] = df["name"].fillna("").astype(str)
    df = df.sort_values(["__cat","__name"], kind="stable")
    groups = {}
    for cat, g in df.groupby("__cat", sort=False):
        groups[cat] = g.drop(columns=["__cat","__name"], errors="ignore")
    return groups


# ---------------------------
# Resumable API loading (batch per run)
# ---------------------------

def cache_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)

def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)

def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def api_cache_key(base: str, include_private: bool) -> str:
    return f"api_cache_{sha1(base + str(include_private))}.json"

def load_cached_api(base: str, include_private: bool):
    key = api_cache_key(base, include_private)
    return load_json(cache_path(key))

def save_cached_api(base: str, include_private: bool, payload: dict):
    key = api_cache_key(base, include_private)
    save_json(cache_path(key), payload)

def api_fetch_step(
    wc: WCClient,
    include_private: bool,
    per_page: int,
    pages_per_run: int,
):
    """
    Fetch a few pages, store progress in cache + session_state.
    Then rerun until done.
    """
    status = "any" if include_private else "publish"
    base = wc.base_url

    cached = load_cached_api(base, include_private)
    if not cached:
        cached = {
            "done": False,
            "next_page": 1,
            "products": [],
            "fetched_at": time.time(),
            "include_private": include_private,
        }

    next_page = int(cached.get("next_page", 1))
    done = bool(cached.get("done", False))
    if done:
        return cached

    # Fetch categories once per run if missing in session
    # (categories cached separately in session)
    fetched_pages = 0
    while fetched_pages < pages_per_run:
        batch = wc.fetch_products_page(page=next_page, per_page=per_page, status=status)
        if not batch:
            cached["done"] = True
            break
        cached["products"].extend(batch)
        next_page += 1
        cached["next_page"] = next_page
        fetched_pages += 1

        save_cached_api(base, include_private, cached)

        # yield to UI a bit
        time.sleep(0.05)

        # If last page smaller than per_page, we're done
        if len(batch) < per_page:
            cached["done"] = True
            save_cached_api(base, include_private, cached)
            break

    return cached


# ---------------------------
# UI
# ---------------------------

def render_header():
    st.set_page_config(page_title="B-Kosher Catalog Builder", layout="wide")
    st.markdown(
        """
        <style>
          .bk-card { border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; padding: 14px; background: white; }
          .bk-title { font-weight: 800; font-size: 28px; color: #004C97; margin: 0; }
          .bk-sub { color: #6b7280; margin-top: 4px; }
          .bk-pill { display:inline-block; padding: 4px 10px; border-radius:999px; background:#004C97; color:white; font-size:12px; }
          .bk-btn > button { background:#C8102E !important; border: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    logo_bytes = read_logo_bytes()
    st.markdown('<div class="bk-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 6, 2])
    with c1:
        if logo_bytes:
            st.image(logo_bytes, width=90)
    with c2:
        st.markdown('<div class="bk-title">B-Kosher Catalog Builder</div>', unsafe_allow_html=True)
        st.markdown('<div class="bk-sub">Build printable product catalogs from WooCommerce (API) or CSV export.</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="bk-pill">Customer-facing</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")


def live_logs_box():
    with st.expander("Live logs", expanded=True):
        logs = st.session_state.get("live_logs", [])
        if not logs:
            st.info("Logs will appear here during import and PDF generation.")
        else:
            st.code("\n".join(logs), language="text")


def main():
    require_login()
    render_header()

    base, ck, cs = get_wc_config()

    # Step 1 - data source
    st.markdown("## Step 1 — Choose data source")
    source = st.radio("Source", ["WooCommerce API", "CSV Upload"], index=0, horizontal=True)

    st.session_state.setdefault("products_df", None)
    st.session_state.setdefault("categories", None)
    st.session_state.setdefault("cat_maps", None)

    st.markdown("---")
    st.markdown("## Step 2 — Load products")

    api_timeout = st.slider("API timeout (seconds)", 10, 60, 30)

    include_private_fetch = st.checkbox(
        "Include private/unpublished products (requires API user permission)",
        value=False,
        help="If enabled, API fetch uses status=any. If your store returns 500/401 for status=any, turn this off.",
    )

    if source == "WooCommerce API":
        if not (base and ck and cs):
            st.error("WooCommerce API secrets missing. Add WC_BASE_URL/WC_CONSUMER_KEY/WC_CONSUMER_SECRET (or WC_URL/WC_CK/WC_CS) to secrets.")
        else:
            wc = WCClient(base, ck, cs, timeout=api_timeout)

            # Load categories (cached in session)
            if st.session_state.get("categories") is None:
                try:
                    st_log("Loading category hierarchy…")
                    cats = wc.fetch_categories_all(per_page=100)
                    st.session_state["categories"] = cats
                    st_log(f"Loaded {len(cats)} categories.")
                except Exception as e:
                    st.error(f"Failed to load categories: {e}")
                    cats = None
            else:
                cats = st.session_state["categories"]

            if cats:
                nodes, children, path_str, descendants = build_category_maps(cats)
                st.session_state["cat_maps"] = (nodes, children, path_str, descendants)

            # Controls for resilient fetching
            per_page = 25
            pages_per_run = 8  # keep run short to avoid health-check timeouts

            colA, colB = st.columns([1, 1])
            with colA:
                load_btn = st.button("Load (use cache if available)", use_container_width=True)
            with colB:
                refresh_btn = st.button("Refresh cache (fetch again)", use_container_width=True)

            # Cache handling
            if refresh_btn:
                st_log("Clearing API cache…")
                payload = {"done": False, "next_page": 1, "products": [], "fetched_at": time.time(), "include_private": include_private_fetch}
                save_cached_api(base, include_private_fetch, payload)
                st.session_state["products_df"] = None
                st.success("Cache cleared. Click Load to fetch again.")

            # Load from cache if done
            if load_btn:
                st_log("Starting/resuming API fetch…")
                try:
                    # Ensure cache exists
                    if not load_cached_api(base, include_private_fetch):
                        save_cached_api(base, include_private_fetch, {"done": False, "next_page": 1, "products": [], "fetched_at": time.time(), "include_private": include_private_fetch})

                    # Fetch a batch
                    payload = api_fetch_step(wc, include_private_fetch, per_page=per_page, pages_per_run=pages_per_run)
                    total = len(payload.get("products", []))
                    done = bool(payload.get("done", False))
                    next_page = int(payload.get("next_page", 1))

                    st.info(f"Fetched so far: {total} products. Next page: {next_page}. Done: {done}")

                    # If done, build df now
                    if done:
                        _, _, path_str, _ = st.session_state["cat_maps"]
                        df = wc_products_to_df(payload["products"], path_str)
                        st.session_state["products_df"] = df
                        st.success(f"Loaded {len(df):,} products from API.")
                    else:
                        # Continue automatically via rerun (keeps runs short)
                        st_log("Continuing fetch…")
                        time.sleep(0.1)
                        st.rerun()

                except Exception as e:
                    st.error(str(e))

    else:
        up = st.file_uploader("Upload WooCommerce product export CSV", type=["csv"])
        if up is not None:
            try:
                df = load_products_from_csv(up)
                st.session_state["products_df"] = df
                st.success(f"Loaded {len(df):,} products from CSV.")
            except Exception as e:
                st.error(f"CSV load failed: {e}")

    live_logs_box()

    df = st.session_state.get("products_df")
    if df is None or len(df) == 0:
        st.warning("Load products to continue.")
        return

    st.markdown("---")
    st.markdown("## Step 3 — Filter & options")

    # Defaults requested: SKU unticked, description unticked
    show_price = st.checkbox("Show price", value=True)
    show_sku = st.checkbox("Show SKU", value=False)
    show_desc = st.checkbox("Show description", value=False)
    show_attrs = st.checkbox("Show attributes", value=True)

    exclude_oos = st.checkbox("Exclude out-of-stock", value=True)
    only_sale = st.checkbox("Only sale items", value=False)

    # IMPORTANT: include private in PDF (separate from fetch)
    include_private_pdf = st.checkbox(
        "Include private/unpublished items in PDF",
        value=False,
        help="If unchecked, private/draft items are removed before PDF generation (even if API fetched them).",
    )

    # Category selection (tree)
    cat_sel = []
    nodes_children = st.session_state.get("cat_maps")
    if nodes_children:
        nodes, children, path_str, descendants = nodes_children

        # Build display list including parents
        all_ids = sorted(nodes.keys(), key=lambda cid: path_str(cid).lower())
        options = [path_str(cid) for cid in all_ids]
        id_by_label = {path_str(cid): cid for cid in all_ids}

        selected_labels = st.multiselect(
            "Categories (tree) — select a parent to include all children",
            options=options,
            default=[],
        )
        sel_ids = [id_by_label[lbl] for lbl in selected_labels if lbl in id_by_label]

        # If any parent selected, include descendants
        expanded_ids = set()
        for cid in sel_ids:
            for d in descendants(cid):
                expanded_ids.add(d)

        cat_sel = sorted(expanded_ids)

    search = st.text_input("Search (name or SKU)", value="")

    # Grid density + orientation
    density = st.selectbox("Grid density", ["Standard (3×3)", "Compact (6×5) — recommended", "Ultra (6×6) — experimental"], index=0)
    currency = st.text_input("Currency symbol", value="£")

    orientation = st.selectbox("Page orientation", ["Portrait", "Landscape"], index=0)

    # Image preset
    preset = st.selectbox("Image download preset", ["Reliable", "Fast"], index=0)
    img_timeout = 25 if preset == "Reliable" else 15
    img_retries = 6 if preset == "Reliable" else 2

    # Apply filters
    f = df.copy()

    # Normalize strings
    f["name"] = f["name"].fillna("").astype(str).map(safe_unescape)
    f["sku"] = f["sku"].fillna("").astype(str)
    f["short_description"] = f["short_description"].fillna("").astype(str).map(safe_unescape)
    f["status"] = f["status"].fillna("").astype(str)

    if exclude_oos:
        f = f[~f.apply(product_is_oos, axis=1)]

    if only_sale:
        f = f[f.apply(product_is_on_sale, axis=1)]

    if not include_private_pdf:
        f = f[~f.apply(product_is_private, axis=1)]

    if search.strip():
        q = search.strip().lower()
        f = f[
            f["name"].str.lower().str.contains(q, na=False)
            | f["sku"].str.lower().str.contains(q, na=False)
        ]

    if cat_sel and "category_ids" in f.columns:
        def has_any_cat(row):
            ids = row.get("category_ids", [])
            if isinstance(ids, str):
                return False
            return any((cid in cat_sel) for cid in ids)
        f = f[f.apply(has_any_cat, axis=1)]

    st.info(f"Selected products: {len(f):,}")

    # Preview
    with st.expander("Preview (first 9 products)"):
        st.dataframe(f.head(9)[["name","sku","regular_price","sale_price","stock_status","status","permalink"]], use_container_width=True)

    st.markdown("---")
    st.markdown("## Step 4 — Generate PDF")

    title = st.text_input("Catalog title", value=DEFAULT_TITLE)

    generate = st.button("Generate PDF", type="primary", use_container_width=True)
    if not generate:
        return

    st.session_state["live_logs"] = []  # reset logs
    st_log("Starting PDF generation…")

    # Decide grid
    dense = False
    cols = 3
    rows = 3
    if density.startswith("Standard"):
        cols, rows = (3, 3) if orientation == "Portrait" else (4, 2)
        dense = False
    elif density.startswith("Compact"):
        # recommended: 6x5 portrait, 7x4 landscape
        cols, rows = (6, 5) if orientation == "Portrait" else (7, 4)
        dense = True
    else:
        # Ultra: 6x6 portrait, 8x5 landscape (may overflow text)
        cols, rows = (6, 6) if orientation == "Portrait" else (8, 5)
        dense = True

    # Group by categories for divider bars
    groups = group_products(f)

    pdf = CatalogPDF(orientation=("P" if orientation == "Portrait" else "L"), title=title, currency=currency)

    margin = 10.0
    top_y = 42.0
    bottom_y = 280.0
    usable_h = bottom_y - top_y

    # Adjust for category bar
    bar_h = 12.0
    gap_after_bar = 6.0

    tile_gap_x = 4.0 if not dense else 2.6
    tile_gap_y = 5.0 if not dense else 2.8

    usable_w = (297.0 if orientation == "Landscape" else 210.0) - 2 * margin
    tile_w = (usable_w - (cols - 1) * tile_gap_x) / cols

    # allocate vertical space under bar
    grid_top = top_y + bar_h + gap_after_bar
    grid_h = (bottom_y - grid_top)
    tile_h = (grid_h - (rows - 1) * tile_gap_y) / rows

    progress = st.progress(0.0)
    status = st.empty()

    total_items = len(f)
    done_items = 0

    for cat_name, gdf in groups.items():
        # new page for each category
        pdf.add_page()
        pdf.category_bar(cat_name)

        # positions
        start_x = margin
        start_y = grid_top

        i = 0
        for _, row in gdf.iterrows():
            r = i // cols
            c = i % cols

            # if exceeds page grid, start new page with same category bar
            if r >= rows:
                pdf.add_page()
                pdf.category_bar(cat_name)
                i = 0
                r = 0
                c = 0

            x = start_x + c * (tile_w + tile_gap_x)
            y = start_y + r * (tile_h + tile_gap_y)

            product = row.to_dict()

            pdf.product_tile(
                x=x, y=y, w=tile_w, h=tile_h,
                product=product,
                show_price=show_price,
                show_sku=show_sku,
                show_desc=show_desc,
                show_attrs=show_attrs,
                currency=currency,
                img_timeout=img_timeout,
                img_retries=img_retries,
                dense=dense,
                add_link=True,
            )

            i += 1
            done_items += 1
            if total_items:
                p = min(1.0, done_items / total_items)
                progress.progress(p)
                status.info(f"Rendering PDF… {done_items:,}/{total_items:,}")

    status.success("PDF ready.")
    progress.progress(1.0)

    b = pdf_bytes(pdf)  # ensures bytes not bytearray
    st.download_button(
        "Download PDF",
        data=b,
        file_name=f"bkosher_catalog_{slugify(title) or 'catalog'}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
    st.caption("Tip: if links don’t work in your viewer, test opening the PDF in Chrome/Edge.")


if __name__ == "__main__":
    main()