# app.py — B-Kosher Catalog Builder (Streamlit Cloud safe)
# Fixes in this build:
# ✅ NO status=any (prevents Woo 500 on your server)
# ✅ Uses HTTP Basic Auth (keys not exposed in URLs/logs/errors)
# ✅ Resumable API import (continues from last saved page/phase)
# ✅ Parent category selection works (select "Alcohol" includes "Alcohol > Beer", etc.)
# ✅ Explicit checkbox to INCLUDE/EXCLUDE PRIVATE products in the PDF
# ✅ 6×6 (Compact) text containment: hard caps + priority layout (no overflow)
# ✅ Download button always gets bytes (no bytearray crash)
#
# requirements.txt:
# streamlit==1.31.1
# requests==2.31.0
# Pillow==10.4.0
# fpdf2==2.7.9

import io
import re
import csv
import json
import time
import math
import html as htmlmod
import hashlib
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.auth import HTTPBasicAuth
import streamlit as st
from PIL import Image

from fpdf import FPDF
from fpdf.enums import XPos, YPos


# ============================
# VERSION
# ============================
APP_VERSION = "2026-02-08-build-no-status-any-basic-auth-private-pdf-6x6-fix"


# ============================
# BRANDING
# ============================
BRAND_RED_HEX = "#C8102E"
BRAND_BLUE_HEX = "#004C97"

DEFAULT_TITLE = "B-Kosher Product Catalog"
DEFAULT_SITE = "www.b-kosher.co.uk"
DEFAULT_BASE_URL = "https://www.b-kosher.co.uk"

# Repo root file:
LOGO_PNG_PATH = "B-kosher logo high q.png"


# ============================
# STREAMLIT PAGE
# ============================
st.set_page_config(page_title="B-Kosher Catalog Builder", layout="wide")

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
  section[data-testid="stSidebar"] {{ border-right: 3px solid {BRAND_BLUE_HEX}; }}
  div[data-testid="stProgressBar"] > div > div {{ background-color: {BRAND_BLUE_HEX} !important; }}
  .bk_log {{
    border-radius: 14px;
    padding: 10px 12px;
    background: rgba(0,0,0,0.03);
    border: 1px solid rgba(0,0,0,0.06);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
    font-size: 0.85rem;
    white-space: pre-wrap;
    max-height: 360px;
    overflow: auto;
  }}
  .bk_hint {{ color: #6b7280; font-size: 0.92rem; }}
</style>
""",
    unsafe_allow_html=True,
)


# ============================
# SECRETS
# ============================
def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


APP_PASSWORD = get_secret("APP_PASSWORD", "")
WC_URL = get_secret("WC_URL", "")
WC_CK = get_secret("WC_CK", "")
WC_CS = get_secret("WC_CS", "")


# ============================
# LOGIN
# ============================
def login_gate():
    if not APP_PASSWORD:
        st.error("APP_PASSWORD is not set in Streamlit secrets.")
        st.stop()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return

    st.title("B-Kosher Catalog Login")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if pw == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


# ============================
# BYTES SAFETY
# ============================
def strict_bytes(x) -> bytes:
    if x is None:
        return b""
    if isinstance(x, bytes):
        return x
    if isinstance(x, bytearray):
        return bytes(x)
    if isinstance(x, str):
        return x.encode("latin-1", "ignore")
    return bytes(x)


# ============================
# TEXT HELPERS
# ============================
def pdf_safe(s: str) -> str:
    if s is None:
        return ""
    s = htmlmod.unescape(str(s))  # &amp; -> &
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


def parse_money(v):
    s = safe_text(v)
    if not s:
        return None
    s = s.replace("£", "").replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def fmt_money(symbol: str, v):
    if v is None:
        return ""
    return f"{symbol}{float(v):.2f}"  # force 2dp


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def truncate_to_fit(pdf: FPDF, text: str, max_w: float) -> str:
    t = pdf_safe(text)
    if pdf.get_string_width(t) <= max_w:
        return t
    ell = "..."
    while t and pdf.get_string_width(t + ell) > max_w:
        t = t[:-1]
    return (t + ell) if t else ell


def wrap_lines(pdf: FPDF, text: str, max_w: float, max_lines: int) -> list[str]:
    text = pdf_safe(text).replace("\n", " ").strip()
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    for _ in range(max_lines):
        if not words:
            break
        line = ""
        while words:
            cand = (line + " " + words[0]).strip()
            if pdf.get_string_width(cand) <= max_w:
                line = cand
                words.pop(0)
            else:
                break
        if not line:
            line = truncate_to_fit(pdf, words.pop(0), max_w)
        lines.append(line)

    if words and lines:
        last = lines[-1]
        ell = "..."
        while last and pdf.get_string_width(last + ell) > max_w:
            last = last[:-1]
        lines[-1] = (last + ell) if last else ell

    return lines


# ============================
# CACHE DIRS
# ============================
IMAGE_CACHE_DIR = Path("./image_cache")
IMAGE_CACHE_DIR.mkdir(exist_ok=True)

API_CACHE_DIR = Path("./api_cache")
API_CACHE_DIR.mkdir(exist_ok=True)

API_CATEGORIES_CACHE = API_CACHE_DIR / "categories.json"
API_PRODUCTS_CACHE = API_CACHE_DIR / "products_resumable.json"


# ============================
# IMAGE CACHE
# ============================
def cache_path_for_url(url: str) -> Path:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return IMAGE_CACHE_DIR / f"{h}.jpg"


def read_image_cache(url: str):
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


def resize_to_jpeg_bytes(raw: bytes, max_px: int = 900, quality: int = 82):
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


def download_with_retries(url: str, *, timeout: int, retries: int, backoff: float, append_log=None):
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
            cooked = resize_to_jpeg_bytes(r.content)
            final = cooked if cooked else r.content
            write_image_cache(url, final)
            return final
        except Exception as e:
            if append_log and attempt == retries:
                append_log(f"Image failed: {url[:80]}… ({type(e).__name__})")
            continue
    return None


# ============================
# RESUMABLE API CACHE
# ============================
def cache_load_products():
    if not API_PRODUCTS_CACHE.exists():
        return {
            "fetched_at": "",
            "in_progress": False,
            "phase": "publish",
            "next_page": 1,
            "include_private_fetch": False,
            "per_page": 25,
            "data": [],
        }
    try:
        payload = json.loads(API_PRODUCTS_CACHE.read_text(encoding="utf-8"))
        return {
            "fetched_at": payload.get("fetched_at", "") or "",
            "in_progress": bool(payload.get("in_progress", False)),
            "phase": payload.get("phase", "publish") or "publish",
            "next_page": int(payload.get("next_page", 1) or 1),
            "include_private_fetch": bool(payload.get("include_private_fetch", False)),
            "per_page": int(payload.get("per_page", 25) or 25),
            "data": payload.get("data", []) or [],
        }
    except Exception:
        return {
            "fetched_at": "",
            "in_progress": False,
            "phase": "publish",
            "next_page": 1,
            "include_private_fetch": False,
            "per_page": 25,
            "data": [],
        }


def cache_save_products(state: dict):
    state = dict(state)
    state["fetched_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    API_PRODUCTS_CACHE.write_text(json.dumps(state), encoding="utf-8")


def cache_clear_products():
    try:
        API_PRODUCTS_CACHE.unlink(missing_ok=True)
    except Exception:
        pass


def cache_load_categories():
    if not API_CATEGORIES_CACHE.exists():
        return None, ""
    try:
        payload = json.loads(API_CATEGORIES_CACHE.read_text(encoding="utf-8"))
        data = payload.get("data")
        at = payload.get("fetched_at", "")
        if isinstance(data, list):
            return data, at
    except Exception:
        pass
    return None, ""


def cache_save_categories(data_list: list):
    payload = {
        "fetched_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": data_list,
    }
    API_CATEGORIES_CACHE.write_text(json.dumps(payload), encoding="utf-8")


# ============================
# WOO API (Basic Auth, no keys in URLs/logs)
# ============================
def wc_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "bkosher-catalog/1.0"})
    return s


def wc_fetch_categories(base_url: str, ck: str, cs: str, *, timeout=30, append_log=None, set_progress=None):
    base_url = base_url.rstrip("/")
    endpoint = f"{base_url}/wp-json/wc/v3/products/categories"
    session = wc_session()
    auth = HTTPBasicAuth(ck, cs)

    per_page = 100
    page = 1
    out = []
    retries = 6

    while True:
        if append_log:
            append_log(f"API: categories page {page}…")
        if set_progress:
            set_progress(min(0.30, 0.05 + page * 0.02))

        params = {"per_page": per_page, "page": page, "hide_empty": "false"}

        last_exc = None
        for attempt in range(retries + 1):
            if attempt > 0:
                time.sleep(min(0.7 * (2 ** (attempt - 1)), 10.0))
            try:
                r = session.get(endpoint, params=params, auth=auth, timeout=timeout)

                if r.status_code in (401, 403):
                    raise RuntimeError("Unauthorized when listing categories. Check REST API permissions for the key.")

                if r.status_code >= 500:
                    last_exc = RuntimeError(f"Server error ({r.status_code}) while fetching categories.")
                    continue

                if r.status_code >= 400:
                    raise RuntimeError(f"API error ({r.status_code}) while fetching categories.")

                batch = r.json()
                if not batch:
                    return out

                out.extend(batch)
                if len(batch) < per_page:
                    return out

                page += 1
                last_exc = None
                break

            except Exception as e:
                last_exc = e
                continue

        if last_exc is not None:
            # return what we have rather than hard-fail
            return out


def build_category_paths(categories: list[dict]):
    by_id = {}
    for c in categories:
        try:
            cid = int(c.get("id"))
        except Exception:
            continue
        by_id[cid] = {
            "id": cid,
            "name": safe_text(c.get("name")),
            "parent": int(c.get("parent") or 0),
        }

    paths = {}

    def resolve(cid: int, depth=0):
        if cid in paths:
            return paths[cid]
        if depth > 18:
            return []
        node = by_id.get(cid)
        if not node:
            return []
        name = node["name"] or ""
        parent = node["parent"]
        if parent and parent in by_id:
            ppath = resolve(parent, depth + 1)
            out = [x for x in ppath if x] + ([name] if name else [])
        else:
            out = [name] if name else []
        paths[cid] = out
        return out

    for cid in list(by_id.keys()):
        resolve(cid)

    return paths


def category_path_to_str(path: list[str]) -> str:
    return " > ".join([p for p in path if p])


def wc_fetch_products_resumable(
    base_url: str,
    ck: str,
    cs: str,
    *,
    include_private_fetch: bool,
    resume: bool,
    timeout: int,
    per_page: int,
    append_log=None,
    set_progress=None,
    set_count=None,
):
    """
    Two-pass fetch:
      1) status=publish
      2) status=private (optional)
    NEVER uses status=any (your server returns 500 for that).
    Uses Basic Auth so keys don't appear in URLs/logs.
    Saves progress after every page for resume.
    """
    if not (base_url and ck and cs):
        raise RuntimeError("Woo API secrets missing. Set WC_URL, WC_CK, WC_CS in Streamlit Secrets.")

    base_url = base_url.rstrip("/")
    endpoint = f"{base_url}/wp-json/wc/v3/products"
    session = wc_session()
    auth = HTTPBasicAuth(ck, cs)

    if resume:
        state = cache_load_products()
    else:
        state = {
            "fetched_at": "",
            "in_progress": False,
            "phase": "publish",
            "next_page": 1,
            "include_private_fetch": include_private_fetch,
            "per_page": per_page,
            "data": [],
        }

    # If settings changed, start fresh to avoid mixing incompatible states
    if resume:
        if bool(state.get("include_private_fetch")) != bool(include_private_fetch) or int(state.get("per_page", per_page)) != int(per_page):
            if append_log:
                append_log("Fetch settings changed (private/per_page). Starting fresh import.")
            state = {
                "fetched_at": "",
                "in_progress": False,
                "phase": "publish",
                "next_page": 1,
                "include_private_fetch": include_private_fetch,
                "per_page": per_page,
                "data": [],
            }

    out = state.get("data", []) or []
    phase = state.get("phase", "publish") or "publish"
    page = int(state.get("next_page", 1) or 1)

    # De-dupe
    seen_ids = set()
    for p in out:
        pid = p.get("id")
        if pid is not None:
            seen_ids.add(pid)

    phases = ["publish"] + (["private"] if include_private_fetch else [])
    start_phase_index = phases.index(phase) if phase in phases else 0

    cache_save_products({
        "in_progress": True,
        "phase": phase,
        "next_page": page,
        "include_private_fetch": include_private_fetch,
        "per_page": per_page,
        "data": out,
    })

    retries = 10

    # adaptive backoff; if repeated 500s, reduce per_page temporarily
    adaptive_per_page = int(per_page)

    for phase_index in range(start_phase_index, len(phases)):
        status_value = phases[phase_index]
        if phase_index != start_phase_index:
            page = 1

        while True:
            if append_log:
                append_log(f"API: products page {page} (status={status_value}, per_page={adaptive_per_page})…")

            if set_progress:
                base = 0.10 if status_value == "publish" else 0.60
                set_progress(min(0.95, base + page * 0.01))

            params = {
                "per_page": adaptive_per_page,
                "page": page,
                "status": status_value,
            }

            last_exc = None
            for attempt in range(retries + 1):
                if attempt > 0:
                    time.sleep(min(0.8 * (2 ** (attempt - 1)), 25.0))
                try:
                    r = session.get(endpoint, params=params, auth=auth, timeout=timeout)

                    if r.status_code in (401, 403):
                        raise RuntimeError("Unauthorized when listing products. Ensure this key can 'read' products.")

                    if r.status_code >= 500:
                        # Common on some hosts: 500 for certain per_page / load spikes
                        last_exc = RuntimeError(f"Server error ({r.status_code}) while fetching products.")
                        # After a couple failed attempts, reduce per_page to reduce server load
                        if attempt >= 2 and adaptive_per_page > 10:
                            adaptive_per_page = 10
                            if append_log:
                                append_log("API: server errors detected — reducing per_page to 10 and retrying…")
                        continue

                    if r.status_code >= 400:
                        # Don't show URL (no keys anyway, but keep clean)
                        raise RuntimeError(f"API request failed ({r.status_code}) while fetching products (status={status_value}, page={page}).")

                    batch = r.json()
                    if not batch:
                        break

                    for item in batch:
                        pid = item.get("id")
                        if pid is None or pid not in seen_ids:
                            out.append(item)
                            if pid is not None:
                                seen_ids.add(pid)

                    if set_count:
                        set_count(len(out))

                    cache_save_products({
                        "in_progress": True,
                        "phase": status_value,
                        "next_page": page + 1,
                        "include_private_fetch": include_private_fetch,
                        "per_page": per_page,
                        "data": out,
                    })

                    if len(batch) < adaptive_per_page:
                        break

                    page += 1
                    last_exc = None
                    break

                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    last_exc = e
                    continue
                except Exception as e:
                    last_exc = e
                    break

            if last_exc is not None:
                # Save resume point
                cache_save_products({
                    "in_progress": True,
                    "phase": status_value,
                    "next_page": page,
                    "include_private_fetch": include_private_fetch,
                    "per_page": per_page,
                    "data": out,
                })
                # Helpful message; no URL or keys
                raise RuntimeError(
                    "API import interrupted.\n"
                    "Progress was saved — press Load again to continue.\n"
                    f"Details: {type(last_exc).__name__}: {str(last_exc)[:220]}"
                )

            if append_log:
                append_log(f"API: phase {status_value} complete.")
            break

    cache_save_products({
        "in_progress": False,
        "phase": "publish",
        "next_page": 1,
        "include_private_fetch": include_private_fetch,
        "per_page": per_page,
        "data": out,
    })
    if set_progress:
        set_progress(1.0)
    return out


# ============================
# NORMALIZE PRODUCTS (parent category selection)
# ============================
def extract_brand_from_attributes(attrs: list[tuple[str, str]]) -> str:
    for n, v in attrs:
        nn = safe_text(n).lower()
        if nn in ("brand", "manufacturer", "make"):
            return safe_text(v)
    return ""


def wc_to_product(p: dict, cat_paths: dict[int, list[str]]):
    cat_ids = []
    cat_names = []
    for c in (p.get("categories") or []):
        try:
            cid = int(c.get("id"))
            cat_ids.append(cid)
        except Exception:
            pass
        n = safe_text(c.get("name"))
        if n:
            cat_names.append(n)

    full_paths = []
    for cid in cat_ids:
        pl = cat_paths.get(cid) or []
        if pl:
            full_paths.append(category_path_to_str(pl))

    if not full_paths and cat_names:
        full_paths = [cat_names[0]]

    expanded = set()
    for full in full_paths:
        parts = [x.strip() for x in full.split(">")]
        parts = [x for x in parts if x]
        for i in range(1, len(parts) + 1):
            expanded.add(" > ".join(parts[:i]))

    category_paths = sorted(expanded, key=lambda s: s.lower())

    primary_path = "Other"
    if full_paths:
        primary_path = max(full_paths, key=lambda s: len(s.split(">")))

    attrs = []
    for a in (p.get("attributes") or []):
        n = safe_text(a.get("name"))
        opts = a.get("options") or []
        opts_s = ", ".join([safe_text(x) for x in opts if safe_text(x)])
        if n and opts_s:
            attrs.append((n, opts_s))

    brand = extract_brand_from_attributes(attrs)

    reg = parse_money(p.get("regular_price"))
    sale = parse_money(p.get("sale_price"))
    on_sale = bool(p.get("on_sale")) or (sale is not None and reg is not None and sale < reg)

    image_urls = []
    for im in (p.get("images") or []):
        u = safe_text(im.get("src"))
        if u and u.startswith("http"):
            image_urls.append(u)

    stock_status = safe_text(p.get("stock_status")) or ""
    status = safe_text(p.get("status")) or ""

    return {
        "id": p.get("id"),
        "status": status,  # publish/private/draft/etc
        "name": safe_text(p.get("name")),
        "sku": safe_text(p.get("sku")),
        "brand": brand,
        "category_path": primary_path,         # grouping label
        "category_paths": category_paths,      # for filtering (parent works!)
        "categories_flat": cat_names,
        "short_desc": strip_html(p.get("short_description") or ""),
        "regular_price": reg,
        "sale_price": sale,
        "on_sale": on_sale,
        "attributes": attrs,
        "url": safe_text(p.get("permalink")) or "",
        "stock_status": stock_status,
        "_image_urls": image_urls,
        "_image_path": None,
    }


def load_products_and_categories(
    *,
    include_private_fetch: bool,
    timeout: int,
    force_refresh: bool,
    resume: bool,
    per_page: int,
    append_log=None,
    set_progress=None,
    set_count=None,
):
    cat_raw, cat_at = cache_load_categories()
    if cat_raw is None or force_refresh:
        if append_log:
            append_log("Loading category hierarchy…")
        cat_raw = wc_fetch_categories(WC_URL, WC_CK, WC_CS, timeout=timeout, append_log=append_log, set_progress=set_progress) or []
        cache_save_categories(cat_raw)
        cat_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cat_paths = build_category_paths(cat_raw or [])

    if not force_refresh and resume:
        state = cache_load_products()
        if state.get("data") and not state.get("in_progress") and bool(state.get("include_private_fetch")) == bool(include_private_fetch):
            if append_log:
                append_log(f"Loaded {len(state['data'])} products from completed cache ({state['fetched_at']}).")
            normalized = [wc_to_product(x, cat_paths) for x in state["data"]]
            return normalized, state["fetched_at"], "disk_cache", cat_at

    raw = wc_fetch_products_resumable(
        WC_URL, WC_CK, WC_CS,
        include_private_fetch=include_private_fetch,
        resume=resume,
        timeout=timeout,
        per_page=per_page,
        append_log=append_log,
        set_progress=set_progress,
        set_count=set_count,
    )
    fetched_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    normalized = [wc_to_product(x, cat_paths) for x in raw]
    return normalized, fetched_at, "api_or_resume", cat_at


# ============================
# CSV BACKUP (kept)
# ============================
def read_csv_rows(uploaded_file):
    b = uploaded_file.getvalue()
    try:
        text = b.decode("utf-8")
    except Exception:
        text = b.decode("latin-1", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def best_key(keys, candidates):
    lower = {k.lower(): k for k in keys}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def csv_to_products(rows, base_url: str):
    if not rows:
        return []
    keys = list(rows[0].keys())

    col_id = best_key(keys, ["ID", "Id"])
    col_name = best_key(keys, ["Name", "Product name", "Title"])
    col_sku = best_key(keys, ["SKU"])
    col_brand = best_key(keys, ["Brand", "brand"])
    col_reg = best_key(keys, ["Regular price", "Regular Price", "Price"])
    col_sale = best_key(keys, ["Sale price", "Sale Price"])
    col_cats = best_key(keys, ["Categories", "Category"])
    col_desc = best_key(keys, ["Short description", "Short Description", "Description"])
    col_imgs = best_key(keys, ["Images", "Image", "Image URLs"])
    col_url = best_key(keys, ["Permalink", "Product URL", "URL", "Link"])
    col_stock = best_key(keys, ["Stock status", "Stock Status", "stock_status"])
    col_status = best_key(keys, ["Status", "Post status", "Visibility", "Published"])

    products = []
    for row in rows:
        name = safe_text(row.get(col_name)) if col_name else ""
        if not name:
            continue

        sku = safe_text(row.get(col_sku)) if col_sku else ""
        brand = safe_text(row.get(col_brand)) if col_brand else ""
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

        cats_flat = []
        if col_cats:
            cats_flat = [c.strip() for c in safe_text(row.get(col_cats)).split(",") if c.strip()]
        primary = cats_flat[0] if cats_flat else "Other"

        expanded = set()
        if primary and ">" in primary:
            parts = [x.strip() for x in primary.split(">") if x.strip()]
            for i in range(1, len(parts) + 1):
                expanded.add(" > ".join(parts[:i]))
        elif primary:
            expanded.add(primary)

        category_paths = sorted(expanded, key=lambda s: s.lower())

        image_urls = []
        if col_imgs:
            s = safe_text(row.get(col_imgs))
            if s:
                found = re.findall(r"https?://\S+", s)
                for u in found:
                    u = u.strip().strip('",').strip("'").strip()
                    if u.startswith("http"):
                        image_urls.append(u)

        url = safe_text(row.get(col_url)) if col_url else ""
        if not url.startswith("http"):
            pid = safe_text(row.get(col_id)) if col_id else ""
            if pid.isdigit():
                url = f"{base_url.rstrip('/')}/?post_type=product&p={pid}"

        status = safe_text(row.get(col_status)) if col_status else ""
        status_l = status.lower()
        if "priv" in status_l:
            status = "private"
        elif "publish" in status_l or "public" in status_l or status_l == "1":
            status = "publish"

        products.append(
            {
                "id": safe_text(row.get(col_id)) if col_id else "",
                "status": status,
                "name": name,
                "sku": sku,
                "brand": brand,
                "category_path": primary,
                "category_paths": category_paths or ([primary] if primary else ["Other"]),
                "categories_flat": cats_flat,
                "short_desc": desc,
                "regular_price": reg,
                "sale_price": sale,
                "on_sale": bool(on_sale),
                "attributes": [],
                "url": url,
                "stock_status": stock_status,
                "_image_urls": image_urls,
                "_image_path": None,
            }
        )

    return products


# ============================
# CATEGORY OPTIONS
# ============================
def build_category_options_from_products(products: list[dict]) -> list[str]:
    all_paths = set()
    for p in products:
        for cp in (p.get("category_paths") or []):
            cp = safe_text(cp)
            if cp:
                all_paths.add(cp)
    return sorted(all_paths, key=lambda s: s.lower())


def product_matches_selected_categories(p: dict, selected: set[str]) -> bool:
    paths = p.get("category_paths") or []
    for pp in paths:
        pp = safe_text(pp)
        if not pp:
            continue
        for sel in selected:
            if pp == sel or pp.startswith(sel + " > "):
                return True
    return False


# ============================
# PDF
# ============================
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
        title_x = left
        try:
            self.image(self.logo_path, x=left, y=top, h=11)
            title_x = left + 38
        except Exception:
            pass

        self.set_text_color(*self.blue)
        self.set_font("Helvetica", "B", 13)
        self.set_xy(title_x, top + 1)
        self.cell(0, 8, self.catalog_title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 9)
        self.set_xy(-12 - 25, top + 2)
        self.cell(25, 8, f"Page {self.page_no()}", align="R")

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
        self.cell(0, 8, f"{self.brand_site} | {self.disclaimer}")


def group_products_by_category_path(products):
    groups = {}
    for p in products:
        key = safe_text(p.get("category_path")) or "Other"
        groups.setdefault(key, []).append(p)
    out = []
    for k in sorted(groups.keys(), key=lambda s: s.lower()):
        items = groups[k]
        items.sort(key=lambda p: safe_text(p.get("name")).lower())
        out.append((k, items))
    return out


def make_catalog_pdf_bytes(
    products,
    *,
    title: str,
    orientation: str,   # Portrait / Landscape
    grid_mode: str,     # Standard / Compact
    currency_symbol: str,
    show_price: bool,
    show_sku: bool,
    show_desc: bool,
    show_attrs: bool,
    exclude_oos: bool,
    only_sale: bool,
    include_private_in_pdf: bool,
):
    today_str = datetime.date.today().strftime("%d %b %Y")
    disclaimer = pdf_safe(f"Prices correct as of {today_str}")
    orient_flag = "L" if orientation == "Landscape" else "P"

    pdf = CatalogPDF(
        title=title or DEFAULT_TITLE,
        logo_path=LOGO_PNG_PATH,
        brand_site=DEFAULT_SITE,
        disclaimer=disclaimer,
        orientation=orient_flag,
        unit="mm",
        format="A4",
    )

    # Final PDF filter
    working = []
    for p in products:
        status = safe_text(p.get("status")).lower()
        if (not include_private_in_pdf) and status == "private":
            continue
        if exclude_oos and safe_text(p.get("stock_status")).lower() == "outofstock":
            continue
        if only_sale and not bool(p.get("on_sale")):
            continue
        working.append(p)

    grouped = group_products_by_category_path(working)

    margin = 12
    header_space = 28
    footer_space = 18
    category_bar_h = 10
    gutter = 5 if grid_mode == "Compact" else 6

    # Grid rules
    if orientation == "Portrait":
        cols, rows = (6, 6) if grid_mode == "Compact" else (3, 3)
    else:
        # landscape keeps cards readable
        cols, rows = (8, 4) if grid_mode == "Compact" else (4, 3)

    usable_w = pdf.w - 2 * margin
    usable_h = pdf.h - header_space - footer_space - category_bar_h - 8
    card_w = (usable_w - (cols - 1) * gutter) / cols
    card_h = (usable_h - (rows - 1) * gutter) / rows

    blue_rgb = hex_to_rgb(BRAND_BLUE_HEX)
    red_rgb = hex_to_rgb(BRAND_RED_HEX)

    # Contents
    pdf.add_page()
    pdf.set_text_color(*blue_rgb)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(margin, 36)
    pdf.cell(0, 10, "Contents", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_text_color(55, 65, 81)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(margin)
    pdf.cell(0, 7, disclaimer, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    per_page = rows * cols
    page_no = 2
    first_page_for_group = {}
    for cat, items in grouped:
        if items:
            first_page_for_group[cat] = page_no
            page_no += math.ceil(len(items) / per_page)

    pdf.set_text_color(17, 24, 39)
    pdf.set_font("Helvetica", "", 11)
    y = 60
    for cat, items in grouped:
        if not items:
            continue
        if y > pdf.h - 30:
            pdf.add_page()
            y = 40
        pdf.set_xy(margin, y)
        pdf.cell(0, 7, pdf_safe(cat))
        pdf.set_xy(pdf.w - margin - 25, y)
        pdf.set_text_color(107, 114, 128)
        pdf.cell(25, 7, str(first_page_for_group.get(cat, "")), align="R")
        pdf.set_text_color(17, 24, 39)
        y += 8

    def clamp_draw_text_lines(x, y, bottom_y, line_h, lines, font_setter):
        cy = y
        for ln in lines:
            if cy + line_h > bottom_y:
                break
            font_setter()
            pdf.set_xy(x, cy)
            pdf.cell(0, line_h, ln)
            cy += line_h
        return cy

    # Pages
    for cat, items in grouped:
        if not items:
            continue

        idx = 0
        while idx < len(items):
            pdf.add_page()

            # Category divider bar
            pdf.set_fill_color(*blue_rgb)
            bar_y = header_space
            pdf.rect(margin, bar_y, pdf.w - 2 * margin, category_bar_h, style="F")
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 10.5 if grid_mode == "Compact" else 11)
            pdf.set_xy(margin + 4, bar_y + 2.2)
            pdf.cell(0, 6, pdf_safe(cat))

            start_y = header_space + category_bar_h + 6

            for r in range(rows):
                yy = start_y + r * (card_h + gutter)
                xx = margin

                for c in range(cols):
                    if idx >= len(items):
                        break
                    p = items[idx]
                    idx += 1

                    # Card border
                    pdf.set_draw_color(*blue_rgb)
                    pdf.set_line_width(0.4 if grid_mode == "Compact" else 0.5)
                    pdf.rect(xx, yy, card_w, card_h, style="D")

                    # Clickable card
                    url = safe_text(p.get("url"))
                    if url.startswith("http"):
                        pdf.link(x=xx, y=yy, w=card_w, h=card_h, link=url)

                    pad = 2.2 if grid_mode == "Compact" else 3.5
                    img_h = card_h * (0.54 if grid_mode == "Compact" else 0.48)
                    img_w = card_w - 2 * pad
                    img_x = xx + pad
                    img_y = yy + pad + 1.6

                    # Image
                    if p.get("_image_path"):
                        try:
                            pdf.image(p["_image_path"], x=img_x, y=img_y, w=img_w, h=img_h)
                        except Exception:
                            pass
                    else:
                        pdf.set_text_color(120, 120, 120)
                        pdf.set_font("Helvetica", "I", 6.3 if grid_mode == "Compact" else 8)
                        pdf.set_xy(xx, img_y + img_h / 2 - 2)
                        pdf.cell(card_w, 4, "No image", align="C")

                    # SALE badge
                    if bool(p.get("on_sale")):
                        badge_w = min(18, card_w * 0.45)
                        badge_h = 6
                        bx = xx + card_w - badge_w - 1.1
                        by = yy + 1.1
                        pdf.set_fill_color(*red_rgb)
                        pdf.rect(bx, by, badge_w, badge_h, style="F")
                        pdf.set_text_color(255, 255, 255)
                        pdf.set_font("Helvetica", "B", 7.3)
                        pdf.set_xy(bx, by + 1.3)
                        pdf.cell(badge_w, 3.4, "SALE", align="C")

                    # Text bounds
                    tx = xx + pad
                    max_w = card_w - 2 * pad
                    ycur = img_y + img_h + (1.8 if grid_mode == "Compact" else 2.2)
                    bottom = yy + card_h - pad

                    # Compact mode: strict priority layout to prevent messy text
                    compact = (grid_mode == "Compact")

                    # Font sizes
                    name_fs = 6.6 if compact else 9.5
                    price_fs = 7.0 if compact else 10
                    meta_fs = 5.6 if compact else 8.5
                    desc_fs = 5.3 if compact else 7.6

                    # 1) NAME
                    def set_name_font():
                        pdf.set_text_color(0, 0, 0)
                        pdf.set_font("Helvetica", "B", name_fs)

                    name_max_lines = 1 if compact else 2
                    name_lines = wrap_lines(pdf, safe_text(p.get("name")), max_w, max_lines=name_max_lines)
                    ycur = clamp_draw_text_lines(
                        tx, ycur, bottom, 3.2 if compact else 5.0, name_lines, set_name_font
                    )
                    ycur += 0.4 if compact else 0.6

                    # 2) PRICE (always if enabled)
                    if show_price and ycur + 3.8 <= bottom:
                        reg = p.get("regular_price")
                        sale = p.get("sale_price")
                        on_sale = bool(p.get("on_sale")) and sale is not None and reg is not None and sale < reg

                        if on_sale:
                            pdf.set_text_color(*red_rgb)
                            pdf.set_font("Helvetica", "B", price_fs)
                            pdf.set_xy(tx, ycur)
                            pdf.cell(0, 3.8 if compact else 4.8, pdf_safe(fmt_money(currency_symbol, sale)))

                            # in Compact mode we DO NOT draw the struck-through regular (too cramped)
                            if not compact:
                                pdf.set_text_color(107, 114, 128)
                                pdf.set_font("Helvetica", "", meta_fs)
                                reg_txt = pdf_safe(fmt_money(currency_symbol, reg))
                                rx = tx + 18
                                pdf.set_xy(rx, ycur + 0.6)
                                pdf.cell(0, 3.6, reg_txt)
                                wtxt = pdf.get_string_width(reg_txt)
                                pdf.set_draw_color(107, 114, 128)
                                pdf.set_line_width(0.25)
                                pdf.line(rx, ycur + 2.2, rx + wtxt, ycur + 2.2)

                            ycur += 4.0 if compact else 5.3

                        else:
                            if reg is not None:
                                pdf.set_text_color(*red_rgb)
                                pdf.set_font("Helvetica", "B", price_fs)
                                pdf.set_xy(tx, ycur)
                                pdf.cell(0, 3.8 if compact else 4.8, pdf_safe(fmt_money(currency_symbol, reg)))
                                ycur += 4.0 if compact else 5.3

                        pdf.set_text_color(0, 0, 0)

                    # 3) SKU (compact only if there is room + user enabled)
                    if show_sku and (not compact) and ycur + 3.6 <= bottom:
                        sku = safe_text(p.get("sku"))
                        if sku:
                            pdf.set_text_color(31, 41, 55)
                            pdf.set_font("Helvetica", "", meta_fs)
                            line = truncate_to_fit(pdf, f"SKU: {sku}", max_w)
                            pdf.set_xy(tx, ycur)
                            pdf.cell(0, 3.6, pdf_safe(line))
                            ycur += 3.8

                    # 4) META LINE: Brand OR first attribute (compact shows only 1 line max)
                    if ycur + 3.2 <= bottom:
                        meta_line = ""
                        if safe_text(p.get("brand")):
                            meta_line = f"Brand: {safe_text(p.get('brand'))}"
                        elif show_attrs and (p.get("attributes") or []):
                            an, av = (p.get("attributes") or [("", "")])[0]
                            an = safe_text(an)
                            av = safe_text(av)
                            if an and av:
                                meta_line = f"{an}: {av}"

                        if meta_line:
                            pdf.set_text_color(55, 65, 81)
                            pdf.set_font("Helvetica", "", meta_fs)
                            meta_line = truncate_to_fit(pdf, meta_line, max_w)
                            pdf.set_xy(tx, ycur)
                            pdf.cell(0, 3.2 if compact else 3.6, pdf_safe(meta_line))
                            ycur += 3.2 if compact else 3.8

                    # 5) DESCRIPTION (never in compact unless user forces it; even then clamp to 1 line)
                    if show_desc and ycur + 3.0 <= bottom:
                        desc = strip_html(p.get("short_desc"))
                        if desc:
                            max_lines = 1 if compact else 2

                            def set_desc_font():
                                pdf.set_text_color(75, 85, 99)
                                pdf.set_font("Helvetica", "", desc_fs)

                            desc_lines = wrap_lines(pdf, desc, max_w, max_lines=max_lines)
                            ycur = clamp_draw_text_lines(tx, ycur, bottom, 3.0 if compact else 3.6, desc_lines, set_desc_font)

                    xx += card_w + gutter

    out = pdf.output()
    return strict_bytes(out)


# ============================
# APP STATE + LOGIN
# ============================
login_gate()

if "step" not in st.session_state:
    st.session_state.step = 1
if "products_raw" not in st.session_state:
    st.session_state.products_raw = []
if "products_filtered" not in st.session_state:
    st.session_state.products_filtered = []
if "_settings" not in st.session_state:
    st.session_state._settings = {}


# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.caption(f"App version: **{APP_VERSION}**")

    try:
        st.image(LOGO_PNG_PATH, width=200)
    except Exception:
        st.warning(f"Logo missing. Upload '{LOGO_PNG_PATH}' to repo root.")

    if WC_URL and WC_CK and WC_CS:
        st.success("Woo API configured")
    else:
        st.error("Woo API secrets missing (WC_URL / WC_CK / WC_CS)")

    state = cache_load_products()
    if state.get("data"):
        st.info(
            f"Cache: {len(state['data']):,} products "
            f"• in_progress={state['in_progress']} "
            f"• phase={state['phase']} "
            f"• next_page={state['next_page']}"
        )

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    if st.button("Reset app state"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ============================
# MAIN UI
# ============================
st.title("Customer Catalog Builder")
st.caption("Default = WooCommerce API. CSV upload is a backup option.")
st.markdown(
    "<div class='bk_hint'>Streamlit Cloud may pause when your iPhone browser is backgrounded. Imports are resumable and continue next run.</div>",
    unsafe_allow_html=True,
)

# ============================
# STEP 1 — SOURCE
# ============================
st.subheader("Step 1 — Choose data source")
data_source = st.radio("Source", ["WooCommerce API", "CSV Upload"], index=0, horizontal=True)

if st.button("Continue → Load products"):
    st.session_state.step = 2
    st.rerun()

# ============================
# STEP 2 — LOAD
# ============================
if st.session_state.step >= 2:
    st.subheader("Step 2 — Load products")

    if data_source == "WooCommerce API":
        api_timeout = st.slider("API timeout (seconds)", 10, 90, 45)

        include_private_fetch = st.checkbox(
            "Include private/unpublished products (requires API user permission)",
            value=False,
            help="This fetches private products in a separate pass (status=private). No status=any is used.",
        )

        resume_import = st.checkbox(
            "Resume import if previously interrupted",
            value=True,
            help="If Cloud restarted mid-import, continue from the last saved page.",
        )

        per_page = st.selectbox("API page size (smaller = more reliable)", [10, 25, 50], index=1)

        c1, c2, c3 = st.columns(3)
        load_btn = c1.button("Load (use cache / resume)")
        refresh_btn = c2.button("Start fresh (clear progress)")
        clear_img_cache = c3.button("Clear image cache")

        if clear_img_cache:
            try:
                for p in IMAGE_CACHE_DIR.glob("*.jpg"):
                    p.unlink(missing_ok=True)
                st.success("Image cache cleared.")
            except Exception:
                st.warning("Could not clear some cached images.")

        if refresh_btn:
            cache_clear_products()
            st.success("Progress cleared. Now press Load.")
            st.stop()

        if load_btn:
            prog = st.progress(0.0)
            status = st.empty()
            count_box = st.empty()
            log_box = st.empty()
            logs = []
            t0 = time.time()

            def append_log(msg: str):
                logs.append(f"[{time.time()-t0:6.1f}s] {msg}")
                log_box.markdown(
                    "<div class='bk_log'>" + htmlmod.escape("\n".join(logs[-40:])) + "</div>",
                    unsafe_allow_html=True,
                )

            def set_progress(v: float):
                prog.progress(max(0.0, min(1.0, float(v))))

            def set_count(n: int):
                count_box.info(f"Products loaded so far: {n:,}")

            try:
                status.info("Loading products (resumable)…")
                append_log("NOTE: This build does NOT use status=any anywhere.")

                products, fetched_at, source, cat_at = load_products_and_categories(
                    include_private_fetch=bool(include_private_fetch),
                    timeout=int(api_timeout),
                    force_refresh=False,
                    resume=bool(resume_import),
                    per_page=int(per_page),
                    append_log=append_log,
                    set_progress=set_progress,
                    set_count=set_count,
                )

                status.success("Products loaded.")
                set_progress(1.0)

                st.session_state.products_raw = products
                st.session_state.step = 3
                st.success(f"Loaded {len(products):,} products • {source} • {fetched_at}")
                st.rerun()

            except Exception as e:
                # no URLs printed
                st.error(str(e))
                st.stop()

    else:
        base_url = st.text_input("Base URL", value=DEFAULT_BASE_URL)
        csv_file = st.file_uploader("Upload WooCommerce CSV export", type=["csv"])
        if st.button("Load from CSV", disabled=(csv_file is None)):
            rows = read_csv_rows(csv_file)
            products = csv_to_products(rows, base_url=base_url)
            if not products:
                st.error("No products found in CSV.")
                st.stop()
            st.session_state.products_raw = products
            st.session_state.step = 3
            st.success(f"Loaded {len(products):,} products from CSV.")
            st.rerun()

# ============================
# STEP 3 — FILTERS & LAYOUT
# ============================
if st.session_state.step >= 3:
    st.subheader("Step 3 — Filters & layout")
    products = st.session_state.products_raw

    title = st.text_input("Catalog title", value=DEFAULT_TITLE)
    orientation = st.selectbox("Page orientation", ["Portrait", "Landscape"], index=0)

    grid_mode = st.selectbox(
        "Products per page",
        ["Standard (3×3)", "Compact (6×6)"],
        index=0,
        help="Compact has strict text limits to prevent messy overflow.",
    )
    grid_mode_value = "Compact" if grid_mode.startswith("Compact") else "Standard"

    currency = st.text_input("Currency symbol", value="£")

    preset = st.selectbox("Image download preset", ["Reliable", "Normal", "Fast"], index=0)

    # Defaults: SKU & Description OFF
    show_price = st.checkbox("Show price", value=True)
    show_sku = st.checkbox("Show SKU", value=False)
    show_desc = st.checkbox("Show description", value=False)
    show_attrs = st.checkbox("Show attributes", value=True)

    exclude_oos = st.checkbox("Exclude out-of-stock", value=True)
    only_sale = st.checkbox("Only sale items", value=False)

    # ✅ Explicit private-in-PDF option (this was missing for you)
    include_private_in_pdf = st.checkbox(
        "Include PRIVATE (unpublished) products in the PDF",
        value=False,
        help="You must also have fetched private products in Step 2 for this to have any effect.",
    )
    private_loaded = sum(1 for p in products if safe_text(p.get("status")).lower() == "private")
    if private_loaded == 0:
        st.caption("Private products loaded: 0 (enable private fetching in Step 2 if needed).")
    else:
        st.caption(f"Private products loaded: {private_loaded}")

    # Category selection (parents selectable)
    tree_options = build_category_options_from_products(products)
    selected_paths = st.multiselect(
        "Categories (select a parent like 'Alcohol' to include all its children)",
        tree_options,
        default=[],
    )

    q = st.text_input("Search (name or SKU)", value="").strip().lower()

    filtered = products[:]

    if selected_paths:
        sset = set(selected_paths)
        filtered = [p for p in filtered if product_matches_selected_categories(p, sset)]

    if q:
        filtered = [
            p for p in filtered
            if (q in safe_text(p.get("name")).lower()) or (q in safe_text(p.get("sku")).lower())
        ]

    if exclude_oos:
        filtered = [p for p in filtered if safe_text(p.get("stock_status")).lower() != "outofstock"]

    if only_sale:
        filtered = [p for p in filtered if bool(p.get("on_sale"))]

    # Filter private out unless including in PDF
    if not include_private_in_pdf:
        filtered = [p for p in filtered if safe_text(p.get("status")).lower() != "private"]

    filtered.sort(key=lambda p: ((safe_text(p.get("category_path")) or "Other").lower(), safe_text(p.get("name")).lower()))

    st.session_state.products_filtered = filtered

    st.info(f"Selected products: {len(filtered):,}")

    if st.button("Continue → Generate PDF", disabled=(len(filtered) == 0)):
        st.session_state.step = 4
        st.session_state._settings = {
            "title": title,
            "orientation": orientation,
            "grid_mode": grid_mode_value,
            "currency": currency,
            "preset": preset,
            "show_price": bool(show_price),
            "show_sku": bool(show_sku),
            "show_desc": bool(show_desc),
            "show_attrs": bool(show_attrs),
            "exclude_oos": bool(exclude_oos),
            "only_sale": bool(only_sale),
            "include_private_in_pdf": bool(include_private_in_pdf),
        }
        st.rerun()

# ============================
# STEP 4 — GENERATE
# ============================
if st.session_state.step >= 4:
    st.subheader("Step 4 — Generate & download")
    filtered = st.session_state.products_filtered
    settings = st.session_state._settings

    preset = settings.get("preset", "Reliable")

    # Cloud-safe caps
    cloud_cap = 4
    if preset == "Reliable":
        dl_workers, dl_retries, dl_timeout, dl_backoff = 4, 10, 25, 0.9
    elif preset == "Normal":
        dl_workers, dl_retries, dl_timeout, dl_backoff = 6, 6, 18, 0.7
    else:
        dl_workers, dl_retries, dl_timeout, dl_backoff = 10, 3, 12, 0.5
    dl_workers = min(dl_workers, cloud_cap)

    progress = st.progress(0.0)
    status = st.empty()
    log_box = st.empty()
    logs = []
    t0 = time.time()

    def append_log(msg: str):
        logs.append(f"[{time.time()-t0:6.1f}s] {msg}")
        log_box.markdown("<div class='bk_log'>" + htmlmod.escape("\n".join(logs[-40:])) + "</div>", unsafe_allow_html=True)

    status.info("Stage 1/2 — Downloading images…")
    append_log(f"Preset={preset} workers={dl_workers} retries={dl_retries}")

    items = [p for p in filtered if (p.get("_image_urls") or [])]
    total = max(1, len(items))
    done = 0
    ok = 0

    def dl_task(p: dict):
        urls = p.get("_image_urls") or []
        for u in urls:
            cached_path = cache_path_for_url(u)
            if cached_path.exists():
                return p, u, True
            b = download_with_retries(u, timeout=dl_timeout, retries=dl_retries, backoff=dl_backoff, append_log=append_log)
            if b:
                return p, u, True
        return p, None, False

    with ThreadPoolExecutor(max_workers=dl_workers) as ex:
        futures = [ex.submit(dl_task, p) for p in items]
        for fut in as_completed(futures):
            p, u, success = fut.result()
            if success and u:
                p["_image_path"] = str(cache_path_for_url(u))
                ok += 1
            done += 1
            progress.progress(min(0.75, (done / total) * 0.75))
            if done % 50 == 0 or done == total:
                append_log(f"Images: {done}/{total} ok={ok} missing={done-ok}")

    status.info("Stage 2/2 — Building PDF…")
    progress.progress(0.85)

    pdf_bytes = make_catalog_pdf_bytes(
        filtered,
        title=settings.get("title", DEFAULT_TITLE),
        orientation=settings.get("orientation", "Portrait"),
        grid_mode=settings.get("grid_mode", "Standard"),
        currency_symbol=settings.get("currency", "£"),
        show_price=bool(settings.get("show_price", True)),
        show_sku=bool(settings.get("show_sku", False)),
        show_desc=bool(settings.get("show_desc", False)),
        show_attrs=bool(settings.get("show_attrs", True)),
        exclude_oos=bool(settings.get("exclude_oos", True)),
        only_sale=bool(settings.get("only_sale", False)),
        include_private_in_pdf=bool(settings.get("include_private_in_pdf", False)),
    )

    pdf_bytes = strict_bytes(pdf_bytes)

    progress.progress(1.0)
    status.success("PDF ready.")

    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name="bkosher_catalog.pdf",
        mime="application/pdf",
    )

    st.caption("Tip: On iPhone, don’t background the tab during long runs. If it restarts, use Resume in Step 2.")