# app.py — B-Kosher Catalog Builder (Streamlit)
# - WooCommerce API (default) + CSV upload (backup)
# - Login gate (password from secrets)
# - Resumable API import with true totals + % progress + live logs
# - Category tree selection (parents + children + grandchildren) + search + sale-only
# - Include/exclude private products in *catalog filtering* (separate from API-load toggle)
# - PDF generator (fpdf2) with B-Kosher branding, clickable product tiles
# - Grid density: Standard (3×3) or Compact (6×5) (no text overflow)
# - Orientation: Portrait / Landscape (auto-tunes grid)
# - API import improved: per_page=100 option + continuous progress + auto-continue
# - PDF build improved:
#     - FAST image prefetch with concurrency (configurable > 32)
#     - PDF build progress bar (per part + overall)
#     - Large builds: splits into multiple PDFs, zips to disk (no RAM blowups)
#     - Optional merge-to-one PDF if pypdf/PyPDF2 is available (also disk-based)
#
# Secrets supported (Streamlit Cloud -> Settings -> Secrets):
#   APP_PASSWORD = "..."
#
# WooCommerce credentials (ANY ONE naming scheme works):
#   WC_URL = "https://www.b-kosher.co.uk"
#   WC_CK  = "ck_..."
#   WC_CS  = "cs_..."
#
# OR:
#   WC_BASE_URL = "https://www.b-kosher.co.uk"
#   WC_CONSUMER_KEY = "ck_..."
#   WC_CONSUMER_SECRET = "cs_..."
#
# OR legacy:
#   WOOCOMMERCE_URL = "..."
#
# Notes:
# - Streamlit Cloud will restart a script if it runs too long without yielding.
#   We therefore fetch in chunks and auto-rerun until finished.
# - iPhone backgrounding/sleep can pause the browser -> you may need to open the app again.
#   The import RESUMES from cache; it does not restart.

from __future__ import annotations

import hashlib
import html
import io
import json
import os
import re
import time
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth
from fpdf import FPDF  # fpdf2


# ----------------------------
# Brand config
# ----------------------------
BRAND_NAME = "B-Kosher"
DEFAULT_TITLE = "B-Kosher Product Catalog"
BRAND_SITE = "www.b-kosher.co.uk"

# Pantone screenshot provided:
BRAND_RED = "#C8102E"
BRAND_BLUE = "#004C97"

# PDF layout
PDF_MARGIN_MM = 10.0
HEADER_H_MM = 14.0
FOOTER_H_MM = 10.0
CATEGORY_BAR_H_MM = 8.0

# Cache dirs (Streamlit Cloud ephemeral but persists within container lifetime)
APP_DIR = Path(__file__).parent.resolve()
CACHE_DIR = APP_DIR / ".cache"
IMG_DIR = CACHE_DIR / "images"
API_CACHE_DIR = CACHE_DIR / "api"
OUT_DIR = CACHE_DIR / "output"
CACHE_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)
API_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Utilities
# ----------------------------
def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def safe_unescape(s: str) -> str:
    try:
        return html.unescape(s)
    except Exception:
        return s


def sanitize_latin1(s: Any) -> str:
    """Make text safe for built-in PDF fonts (latin-1)."""
    if s is None:
        return ""
    s = safe_unescape(str(s))

    # Replace problematic unicode punctuation with latin-1 friendly forms
    s = s.replace("\u2026", "...")  # ellipsis
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201C", '"').replace("\u201D", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\xa0", " ")

    # Strip HTML (defensive; descriptions sometimes leak tags)
    s = re.sub(r"<[^>]+>", "", s)

    # Clean stray control chars
    s = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", s)

    # Finally enforce latin-1
    return s.encode("latin-1", "ignore").decode("latin-1")


def money_2dp(val: Any) -> Optional[str]:
    if val is None:
        return None
    try:
        if isinstance(val, str):
            val = val.strip()
            if val == "":
                return None
        x = float(val)
        return f"{x:.2f}"
    except Exception:
        return None


def boolish(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y", "on", "publish", "published")


def now_utc_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ----------------------------
# Logging (live logs panel)
# ----------------------------
def log(msg: str) -> None:
    st.session_state.setdefault("logs", [])
    st.session_state["logs"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    if len(st.session_state["logs"]) > 800:
        st.session_state["logs"] = st.session_state["logs"][-800:]


def logs_text() -> str:
    return "\n".join(st.session_state.get("logs", []))


# ----------------------------
# Secrets + auth
# ----------------------------
def get_secret(key: str) -> Optional[str]:
    try:
        v = st.secrets.get(key)  # type: ignore[attr-defined]
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None
    except Exception:
        return None


def get_wc_creds() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Support multiple secret key naming schemes
    url = (
        get_secret("WC_URL")
        or get_secret("WC_BASE_URL")
        or get_secret("WOOCOMMERCE_URL")
        or None
    )
    ck = (
        get_secret("WC_CK")
        or get_secret("WC_CONSUMER_KEY")
        or None
    )
    cs = (
        get_secret("WC_CS")
        or get_secret("WC_CONSUMER_SECRET")
        or None
    )
    return url, ck, cs


def require_login() -> None:
    """Render login page and stop the app until authenticated."""
    app_pw = get_secret("APP_PASSWORD")

    # If no password set, allow locally (but warn)
    if not app_pw:
        st.warning("APP_PASSWORD is not set in secrets. Login is disabled.")
        st.session_state["authed"] = True
        return

    if st.session_state.get("authed"):
        return

    st.set_page_config(page_title="B-Kosher Catalog Builder", page_icon="🧾", layout="wide")
    st.title("Login")
    st.caption("Enter the password to access the catalog builder.")
    pw = st.text_input("Password", type="password")

    if st.button("Login", use_container_width=True):
        if pw == app_pw:
            st.session_state["authed"] = True
            st.success("Logged in.")
            time.sleep(0.2)
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


# ----------------------------
# Logo loader (local repo files)
# ----------------------------
def load_logo_bytes() -> Optional[bytes]:
    candidates = [
        APP_DIR / "B-kosher logo high q.png",
        APP_DIR / "Bkosher.png",
        APP_DIR / "bkosher.png",
        APP_DIR / "bkosher.svg",
        APP_DIR / "logo.png",
        APP_DIR / "logo.jpg",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            try:
                return p.read_bytes()
            except Exception:
                continue
    return None


# ----------------------------
# WooCommerce API fetcher (resumable)
# ----------------------------
@dataclass
class WCFetchResult:
    products: List[dict]
    categories: List[dict]
    total_products: int
    total_pages: int
    used_status: str


def wc_api_base(wc_url: str) -> str:
    return wc_url.rstrip("/") + "/wp-json/wc/v3"


def wc_get_json(
    session: requests.Session,
    url: str,
    auth: HTTPBasicAuth,
    params: dict,
    timeout: int,
    retries: int,
    backoff_base: float,
) -> Tuple[Optional[Any], Optional[requests.Response], Optional[str]]:
    """Return (data, response, error_text). Never raises JSON decode errors."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, auth=auth, timeout=timeout)
            if r.status_code != 200:
                snippet = (r.text or "")[:500]
                last_err = f"HTTP {r.status_code}: {snippet}"
                log(f"⚠️ API error {r.status_code} (attempt {attempt}/{retries})")
            else:
                try:
                    return r.json(), r, None
                except ValueError as e:
                    last_err = f"Invalid JSON: {e}"
                    log(f"⚠️ Invalid JSON (attempt {attempt}/{retries}) — retrying")
            time.sleep(backoff_base * attempt)
        except requests.RequestException as e:
            last_err = str(e)
            log(f"⚠️ Request error (attempt {attempt}/{retries}): {e}")
            time.sleep(backoff_base * attempt)

    return None, None, last_err


def cache_paths_for(status_key: str) -> Tuple[Path, Path]:
    meta = API_CACHE_DIR / f"meta_{status_key}.json"
    data = API_CACHE_DIR / f"products_{status_key}.jsonl"
    return meta, data


def load_cached_products(status_key: str) -> Tuple[List[dict], dict]:
    meta_path, data_path = cache_paths_for(status_key)
    meta: dict = {}
    items: List[dict] = []

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
        except Exception:
            meta = {}

    if data_path.exists():
        try:
            with data_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            items = []

    return items, meta


def append_cached_products(status_key: str, new_items: List[dict]) -> None:
    _, data_path = cache_paths_for(status_key)
    with data_path.open("a", encoding="utf-8") as f:
        for it in new_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def save_cache_meta(status_key: str, meta: dict) -> None:
    meta_path, _ = cache_paths_for(status_key)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")


def clear_cache(status_key: str) -> None:
    meta_path, data_path = cache_paths_for(status_key)
    if meta_path.exists():
        meta_path.unlink()
    if data_path.exists():
        data_path.unlink()


def fetch_all_categories(wc_url: str, ck: str, cs: str, timeout: int) -> List[dict]:
    base = wc_api_base(wc_url)
    session = requests.Session()
    auth = HTTPBasicAuth(ck, cs)

    cats: List[dict] = []
    page = 1
    per_page = 100

    while True:
        log(f"API: categories page {page}…")
        data, _, err = wc_get_json(
            session=session,
            url=f"{base}/products/categories",
            auth=auth,
            params={"per_page": per_page, "page": page, "hide_empty": False},
            timeout=timeout,
            retries=5,
            backoff_base=0.6,
        )
        if data is None:
            raise RuntimeError(f"Failed to fetch categories: {err}")
        if not isinstance(data, list):
            raise RuntimeError("Unexpected categories payload.")
        cats.extend(data)
        if len(data) < per_page:
            break
        page += 1

    log(f"Loaded {len(cats)} categories.")
    return cats


def probe_total_products(wc_url: str, ck: str, cs: str, timeout: int, status: str) -> int:
    """
    Get the TOTAL product count from X-WP-Total.
    IMPORTANT: We compute total_pages ourselves based on the chosen per_page,
    because X-WP-TotalPages depends on per_page and can cause “never finishes” loops
    if you probe with per_page=1 but fetch with per_page=100.
    """
    base = wc_api_base(wc_url)
    session = requests.Session()
    auth = HTTPBasicAuth(ck, cs)
    r = session.get(
        f"{base}/products",
        params={"per_page": 1, "page": 1, "status": status},
        auth=auth,
        timeout=timeout,
    )
    r.raise_for_status()
    total = int(r.headers.get("X-WP-Total", "0") or "0")
    return total


def resumable_fetch_products(
    wc_url: str,
    ck: str,
    cs: str,
    timeout: int,
    include_private: bool,
    per_page: int,
    max_pages_per_run: int,
    retries: int = 6,
) -> WCFetchResult:
    """
    Resumable fetch using disk cache:
      - meta stores last_completed_page, total_products, total_pages, status, per_page_used
      - products stored in jsonl append-only

    IMPORTANT:
    - If total_pages was computed using a different per_page, the import may never “finish”.
      We therefore store per_page_used in meta, and recompute total_pages when per_page changes.
    """
    status = "any" if include_private else "publish"
    status_key = f"status_{status}"
    base = wc_api_base(wc_url)
    session = requests.Session()
    auth = HTTPBasicAuth(ck, cs)

    cached_products, meta = load_cached_products(status_key)
    last_page = int(meta.get("last_completed_page", 0) or 0)
    total_products = int(meta.get("total_products", 0) or 0)
    total_pages = int(meta.get("total_pages", 0) or 0)
    per_page_used = int(meta.get("per_page_used", 0) or 0)

    # Ensure we have total_products
    if total_products <= 0:
        log("Probing total product count…")
        total_products = probe_total_products(wc_url, ck, cs, timeout, status=status)
        meta["total_products"] = total_products
        meta["status"] = status
        meta["created_utc"] = now_utc_str()

    # Ensure total_pages matches current per_page
    if per_page_used != per_page or total_pages <= 0:
        total_pages = int(math.ceil(total_products / max(per_page, 1)))
        meta["total_pages"] = total_pages
        meta["per_page_used"] = per_page
        meta["updated_utc"] = now_utc_str()
        save_cache_meta(status_key, meta)
        log(f"Totals: total_products={total_products:,} total_pages={total_pages:,} (per_page={per_page}, status={status})")

        # If per_page changed, last_page might no longer be correct; safest is to keep going
        # from last_page+1. (Cache is append-only; duplicates are possible if you changed per_page mid-run.
        # Best practice: press “Refresh cache” if you change per_page.)
        last_page = int(meta.get("last_completed_page", 0) or 0)

    start_page = max(1, last_page + 1)

    # Continuous progress UI
    progress_bar = st.progress(0.0)
    count_line = st.empty()
    pages_line = st.empty()

    def render_progress(done_count: int, current_page: int) -> None:
        denom = total_products if total_products > 0 else max(done_count, 1)
        pct = min(done_count / denom, 1.0)
        progress_bar.progress(pct)
        count_line.markdown(
            f"**Imported:** {done_count:,} / {total_products:,} products (**{pct*100:.1f}%**)"
        )
        pages_line.caption(f"Page {min(max(current_page,0), total_pages):,} / {total_pages:,} • per_page={per_page}")

    render_progress(len(cached_products), last_page)

    fetched_this_run = 0
    page = start_page

    while page <= total_pages and fetched_this_run < max_pages_per_run:
        if st.session_state.get("fetch_stop"):
            log("Stop flag set — halting fetch loop.")
            break

        log(f"API: products page {page} (status={status})…")

        data, _, err = wc_get_json(
            session=session,
            url=f"{base}/products",
            auth=auth,
            params={"per_page": per_page, "page": page, "status": status},
            timeout=timeout,
            retries=retries,
            backoff_base=0.8,
        )

        if data is None:
            log(f"⚠️ Skipping page {page} after retries: {err}")
            meta.setdefault("skipped_pages", [])
            meta["skipped_pages"].append(page)
            meta["last_completed_page"] = page
            meta["updated_utc"] = now_utc_str()
            save_cache_meta(status_key, meta)
            page += 1
            fetched_this_run += 1
            render_progress(len(cached_products), page - 1)
            continue

        if not isinstance(data, list):
            log(f"⚠️ Unexpected payload on page {page}; skipping")
            page += 1
            fetched_this_run += 1
            continue

        append_cached_products(status_key, data)
        cached_products.extend(data)

        meta["last_completed_page"] = page
        meta["updated_utc"] = now_utc_str()
        save_cache_meta(status_key, meta)

        fetched_this_run += 1
        render_progress(len(cached_products), page)
        page += 1

        time.sleep(0.06)

    cats = st.session_state.get("wc_categories") or []
    return WCFetchResult(
        products=cached_products,
        categories=cats,
        total_products=total_products,
        total_pages=total_pages,
        used_status=status,
    )


# ----------------------------
# Category tree helpers
# ----------------------------
def build_category_maps(
    categories: List[dict],
) -> Tuple[Dict[int, dict], Dict[int, List[int]], Dict[int, int]]:
    by_id: Dict[int, dict] = {}
    children: Dict[int, List[int]] = {}
    parent: Dict[int, int] = {}

    for c in categories:
        try:
            cid = int(c.get("id"))
        except Exception:
            continue
        by_id[cid] = c
        pid = int(c.get("parent") or 0)
        parent[cid] = pid
        children.setdefault(pid, []).append(cid)

    for pid, kids in children.items():
        kids.sort(key=lambda k: sanitize_latin1(by_id.get(k, {}).get("name", "")).lower())

    return by_id, children, parent


def category_path(cid: int, by_id: Dict[int, dict], parent: Dict[int, int]) -> str:
    parts = []
    cur = cid
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        nm = sanitize_latin1(by_id.get(cur, {}).get("name", f"#{cur}"))
        parts.append(nm)
        cur = parent.get(cur, 0)
    parts.reverse()
    return " > ".join(parts)


def descendants(cid: int, children: Dict[int, List[int]]) -> Set[int]:
    out: Set[int] = set()
    stack = [cid]
    while stack:
        x = stack.pop()
        for ch in children.get(x, []):
            if ch not in out:
                out.add(ch)
                stack.append(ch)
    return out


# ----------------------------
# CSV loader (backup)
# ----------------------------
def load_products_from_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    col_map = {c.lower(): c for c in df.columns}

    def col(name: str) -> Optional[str]:
        return col_map.get(name.lower())

    out = pd.DataFrame()
    out["id"] = df[col("ID")] if col("ID") else None
    out["sku"] = df[col("SKU")] if col("SKU") else ""
    out["name"] = df[col("Name")] if col("Name") else ""
    out["description"] = df[col("Description")] if col("Description") else ""
    out["short_description"] = df[col("Short description")] if col("Short description") else ""
    out["categories_raw"] = df[col("Categories")] if col("Categories") else ""
    out["regular_price"] = df[col("Regular price")] if col("Regular price") else ""
    out["sale_price"] = df[col("Sale price")] if col("Sale price") else ""
    out["in_stock"] = df[col("In stock?")] if col("In stock?") else ""
    out["published"] = df[col("Published")] if col("Published") else ""
    out["images_raw"] = df[col("Images")] if col("Images") else ""
    out["status"] = out["published"].apply(lambda x: "publish" if boolish(x) else "private")

    def first_img(x: Any) -> str:
        if x is None:
            return ""
        s = str(x).strip()
        if not s:
            return ""
        return s.split(",")[0].strip()

    out["image_url"] = out["images_raw"].apply(first_img)
    out["permalink"] = ""
    return out


# ----------------------------
# Product normalization (API -> dataframe)
# ----------------------------
def normalize_api_products(items: List[dict]) -> pd.DataFrame:
    rows = []
    for p in items:
        rows.append(
            {
                "id": p.get("id"),
                "sku": p.get("sku") or "",
                "name": p.get("name") or "",
                "status": p.get("status") or "",
                "permalink": p.get("permalink") or "",
                "regular_price": p.get("regular_price") or "",
                "sale_price": p.get("sale_price") or "",
                "on_sale": bool(p.get("on_sale")) if p.get("on_sale") is not None else False,
                "stock_status": p.get("stock_status") or "",
                "short_description": p.get("short_description") or "",
                "description": p.get("description") or "",
                "categories": p.get("categories") or [],
                "images": p.get("images") or [],
                "attributes": p.get("attributes") or [],
            }
        )
    df = pd.DataFrame(rows)

    def first_img(images: Any) -> str:
        if not images:
            return ""
        try:
            if isinstance(images, list) and len(images) > 0:
                return images[0].get("src") or ""
        except Exception:
            return ""
        return ""

    df["image_url"] = df["images"].apply(first_img)
    return df


def extract_brand_and_kashrut(attrs: Any) -> Tuple[str, str, str]:
    brand = ""
    kash = ""
    other_bits: List[str] = []

    if isinstance(attrs, list):
        for a in attrs:
            nm = sanitize_latin1(a.get("name", "")).strip()
            opts = a.get("options") or []
            if isinstance(opts, list):
                v = ", ".join(sanitize_latin1(x) for x in opts if x)
            else:
                v = sanitize_latin1(opts)
            nm_l = nm.lower()

            if nm_l in ("brand", "manufacturer"):
                brand = v
            elif "kash" in nm_l or "kosher" in nm_l:
                kash = v
            else:
                if nm and v:
                    other_bits.append(f"{nm}: {v}")

    attrs_text = " | ".join(other_bits[:2])
    return brand, kash, attrs_text


# ----------------------------
# Image download + caching
# ----------------------------
def image_cache_path(url: str) -> Path:
    ext = ".jpg"
    m = re.search(r"\.(png|jpg|jpeg|webp)(\?|$)", str(url).lower())
    if m:
        ext = "." + m.group(1).replace("jpeg", "jpg")
    return IMG_DIR / f"{sha1_text(str(url))}{ext}"


def _download_one_image(url: str, timeout: int, retries: int) -> bool:
    """Worker-safe single download, returns True if downloaded/valid in cache."""
    if not url:
        return False
    path = image_cache_path(url)
    if path.exists() and path.stat().st_size > 2_000:
        return True

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.content and len(r.content) > 1_000:
                path.write_bytes(r.content)
                return True
        except requests.RequestException:
            pass
        time.sleep(0.25 * attempt)
    return False


def prefetch_images_with_progress(
    df: pd.DataFrame,
    timeout: int = 25,
    retries: int = 6,
    max_workers: int = 32,
) -> None:
    """
    FAST prefetch unique image URLs with concurrency.
    This is the ONLY section changed to speed up image downloading:
    - downloads in parallel using ThreadPoolExecutor
    - can set max_workers > 32
    """
    series = df.get("image_url", pd.Series([], dtype=str))
    urls = [u for u in series.astype(str).tolist() if u and u != "nan"]
    # unique preserve order
    seen: Set[str] = set()
    uniq: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    total = len(uniq)
    if total == 0:
        return

    st.subheader("Image download progress")
    bar = st.progress(0.0)
    line = st.empty()

    cached = 0
    to_get: List[str] = []
    for u in uniq:
        p = image_cache_path(u)
        if p.exists() and p.stat().st_size > 2_000:
            cached += 1
        else:
            to_get.append(u)

    done = cached
    downloaded = 0

    # initial render
    bar.progress(done / max(total, 1))
    line.markdown(f"**Images:** {done:,} / {total:,}  •  cached {cached:,}  •  downloaded {downloaded:,}")

    if not to_get:
        log(f"Images prefetched: total={total} cached={cached} downloaded=0")
        return

    workers = max(1, int(max_workers))
    log(f"Prefetching images with concurrency: max_workers={workers} (missing={len(to_get):,})")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_download_one_image, u, timeout, retries): u for u in to_get}
        for fut in as_completed(futures):
            ok = False
            try:
                ok = bool(fut.result())
            except Exception:
                ok = False
            if ok:
                downloaded += 1
            done += 1

            # update progress
            bar.progress(done / max(total, 1))
            line.markdown(
                f"**Images:** {done:,} / {total:,}  •  cached {cached:,}  •  downloaded {downloaded:,}"
            )

    log(f"Images prefetched: total={total} cached={cached} downloaded={downloaded}")


# ----------------------------
# PDF builder (fpdf2)
# ----------------------------
class CatalogPDF(FPDF):
    def __init__(
        self,
        title_txt: str,
        orientation: str,
        show_site_footer: bool = True,
        logo_bytes: Optional[bytes] = None,
    ):
        super().__init__(orientation=orientation, unit="mm", format="A4")
        self.title_txt = sanitize_latin1(title_txt)
        self.show_site_footer = show_site_footer
        self.logo_bytes = logo_bytes

        self.brand_red = hex_to_rgb(BRAND_RED)
        self.brand_blue = hex_to_rgb(BRAND_BLUE)

        self.set_auto_page_break(auto=False, margin=PDF_MARGIN_MM)

        self._logo_path: Optional[Path] = None
        if self.logo_bytes:
            ext = ".png"
            if self.logo_bytes[:2] == b"\xff\xd8":
                ext = ".jpg"
            tmp = CACHE_DIR / f"logo{ext}"
            try:
                tmp.write_bytes(self.logo_bytes)
                self._logo_path = tmp
            except Exception:
                self._logo_path = None

    def header(self):
        self.set_fill_color(255, 255, 255)
        self.rect(0, 0, self.w, HEADER_H_MM + 6, style="F")

        x = PDF_MARGIN_MM
        y = 6
        if self._logo_path and self._logo_path.exists():
            try:
                self.image(str(self._logo_path), x=x, y=y, w=24)
            except Exception:
                pass

        self.set_xy(x + 28, 8)
        self.set_text_color(*self.brand_blue)
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 7, self.title_txt)

        self.set_draw_color(*self.brand_blue)
        self.set_line_width(0.6)
        self.line(PDF_MARGIN_MM, HEADER_H_MM + 6, self.w - PDF_MARGIN_MM, HEADER_H_MM + 6)

        # Page number in header area
        self.set_text_color(50, 50, 50)
        self.set_font("Helvetica", "", 9)
        self.set_xy(self.w - PDF_MARGIN_MM - 25, 8)
        self.cell(25, 7, f"Page {self.page_no()}", align="R")

        self.set_text_color(0, 0, 0)

    def footer(self):
        if not self.show_site_footer:
            return
        y = self.h - FOOTER_H_MM
        self.set_draw_color(*self.brand_red)
        self.set_line_width(0.4)
        self.line(PDF_MARGIN_MM, y, self.w - PDF_MARGIN_MM, y)

        self.set_xy(PDF_MARGIN_MM, y + 2.5)
        self.set_text_color(50, 50, 50)
        self.set_font("Helvetica", "", 8)
        left = sanitize_latin1(f"{BRAND_SITE} | Prices correct as of {time.strftime('%d %b %Y')}")
        self.cell(0, 6, left)
        self.set_text_color(0, 0, 0)

    def category_bar(self, text: str):
        text = sanitize_latin1(text)
        y = HEADER_H_MM + 10
        self.set_xy(PDF_MARGIN_MM, y)
        self.set_fill_color(*self.brand_blue)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 11)
        self.rect(PDF_MARGIN_MM, y, self.w - 2 * PDF_MARGIN_MM, CATEGORY_BAR_H_MM, style="F")
        self.set_xy(PDF_MARGIN_MM + 4, y + 1.2)
        self.cell(0, CATEGORY_BAR_H_MM - 2, text)
        self.set_text_color(0, 0, 0)

    def wrap_lines(
        self,
        text: str,
        max_w: float,
        max_lines: int,
        font_family: str,
        font_style: str,
        font_size: float,
    ) -> List[str]:
        text = sanitize_latin1(text).strip()
        if not text:
            return []
        self.set_font(font_family, font_style, font_size)

        words = text.split()
        lines: List[str] = []
        cur = ""

        for w in words:
            test = (cur + " " + w).strip()
            if self.get_string_width(test) <= max_w or not cur:
                cur = test
            else:
                lines.append(cur)
                cur = w
                if len(lines) >= max_lines:
                    break

        if len(lines) < max_lines and cur:
            lines.append(cur)

        if lines:
            ell = "..."  # latin-1 safe
            last = lines[-1]
            while self.get_string_width(last + ell) > max_w and len(last) > 1:
                last = last[:-1]
            if last != lines[-1]:
                lines[-1] = (last + ell).strip()

        return lines[:max_lines]

    def tile(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        product: dict,
        dense: bool,
        currency: str,
        show_price: bool,
        show_sku: bool,
        show_desc: bool,
        show_attrs: bool,
        show_brand_kashrut: bool,
    ):
        self.set_draw_color(*hex_to_rgb(BRAND_BLUE))
        self.set_line_width(0.35)
        self.rect(x, y, w, h)

        pad = 2.2 if dense else 2.8

        # Image area
        img_h = h * (0.58 if dense else 0.62)
        img_x = x + pad
        img_y = y + pad
        img_w = w - 2 * pad
        img_box_h = img_h - pad

        img_path = None
        if product.get("image_url"):
            img_path = image_cache_path(product["image_url"])
            if not img_path.exists() or img_path.stat().st_size <= 2_000:
                img_path = None

        if img_path and Path(img_path).exists():
            try:
                self.image(str(img_path), x=img_x, y=img_y, w=img_w)
            except Exception:
                pass
        else:
            self.set_font("Helvetica", "I", 7 if dense else 8)
            self.set_text_color(120, 120, 120)
            self.set_xy(img_x, img_y + img_box_h / 2)
            self.cell(img_w, 4, "No image", align="C")
            self.set_text_color(0, 0, 0)

        # Sale badge
        on_sale = bool(product.get("on_sale")) or (
            money_2dp(product.get("sale_price")) is not None
            and money_2dp(product.get("regular_price")) is not None
            and float(money_2dp(product.get("sale_price")) or 0) < float(money_2dp(product.get("regular_price")) or 0)
        )
        if on_sale:
            self.set_fill_color(*hex_to_rgb(BRAND_RED))
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 7 if dense else 8)
            bw, bh = ((14, 6) if dense else (16, 6.5))
            self.rect(x + w - bw - 1.2, y + 1.2, bw, bh, style="F")
            self.set_xy(x + w - bw - 1.2, y + 1.2 + 1.0)
            self.cell(bw, bh - 2, "SALE", align="C")
            self.set_text_color(0, 0, 0)

        # Text area
        cursor_y = y + img_h
        max_w = w - 2 * pad

        # Name
        name_font = 7.2 if dense else 9.2
        name_lines = self.wrap_lines(
            product.get("name", ""),
            max_w=max_w,
            max_lines=2,
            font_family="Helvetica",
            font_style="B",
            font_size=name_font,
        )
        self.set_font("Helvetica", "B", name_font)
        for ln in name_lines:
            self.set_xy(x + pad, cursor_y)
            self.cell(max_w, 3.6 if dense else 4.4, ln)
            cursor_y += 3.6 if dense else 4.4

        # Price
        if show_price:
            sale = money_2dp(product.get("sale_price"))
            reg = money_2dp(product.get("regular_price"))
            if sale and (not reg or float(sale) < float(reg)):
                self.set_font("Helvetica", "B", 7.2 if dense else 8.8)
                self.set_text_color(*hex_to_rgb(BRAND_RED))
                self.set_xy(x + pad, cursor_y)
                self.cell(max_w, 3.8 if dense else 4.3, f"{currency}{sale}")

                if reg:
                    self.set_text_color(120, 120, 120)
                    self.set_font("Helvetica", "", 6.6 if dense else 8.0)
                    self.set_xy(x + pad + 16, cursor_y)
                    self.cell(max_w, 3.8 if dense else 4.3, f"{currency}{reg}")

                self.set_text_color(0, 0, 0)
                cursor_y += 3.9 if dense else 4.6
            else:
                val = sale or reg
                if val:
                    self.set_font("Helvetica", "B", 7.2 if dense else 8.8)
                    self.set_text_color(*hex_to_rgb(BRAND_RED))
                    self.set_xy(x + pad, cursor_y)
                    self.cell(max_w, 3.8 if dense else 4.3, f"{currency}{val}")
                    self.set_text_color(0, 0, 0)
                    cursor_y += 3.9 if dense else 4.6

        # SKU
        if show_sku and product.get("sku"):
            self.set_font("Helvetica", "", 6.2 if dense else 7.0)
            self.set_text_color(90, 90, 90)
            self.set_xy(x + pad, cursor_y)
            self.cell(max_w, 3.2 if dense else 3.6, f"SKU: {sanitize_latin1(product.get('sku'))}")
            self.set_text_color(0, 0, 0)
            cursor_y += 3.2 if dense else 3.6

        brand, kash, attrs_text = extract_brand_and_kashrut(product.get("attributes"))

        # Brand + Kashrut
        if show_brand_kashrut:
            self.set_font("Helvetica", "", 6.2 if dense else 7.0)
            self.set_text_color(70, 70, 70)

            if brand:
                self.set_xy(x + pad, cursor_y)
                self.cell(max_w, 3.1 if dense else 3.5, sanitize_latin1(f"Brand: {brand}"))
                cursor_y += 3.1 if dense else 3.5

            if kash:
                ln = self.wrap_lines(f"Kashrus: {kash}", max_w, 1, "Helvetica", "", 6.2 if dense else 7.0)
                if ln:
                    self.set_xy(x + pad, cursor_y)
                    self.cell(max_w, 3.1 if dense else 3.5, ln[0])
                    cursor_y += 3.1 if dense else 3.5

            self.set_text_color(0, 0, 0)

        # Attributes (short)
        if show_attrs and attrs_text:
            self.set_font("Helvetica", "", 6.0 if dense else 6.8)
            self.set_text_color(70, 70, 70)
            ln = self.wrap_lines(attrs_text, max_w, 1, "Helvetica", "", 6.0 if dense else 6.8)
            if ln:
                self.set_xy(x + pad, cursor_y)
                self.cell(max_w, 3.0 if dense else 3.4, ln[0])
                cursor_y += 3.0 if dense else 3.4
            self.set_text_color(0, 0, 0)

        # Description
        if show_desc:
            desc = product.get("short_description") or product.get("description") or ""
            desc = sanitize_latin1(desc).strip()
            if desc:
                self.set_font("Helvetica", "", 5.8 if dense else 6.6)
                self.set_text_color(80, 80, 80)
                lines = self.wrap_lines(desc, max_w, 2, "Helvetica", "", 5.8 if dense else 6.6)
                for ln in lines:
                    self.set_xy(x + pad, cursor_y)
                    self.cell(max_w, 2.8 if dense else 3.2, ln)
                    cursor_y += 2.8 if dense else 3.2
                self.set_text_color(0, 0, 0)

        # Clickable link over entire tile
        url = product.get("permalink") or ""
        if url:
            try:
                self.link(x=x, y=y, w=w, h=h, link=url)
            except Exception:
                pass


def make_catalog_pdf_bytes(
    df: pd.DataFrame,
    title: str,
    orientation: str,
    currency: str,
    grid_mode: str,
    show_price: bool,
    show_sku: bool,
    show_desc: bool,
    show_attrs: bool,
    show_brand_kashrut: bool,
    category_label_mode: str,
    build_progress: Optional[Tuple[st.delta_generator.DeltaGenerator, st.delta_generator.DeltaGenerator, int]] = None,
) -> bytes:
    """
    Builds a PDF in-memory and returns bytes.
    NOTE: For huge catalogs we instead build PART PDFs to disk and zip them
    to avoid Streamlit/RAM crashes. (See UI section.)
    """
    logo_b = load_logo_bytes()

    pdf = CatalogPDF(
        title_txt=title,
        orientation="P" if orientation == "Portrait" else "L",
        show_site_footer=True,
        logo_bytes=logo_b,
    )
    pdf.add_page()

    dense = (grid_mode == "Compact")
    if orientation == "Portrait":
        cols, rows = (3, 3) if not dense else (6, 5)
    else:
        cols, rows = (3, 3) if not dense else (7, 4)

    margin = PDF_MARGIN_MM
    start_y = HEADER_H_MM + 10 + CATEGORY_BAR_H_MM + 4
    usable_w = pdf.w - 2 * margin
    usable_h = pdf.h - start_y - FOOTER_H_MM - 6

    gap_x = 2.2 if dense else 4.0
    gap_y = 2.2 if dense else 5.0

    tile_w = (usable_w - gap_x * (cols - 1)) / cols
    tile_h = (usable_h - gap_y * (rows - 1)) / rows

    def cat_label(row: pd.Series) -> str:
        if category_label_mode == "Full path":
            return row.get("category_path_best") or "Uncategorised"
        return row.get("category_top") or "Uncategorised"

    df2 = df.copy()
    df2["group_label"] = df2.apply(cat_label, axis=1)
    df2 = df2.sort_values(["category_path_best", "name"], kind="stable")

    current_group = None
    slot = 0

    total_rows = int(df2.shape[0])
    bar = None
    line = None
    if build_progress is not None:
        bar, line, _ = build_progress

    for i, (_, row) in enumerate(df2.iterrows(), start=1):
        g = sanitize_latin1(row.get("group_label") or "Uncategorised")
        if g != current_group:
            if current_group is not None:
                pdf.add_page()
            current_group = g
            pdf.category_bar(g)
            slot = 0

        r = slot // cols
        c = slot % cols
        if r >= rows:
            pdf.add_page()
            pdf.category_bar(g)
            slot = 0
            r = 0
            c = 0

        x = margin + c * (tile_w + gap_x)
        y = start_y + r * (tile_h + gap_y)

        pdf.tile(
            x=x,
            y=y,
            w=tile_w,
            h=tile_h,
            product=row.to_dict(),
            dense=dense,
            currency=currency,
            show_price=show_price,
            show_sku=show_sku,
            show_desc=show_desc,
            show_attrs=show_attrs,
            show_brand_kashrut=show_brand_kashrut,
        )
        slot += 1

        # build progress update (throttled)
        if bar is not None and line is not None:
            if i == 1 or i == total_rows or (i % 50 == 0):
                bar.progress(i / max(total_rows, 1))
                line.markdown(f"**Building PDF:** {i:,} / {total_rows:,}")

    # fpdf2: dest parameter deprecated; output() returns bytes/bytearray in newer versions
    out = pdf.output()
    return bytes(out)


def make_catalog_pdf_file(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    orientation: str,
    currency: str,
    grid_mode: str,
    show_price: bool,
    show_sku: bool,
    show_desc: bool,
    show_attrs: bool,
    show_brand_kashrut: bool,
    category_label_mode: str,
    build_bar: st.delta_generator.DeltaGenerator,
    build_line: st.delta_generator.DeltaGenerator,
) -> Path:
    b = make_catalog_pdf_bytes(
        df=df,
        title=title,
        orientation=orientation,
        currency=currency,
        grid_mode=grid_mode,
        show_price=show_price,
        show_sku=show_sku,
        show_desc=show_desc,
        show_attrs=show_attrs,
        show_brand_kashrut=show_brand_kashrut,
        category_label_mode=category_label_mode,
        build_progress=(build_bar, build_line, int(df.shape[0])),
    )
    out_path.write_bytes(b)
    return out_path


def zip_files_to_disk(files: List[Path], zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=p.name)
    return zip_path


def try_merge_pdfs_to_disk(part_paths: List[Path], out_path: Path) -> Optional[Path]:
    """
    Optional: merges parts into one file if pypdf or PyPDF2 is available.
    If not available, returns None (you can still download the ZIP).
    """
    merger = None
    try:
        from pypdf import PdfMerger  # type: ignore
        merger = PdfMerger()
    except Exception:
        try:
            from PyPDF2 import PdfMerger  # type: ignore
            merger = PdfMerger()
        except Exception:
            return None

    try:
        for p in part_paths:
            merger.append(str(p))
        with out_path.open("wb") as f:
            merger.write(f)
        merger.close()
        return out_path
    except Exception:
        try:
            merger.close()
        except Exception:
            pass
        return None


# ----------------------------
# App UI
# ----------------------------
def main():
    st.set_page_config(page_title="B-Kosher Catalog Builder", page_icon="🧾", layout="wide")

    # Header
    col1, col2 = st.columns([1, 5])
    with col1:
        lb = load_logo_bytes()
        if lb:
            st.image(lb, width=120)
    with col2:
        st.title("B-Kosher Catalog Builder")
        st.caption("WooCommerce API is the default. CSV upload is a backup option.")

    # Sidebar logs
    with st.sidebar:
        st.subheader("Live logs")
        st.text_area("Logs", value=logs_text(), height=420, label_visibility="collapsed")
        st.caption("If the import pauses (phone sleep/background), re-open the app and press “Continue importing” — it resumes from cache.")

    wc_url, wc_ck, wc_cs = get_wc_creds()

    # Step 1
    st.header("Step 1 — Choose data source")
    source = st.radio("Source", ["WooCommerce API", "CSV Upload"], index=0, horizontal=True)

    if source == "WooCommerce API":
        st.header("Step 2 — Load products (API)")

        if not wc_url or not wc_ck or not wc_cs:
            st.error(
                "WooCommerce API secrets missing.\n\n"
                "Add ONE of these sets to Streamlit Secrets:\n"
                "- WC_URL, WC_CK, WC_CS\n"
                "- WC_BASE_URL, WC_CONSUMER_KEY, WC_CONSUMER_SECRET"
            )
            st.stop()

        timeout = int(st.slider("API timeout (seconds)", 10, 60, 30))
        include_private_load = st.checkbox(
            "Include private/unpublished products while LOADING (API status=any)",
            value=False,
            help="If your API user lacks permission for private products, the server may return errors.",
        )

        # per_page option + chunk size control
        per_page = st.selectbox("Products per API page", [25, 50, 100], index=2)
        max_pages_per_run = st.slider(
            "Pages per run (auto-continues until done)",
            min_value=1,
            max_value=30,
            value=3,
            help="Higher = faster per click, but too high can trigger Streamlit Cloud timeouts.",
        )

        colA, colB, colC, colD = st.columns([1, 1, 1, 1])
        with colA:
            start_btn = st.button("Start importing (auto-continue)", use_container_width=True)
        with colB:
            cont_btn = st.button("Continue importing", use_container_width=True)
        with colC:
            refresh_btn = st.button("Refresh cache (start over)", use_container_width=True)
        with colD:
            stop_btn = st.button("Stop importing", use_container_width=True)

        if stop_btn:
            st.session_state["fetch_stop"] = True
            st.session_state["auto_continue_import"] = False
            log("Stop requested.")

        status_key = f"status_{'any' if include_private_load else 'publish'}"

        if refresh_btn:
            clear_cache(status_key)
            st.session_state.pop("wc_products_df", None)
            st.session_state.pop("wc_categories", None)
            st.session_state["fetch_stop"] = False
            st.session_state["auto_continue_import"] = False
            log("Cache cleared for this mode. (If you changed per_page, this is recommended.)")

        if start_btn:
            st.session_state["fetch_stop"] = False
            st.session_state["auto_continue_import"] = True

        if cont_btn:
            st.session_state["fetch_stop"] = False
            st.session_state.setdefault("auto_continue_import", False)

        # Always ensure categories are loaded before importing products
        if (st.session_state.get("auto_continue_import") or cont_btn) and not st.session_state.get("wc_categories"):
            st.info("Loading categories…")
            log("Loading categories…")
            try:
                cats = fetch_all_categories(wc_url, wc_ck, wc_cs, timeout)
                st.session_state["wc_categories"] = cats
            except Exception as e:
                st.error(f"Failed to load categories: {e}")
                st.stop()

        # Run one import chunk if asked
        if st.session_state.get("auto_continue_import") or cont_btn:
            try:
                res = resumable_fetch_products(
                    wc_url=wc_url,
                    ck=wc_ck,
                    cs=wc_cs,
                    timeout=timeout,
                    include_private=include_private_load,
                    per_page=int(per_page),
                    max_pages_per_run=int(max_pages_per_run),
                    retries=6,
                )
                df = normalize_api_products(res.products)
                st.session_state["wc_products_df"] = df
                st.success(f"Cached: {len(df):,} products (status used: {res.used_status}).")
            except Exception as e:
                st.error(str(e))
                st.session_state["auto_continue_import"] = False
                st.stop()

            # Auto-continue until done
            _, meta = load_cached_products(status_key)
            last = int(meta.get("last_completed_page", 0) or 0)
            total_pages = int(meta.get("total_pages", 0) or 0)

            if st.session_state.get("auto_continue_import") and not st.session_state.get("fetch_stop"):
                if total_pages > 0 and last < total_pages:
                    st.info("Auto-continuing… (the app will refresh itself while importing)")
                    time.sleep(0.25)
                    st.rerun()
                else:
                    st.session_state["auto_continue_import"] = False
                    st.success("Import complete.")

        df_loaded = st.session_state.get("wc_products_df")
        if df_loaded is None:
            st.info("Press **Start importing (auto-continue)** to begin.")
            st.stop()

        products_df = df_loaded

    else:
        st.header("Step 2 — Load products (CSV)")
        up = st.file_uploader("Upload WooCommerce product export (.csv)", type=["csv"])
        if not up:
            st.stop()
        products_df = load_products_from_csv(up.read())
        st.session_state["wc_categories"] = []
        st.session_state["wc_products_df"] = products_df
        st.success(f"Loaded {len(products_df):,} products from CSV.")

    # Step 3
    st.header("Step 3 — Choose what goes into the catalog")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        orientation = st.selectbox("Page orientation", ["Portrait", "Landscape"], index=0)
        grid_mode = st.selectbox("Grid density", ["Standard (3×3)", "Compact (6×5)"], index=0)
    with col2:
        currency = st.text_input("Currency symbol", value="£")
    with col3:
        show_price = st.checkbox("Show price", value=True)
        show_sku = st.checkbox("Show SKU", value=False)
        show_desc = st.checkbox("Show description", value=False)
        show_attrs = st.checkbox("Show attributes", value=True)

    col4, col5, col6 = st.columns([1, 1, 1])
    with col4:
        exclude_oos = st.checkbox("Exclude out-of-stock", value=True)
        only_sale = st.checkbox("Only sale items", value=False)
    with col5:
        include_private_in_pdf = st.checkbox(
            "Include private/unpublished products in catalog (PDF)",
            value=False,
            help="This filter applies after loading. If you did not load private products from API, turning this on cannot add them.",
        )
    with col6:
        category_label_mode = st.selectbox("Category divider label", ["Full path", "Top level"], index=0)

    title = st.text_input("Catalog title", value=DEFAULT_TITLE)

    # Category selection
    cats = st.session_state.get("wc_categories") or []
    by_id, children, parent = build_category_maps(cats) if cats else ({}, {}, {})
    paths: List[Tuple[str, int]] = []

    expanded: Set[int] = set()
    if by_id:
        for cid in by_id.keys():
            paths.append((category_path(cid, by_id, parent), cid))
        paths.sort(key=lambda t: t[0].lower())

        st.subheader("Categories (tree)")
        selected_paths = st.multiselect(
            "Choose categories (selecting a parent includes all children & grandchildren)",
            options=[p for p, _ in paths],
            default=[],
        )
        selected_ids = [cid for p, cid in paths if p in set(selected_paths)]
        for cid in selected_ids:
            expanded.add(cid)
            expanded |= descendants(cid, children)
    else:
        st.subheader("Categories")
        st.caption("CSV mode does not include a category tree; category filtering is limited.")

    search = st.text_input("Search (name or SKU)", value="")

    # Apply filters
    df = products_df.copy()
    df["name"] = df["name"].astype(str).map(safe_unescape)
    df["sku"] = df.get("sku", "").astype(str).fillna("")

    if exclude_oos:
        if "stock_status" in df.columns:
            df = df[df["stock_status"].astype(str).str.lower().eq("instock")]

    if not include_private_in_pdf and "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("publish")]

    if only_sale:
        def is_sale_row(r: pd.Series) -> bool:
            sp = money_2dp(r.get("sale_price"))
            rp = money_2dp(r.get("regular_price"))
            if sp and rp:
                try:
                    return float(sp) < float(rp)
                except Exception:
                    return True
            return bool(r.get("on_sale"))
        df = df[df.apply(is_sale_row, axis=1)]

    # Category mapping for API mode
    if by_id and "categories" in df.columns:
        def best_cat_path(cat_list: Any) -> str:
            if not isinstance(cat_list, list) or not cat_list:
                return ""
            best = ""
            best_len = -1
            for c in cat_list:
                cid = c.get("id")
                if cid is None:
                    continue
                try:
                    cid_i = int(cid)
                except Exception:
                    continue
                pth = category_path(cid_i, by_id, parent)
                depth = len(pth.split(" > "))
                if depth > best_len:
                    best = pth
                    best_len = depth
            return best

        def top_cat(cat_list: Any) -> str:
            if not isinstance(cat_list, list) or not cat_list:
                return ""
            try:
                cid = int(cat_list[0].get("id"))
            except Exception:
                return ""
            cur = cid
            seen = set()
            while cur and cur not in seen:
                seen.add(cur)
                p = parent.get(cur, 0)
                if p == 0:
                    break
                cur = p
            return sanitize_latin1(by_id.get(cur, {}).get("name", ""))

        df["category_path_best"] = df["categories"].apply(best_cat_path)
        df["category_top"] = df["categories"].apply(top_cat)

        if expanded:
            def has_selected(cat_list: Any) -> bool:
                if not isinstance(cat_list, list):
                    return False
                for c in cat_list:
                    try:
                        cid = int(c.get("id"))
                    except Exception:
                        continue
                    if cid in expanded:
                        return True
                return False
            df = df[df["categories"].apply(has_selected)]
    else:
        df["category_path_best"] = df.get("categories_raw", "")
        df["category_top"] = df.get("categories_raw", "")

    if search.strip():
        q = search.strip().lower()
        df = df[
            df["name"].astype(str).str.lower().str.contains(q, na=False)
            | df["sku"].astype(str).str.lower().str.contains(q, na=False)
        ]

    st.info(f"Selected products: **{len(df):,}**")

    with st.expander("Preview (first 12 products)"):
        prev = df.head(12).copy()
        prev["regular_price"] = prev["regular_price"].map(money_2dp)
        prev["sale_price"] = prev["sale_price"].map(money_2dp)
        cols = ["name", "sku", "regular_price", "sale_price"]
        if "status" in prev.columns:
            cols.append("status")
        st.dataframe(prev[cols], use_container_width=True)

    # Step 4 — PDF
    st.header("Step 4 — Generate PDF")

    # Controls for huge builds (disk-based)
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        chunk_size = st.number_input(
            "PDF part size (products per PDF)",
            min_value=200,
            max_value=5000,
            value=1000,
            step=100,
            help="Large catalogs are built as multiple PDFs to avoid crashes.",
        )
    with colB:
        max_workers = st.number_input(
            "Image download concurrency (max workers)",
            min_value=4,
            max_value=256,
            value=64,
            step=4,
            help="Increase to download more than 32 images at once. Too high can trigger rate limits.",
        )
    with colC:
        want_merge = st.checkbox(
            "Also try to create ONE merged PDF (optional)",
            value=False,
            help="Requires pypdf or PyPDF2 installed. If not available, you'll still get the ZIP of parts.",
        )

    if st.button("Generate PDF", type="primary", use_container_width=True):
        if len(df) == 0:
            st.error("No products selected.")
            st.stop()

        dense = grid_mode.startswith("Compact")
        grid_key = "Compact" if dense else "Standard"

        # Stage 1/2 — images
        log("Prefetching images…")
        st.info("Stage 1/3 — Downloading images…")
        prefetch_images_with_progress(df, timeout=25, retries=6, max_workers=int(max_workers))

        # Stage 2/3 — build PDFs in parts to disk + progress
        total_products = int(df.shape[0])
        part_n = int(math.ceil(total_products / max(int(chunk_size), 1)))

        st.info("Stage 2/3 — Building PDF part files…")
        overall_bar = st.progress(0.0)
        overall_line = st.empty()

        part_paths: List[Path] = []

        # stable ordering
        df_build = df.copy()
        if "category_path_best" in df_build.columns:
            df_build = df_build.sort_values(["category_path_best", "name"], kind="stable")
        else:
            df_build = df_build.sort_values(["name"], kind="stable")

        for idx in range(part_n):
            start = idx * int(chunk_size)
            end = min((idx + 1) * int(chunk_size), total_products)
            part_df = df_build.iloc[start:end].copy()

            overall_line.markdown(f"**Building part {idx+1} / {part_n}**  •  products {start+1:,}–{end:,}")
            overall_bar.progress(idx / max(part_n, 1))

            part_bar = st.progress(0.0)
            part_line = st.empty()

            out_path = OUT_DIR / f"bkosher_catalog_part_{idx+1:02d}_of_{part_n:02d}.pdf"
            log(f"Building PDF part {idx+1}/{part_n} → {out_path.name}")

            make_catalog_pdf_file(
                df=part_df,
                out_path=out_path,
                title=title,
                orientation=orientation,
                currency=currency,
                grid_mode=grid_key,
                show_price=show_price,
                show_sku=show_sku,
                show_desc=show_desc,
                show_attrs=show_attrs,
                show_brand_kashrut=True,
                category_label_mode=category_label_mode,
                build_bar=part_bar,
                build_line=part_line,
            )
            part_paths.append(out_path)

            # clean up per-part UI a bit
            part_line.success(f"Part {idx+1} saved: {out_path.name}")
            part_bar.progress(1.0)

            # yield a tiny bit to keep Streamlit responsive
            time.sleep(0.05)

        overall_bar.progress(1.0)
        overall_line.success("All PDF parts built.")

        # Stage 3/3 — package downloads (disk-based, avoids RAM crash)
        st.info("Stage 3/3 — Preparing downloads…")

        zip_path = OUT_DIR / "bkosher_catalog_parts.zip"
        if zip_path.exists():
            try:
                zip_path.unlink()
            except Exception:
                pass
        zip_files_to_disk(part_paths, zip_path)
        log(f"ZIP ready: {zip_path.name}")

        st.subheader("Downloads")

        with zip_path.open("rb") as f:
            st.download_button(
                "Download ZIP of PDF parts",
                data=f,
                file_name=zip_path.name,
                mime="application/zip",
                use_container_width=True,
            )

        # Optional merge
        if want_merge:
            merged_path = OUT_DIR / "bkosher_catalog_merged.pdf"
            if merged_path.exists():
                try:
                    merged_path.unlink()
                except Exception:
                    pass

            merged = try_merge_pdfs_to_disk(part_paths, merged_path)
            if merged:
                log("Merged PDF created.")
                with merged.open("rb") as f:
                    st.download_button(
                        "Download merged PDF",
                        data=f,
                        file_name=merged.name,
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                st.warning("Merge not available (pypdf/PyPDF2 not installed). Download the ZIP of parts instead.")

        st.caption("If links don’t work in your viewer, test in Chrome/Edge — some mobile viewers ignore PDF links.")

    st.caption("Done.")


# ----------------------------
# Boot
# ----------------------------
require_login()
main()