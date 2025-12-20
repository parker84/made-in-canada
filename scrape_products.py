#!/usr/bin/env python3
"""
Scrape product pages + basic info + images from an ecommerce site.

Strategy:
1) Discover sitemaps (robots.txt -> Sitemap: lines, else common sitemap URLs)
2) Parse sitemap(s) for product URLs (default: contains '/products/' but configurable)
3) For each product URL:
   - Fetch HTML
   - Extract Product data from JSON-LD (schema.org/Product) if present
   - Fallback: parse common meta tags
   - Collect images from JSON-LD and/or og:image
4) Save JSON output + download images

Usage:
  python scrape_products.py --base https://www.roots.com --out ./out --pattern "/products/"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import colorlog
import httpx
from bs4 import BeautifulSoup
from tqdm import tqdm
import xml.etree.ElementTree as ET


# Set up colored logging
def setup_logger() -> logging.Logger:
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    ))
    logger = logging.getLogger("scraper")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

log = setup_logger()

DEFAULT_UA = "Mozilla/5.0 (compatible; ProductScraper/1.0; +https://example.com/bot)"


@dataclass
class Product:
    url: str
    name: Optional[str] = None
    brand: Optional[str] = None
    sku: Optional[str] = None
    description: Optional[str] = None
    price: Optional[str] = None
    currency: Optional[str] = None
    availability: Optional[str] = None
    images: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "name": self.name,
            "brand": self.brand,
            "sku": self.sku,
            "description": self.description,
            "price": self.price,
            "currency": self.currency,
            "availability": self.availability,
            "images": self.images or [],
        }


def same_host(url: str, base: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc


async def fetch_text(client: httpx.AsyncClient, url: str, timeout: float = 30) -> Optional[str]:
    try:
        r = await client.get(url, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        return r.text
    except Exception:
        return None


async def discover_sitemaps(client: httpx.AsyncClient, base: str) -> List[str]:
    """Try robots.txt Sitemap directives first; else try common sitemap paths."""
    sitemaps: List[str] = []

    robots_url = urljoin(base, "/robots.txt")
    robots = await fetch_text(client, robots_url)
    if robots:
        for line in robots.splitlines():
            if line.lower().startswith("sitemap:"):
                sm = line.split(":", 1)[1].strip()
                if sm:
                    sitemaps.append(sm)

    # Common fallbacks
    fallbacks = [
        urljoin(base, "/sitemap.xml"),
        urljoin(base, "/sitemap_index.xml"),
        urljoin(base, "/sitemap/sitemap.xml"),
    ]
    for f in fallbacks:
        if f not in sitemaps:
            sitemaps.append(f)

    # Keep unique, preserve order
    uniq = []
    seen = set()
    for s in sitemaps:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def parse_sitemap_xml(xml_text: str) -> Tuple[List[str], List[str]]:
    """
    Returns (urls, nested_sitemaps)
    Supports sitemapindex + urlset.
    Falls back to BeautifulSoup href extraction if XML parsing fails.
    """
    urls: List[str] = []
    nested: List[str] = []
    
    # First try XML parsing for proper sitemaps
    try:
        root = ET.fromstring(xml_text)
        
        # Handle namespaces
        def strip_ns(t: str) -> str:
            return t.split("}", 1)[-1] if "}" in t else t

        root_name = strip_ns(root.tag).lower()

        if root_name == "sitemapindex":
            for sm in root.findall(".//{*}sitemap/{*}loc"):
                if sm.text:
                    nested.append(sm.text.strip())
        elif root_name == "urlset":
            for loc in root.findall(".//{*}url/{*}loc"):
                if loc.text:
                    urls.append(loc.text.strip())
        
        return urls, nested
    except ET.ParseError:
        pass
    
    # Fallback: parse as HTML and extract all hrefs
    soup = BeautifulSoup(xml_text, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href and href.startswith("http"):
            # URLs ending with / are likely category pages -> crawl them
            # URLs ending with .html are likely product pages -> collect them
            if href.endswith("/"):
                nested.append(href)
            else:
                urls.append(href)
    
    # Also check for <loc> tags in case it's malformed XML
    for loc in soup.find_all("loc"):
        if loc.string:
            urls.append(loc.string.strip())
    
    return urls, nested


async def collect_product_urls(
    client: httpx.AsyncClient,
    base: str,
    pattern: str,
    max_sitemaps: int = 50,
) -> List[str]:
    """
    Crawl sitemaps (with nesting) and return URLs matching pattern.
    """
    sitemaps = await discover_sitemaps(client, base)
    to_visit = sitemaps[:]
    visited: Set[str] = set()
    product_urls: Set[str] = set()

    with tqdm(total=len(to_visit), desc="🔍 Crawling pages", unit="page") as pbar:
        while to_visit and len(visited) < max_sitemaps:
            sm_url = to_visit.pop(0)
            if sm_url in visited:
                continue
            visited.add(sm_url)

            xml_text = await fetch_text(client, sm_url)
            if not xml_text:
                pbar.update(1)
                continue

            urls, nested = parse_sitemap_xml(xml_text)

            # Add nested sitemaps
            new_pages = 0
            for n in nested:
                if n not in visited and n not in to_visit:
                    to_visit.append(n)
                    new_pages += 1
            
            # Update progress bar total if we discovered new pages
            if new_pages > 0:
                pbar.total += new_pages
                pbar.refresh()

            # Collect product urls
            for u in urls:
                if pattern in u and same_host(u, base):
                    product_urls.add(u)

            pbar.update(1)
            pbar.set_postfix({"🛍️": len(product_urls), "📋": len(to_visit)})

    log.info(f"✅ Discovered {len(product_urls)} product URLs")
    return sorted(product_urls)


def extract_json_ld_products(html: str) -> List[Dict[str, Any]]:
    """
    Extract schema.org Product objects from JSON-LD blocks.
    Returns list of product dicts found.
    """
    soup = BeautifulSoup(html, "html.parser")
    blocks = soup.find_all("script", attrs={"type": "application/ld+json"})
    found: List[Dict[str, Any]] = []

    for b in blocks:
        if not b.string:
            continue
        raw = b.string.strip()
        # Some sites output multiple JSON objects or invalid trailing commas; be conservative
        try:
            data = json.loads(raw)
        except Exception:
            # Try to salvage common cases (very light cleanup)
            raw2 = raw.strip()
            raw2 = re.sub(r",\s*}", "}", raw2)
            raw2 = re.sub(r",\s*]", "]", raw2)
            try:
                data = json.loads(raw2)
            except Exception:
                continue

        def walk(obj: Any):
            if isinstance(obj, dict):
                t = obj.get("@type") or obj.get("type")
                if t:
                    # @type can be list
                    types = t if isinstance(t, list) else [t]
                    if any(str(x).lower() == "product" for x in types):
                        found.append(obj)
                # Graph pattern
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for it in obj:
                    walk(it)

        walk(data)

    return found


def pick_best_product_schema(schemas: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Prefer schema with offers + name."""
    if not schemas:
        return None
    scored = []
    for s in schemas:
        score = 0
        if s.get("name"):
            score += 2
        if s.get("offers"):
            score += 2
        if s.get("image"):
            score += 1
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def parse_product_from_html(url: str, html: str) -> Product:
    soup = BeautifulSoup(html, "html.parser")
    p = Product(url=url, images=[])

    schemas = extract_json_ld_products(html)
    schema = pick_best_product_schema(schemas)

    if schema:
        p.name = schema.get("name")
        p.description = schema.get("description")
        p.sku = schema.get("sku")

        brand = schema.get("brand")
        if isinstance(brand, dict):
            p.brand = brand.get("name")
        elif isinstance(brand, str):
            p.brand = brand

        # Images can be str or list
        img = schema.get("image")
        if isinstance(img, str):
            p.images.append(img)
        elif isinstance(img, list):
            p.images.extend([x for x in img if isinstance(x, str)])

        offers = schema.get("offers")
        # offers can be dict or list
        offer = offers[0] if isinstance(offers, list) and offers else offers
        if isinstance(offer, dict):
            p.price = str(offer.get("price")) if offer.get("price") is not None else None
            p.currency = offer.get("priceCurrency")
            p.availability = offer.get("availability")

    # Fallback meta tags
    if not p.name:
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            p.name = og_title["content"].strip()

    if not p.description:
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            p.description = og_desc["content"].strip()

    # Add og:image if we have none
    if not p.images:
        og_img = soup.find("meta", property="og:image")
        if og_img and og_img.get("content"):
            p.images.append(og_img["content"].strip())

    # De-dupe images
    seen = set()
    imgs = []
    for i in p.images:
        if i and i not in seen:
            imgs.append(i)
            seen.add(i)
    p.images = imgs

    return p


async def download_file(client: httpx.AsyncClient, url: str, path: str) -> bool:
    try:
        r = await client.get(url, follow_redirects=True)
        r.raise_for_status()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except Exception:
        return False


def safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s.strip("_")[:180] or "file"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base site URL, e.g. https://www.roots.com")
    ap.add_argument("--out", default="./out", help="Output directory")
    ap.add_argument("--pattern", default="/", help="URL substring to filter (default: '/' matches all)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of products (0 = no limit)")
    ap.add_argument("--concurrency", type=int, default=6, help="Concurrent requests")
    ap.add_argument("--delay", type=float, default=0.3, help="Delay between requests per worker (seconds)")
    ap.add_argument("--download-images", action="store_true", help="Download images locally")
    ap.add_argument("--url-regex", default="", help="Regex to filter discovered URLs (applied after --pattern)")
    ap.add_argument("--dump-urls", action="store_true", help="Dump all discovered URLs to a file before filtering")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    headers = {"User-Agent": DEFAULT_UA}

    async with httpx.AsyncClient(headers=headers, limits=limits) as client:
        # Step 1: Discover URLs
        log.info(f"🌐 [1/3] Discovering product URLs from {base} ...")
        step1_start = time.time()
        urls = await collect_product_urls(client, base, args.pattern)
        step1_elapsed = time.time() - step1_start
        
        # Dump all URLs before filtering if requested
        if args.dump_urls:
            dump_path = os.path.join(out_dir, "discovered_urls.txt")
            with open(dump_path, "w") as f:
                for u in sorted(urls):
                    f.write(u + "\n")
            log.info(f"📝 Dumped {len(urls)} URLs to {dump_path}")
        
        # Apply regex filter if provided
        if args.url_regex:
            rx = re.compile(args.url_regex)
            urls = [u for u in urls if rx.search(u)]
            log.info(f"🔎 After regex filter '{args.url_regex}': {len(urls)} URLs remain")
        
        if args.limit and args.limit > 0:
            urls = urls[: args.limit]

        log.info(f"🛒 Found {len(urls)} product URLs matching pattern '{args.pattern}' ⏱️  {step1_elapsed:.1f}s")
        if not urls:
            log.error("❌ No product URLs found. You may need a different --pattern or a category crawl approach.")
            sys.exit(1)

        sem = asyncio.Semaphore(args.concurrency)
        products: List[Dict[str, Any]] = []

        async def scrape_one(u: str):
            async with sem:
                html = await fetch_text(client, u)
                await asyncio.sleep(args.delay)
                if not html:
                    return
                prod = parse_product_from_html(u, html)
                products.append(prod.to_dict())

        # Step 2: Scrape product pages
        log.info(f"📦 [2/3] Scraping product pages (concurrency={args.concurrency}) ...")
        step2_start = time.time()
        await asyncio.gather(*[scrape_one(u) for u in urls])
        step2_elapsed = time.time() - step2_start

        products.sort(key=lambda x: x.get("url", ""))
        out_json = os.path.join(out_dir, "products.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(products, f, ensure_ascii=False, indent=2)

        log.info(f"💾 Saved {len(products)} products -> {out_json} ⏱️  {step2_elapsed:.1f}s")

        if args.download_images:
            # Step 3: Download images
            log.info(f"🖼️  [3/3] Downloading images ...")
            step3_start = time.time()
            img_dir = os.path.join(out_dir, "images")
            os.makedirs(img_dir, exist_ok=True)

            # simple sequential download to be gentle (you can parallelize if needed)
            ok = 0
            for prod in products:
                purl = prod["url"]
                for idx, img_url in enumerate(prod.get("images", [])[:8]):  # cap per product
                    parsed = urlparse(img_url)
                    ext = os.path.splitext(parsed.path)[1] or ".jpg"
                    fname = safe_filename(f"{purl}_img{idx}{ext}")
                    path = os.path.join(img_dir, fname)
                    if await download_file(client, img_url, path):
                        ok += 1
                    await asyncio.sleep(args.delay)
            step3_elapsed = time.time() - step3_start

            log.info(f"🎉 Downloaded {ok} images -> {img_dir} ⏱️  {step3_elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
