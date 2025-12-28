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
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import colorlog
import httpx
from bs4 import BeautifulSoup
from decouple import config
from tqdm import tqdm
import xml.etree.ElementTree as ET

# Playwright is optional - only imported if --use-browser is set
playwright_available = False
try:
    from playwright.async_api import async_playwright
    playwright_available = True
except ImportError:
    pass

# Psycopg is optional - only imported if --use-postgres is set
psycopg_available = False
try:
    import psycopg
    from psycopg.types.json import Json
    psycopg_available = True
except ImportError:
    pass

# Cohere is optional - only imported if --use-postgres is set (for embeddings)
cohere_available = False
try:
    import cohere
    cohere_available = True
except ImportError:
    pass

# Embedding Configuration
EMBEDDING_MODEL = "embed-v4.0"
EMBEDDING_DIMENSIONS = 1536

# Sleep time on error
SLEEP_TIME_ON_ERROR = 60*20 # 20 minutes
WAIT_TIME_BETWEEN_REQUESTS = 10 # 10 seconds

# Pure.md API for markdown conversion
PUREMD_API_URL = "https://pure.md"
PUREMD_API_KEY = config("PUREMD_API_KEY", default=None)
PUREMD_HEADERS = {"x-puremd-api-token": PUREMD_API_KEY} if PUREMD_API_KEY else {}

# Store type configurations - patterns for different ecommerce platforms
STORE_CONFIGS = {
    "shopify": {
        "category_patterns": ["/collections/"],
        "product_pattern": r"/products/[^/]+$",
        "product_js": "href.includes('/products/')",
    },
    "roots": {
        "category_patterns": ["/women/", "/men/", "/kids/", "/leather/", "/sale/", "/gifts/"],
        "product_pattern": r"-\d+\.html",
        "product_js": "/-\\d+\\.html/.test(href)",
    },
    "generic": {
        "category_patterns": [
            "/collections/", "/category/", "/categories/", "/shop/",
            "/women/", "/men/", "/kids/", "/sale/", "/new/",
            "/accessories/", "/clothing/", "/shoes/", "/bags/",
        ],
        "product_pattern": r"(/products/|/product/|-\d+\.html|/p/)",
        "product_js": "href.includes('/products/') || href.includes('/product/') || /-\\d+\\.html/.test(href) || /\\/p\\/[^/]+/.test(href)",
    },
}

# TODO: get this running for multiple brands (ex: airflow)
# 1. on a schedule every day 


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

# Database Configuration (from environment variables)
DB_CONFIG = {
    "host": config("POSTGRES_HOST"),
    "dbname": config("POSTGRES_DB"),
    "user": config("POSTGRES_USER"),
    "password": config("POSTGRES_PASSWORD", default=""),
}


class DatabaseManager:
    """Manages PostgreSQL database operations for products with embeddings"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.conn = None
        self.cohere_client = None
    
    async def connect(self):
        """Connect to the database and initialize Cohere client"""
        if not psycopg_available:
            log.error("❌ psycopg not installed. Run: uv add 'psycopg[binary]'")
            sys.exit(1)
        
        if not cohere_available:
            log.error("❌ cohere not installed. Run: uv add cohere")
            sys.exit(1)
        
        # Initialize Cohere client for embeddings
        cohere_api_key = config("COHERE_API_KEY", default="")
        if not cohere_api_key:
            log.error("❌ COHERE_API_KEY not set in environment")
            sys.exit(1)
        self.cohere_client = cohere.AsyncClientV2(api_key=cohere_api_key)
        
        log.info("🔌 Connecting to database...")
        start = time.time()
        
        conn_string = f"host={self.db_config['host']} dbname={self.db_config['dbname']} user={self.db_config['user']} password={self.db_config['password']}"
        self.conn = await psycopg.AsyncConnection.connect(conn_string)
        
        elapsed = time.time() - start
        log.info(f"✅ Database connected in {elapsed:.2f}s")
    
    async def initialize_schema(self):
        """Create necessary tables if they don't exist"""
        log.info("📋 Initializing database schema...")
        start = time.time()
        
        async with self.conn.cursor() as cur:
            # Enable pgvector extension
            await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create products table with embedding column
            await cur.execute(f"""
                CREATE TABLE IF NOT EXISTS products (
                    id SERIAL PRIMARY KEY,
                    url TEXT UNIQUE NOT NULL,
                    name TEXT,
                    brand TEXT,
                    sku TEXT,
                    description TEXT,
                    price TEXT,
                    currency TEXT,
                    availability TEXT,
                    images JSONB,
                    html TEXT,
                    markdown TEXT,
                    source_site TEXT,
                    embedding vector({EMBEDDING_DIMENSIONS}),
                    scraped_at TIMESTAMP DEFAULT NOW(),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Add html column if it doesn't exist (for existing tables)
            await cur.execute("""
                ALTER TABLE products ADD COLUMN IF NOT EXISTS html TEXT
            """)
            
            # Add markdown column if it doesn't exist (for existing tables)
            await cur.execute("""
                ALTER TABLE products ADD COLUMN IF NOT EXISTS markdown TEXT
            """)
            
            # Create index on URL for fast lookups
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS products_url_idx ON products (url)
            """)
            
            # Create index on source_site for filtering
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS products_source_site_idx ON products (source_site)
            """)
            
            # Create vector index for similarity search
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS products_embedding_idx 
                ON products 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            await self.conn.commit()
        
        elapsed = time.time() - start
        log.info(f"✅ Schema initialized in {elapsed:.2f}s")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Cohere"""
        response = await self.cohere_client.embed(
            texts=[text],
            model=EMBEDDING_MODEL,
            input_type="search_document",
            embedding_types=["float"],
            output_dimension=int(EMBEDDING_DIMENSIONS),
        )
        return response.embeddings.float_[0]
    
    def _create_product_text(self, product: Dict[str, Any]) -> str:
        """Create searchable text from product data for embedding"""
        parts = []
        if product.get("name"):
            parts.append(f"Product: {product['name']}")
        # if product.get("brand"):
        #     parts.append(f"Brand: {product['brand']}")
        if product.get("description"):
            parts.append(f"Description: {product['description']}")
        # if product.get("price"):
        #     parts.append(f"Price: {product['price']} {product.get('currency', '')}")
        product_text = " | ".join(parts) if parts else product.get("url", "")
        log.info(f"Product text: {product_text}")
        return product_text
    
    async def save_product(self, product: Dict[str, Any], source_site: str) -> int:
        """Save or update a product with embedding, return its ID"""
        # Generate embedding for the product
        product_text = self._create_product_text(product)
        embedding = await self.generate_embedding(product_text)
        embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
        
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO products (url, name, brand, sku, description, price, currency, availability, images, html, markdown, source_site, embedding, scraped_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, NOW(), NOW())
                ON CONFLICT (url) DO UPDATE SET
                    name = EXCLUDED.name,
                    brand = EXCLUDED.brand,
                    sku = EXCLUDED.sku,
                    description = EXCLUDED.description,
                    price = EXCLUDED.price,
                    currency = EXCLUDED.currency,
                    availability = EXCLUDED.availability,
                    images = EXCLUDED.images,
                    html = EXCLUDED.html,
                    markdown = EXCLUDED.markdown,
                    embedding = EXCLUDED.embedding,
                    scraped_at = NOW(),
                    updated_at = NOW()
                RETURNING id
                """,
                (
                    product["url"],
                    product.get("name"),
                    product.get("brand"),
                    product.get("sku"),
                    product.get("description"),
                    product.get("price"),
                    product.get("currency"),
                    product.get("availability"),
                    Json(product.get("images", [])),
                    product.get("html"),
                    product.get("markdown"),
                    source_site,
                    embedding_str,
                ),
            )
            result = await cur.fetchone()
            await self.conn.commit()
            return result[0]
    
    async def save_products_batch(self, products: List[Dict[str, Any]], source_site: str) -> int:
        """Save multiple products in a batch with rate limiting for embedding API"""
        saved = 0
        failed = 0
        
        for i, product in enumerate(products):
            try:
                await self.save_product(product, source_site)
                saved += 1
                
                # Rate limit: pause every 20 products to avoid API limits
                if saved % 20 == 0:
                    log.info(f"💾 Progress: {saved}/{len(products)} saved...")
                    await asyncio.sleep(1)  # Brief pause to respect rate limits
                    
            except Exception as e:
                failed += 1
                log.warning(f"⚠️  Failed to save product {product.get('url', 'unknown')[:50]}: {str(e)[:100]}")
                
                # If we're getting too many failures, add a longer pause
                if failed > 5 and failed % 5 == 0:
                    log.warning(f"⚠️  Multiple failures ({failed}), pausing 5s to respect rate limits...")
                    await asyncio.sleep(5)
        
        if failed > 0:
            log.warning(f"⚠️  {failed} products failed to save (likely rate limiting or missing data)")
        
        return saved
    
    async def get_product_count(self, source_site: str = None) -> int:
        """Get count of products, optionally filtered by source"""
        async with self.conn.cursor() as cur:
            if source_site:
                await cur.execute(
                    "SELECT COUNT(*) FROM products WHERE source_site = %s",
                    (source_site,)
                )
            else:
                await cur.execute("SELECT COUNT(*) FROM products")
            result = await cur.fetchone()
            return result[0]
    
    async def close(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
            log.info("🔌 Database connection closed")


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
    html: Optional[str] = None  # Raw HTML of the product page
    markdown: Optional[str] = None  # Markdown version of the product page

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
            "html": self.html,
            "markdown": self.markdown,
        }


def same_host(url: str, base: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc


async def fetch_text(client: httpx.AsyncClient, url: str, timeout: float = 30) -> Optional[str]:
    try:
        r = await client.get(url, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        return r.text
    except Exception as e:
        log.error(f"❌ Error fetching {url}: {e}")
        return None


async def fetch_markdown(client: httpx.AsyncClient, url: str, timeout: float = 30) -> Optional[str]:
    """Fetch markdown version of a URL using pure.md API"""
    try:
        puremd_url = f"{PUREMD_API_URL}/{url}"
        r = await client.get(puremd_url, timeout=timeout, headers=PUREMD_HEADERS)
        if r.status_code == 200:
            return r.text
        else:
            log.debug(f"⚠️  pure.md returned {r.status_code} for {url}")
            return None
    except Exception as e:
        log.debug(f"⚠️  Error fetching markdown for {url}: {e}")
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


async def discover_categories_with_browser(
    base_url: str,
    category_patterns: List[str],
    headless: bool = True,
) -> List[str]:
    """
    Discover category page URLs from the homepage navigation.
    
    Args:
        base_url: The homepage URL
        category_patterns: List of URL patterns that indicate category pages (e.g. ["/collections/", "/women/"])
        headless: Whether to run browser in headless mode
    """
    if not playwright_available:
        log.error("❌ Playwright not installed. Run: uv add playwright && playwright install firefox")
        sys.exit(1)
    
    categories: Set[str] = set()
    # Convert patterns to JSON for JavaScript
    patterns_json = json.dumps(category_patterns)
    
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=headless)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
            viewport={"width": 1920, "height": 1080},
            locale="en-CA",
            timezone_id="America/Toronto",
        )
        page = await context.new_page()
        
        try:
            await page.goto(base_url, wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(2)
            
            # Find category links using provided patterns
            links = await page.evaluate(f"""
                () => {{
                    const categoryPatterns = {patterns_json};
                    const links = new Set();
                    const allLinks = Array.from(document.querySelectorAll('a[href]'));
                    
                    for (const a of allLinks) {{
                        const href = a.href;
                        
                        // Skip external links, anchors, query params
                        if (!href.startsWith(window.location.origin)) continue;
                        if (href.includes('#')) continue;
                        if (href.includes('?')) continue;
                        if (href === window.location.origin + '/') continue;
                        
                        // Check if URL matches any category pattern
                        const hrefLower = href.toLowerCase();
                        if (categoryPatterns.some(p => hrefLower.includes(p.toLowerCase()))) {{
                            // For /collections/, exclude product pages
                            if (href.includes('/collections/') && href.includes('/products/')) continue;
                            links.add(href);
                        }}
                    }}
                    
                    return Array.from(links);
                }}
            """)
            
            for link in links:
                categories.add(link)
            
            log.info(f"🔍 Found {len(categories)} category pages")
            
        except Exception as e:
            log.warning(f"⚠️  Failed to discover categories: {e}")
        
        await browser.close()
    
    return sorted(categories)


async def crawl_category_with_browser(
    category_urls: List[str],
    product_js_filter: str,
    url_regex: str = "",
    max_load_more: int = 20,
    headless: bool = True,
    debug_screenshots: bool = False,
    out_dir: str = "./out",
) -> List[str]:
    """
    Use Playwright to crawl category pages, click Load More to get all products,
    and extract product URLs.
    
    Args:
        category_urls: List of category page URLs to crawl
        product_js_filter: JavaScript expression to identify product links (e.g. "href.includes('/products/')")
        url_regex: Optional regex to further filter URLs
        max_load_more: Max scroll attempts
        headless: Run browser headlessly
        debug_screenshots: Save debug screenshots
        out_dir: Output directory
    """
    if not playwright_available:
        log.error("❌ Playwright not installed. Run: uv add playwright && playwright install firefox")
        sys.exit(1)
    
    product_urls: Set[str] = set()
    rx = re.compile(url_regex) if url_regex else None
    
    async with async_playwright() as p:
        # Use Firefox (less likely to be blocked) with realistic settings
        browser = await p.firefox.launch(headless=headless)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
            viewport={"width": 1920, "height": 1080},
            locale="en-CA",
            timezone_id="America/Toronto",
        )
        page = await context.new_page()
        
        for cat_url in tqdm(category_urls, desc="🌐 Crawling categories"):
            try:
                await page.goto(cat_url, wait_until="domcontentloaded", timeout=60000)
                await asyncio.sleep(3)  # Wait for JS to kick in
                
                # Save debug screenshot if requested
                if debug_screenshots:
                    screenshot_dir = os.path.join(out_dir, "debug_screenshots")
                    os.makedirs(screenshot_dir, exist_ok=True)
                    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', cat_url)[-100:]
                    await page.screenshot(path=os.path.join(screenshot_dir, f"{safe_name}.png"), full_page=True)
                
                # Scroll to load all products (infinite scroll + "Load More" buttons)
                last_height = 0
                scroll_attempts = 0
                max_scroll_attempts = 30
                
                while scroll_attempts < max_scroll_attempts:
                    # Get current page height
                    current_height = await page.evaluate("document.body.scrollHeight")
                    
                    # Scroll down in chunks
                    await page.evaluate(f"window.scrollTo(0, {current_height})")
                    await asyncio.sleep(1.5)  # Wait for lazy-loaded content
                    
                    # Try clicking "Load More" / "Show More" buttons
                    load_more_selectors = [
                        "button.show-more",
                        ".show-more-products", 
                        "button:has-text('Show More')",
                        "button:has-text('Load More')",
                        "button:has-text('View More')",
                        ".load-more",
                        "[class*='load-more']",
                        "[class*='show-more']",
                    ]
                    for selector in load_more_selectors:
                        try:
                            btn = page.locator(selector).first
                            if await btn.is_visible(timeout=300):
                                await btn.scroll_into_view_if_needed()
                                await btn.click()
                                await asyncio.sleep(2)
                                break
                        except:
                            continue
                    
                    # Check if page height changed (new content loaded)
                    new_height = await page.evaluate("document.body.scrollHeight")
                    if new_height == last_height:
                        # No new content, try one more scroll to be sure
                        scroll_attempts += 1
                        if scroll_attempts >= 3:
                            break
                    else:
                        scroll_attempts = 0  # Reset counter if we got new content
                    
                    last_height = new_height
                
                # Extract product links using the configured filter
                links = await page.evaluate(f"""
                    () => {{
                        const links = new Set();
                        const allLinks = document.querySelectorAll('a[href]');
                        
                        allLinks.forEach(a => {{
                            const href = a.href;
                            
                            // Apply the store-specific product filter
                            if ({product_js_filter}) {{
                                const cleanUrl = href.split('?')[0];
                                links.add(cleanUrl);
                            }}
                        }});
                        
                        return Array.from(links);
                    }}
                """)
                
                # Debug: show sample links found
                if not links:
                    # Get sample links to help debug
                    sample_links = await page.evaluate("""
                        () => Array.from(document.querySelectorAll('a[href]'))
                            .map(a => a.href)
                            .filter(href => href.startsWith(window.location.origin))
                            .slice(0, 20)
                    """)
                    log.warning(f"   ⚠️  No product links found on this page.")
                    if sample_links:
                        # Look for anything that might be a product
                        product_like = [l for l in sample_links if '/product' in l.lower()]
                        if product_like:
                            log.warning(f"   🔍 Found product-like URLs: {product_like[:3]}")
                        else:
                            log.debug(f"   🔍 Sample links: {sample_links[:5]}")
                
                before_count = len(product_urls)
                for link in links:
                    if rx is None or rx.search(link):
                        product_urls.add(link)
                
                new_products = len(product_urls) - before_count
                log.info(f"📄 {cat_url} → +{new_products} new products ({len(product_urls)} total)")
                
            except Exception as e:
                log.warning(f"⚠️  Failed to crawl {cat_url}: {e}")
        
        await browser.close()
    
    log.info(f"✅ Discovered {len(product_urls)} product URLs via browser")
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


def parse_product_from_html(url: str, html: str, markdown: Optional[str] = None) -> Product:
    soup = BeautifulSoup(html, "html.parser")
    p = Product(url=url, images=[], html=html, markdown=markdown)

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
    ap.add_argument("--download-images", action="store_true", help="Download images locally")
    ap.add_argument("--url-regex", default="", help="Regex to filter discovered URLs (applied after --pattern)")
    ap.add_argument("--dump-urls", action="store_true", help="Dump all discovered URLs to a file before filtering")
    ap.add_argument("--use-browser", action="store_true", help="Use Playwright browser to crawl (for JS-heavy sites)")
    ap.add_argument("--show-browser", action="store_true", help="Show browser window (for debugging, use with --use-browser)")
    ap.add_argument("--category-urls", nargs="+", default=[], help="Category page URLs to crawl (with --use-browser)")
    ap.add_argument("--max-categories", type=int, default=0, help="Max categories to crawl (0 = all)")
    ap.add_argument("--debug-screenshots", action="store_true", help="Save screenshots of crawled pages for debugging")
    ap.add_argument("--use-postgres", action="store_true", help="Save products to PostgreSQL instead of JSON file")
    ap.add_argument("--store-type", choices=["shopify", "roots", "generic"], default="generic",
                    help="Store type for auto-detection patterns (default: generic)")
    args = ap.parse_args()
    
    # Get store configuration
    store_config = STORE_CONFIGS[args.store_type]

    base = args.base.rstrip("/")
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    headers = {"User-Agent": DEFAULT_UA}

    async with httpx.AsyncClient(headers=headers, limits=limits) as client:
        # Step 1: Discover URLs
        step1_start = time.time()
        
        if args.use_browser:
            # Use Playwright to crawl category pages
            category_urls = args.category_urls
            
            # Auto-discover categories if none specified
            if not category_urls:
                log.info(f"🔍 [1/3] Auto-discovering category pages from {base} (store type: {args.store_type}) ...")
                category_urls = await discover_categories_with_browser(
                    base,
                    category_patterns=store_config["category_patterns"],
                    headless=not args.show_browser
                )
                if not category_urls:
                    log.error("❌ No category pages found. Try specifying them with --category-urls or a different --store-type")
                    sys.exit(1)
            
            # Apply max-categories limit
            if args.max_categories > 0:
                category_urls = category_urls[:args.max_categories]
            
            log.info(f"🌐 Crawling {len(category_urls)} category pages with browser ...")
            urls = await crawl_category_with_browser(
                category_urls,
                product_js_filter=store_config["product_js"],
                url_regex=args.url_regex,
                headless=not args.show_browser,
                debug_screenshots=args.debug_screenshots,
                out_dir=out_dir,
            )
        else:
            # Standard sitemap/link crawl
            log.info(f"🌐 [1/3] Discovering product URLs from {base} ...")
            urls = await collect_product_urls(client, base, args.pattern)
            
            # Apply regex filter if provided
            if args.url_regex:
                rx = re.compile(args.url_regex)
                urls = [u for u in urls if rx.search(u)]
                log.info(f"🔎 After regex filter '{args.url_regex}': {len(urls)} URLs remain")
        
        step1_elapsed = time.time() - step1_start
        
        # Dump all URLs before filtering if requested
        if args.dump_urls:
            dump_path = os.path.join(out_dir, "discovered_urls.txt")
            with open(dump_path, "w") as f:
                for u in sorted(urls):
                    f.write(u + "\n")
            log.info(f"📝 Dumped {len(urls)} URLs to {dump_path}")
        
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
                # Fetch HTML and markdown in parallel
                html_task = fetch_text(client, u)
                markdown_task = fetch_markdown(client, u)
                html, markdown = await asyncio.gather(html_task, markdown_task)
                
                await asyncio.sleep(WAIT_TIME_BETWEEN_REQUESTS)
                if not html:
                    log.info(f"No HTML found for {u}, sleeping for {SLEEP_TIME_ON_ERROR} seconds and retrying...")
                    await asyncio.sleep(SLEEP_TIME_ON_ERROR)
                    html = await fetch_text(client, u)
                    if not html:
                        log.warning(f"❌ No HTML found for {u}")
                        return
                    else:
                        log.debug(f"✅ HTML found for {u}")
                else:
                    log.debug(f"✅ HTML found for {u}")
                
                if markdown:
                    log.debug(f"✅ Markdown found for {u}")
                
                prod = parse_product_from_html(u, html, markdown=markdown)
                products.append(prod.to_dict())

        # Step 2: Scrape product pages
        log.info(f"📦 [2/3] Scraping product pages (concurrency={args.concurrency}) ...")
        step2_start = time.time()
        
        # Use tqdm to show progress
        pbar = tqdm(total=len(urls), desc="📦 Scraping products", unit="page")
        
        async def scrape_one_with_progress(u: str):
            await scrape_one(u)
            pbar.update(1)
        
        await asyncio.gather(*[scrape_one_with_progress(u) for u in urls])
        pbar.close()
        step2_elapsed = time.time() - step2_start

        products.sort(key=lambda x: x.get("url", ""))
        
        # Filter out products without useful data (no name = likely failed parse)
        valid_products = [p for p in products if p.get("name")]
        skipped = len(products) - len(valid_products)
        if skipped > 0:
            log.info(f"⏭️  Skipping {skipped} products without names (failed to parse product data)")
        
        # Save to PostgreSQL or JSON file
        if args.use_postgres:
            db = DatabaseManager(DB_CONFIG)
            await db.connect()
            await db.initialize_schema()
            
            source_site = urlparse(base).netloc
            log.info(f"💾 Saving {len(valid_products)} valid products to database...")
            saved_count = await db.save_products_batch(valid_products, source_site)
            total_count = await db.get_product_count(source_site)
            await db.close()
            
            log.info(f"✅ Saved {saved_count} products to PostgreSQL ({total_count} total for {source_site}) ⏱️  {step2_elapsed:.1f}s")
        else:
            out_json = os.path.join(out_dir, "products.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(valid_products, f, ensure_ascii=False, indent=2)
            log.info(f"💾 Saved {len(valid_products)} products -> {out_json} ⏱️  {step2_elapsed:.1f}s")

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
                    await asyncio.sleep(WAIT_TIME_BETWEEN_REQUESTS)
            step3_elapsed = time.time() - step3_start

            log.info(f"🎉 Downloaded {ok} images -> {img_dir} ⏱️  {step3_elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
