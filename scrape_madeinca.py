#!/usr/bin/env python3
"""
Scrape madeinca.ca directory for Canadian product/company listings.

This scraper extracts structured information about Canadian businesses including:
- Name
- Products/Services
- Manufactured In
- Where to Buy
- Website
- Canadian Owned status
- Description

Usage:
  python scrape_madeinca.py --use-postgres
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
from decouple import config
from tqdm import tqdm

# Playwright for JS-heavy pages
playwright_available = False
try:
    from playwright.async_api import async_playwright
    playwright_available = True
except ImportError:
    pass

# Psycopg for database
psycopg_available = False
try:
    import psycopg
    from psycopg.types.json import Json
    psycopg_available = True
except ImportError:
    pass

# Cohere for embeddings
cohere_available = False
try:
    import cohere
    cohere_available = True
except ImportError:
    pass

# OpenAI for AI detection
openai_available = False
try:
    from openai import AsyncOpenAI
    openai_available = True
except ImportError:
    pass

# Configuration
MADE_IN_CANADA_MODEL = config("MADE_IN_CANADA_MODEL", default="gpt-5-nano")
EXTRACTION_MODEL = config("EXTRACTION_MODEL", default="gpt-5-nano")  # For structured data extraction
EMBEDDING_MODEL = "embed-v4.0"
EMBEDDING_DIMENSIONS = 1536
BASE_URL = "https://madeinca.ca"
WAIT_BETWEEN_REQUESTS = 3  # seconds - increased to avoid 429s
MAX_RETRIES = 3
RETRY_DELAY_BASE = 5  # Base delay for exponential backoff
MAX_CONTENT_LENGTH = 8000  # Max chars to send to AI for extraction

# Database Configuration
DB_CONFIG = {
    "host": config("POSTGRES_HOST"),
    "dbname": config("POSTGRES_DB"),
    "user": config("POSTGRES_USER"),
    "password": config("POSTGRES_PASSWORD", default=""),
}


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
    logger = logging.getLogger("madeinca_scraper")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


log = setup_logger()


@dataclass
class ScrapeStats:
    """Track running statistics during scraping"""
    total_processed: int = 0
    
    # Shopify detection
    shopify_yes: int = 0
    shopify_no: int = 0
    shopify_unknown: int = 0
    
    # Canadian owned
    canadian_owned_yes: int = 0
    canadian_owned_no: int = 0
    canadian_owned_unknown: int = 0
    
    # Made in Canada
    made_in_canada_yes: int = 0
    made_in_canada_no: int = 0
    made_in_canada_unknown: int = 0
    
    # Extraction method
    regex_extractions: int = 0
    ai_extractions: int = 0
    
    def log_running_stats(self, name: str = ""):
        """Log running percentages"""
        if self.total_processed == 0:
            return
        
        n = self.total_processed
        
        # Shopify stats
        shopify_pct = (self.shopify_yes / n * 100) if n > 0 else 0
        not_shopify_pct = (self.shopify_no / n * 100) if n > 0 else 0
        
        # Canadian owned stats
        owned_pct = (self.canadian_owned_yes / n * 100) if n > 0 else 0
        not_owned_pct = (self.canadian_owned_no / n * 100) if n > 0 else 0
        
        # Made in Canada stats
        mic_pct = (self.made_in_canada_yes / n * 100) if n > 0 else 0
        not_mic_pct = (self.made_in_canada_no / n * 100) if n > 0 else 0
        
        log.info(f"""
📊 Running Stats ({n} processed):
   🛒 Shopify: {self.shopify_yes} ({shopify_pct:.1f}%) | Not: {self.shopify_no} ({not_shopify_pct:.1f}%) | Unknown: {self.shopify_unknown}
   🇨🇦 Canadian Owned: {self.canadian_owned_yes} ({owned_pct:.1f}%) | Not: {self.canadian_owned_no} ({not_owned_pct:.1f}%) | Unknown: {self.canadian_owned_unknown}
   🍁 Made in Canada: {self.made_in_canada_yes} ({mic_pct:.1f}%) | Not: {self.made_in_canada_no} ({not_mic_pct:.1f}%) | Unknown: {self.made_in_canada_unknown}
   🤖 Extraction: Regex={self.regex_extractions}, AI={self.ai_extractions}
""")


# Global stats tracker
stats = ScrapeStats()


@dataclass
class MadeInCAListing:
    """A business/product listing from madeinca.ca"""
    url: str
    name: Optional[str] = None
    products: Optional[str] = None  # What products/services they offer
    manufactured_in: Optional[str] = None  # Location of manufacturing
    where_to_buy: Optional[str] = None  # Online, Retail, etc.
    website: Optional[str] = None  # Company website
    canadian_owned: Optional[bool] = None  # Is it Canadian owned?
    canadian_owned_text: Optional[str] = None  # Raw text for Canadian owned field
    description: Optional[str] = None  # Full description
    category: Optional[str] = None  # Top-level category (e.g., "Clothing")
    subcategory: Optional[str] = None  # Subcategory (e.g., "Men's Clothing")
    made_in_canada: Optional[bool] = None  # AI-determined: is manufactured_in in Canada?
    made_in_canada_reason: Optional[str] = None  # AI justification
    is_shopify_store: Optional[bool] = None  # Detected via /meta.json endpoint
    shopify_meta: Optional[Dict[str, Any]] = None  # Full Shopify meta.json data
    shopify_reason: Optional[str] = None  # Reason for Shopify detection
    canadian_owned_reason: Optional[str] = None  # Reason for Canadian owned detection
    raw_text: Optional[str] = None  # Raw text content from the page
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "name": self.name,
            "products": self.products,
            "manufactured_in": self.manufactured_in,
            "where_to_buy": self.where_to_buy,
            "website": self.website,
            "canadian_owned": self.canadian_owned,
            "canadian_owned_text": self.canadian_owned_text,
            "canadian_owned_reason": self.canadian_owned_reason,
            "description": self.description,
            "category": self.category,
            "subcategory": self.subcategory,
            "made_in_canada": self.made_in_canada,
            "made_in_canada_reason": self.made_in_canada_reason,
            "is_shopify_store": self.is_shopify_store,
            "shopify_meta": self.shopify_meta,
            "shopify_reason": self.shopify_reason,
            "raw_text": self.raw_text,
        }


class MadeInCADatabaseManager:
    """Manages PostgreSQL database operations for madeinca.ca listings"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.conn = None
        self.cohere_client = None
    
    async def connect(self):
        if not psycopg_available:
            log.error("❌ psycopg not installed. Run: uv add 'psycopg[binary]'")
            sys.exit(1)
        
        if not cohere_available:
            log.error("❌ cohere not installed. Run: uv add cohere")
            sys.exit(1)
        
        cohere_api_key = config("COHERE_API_KEY", default="")
        if not cohere_api_key:
            log.error("❌ COHERE_API_KEY not set in environment")
            sys.exit(1)
        self.cohere_client = cohere.AsyncClientV2(api_key=cohere_api_key)
        
        log.info("🔌 Connecting to database...")
        conn_string = f"host={self.db_config['host']} dbname={self.db_config['dbname']} user={self.db_config['user']} password={self.db_config['password']}"
        self.conn = await psycopg.AsyncConnection.connect(conn_string)
        log.info("✅ Database connected")
    
    async def initialize_schema(self):
        """Create the madeinca_listings table"""
        log.info("📋 Initializing madeinca schema...")
        
        async with self.conn.cursor() as cur:
            await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            await cur.execute(f"""
                CREATE TABLE IF NOT EXISTS madeinca_listings (
                    id SERIAL PRIMARY KEY,
                    url TEXT UNIQUE NOT NULL,
                    name TEXT,
                    products TEXT,
                    manufactured_in TEXT,
                    where_to_buy TEXT,
                    website TEXT,
                    canadian_owned BOOLEAN,
                    canadian_owned_text TEXT,
                    description TEXT,
                    category TEXT,
                    subcategory TEXT,
                    made_in_canada BOOLEAN,
                    made_in_canada_reason TEXT,
                    is_shopify_store BOOLEAN,
                    shopify_meta JSONB,
                    raw_text TEXT,
                    embedding vector({EMBEDDING_DIMENSIONS}),
                    scraped_at TIMESTAMP DEFAULT NOW(),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Add new columns if they don't exist (for migrations)
            await cur.execute("""
                ALTER TABLE madeinca_listings ADD COLUMN IF NOT EXISTS subcategory TEXT
            """)
            await cur.execute("""
                ALTER TABLE madeinca_listings ADD COLUMN IF NOT EXISTS is_shopify_store BOOLEAN
            """)
            await cur.execute("""
                ALTER TABLE madeinca_listings ADD COLUMN IF NOT EXISTS shopify_meta JSONB
            """)
            await cur.execute("""
                ALTER TABLE madeinca_listings ADD COLUMN IF NOT EXISTS raw_text TEXT
            """)
            
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS madeinca_url_idx ON madeinca_listings (url)
            """)
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS madeinca_category_idx ON madeinca_listings (category)
            """)
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS madeinca_subcategory_idx ON madeinca_listings (subcategory)
            """)
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS madeinca_made_in_canada_idx ON madeinca_listings (made_in_canada)
            """)
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS madeinca_is_shopify_idx ON madeinca_listings (is_shopify_store)
            """)
            
            await self.conn.commit()
        
        log.info("✅ Schema initialized")
    
    async def generate_embedding(self, text: str) -> List[float]:
        response = await self.cohere_client.embed(
            texts=[text],
            model=EMBEDDING_MODEL,
            input_type="search_document",
            embedding_types=["float"],
            output_dimension=int(EMBEDDING_DIMENSIONS),
        )
        return response.embeddings.float_[0]
    
    def _create_listing_text(self, listing: Dict[str, Any]) -> str:
        parts = []
        if listing.get("name"):
            parts.append(f"Company: {listing['name']}")
        if listing.get("products"):
            parts.append(f"Products: {listing['products']}")
        if listing.get("description"):
            parts.append(f"Description: {listing['description'][:500]}")
        if listing.get("manufactured_in"):
            parts.append(f"Manufactured in: {listing['manufactured_in']}")
        return " | ".join(parts) if parts else listing.get("url", "")
    
    async def save_listing(self, listing: Dict[str, Any]) -> int:
        listing_text = self._create_listing_text(listing)
        embedding = await self.generate_embedding(listing_text)
        embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
        
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO madeinca_listings (
                    url, name, products, manufactured_in, where_to_buy, website,
                    canadian_owned, canadian_owned_text, description, category, subcategory,
                    made_in_canada, made_in_canada_reason, is_shopify_store, shopify_meta,
                    raw_text, embedding, scraped_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, NOW(), NOW())
                ON CONFLICT (url) DO UPDATE SET
                    name = EXCLUDED.name,
                    products = EXCLUDED.products,
                    manufactured_in = EXCLUDED.manufactured_in,
                    where_to_buy = EXCLUDED.where_to_buy,
                    website = EXCLUDED.website,
                    canadian_owned = EXCLUDED.canadian_owned,
                    canadian_owned_text = EXCLUDED.canadian_owned_text,
                    description = EXCLUDED.description,
                    category = EXCLUDED.category,
                    subcategory = EXCLUDED.subcategory,
                    made_in_canada = EXCLUDED.made_in_canada,
                    made_in_canada_reason = EXCLUDED.made_in_canada_reason,
                    is_shopify_store = EXCLUDED.is_shopify_store,
                    shopify_meta = EXCLUDED.shopify_meta,
                    raw_text = EXCLUDED.raw_text,
                    embedding = EXCLUDED.embedding,
                    scraped_at = NOW(),
                    updated_at = NOW()
                RETURNING id
                """,
                (
                    listing["url"],
                    listing.get("name"),
                    listing.get("products"),
                    listing.get("manufactured_in"),
                    listing.get("where_to_buy"),
                    listing.get("website"),
                    listing.get("canadian_owned"),
                    listing.get("canadian_owned_text"),
                    listing.get("description"),
                    listing.get("category"),
                    listing.get("subcategory"),
                    listing.get("made_in_canada"),
                    listing.get("made_in_canada_reason"),
                    listing.get("is_shopify_store"),
                    Json(listing.get("shopify_meta")) if listing.get("shopify_meta") else None,
                    listing.get("raw_text"),
                    embedding_str,
                ),
            )
            result = await cur.fetchone()
            await self.conn.commit()
            return result[0]
    
    async def save_listings_batch(self, listings: List[Dict[str, Any]]) -> int:
        saved = 0
        for i, listing in enumerate(listings):
            try:
                await self.save_listing(listing)
                saved += 1
                if saved % 20 == 0:
                    log.info(f"💾 Progress: {saved}/{len(listings)} saved...")
                    await asyncio.sleep(1)
            except Exception as e:
                log.warning(f"⚠️ Failed to save {listing.get('url', 'unknown')}: {e}")
        return saved
    
    async def get_listing_count(self) -> int:
        async with self.conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM madeinca_listings")
            result = await cur.fetchone()
            return result[0]
    
    async def close(self):
        if self.conn:
            await self.conn.close()
            log.info("🔌 Database connection closed")


async def fetch_with_retry(client: httpx.AsyncClient, url: str, timeout: float = 30) -> Optional[httpx.Response]:
    """
    Fetch a URL with retry logic for 429 and other transient errors.
    """
    for attempt in range(MAX_RETRIES):
        try:
            r = await client.get(url, timeout=timeout, follow_redirects=True)
            
            if r.status_code == 429:
                # Rate limited - wait with exponential backoff
                delay = RETRY_DELAY_BASE * (2 ** attempt)
                log.warning(f"⏳ Rate limited (429) on {url[:50]}... waiting {delay}s (attempt {attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(delay)
                continue
            
            r.raise_for_status()
            return r
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                delay = RETRY_DELAY_BASE * (2 ** attempt)
                log.warning(f"⏳ Rate limited (429) on {url[:50]}... waiting {delay}s (attempt {attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(delay)
                continue
            else:
                log.debug(f"   ⚠️ HTTP {e.response.status_code} for {url[:50]}")
                return None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY_BASE * (2 ** attempt)
                log.debug(f"   ⚠️ Error fetching {url[:50]}: {e}, retrying in {delay}s")
                await asyncio.sleep(delay)
            else:
                log.debug(f"   ⚠️ Failed to fetch {url[:50]} after {MAX_RETRIES} attempts: {e}")
                return None
    
    return None


async def extract_listing_with_ai(content_text: str, url: str) -> Dict[str, Any]:
    """
    Use AI to extract structured listing information from page content.
    Returns a dict with extracted fields.
    """
    if not openai_available:
        return {}
    
    openai_api_key = config("OPENAI_API_KEY", default=None)
    if not openai_api_key:
        return {}
    
    # Truncate content to avoid token limits
    content = content_text[:MAX_CONTENT_LENGTH]
    
    prompt = f"""Extract the following information from this Canadian business listing page.
Return the data in this exact JSON format (use null for missing fields):

{{
    "name": "Company/Business name",
    "products": "What products or services they offer",
    "manufactured_in": "Where they manufacture (city, province, country)",
    "where_to_buy": "How to purchase (Online, Retail stores, etc)",
    "website": "Company website URL",
    "canadian_owned": true/false/null,
    "description": "Brief description of the business"
}}

Important:
- "canadian_owned" should be true ONLY if explicitly stated as Canadian-owned
- "manufactured_in" should be the manufacturing LOCATION, not the company headquarters
- Extract the actual company website URL, not madeinca.ca links

Page content:
{content}"""

    try:
        client = AsyncOpenAI(api_key=openai_api_key)
        response = await client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured data from web pages. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content.strip()
        data = json.loads(response_text)
        log.debug(f"   🤖 AI extracted: {data.get('name', 'Unknown')}")
        return data
        
    except json.JSONDecodeError as e:
        log.debug(f"   ⚠️ AI returned invalid JSON for {url[:50]}: {e}")
        return {}
    except Exception as e:
        log.debug(f"   ⚠️ AI extraction failed for {url[:50]}: {e}")
        return {}


async def detect_shopify_store(client: httpx.AsyncClient, website_url: str) -> Tuple[Optional[bool], Optional[Dict[str, Any]], str]:
    """
    Detect if a website is a Shopify store by checking /meta.json endpoint.
    
    Returns:
        Tuple of (is_shopify: bool | None, meta_data: dict | None, reason: str)
    """
    if not website_url:
        reason = "No website URL provided"
        log.debug(f"   🛒 Shopify check skipped: {reason}")
        return None, None, reason
    
    # Normalize the URL
    website_url = website_url.strip()
    if not website_url.startswith("http"):
        website_url = "https://" + website_url
    
    # Remove trailing slash and add /meta.json
    base_url = website_url.rstrip("/")
    meta_url = f"{base_url}/meta.json"
    
    # Add a small delay before checking to avoid hammering external sites
    await asyncio.sleep(0.5)
    
    try:
        r = await fetch_with_retry(client, meta_url, timeout=10)
        if r and r.status_code == 200:
            try:
                data = r.json()
                # Verify it looks like Shopify meta.json (has expected fields)
                if isinstance(data, dict) and ("myshopify_domain" in data or "id" in data):
                    myshopify = data.get("myshopify_domain", "N/A")
                    reason = f"Found meta.json (myshopify_domain={myshopify})"
                    return True, data, reason
                else:
                    reason = f"meta.json missing Shopify fields"
                    return False, None, reason
            except json.JSONDecodeError:
                reason = "meta.json not valid JSON"
                return False, None, reason
        elif r:
            reason = f"meta.json HTTP {r.status_code}"
            return False, None, reason
        else:
            reason = "meta.json fetch failed"
            return None, None, reason
    except Exception as e:
        reason = f"Error: {str(e)[:30]}"
        return None, None, reason


async def detect_made_in_canada_from_location(location: str, company_name: str = "") -> Tuple[Optional[bool], Optional[str]]:
    """
    Use AI to determine if a manufacturing location is in Canada.
    Returns (is_made_in_canada, reason)
    """
    if not location:
        reason = "No manufacturing location provided"
        log.debug(f"   🍁 Made in Canada check skipped: {reason}")
        return None, reason
    
    if not openai_available:
        reason = "OpenAI not available for AI detection"
        log.warning(f"⚠️ {reason}")
        return None, reason
    
    openai_api_key = config("OPENAI_API_KEY", default=None)
    if not openai_api_key:
        reason = "OPENAI_API_KEY not set"
        log.debug(f"   🍁 Made in Canada check skipped: {reason}")
        return None, reason
    
    # Quick heuristics first - check for Canadian provinces/territories and major cities
    location_lower = location.lower()
    canadian_indicators = {
        # Provinces/territories
        "ontario": "Ontario",
        "quebec": "Quebec",
        "british columbia": "British Columbia",
        "alberta": "Alberta",
        "manitoba": "Manitoba",
        "saskatchewan": "Saskatchewan",
        "nova scotia": "Nova Scotia",
        "new brunswick": "New Brunswick",
        "newfoundland": "Newfoundland",
        "prince edward island": "PEI",
        "pei": "PEI",
        "yukon": "Yukon",
        "northwest territories": "NWT",
        "nunavut": "Nunavut",
        "canada": "Canada",
        # Province abbreviations
        ", on": "Ontario",
        ", qc": "Quebec",
        ", bc": "British Columbia",
        ", ab": "Alberta",
        ", mb": "Manitoba",
        ", sk": "Saskatchewan",
        ", ns": "Nova Scotia",
        ", nb": "New Brunswick",
        ", nl": "Newfoundland",
        ", pe": "PEI",
        ", yt": "Yukon",
        ", nt": "NWT",
        ", nu": "Nunavut",
        # Major Canadian cities (on madeinca.ca, these almost certainly refer to Canadian cities)
        "toronto": "Toronto, ON",
        "montreal": "Montreal, QC",
        "montréal": "Montreal, QC",
        "vancouver": "Vancouver, BC",
        "calgary": "Calgary, AB",
        "edmonton": "Edmonton, AB",
        "ottawa": "Ottawa, ON",
        "winnipeg": "Winnipeg, MB",
        "quebec city": "Quebec City, QC",
        "hamilton": "Hamilton, ON",
        "kitchener": "Kitchener, ON",
        "london": "London, ON",
        "victoria": "Victoria, BC",
        "halifax": "Halifax, NS",
        "oshawa": "Oshawa, ON",
        "windsor": "Windsor, ON",
        "saskatoon": "Saskatoon, SK",
        "regina": "Regina, SK",
        "st. john's": "St. John's, NL",
        "barrie": "Barrie, ON",
        "kelowna": "Kelowna, BC",
        "abbotsford": "Abbotsford, BC",
        "sudbury": "Sudbury, ON",
        "kingston": "Kingston, ON",
        "guelph": "Guelph, ON",
        "moncton": "Moncton, NB",
        "mississauga": "Mississauga, ON",
        "brampton": "Brampton, ON",
        "markham": "Markham, ON",
        "vaughan": "Vaughan, ON",
        "burnaby": "Burnaby, BC",
        "surrey": "Surrey, BC",
        "laval": "Laval, QC",
        "longueuil": "Longueuil, QC",
        "gta": "Greater Toronto Area",
        "greater toronto": "Greater Toronto Area",
    }
    
    for indicator, province_name in canadian_indicators.items():
        if indicator in location_lower:
            reason = f"Location '{location}' contains Canadian indicator: {province_name}"
            return True, reason
    
    # Check for phrases that imply Canadian manufacturing (common on madeinca.ca)
    canadian_phrases = [
        "completely canadian",
        "100% canadian",
        "canadian made",
        "canadian-made", 
        "made in canada",
        "proudly canadian",
        "canadian owned and operated",
        "canadian company",
        "canadian family",
        "family-owned canadian",
        "canadian manufacturer",
        "manufactured in canada",
        "produced in canada",
        "crafted in canada",
        "handmade in canada",
        "locally made",  # On madeinca.ca, "locally" means Canada
        "locally manufactured",
        "domestic manufacturing",
    ]
    for phrase in canadian_phrases:
        if phrase in location_lower:
            reason = f"Text implies Canadian manufacturing: '{phrase}'"
            return True, reason
    
    # Check for obvious non-Canadian locations
    non_canadian_indicators = ["usa", "united states", "china", "mexico", "germany", "france", "italy", "japan", "korea", "taiwan", "vietnam", "india", "bangladesh"]
    for indicator in non_canadian_indicators:
        if indicator in location_lower:
            reason = f"Location '{location}' indicates non-Canadian manufacturing ({indicator})"
            return False, reason
    
    # Use AI for uncertain cases
    prompt = f"""Is this manufacturing location in Canada?

IMPORTANT CONTEXT: 
- This text is from madeinca.ca, a directory of Canadian-made products and Canadian businesses.
- When a city name could refer to multiple places (e.g., Ottawa, London, Kingston), assume it refers to the Canadian city.
- Phrases like "Completely Canadian", "Canadian family-owned", "proudly Canadian", "100% Canadian" etc. IMPLY the product is manufactured in Canada - answer YES.
- Use your best judgment - if the text suggests Canadian manufacturing in any way, answer YES.

Manufacturing location text: "{location}"

Respond with ONLY:
ANSWER: YES or NO or UNKNOWN
REASON: Brief explanation"""

    try:
        client = AsyncOpenAI(api_key=openai_api_key)
        response = await client.chat.completions.create(
            model=MADE_IN_CANADA_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert at identifying geographic locations. Determine if the given location is in Canada."},
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = response.choices[0].message.content.strip()
        
        answer = None
        reason = None
        
        for line in response_text.split("\n"):
            line = line.strip()
            if line.upper().startswith("ANSWER:"):
                answer_text = line[7:].strip().upper()
                if "YES" in answer_text:
                    answer = True
                elif "NO" in answer_text:
                    answer = False
            elif line.upper().startswith("REASON:"):
                reason = line[7:].strip()
        
        return answer, reason
        
    except Exception as e:
        reason = f"AI detection failed: {str(e)[:50]}"
        log.warning(f"⚠️ {reason}")
        return None, reason


async def discover_all_listing_urls(headless: bool = True, max_categories: int = 0) -> List[Tuple[str, str, str]]:
    """
    Discover all listing URLs from madeinca.ca by crawling category pages.
    
    Args:
        headless: Whether to run browser in headless mode
        max_categories: Maximum number of categories to crawl (0 = all)
    
    Returns list of (url, category, subcategory) tuples.
    """
    if not playwright_available:
        log.error("❌ Playwright not installed. Run: uv add playwright && playwright install firefox")
        sys.exit(1)
    
    listings: List[Tuple[str, str, str]] = []  # (url, category, subcategory)
    visited_categories: Set[str] = set()
    category_hierarchy: Dict[str, Tuple[str, str]] = {}  # url -> (category, subcategory)
    
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=headless)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()
        
        # Step 1: Get all category and subcategory URLs from the navigation
        log.info("🔍 Discovering categories from navigation...")
        await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(2)
        
        # Extract category hierarchy from the navigation menu
        # The navigation has a structure like: Categories > Clothing > Men's Clothing
        category_data = await page.evaluate("""
            () => {
                const categories = [];
                
                // Look for the main navigation menu structure
                // madeinca.ca uses a typical WordPress menu with nested ul/li
                const menuItems = document.querySelectorAll('nav li, .menu li, #menu li');
                
                menuItems.forEach(li => {
                    const link = li.querySelector(':scope > a');
                    if (!link || !link.href.includes('madeinca.ca')) return;
                    
                    const url = link.href;
                    const text = link.textContent.trim();
                    
                    // Check if this is a top-level or nested item
                    const parentLi = li.parentElement?.closest('li');
                    const parentLink = parentLi?.querySelector(':scope > a');
                    
                    if (parentLink && parentLink.href.includes('madeinca.ca')) {
                        // This is a subcategory
                        categories.push({
                            url: url,
                            category: parentLink.textContent.trim(),
                            subcategory: text
                        });
                    } else {
                        // This is a top-level category
                        categories.push({
                            url: url,
                            category: text,
                            subcategory: null
                        });
                    }
                });
                
                // Also get direct category links
                document.querySelectorAll('a[href*="/category/"]').forEach(a => {
                    if (a.href.startsWith('https://madeinca.ca')) {
                        const text = a.textContent.trim();
                        // Try to extract category from URL path
                        const urlParts = new URL(a.href).pathname.split('/').filter(p => p);
                        if (urlParts.length >= 2 && urlParts[0] === 'category') {
                            const cat = urlParts[1].replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                            const subcat = urlParts.length > 2 ? urlParts[2].replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) : null;
                            categories.push({
                                url: a.href,
                                category: cat,
                                subcategory: subcat
                            });
                        }
                    }
                });
                
                return categories;
            }
        """)
        
        # Build category hierarchy map
        for item in category_data:
            url = item.get("url", "")
            cat = item.get("category", "")
            subcat = item.get("subcategory", "")
            if url:
                category_hierarchy[url] = (cat, subcat or "")
        
        category_links = list(category_hierarchy.keys())
        
        log.info(f"📁 Found {len(category_links)} potential category links")
        
        # Apply max_categories limit if specified
        if max_categories > 0 and len(category_links) > max_categories:
            log.info(f"🔢 Limiting to first {max_categories} categories (--max-categories)")
            category_links = category_links[:max_categories]
        
        # Step 2: Visit each category page and extract listing URLs
        for cat_url in tqdm(category_links, desc="🌐 Crawling categories"):
            if cat_url in visited_categories:
                continue
            visited_categories.add(cat_url)
            
            # Get category/subcategory from hierarchy or extract from URL
            cat_info = category_hierarchy.get(cat_url, ("", ""))
            category_name = cat_info[0]
            subcategory_name = cat_info[1]
            
            try:
                await page.goto(cat_url, wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(1)
                
                # If we don't have category info, try to extract from the page
                if not category_name:
                    page_title = await page.evaluate("""
                        () => {
                            const h1 = document.querySelector('h1');
                            return h1 ? h1.textContent.trim() : '';
                        }
                    """)
                    # Try to parse from URL if title not found
                    if page_title:
                        category_name = page_title
                    else:
                        # Extract from URL path
                        url_parts = cat_url.replace("https://madeinca.ca/category/", "").strip("/").split("/")
                        if url_parts:
                            category_name = url_parts[0].replace("-", " ").title()
                            if len(url_parts) > 1:
                                subcategory_name = url_parts[1].replace("-", " ").title()
                
                # Find all listing links on this category page
                listing_links = await page.evaluate("""
                    () => {
                        const links = [];
                        // Look for article links or post links
                        document.querySelectorAll('article a, .post a, .entry a, h2 a, h3 a').forEach(a => {
                            if (a.href && a.href.includes('madeinca.ca') && 
                                !a.href.includes('/category/') && 
                                !a.href.includes('/tag/') &&
                                !a.href.includes('#')) {
                                links.push(a.href);
                            }
                        });
                        return [...new Set(links)];
                    }
                """)
                
                for link in listing_links:
                    listings.append((link, category_name, subcategory_name))
                
                if listing_links:
                    cat_display = f"{category_name}" + (f" > {subcategory_name}" if subcategory_name else "")
                    log.debug(f"   📄 {cat_display}: {len(listing_links)} listings")
                
            except Exception as e:
                log.warning(f"⚠️ Failed to crawl {cat_url}: {e}")
            
            await asyncio.sleep(WAIT_BETWEEN_REQUESTS)
        
        # Step 3: Also try the main page and recent posts
        log.info("🔍 Checking main page for additional listings...")
        await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(2)
        
        # Scroll to load more content
        for _ in range(5):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1)
        
        main_listings = await page.evaluate("""
            () => {
                const links = [];
                document.querySelectorAll('article a, .post a, h2 a, h3 a').forEach(a => {
                    if (a.href && a.href.includes('madeinca.ca') && 
                        !a.href.includes('/category/') && 
                        !a.href.includes('/tag/') &&
                        !a.href.includes('#') &&
                        a.href !== 'https://madeinca.ca/' &&
                        a.href !== 'https://madeinca.ca') {
                        links.push(a.href);
                    }
                });
                return [...new Set(links)];
            }
        """)
        
        for link in main_listings:
            if not any(l[0] == link for l in listings):
                listings.append((link, "Homepage", ""))
        
        await browser.close()
    
    # Deduplicate while preserving first category
    seen = set()
    unique_listings = []
    for url, cat, subcat in listings:
        if url not in seen:
            seen.add(url)
            unique_listings.append((url, cat, subcat))
    
    log.info(f"✅ Discovered {len(unique_listings)} unique listings")
    return unique_listings


async def fetch_listing_with_browser(page, url: str) -> Optional[str]:
    """
    Fetch a listing page using Playwright browser and return the main content text.
    Returns the rendered page content as text.
    """
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(1)  # Let dynamic content load
        
        # Extract the main content - try multiple selectors for Elementor/WordPress
        content_text = await page.evaluate("""
            () => {
                // Try to find the main single post content area
                const selectors = [
                    '.elementor-widget-theme-post-content',  // Elementor post content widget
                    '.entry-content',
                    '.post-content', 
                    '.single-post-content',
                    'article.post .elementor-section',
                    'article.type-post',
                    '.mv-content-wrapper .entry-content',
                    'main article',
                    'main',
                ];
                
                for (const selector of selectors) {
                    const el = document.querySelector(selector);
                    if (el && el.textContent.trim().length > 100) {
                        return el.textContent;
                    }
                }
                
                // Fallback: get body text but exclude nav/header/footer/sidebar
                const body = document.body.cloneNode(true);
                ['nav', 'header', 'footer', 'aside', '.sidebar', '.widget-area', '.elementor-location-header', '.elementor-location-footer'].forEach(sel => {
                    body.querySelectorAll(sel).forEach(el => el.remove());
                });
                return body.textContent;
            }
        """)
        
        return content_text.strip() if content_text else None
        
    except Exception as e:
        log.warning(f"⚠️ Browser fetch failed for {url[:50]}: {e}")
        return None


async def parse_listing_page_with_browser(page, client: httpx.AsyncClient, url: str, category: str, subcategory: str = "") -> Optional[MadeInCAListing]:
    """
    Parse a single listing page using Playwright browser and extract structured information.
    Uses regex first, then falls back to AI extraction for better accuracy.
    """
    content_text = await fetch_listing_with_browser(page, url)
    if not content_text:
        return None
    
    listing = MadeInCAListing(url=url, category=category, subcategory=subcategory or None)
    
    # Extract name from the content (usually first line with company name pattern)
    name_match = re.search(r"(?:Name|Company):\s*(.+?)(?:\n|$)", content_text, re.IGNORECASE)
    if name_match:
        listing.name = name_match.group(1).strip()
    else:
        # Try to get from title pattern like "Category: Company Name 🍁"
        title_match = re.search(r"^[^:]+:\s*([^🍁\n]+)", content_text)
        if title_match:
            listing.name = title_match.group(1).strip()
    
    # Store raw text from the page
    listing.raw_text = content_text[:10000]  # Limit to 10k chars
    
    # Try regex extraction
    field_patterns = {
        "name": r"(?:Name|Company):\s*(.+?)(?:\n|$)",
        "products": r"Products?:\s*(.+?)(?:\n|$)",
        "manufactured_in": r"(?:Manufactured In|Manufacturing Location|Made In):\s*(.+?)(?:\n|$)",
        "where_to_buy": r"(?:Where to Buy|Available At|Buy At):\s*(.+?)(?:\n|$)",
        "website": r"Website:\s*(\S+?)(?:\s|\n|$)",
        "canadian_owned_text": r"(?:Canadian[- ]?Owned|Ownership):\s*(.+?)(?:\n|$)",
    }
    
    regex_found_count = 0
    for field, pattern in field_patterns.items():
        match = re.search(pattern, content_text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            regex_found_count += 1
            if field == "canadian_owned_text":
                listing.canadian_owned_text = value
                value_lower = value.lower()
                if value_lower in ["yes", "true", "✓", "✔"]:
                    listing.canadian_owned = True
                elif value_lower in ["no", "false", "✗", "✘"]:
                    listing.canadian_owned = False
            else:
                setattr(listing, field, value)
    
    # If regex didn't find enough fields OR website is missing, use AI extraction
    needs_ai = regex_found_count < 3 or not listing.website
    if needs_ai:
        reason = "website missing" if listing.website is None else f"regex found only {regex_found_count} fields"
        log.debug(f"   🤖 Using AI extraction for {url[:50]} ({reason})")
        stats.ai_extractions += 1
        ai_data = await extract_listing_with_ai(content_text, url)
        
        if ai_data:
            # Fill in missing fields from AI extraction
            if not listing.name and ai_data.get("name"):
                listing.name = ai_data["name"]
            if not listing.products and ai_data.get("products"):
                listing.products = ai_data["products"]
            if not listing.manufactured_in and ai_data.get("manufactured_in"):
                listing.manufactured_in = ai_data["manufactured_in"]
            if not listing.where_to_buy and ai_data.get("where_to_buy"):
                listing.where_to_buy = ai_data["where_to_buy"]
            if not listing.website and ai_data.get("website"):
                listing.website = ai_data["website"]
            if listing.canadian_owned is None and ai_data.get("canadian_owned") is not None:
                listing.canadian_owned = ai_data["canadian_owned"]
                listing.canadian_owned_reason = "AI extraction"
            if not listing.description and ai_data.get("description"):
                listing.description = ai_data["description"]
    else:
        stats.regex_extractions += 1
    
    # Add Canadian owned detection reason
    if listing.canadian_owned is not None and not listing.canadian_owned_reason:
        if listing.canadian_owned:
            listing.canadian_owned_reason = f"Page explicitly states Canadian-Owned: '{listing.canadian_owned_text}'"
        else:
            listing.canadian_owned_reason = f"Page states not Canadian-Owned: '{listing.canadian_owned_text}'"
    elif listing.canadian_owned is None:
        listing.canadian_owned_reason = "No Canadian ownership information found on page"
    
    # Use AI to determine if manufactured_in is in Canada (silently - we'll log at end)
    if listing.manufactured_in:
        listing.made_in_canada, listing.made_in_canada_reason = await detect_made_in_canada_from_location(
            listing.manufactured_in, company_name=listing.name or ""
        )
    else:
        listing.made_in_canada_reason = "No manufacturing location provided on page"
    
    # Detect if the website is a Shopify store
    if listing.website:
        listing.is_shopify_store, listing.shopify_meta, listing.shopify_reason = await detect_shopify_store(client, listing.website)
    else:
        listing.shopify_reason = "No website URL to check"
    
    # Log consolidated summary for this listing
    name = listing.name or "Unknown"
    website_display = listing.website or "N/A"
    
    # Canadian owned status
    if listing.canadian_owned is True:
        owned_str = "🇨🇦 ✅ Canadian Owned"
    elif listing.canadian_owned is False:
        owned_str = "🇨🇦 ❌ Not Canadian Owned"
    else:
        owned_str = "🇨🇦 ❓ Unknown"
    
    # Made in Canada status
    if listing.made_in_canada is True:
        mic_str = "🍁 ✅ Made in Canada"
    elif listing.made_in_canada is False:
        mic_str = "🍁 ❌ Not Made in Canada"
    else:
        mic_str = "🍁 ❓ Unknown"
    
    # Shopify status
    if listing.is_shopify_store is True:
        shopify_str = "🛒 ✅ Shopify"
    elif listing.is_shopify_store is False:
        shopify_str = "🛒 ❌ Not Shopify"
    else:
        shopify_str = "🛒 ❓ Unknown"
    
    log.info(f"""
┌─ {name}
│  📄 MadeInCA: {url}
│  🌐 Website:  {website_display}
│  {owned_str} ({listing.canadian_owned_reason})
│  {mic_str} ({listing.made_in_canada_reason})
│  {shopify_str} ({listing.shopify_reason})
└─""")
    
    # Update stats
    stats.total_processed += 1
    
    # Shopify stats
    if listing.is_shopify_store is True:
        stats.shopify_yes += 1
    elif listing.is_shopify_store is False:
        stats.shopify_no += 1
    else:
        stats.shopify_unknown += 1
    
    # Canadian owned stats
    if listing.canadian_owned is True:
        stats.canadian_owned_yes += 1
    elif listing.canadian_owned is False:
        stats.canadian_owned_no += 1
    else:
        stats.canadian_owned_unknown += 1
    
    # Made in Canada stats
    if listing.made_in_canada is True:
        stats.made_in_canada_yes += 1
    elif listing.made_in_canada is False:
        stats.made_in_canada_no += 1
    else:
        stats.made_in_canada_unknown += 1
    
    # Log running stats every 25 listings
    if stats.total_processed % 25 == 0:
        stats.log_running_stats(listing.name)
    
    return listing


async def parse_listing_page(client: httpx.AsyncClient, url: str, category: str, subcategory: str = "") -> Optional[MadeInCAListing]:
    """
    Parse a single listing page and extract structured information.
    Uses regex first, then falls back to AI extraction for better accuracy.
    NOTE: This is the httpx-based version. Prefer parse_listing_page_with_browser for madeinca.ca.
    """
    # Fetch with retry logic
    r = await fetch_with_retry(client, url, timeout=30)
    if not r:
        return None
    
    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    listing = MadeInCAListing(url=url, category=category, subcategory=subcategory or None)
    
    # Extract the title/name from h1
    title_elem = soup.find("h1")
    if title_elem:
        title_text = title_elem.get_text(strip=True)
        title_text = title_text.replace("🍁", "").replace("📈", "").strip()
        if ":" in title_text:
            parts = title_text.split(":", 1)
            listing.name = parts[1].strip() if len(parts) > 1 else title_text
        else:
            listing.name = title_text
    
    # Find the main content area - try multiple selectors
    content = None
    for selector in [".entry-content", ".post-content", ".elementor-widget-theme-post-content", "article.type-post"]:
        content = soup.select_one(selector)
        if content and len(content.get_text(strip=True)) > 100:
            break
    if not content:
        content = soup.find("article") or soup
    
    content_text = content.get_text(separator="\n", strip=True)
    
    # Store raw text from the page
    listing.raw_text = content_text[:10000]  # Limit to 10k chars to avoid huge entries
    
    # Try regex extraction first
    field_patterns = {
        "name": r"(?:Name|Company):\s*(.+?)(?:\n|$)",
        "products": r"Products?:\s*(.+?)(?:\n|$)",
        "manufactured_in": r"(?:Manufactured In|Manufacturing Location|Made In):\s*(.+?)(?:\n|$)",
        "where_to_buy": r"(?:Where to Buy|Available At|Buy At):\s*(.+?)(?:\n|$)",
        "website": r"Website:\s*(.+?)(?:\n|$)",
        "canadian_owned_text": r"(?:Canadian[- ]?Owned|Ownership):\s*(.+?)(?:\n|$)",
    }
    
    regex_found_count = 0
    for field, pattern in field_patterns.items():
        match = re.search(pattern, content_text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            regex_found_count += 1
            if field == "canadian_owned_text":
                listing.canadian_owned_text = value
                value_lower = value.lower()
                if value_lower in ["yes", "true", "✓", "✔"]:
                    listing.canadian_owned = True
                elif value_lower in ["no", "false", "✗", "✘"]:
                    listing.canadian_owned = False
            else:
                setattr(listing, field, value)
    
    # Extract website URL from links if not found
    if not listing.website:
        for link in content.find_all("a", href=True):
            href = link.get("href", "")
            if href and not href.startswith("https://madeinca.ca") and href.startswith("http"):
                listing.website = href
                break
    
    # Get description from paragraphs
    paragraphs = content.find_all("p")
    if paragraphs:
        desc_parts = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if not any(text.startswith(f) for f in ["Name:", "Products:", "Manufactured", "Website:", "Where to", "Canadian"]):
                desc_parts.append(text)
        listing.description = " ".join(desc_parts)
    
    # If regex didn't find enough fields OR website is missing, use AI extraction
    needs_ai = regex_found_count < 3 or not listing.website
    if needs_ai:
        reason = "website missing" if listing.website is None else f"regex found only {regex_found_count} fields"
        log.debug(f"   🤖 Using AI extraction for {url[:50]} ({reason})")
        stats.ai_extractions += 1
        ai_data = await extract_listing_with_ai(content_text, url)
        
        if ai_data:
            # Fill in missing fields from AI extraction
            if not listing.name and ai_data.get("name"):
                listing.name = ai_data["name"]
            if not listing.products and ai_data.get("products"):
                listing.products = ai_data["products"]
            if not listing.manufactured_in and ai_data.get("manufactured_in"):
                listing.manufactured_in = ai_data["manufactured_in"]
            if not listing.where_to_buy and ai_data.get("where_to_buy"):
                listing.where_to_buy = ai_data["where_to_buy"]
            if not listing.website and ai_data.get("website"):
                listing.website = ai_data["website"]
            if listing.canadian_owned is None and ai_data.get("canadian_owned") is not None:
                listing.canadian_owned = ai_data["canadian_owned"]
                listing.canadian_owned_reason = "AI extraction"
            if not listing.description and ai_data.get("description"):
                listing.description = ai_data["description"]
    else:
        stats.regex_extractions += 1
    
    # Add Canadian owned detection reason for regex-extracted values
    if listing.canadian_owned is not None and not listing.canadian_owned_reason:
        if listing.canadian_owned:
            listing.canadian_owned_reason = f"Page explicitly states Canadian-Owned: '{listing.canadian_owned_text}'"
            log.info(f"🇨🇦 ✅ Canadian Owned: {listing.name or 'Unknown'} - {listing.canadian_owned_reason}")
        else:
            listing.canadian_owned_reason = f"Page states not Canadian-Owned: '{listing.canadian_owned_text}'"
            log.debug(f"   🇨🇦 ❌ NOT Canadian Owned: {listing.name or 'Unknown'} - {listing.canadian_owned_reason}")
    elif listing.canadian_owned is None:
        listing.canadian_owned_reason = "No Canadian ownership information found on page"
        log.debug(f"   🇨🇦 ❓ Unknown ownership: {listing.name or 'Unknown'}")
    
    # Use AI to determine if manufactured_in is in Canada
    if listing.manufactured_in:
        listing.made_in_canada, listing.made_in_canada_reason = await detect_made_in_canada_from_location(
            listing.manufactured_in, company_name=listing.name or ""
        )
    else:
        listing.made_in_canada_reason = "No manufacturing location provided on page"
        log.debug(f"   🍁 ❓ Unknown origin: {listing.name or 'Unknown'} - {listing.made_in_canada_reason}")
    
    # Detect if the website is a Shopify store (with delay to avoid rate limits)
    if listing.website:
        listing.is_shopify_store, listing.shopify_meta, listing.shopify_reason = await detect_shopify_store(client, listing.website)
    else:
        listing.shopify_reason = "No website URL to check"
        log.debug(f"   🛒 ❓ Unknown platform: {listing.name or 'Unknown'} - {listing.shopify_reason}")
    
    # Update stats
    stats.total_processed += 1
    
    # Shopify stats
    if listing.is_shopify_store is True:
        stats.shopify_yes += 1
    elif listing.is_shopify_store is False:
        stats.shopify_no += 1
    else:
        stats.shopify_unknown += 1
    
    # Canadian owned stats
    if listing.canadian_owned is True:
        stats.canadian_owned_yes += 1
    elif listing.canadian_owned is False:
        stats.canadian_owned_no += 1
    else:
        stats.canadian_owned_unknown += 1
    
    # Made in Canada stats
    if listing.made_in_canada is True:
        stats.made_in_canada_yes += 1
    elif listing.made_in_canada is False:
        stats.made_in_canada_no += 1
    else:
        stats.made_in_canada_unknown += 1
    
    # Log running stats every 25 listings
    if stats.total_processed % 25 == 0:
        stats.log_running_stats(listing.name)
    
    return listing


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-postgres", action="store_true", default=True, help="Save to PostgreSQL (default: True)")
    ap.add_argument("--no-postgres", action="store_true", help="Save to JSON instead of PostgreSQL")
    ap.add_argument("--out", default="./out/madeinca", help="Output directory for JSON")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of listings (0 = all)")
    ap.add_argument("--max-categories", type=int, default=0, help="Limit number of categories to crawl (0 = all)")
    ap.add_argument("--show-browser", action="store_true", help="Show browser window")
    ap.add_argument("--concurrency", type=int, default=3, help="Concurrent requests (lower = fewer 429 errors)")
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    # Step 1: Discover all listing URLs
    log.info("🔍 [1/3] Discovering listing URLs from madeinca.ca...")
    step1_start = time.time()
    
    listing_urls = await discover_all_listing_urls(
        headless=not args.show_browser,
        max_categories=args.max_categories
    )
    
    if args.limit > 0:
        listing_urls = listing_urls[:args.limit]
    
    step1_elapsed = time.time() - step1_start
    log.info(f"✅ Found {len(listing_urls)} listings in {step1_elapsed:.1f}s")
    
    if not listing_urls:
        log.error("❌ No listings found")
        sys.exit(1)
    
    # Step 2: Parse each listing page using Playwright browser
    log.info(f"📦 [2/3] Parsing listing pages with browser...")
    step2_start = time.time()
    
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MadeInCABot/1.0)"}
    limits = httpx.Limits(max_connections=args.concurrency)
    
    listings: List[Dict[str, Any]] = []
    
    async with httpx.AsyncClient(headers=headers, limits=limits) as client:
        async with async_playwright() as p:
            browser = await p.firefox.launch(headless=not args.show_browser)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
                viewport={"width": 1920, "height": 1080},
            )
            page = await context.new_page()
            
            pbar = tqdm(total=len(listing_urls), desc="📦 Parsing listings", unit="page")
            
            for url, cat, subcat in listing_urls:
                result = await parse_listing_page_with_browser(page, client, url, cat, subcat)
                if result and result.name:
                    listings.append(result.to_dict())
                pbar.update(1)
                await asyncio.sleep(WAIT_BETWEEN_REQUESTS)
            
            pbar.close()
            await browser.close()
    
    step2_elapsed = time.time() - step2_start
    log.info(f"✅ Parsed {len(listings)} valid listings in {step2_elapsed:.1f}s")
    
    # Step 3: Save to database or JSON
    use_postgres = args.use_postgres and not args.no_postgres
    if use_postgres:
        log.info("💾 [3/3] Saving to PostgreSQL...")
        db = MadeInCADatabaseManager(DB_CONFIG)
        await db.connect()
        await db.initialize_schema()
        
        saved_count = await db.save_listings_batch(listings)
        total_count = await db.get_listing_count()
        await db.close()
        
        log.info(f"✅ Saved {saved_count} listings to PostgreSQL ({total_count} total)")
    else:
        out_json = os.path.join(args.out, "madeinca_listings.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(listings, f, ensure_ascii=False, indent=2)
        log.info(f"💾 Saved {len(listings)} listings to {out_json}")
    
    # Final summary with detailed stats
    made_in_canada_count = sum(1 for l in listings if l.get("made_in_canada") is True)
    canadian_owned_count = sum(1 for l in listings if l.get("canadian_owned") is True)
    shopify_store_count = sum(1 for l in listings if l.get("is_shopify_store") is True)
    
    # Calculate percentages
    total = len(listings) if listings else 1
    mic_pct = made_in_canada_count / total * 100
    owned_pct = canadian_owned_count / total * 100
    shopify_pct = shopify_store_count / total * 100
    
    log.info(f"""
🎉 Scraping complete!

📊 FINAL STATISTICS
{'='*50}
Total listings scraped: {len(listings)}

🍁 Made in Canada:
   ✅ Yes: {made_in_canada_count} ({mic_pct:.1f}%)
   ❌ No: {stats.made_in_canada_no} ({stats.made_in_canada_no / total * 100:.1f}%)
   ❓ Unknown: {stats.made_in_canada_unknown} ({stats.made_in_canada_unknown / total * 100:.1f}%)

🇨🇦 Canadian Owned:
   ✅ Yes: {canadian_owned_count} ({owned_pct:.1f}%)
   ❌ No: {stats.canadian_owned_no} ({stats.canadian_owned_no / total * 100:.1f}%)
   ❓ Unknown: {stats.canadian_owned_unknown} ({stats.canadian_owned_unknown / total * 100:.1f}%)

🛒 Shopify Stores:
   ✅ Yes: {shopify_store_count} ({shopify_pct:.1f}%)
   ❌ No: {stats.shopify_no} ({stats.shopify_no / total * 100:.1f}%)
   ❓ Unknown: {stats.shopify_unknown} ({stats.shopify_unknown / total * 100:.1f}%)

🤖 Extraction Methods:
   Regex: {stats.regex_extractions}
   AI: {stats.ai_extractions}
{'='*50}
    """)


if __name__ == "__main__":
    asyncio.run(main())

