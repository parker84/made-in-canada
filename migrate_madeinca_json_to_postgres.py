#!/usr/bin/env python3
"""
Migrate madeinca listings from JSON file to PostgreSQL.

Usage:
    uv run python migrate_madeinca_json_to_postgres.py --json ./out/madeinca/madeinca_listings.json
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import colorlog
from decouple import config

# Import the database manager from scrape_madeinca
try:
    import psycopg
    from psycopg.types.json import Json
    import cohere
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: uv add 'psycopg[binary]' cohere")
    sys.exit(1)


# Setup logging
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
    logger = logging.getLogger("migrate")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


log = setup_logger()

# Configuration
EMBEDDING_MODEL = "embed-v4.0"
EMBEDDING_DIMENSIONS = 1536

DB_CONFIG = {
    "host": config("POSTGRES_HOST"),
    "dbname": config("POSTGRES_DB"),
    "user": config("POSTGRES_USER"),
    "password": config("POSTGRES_PASSWORD", default=""),
}


class MigrationDatabaseManager:
    """Simplified database manager for migration"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.cohere_client = None
    
    async def connect(self):
        cohere_api_key = config("COHERE_API_KEY", default="")
        if not cohere_api_key:
            log.error("❌ COHERE_API_KEY not set")
            sys.exit(1)
        self.cohere_client = cohere.AsyncClientV2(api_key=cohere_api_key)
        
        log.info("🔌 Connecting to database...")
        conn_string = f"host={self.db_config['host']} dbname={self.db_config['dbname']} user={self.db_config['user']} password={self.db_config['password']}"
        self.conn = await psycopg.AsyncConnection.connect(conn_string)
        log.info("✅ Database connected")
    
    async def initialize_schema(self):
        """Ensure table exists with all columns"""
        log.info("📋 Initializing schema...")
        
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
            
            # Migrations
            await cur.execute("ALTER TABLE madeinca_listings ADD COLUMN IF NOT EXISTS subcategory TEXT")
            await cur.execute("ALTER TABLE madeinca_listings ADD COLUMN IF NOT EXISTS is_shopify_store BOOLEAN")
            await cur.execute("ALTER TABLE madeinca_listings ADD COLUMN IF NOT EXISTS shopify_meta JSONB")
            await cur.execute("ALTER TABLE madeinca_listings ADD COLUMN IF NOT EXISTS raw_text TEXT")
            
            await self.conn.commit()
        
        log.info("✅ Schema ready")
    
    async def generate_embedding(self, text: str):
        response = await self.cohere_client.embed(
            texts=[text],
            model=EMBEDDING_MODEL,
            input_type="search_document",
            embedding_types=["float"],
            output_dimension=int(EMBEDDING_DIMENSIONS),
        )
        return response.embeddings.float_[0]
    
    def _create_listing_text(self, listing):
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
    
    async def save_listing(self, listing):
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
            await self.conn.commit()
            return True
    
    async def get_count(self):
        async with self.conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM madeinca_listings")
            result = await cur.fetchone()
            return result[0]
    
    async def close(self):
        if self.conn:
            await self.conn.close()


async def main():
    ap = argparse.ArgumentParser(description="Migrate madeinca JSON to PostgreSQL")
    ap.add_argument("--json", required=True, help="Path to JSON file")
    ap.add_argument("--dry-run", action="store_true", help="Don't actually save, just validate")
    args = ap.parse_args()
    
    json_path = Path(args.json)
    if not json_path.exists():
        log.error(f"❌ JSON file not found: {json_path}")
        sys.exit(1)
    
    # Load JSON
    log.info(f"📂 Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        listings = json.load(f)
    
    log.info(f"📦 Found {len(listings)} listings in JSON")
    
    if args.dry_run:
        log.info("🧪 Dry run mode - not saving to database")
        for i, listing in enumerate(listings[:5]):
            log.info(f"   {i+1}. {listing.get('name', 'Unknown')} - {listing.get('url', 'N/A')}")
        if len(listings) > 5:
            log.info(f"   ... and {len(listings) - 5} more")
        return
    
    # Connect to database
    db = MigrationDatabaseManager(DB_CONFIG)
    await db.connect()
    await db.initialize_schema()
    
    # Get initial count
    initial_count = await db.get_count()
    log.info(f"📊 Current listings in database: {initial_count}")
    
    # Migrate listings
    log.info("🚀 Starting migration...")
    saved = 0
    errors = 0
    
    for i, listing in enumerate(listings):
        try:
            await db.save_listing(listing)
            saved += 1
            if saved % 10 == 0:
                log.info(f"   💾 Progress: {saved}/{len(listings)} saved...")
            # Small delay to avoid rate limiting on Cohere
            await asyncio.sleep(0.5)
        except Exception as e:
            errors += 1
            log.warning(f"   ⚠️ Failed to save {listing.get('name', 'Unknown')}: {e}")
    
    # Final count
    final_count = await db.get_count()
    await db.close()
    
    log.info("")
    log.info("=" * 50)
    log.info("✅ Migration complete!")
    log.info(f"   📥 Imported: {saved}")
    log.info(f"   ❌ Errors: {errors}")
    log.info(f"   📊 Total in database: {final_count}")
    log.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())

