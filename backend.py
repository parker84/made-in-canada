"""
Made in Canada - Backend API

FastAPI backend for:
- Click tracking with UTM parameters
- Future: Agent API, webhooks, etc.

Run with: uvicorn backend:app --port 8000 --reload
"""

from fastapi import FastAPI, Query, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse
from datetime import datetime
from typing import Optional
import logging
import os

from decouple import config

# Optional: PostgreSQL for persistent storage
psycopg_available = False
try:
    import psycopg
    psycopg_available = True
except ImportError:
    pass

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backend")

# Database Configuration
DB_CONFIG = {
    "host": config("POSTGRES_HOST", default="localhost"),
    "dbname": config("POSTGRES_DB", default="madeinca"),
    "user": config("POSTGRES_USER", default="postgres"),
    "password": config("POSTGRES_PASSWORD", default=""),
}

# UTM Configuration
UTM_SOURCE = "madeincanada.dev"
UTM_MEDIUM = "referral"
UTM_CAMPAIGN = config("UTM_CAMPAIGN", default="madeincanada.dev")
REFERRER = config("REFERRER", default="madeincanada.dev")

# Environment Configuration
ENVIRONMENT = config("ENVIRONMENT", default="development")  # "development" or "production"

app = FastAPI(
    title="Made in Canada API",
    description="Backend API for the Made in Canada shopping assistant",
    version="1.0.0",
)

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your Streamlit domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db_connection_string() -> str:
    """Build PostgreSQL connection string"""
    return f"host={DB_CONFIG['host']} dbname={DB_CONFIG['dbname']} user={DB_CONFIG['user']} password={DB_CONFIG['password']}"


async def init_click_tracking_table():
    """Initialize the click tracking table if it doesn't exist"""
    if not psycopg_available:
        log.warning("psycopg not available - click tracking will only log to console")
        return
    
    try:
        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS link_clicks (
                        id SERIAL PRIMARY KEY,
                        url TEXT NOT NULL,
                        tracked_url TEXT,
                        final_url TEXT,
                        source TEXT,
                        source_type TEXT,
                        product_name TEXT,
                        product_id TEXT,
                        user_id TEXT,
                        session_id TEXT,
                        referrer TEXT,
                        user_agent TEXT,
                        referer TEXT,
                        ip_address TEXT,
                        environment TEXT DEFAULT 'development',
                        clicked_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                # Add columns if they don't exist (migrations)
                await cur.execute("""
                    ALTER TABLE link_clicks ADD COLUMN IF NOT EXISTS environment TEXT DEFAULT 'development'
                """)
                await cur.execute("""
                    ALTER TABLE link_clicks ADD COLUMN IF NOT EXISTS tracked_url TEXT
                """)
                await cur.execute("""
                    ALTER TABLE link_clicks ADD COLUMN IF NOT EXISTS final_url TEXT
                """)
                await cur.execute("""
                    ALTER TABLE link_clicks ADD COLUMN IF NOT EXISTS referrer TEXT
                """)
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_clicks_url ON link_clicks(url)")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_clicks_user ON link_clicks(user_id)")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_clicks_time ON link_clicks(clicked_at)")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_clicks_source ON link_clicks(source)")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_clicks_environment ON link_clicks(environment)")
                await conn.commit()
        log.info("✅ Click tracking table initialized")
    except Exception as e:
        log.error(f"❌ Failed to initialize click tracking table: {e}")


@app.on_event("startup")
async def startup():
    """Initialize database on startup"""
    await init_click_tracking_table()


def add_utm_params(url: str, extra_params: Optional[dict] = None) -> str:
    """Add UTM parameters to a URL"""
    parsed = urlparse(url)
    existing_params = parse_qs(parsed.query)
    
    # Add UTM params (don't override if already present)
    utm_params = {
        "utm_source": UTM_SOURCE,
        "utm_medium": UTM_MEDIUM,
        "utm_campaign": UTM_CAMPAIGN,
    }
    
    # Merge with extra params
    if extra_params:
        utm_params.update(extra_params)
    
    # Only add if not already present
    for key, value in utm_params.items():
        if key not in existing_params:
            existing_params[key] = [value]
    
    # Flatten the params (parse_qs returns lists)
    flat_params = {k: v[0] if isinstance(v, list) else v for k, v in existing_params.items()}
    
    # Rebuild URL
    new_query = urlencode(flat_params)
    new_url = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment,
    ))
    
    return new_url


async def log_click(
    url: str,
    tracked_url: str,
    final_url: str,
    source: Optional[str],
    source_type: Optional[str],
    product_name: Optional[str],
    product_id: Optional[str],
    user_id: Optional[str],
    session_id: Optional[str],
    referrer: Optional[str],
    user_agent: Optional[str],
    referer: Optional[str],
    ip_address: Optional[str],
):
    """Log a click to the database"""
    log.info(f"🔗 Click [{ENVIRONMENT}]: {url} | user={user_id} | product={product_name}")
    
    if not psycopg_available:
        return
    
    try:
        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO link_clicks 
                    (url, tracked_url, final_url, source, source_type, product_name, product_id, user_id, session_id, referrer, user_agent, referer, ip_address, environment, clicked_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    url,
                    tracked_url,
                    final_url,
                    source,
                    source_type,
                    product_name,
                    product_id,
                    user_id,
                    session_id,
                    referrer,
                    user_agent,
                    referer,
                    ip_address,
                    ENVIRONMENT,
                    datetime.now(),
                ))
                await conn.commit()
    except Exception as e:
        log.error(f"❌ Failed to log click: {e}")


@app.get("/click")
async def track_click(
    request: Request,
    url: str = Query(..., description="Target URL to redirect to"),
    source: Optional[str] = Query(None, description="Source site (e.g., 'roots', 'madeinca')"),
    source_type: Optional[str] = Query(None, description="Type of source (e.g., 'brand', 'directory')"),
    product_name: Optional[str] = Query(None, description="Product name"),
    product_id: Optional[str] = Query(None, description="Product ID"),
    user_id: Optional[str] = Query(None, description="User ID"),
    session_id: Optional[str] = Query(None, description="Session ID"),
    referrer: Optional[str] = Query(None, description="Referrer site (e.g., 'madeincanada.dev')"),
    utm_content: Optional[str] = Query(None, description="Additional UTM content tag"),
):
    """
    Track a link click and redirect to the target URL with UTM parameters.
    
    Example:
        /click?url=https://nrsbrakes.com&source=madeinca&product_name=NRS%20Brakes&referrer=madeincanada.dev
    """
    # Get request metadata
    user_agent = request.headers.get("user-agent")
    referer = request.headers.get("referer")
    ip_address = request.client.host if request.client else None
    
    # Reconstruct the full tracked URL from the request
    tracked_url = str(request.url)
    
    # Add UTM params to create final URL
    extra_params = {}
    if utm_content:
        extra_params["utm_content"] = utm_content
    if product_name:
        extra_params["utm_term"] = product_name.replace(" ", "_").lower()[:50]
    # Add referrer as a query param so brands can see where traffic came from
    if referrer:
        extra_params["referrer"] = referrer
    elif REFERRER:
        extra_params["referrer"] = REFERRER
    
    final_url = add_utm_params(url, extra_params)
    
    # Log the click with all URLs
    await log_click(
        url=url,
        tracked_url=tracked_url,
        final_url=final_url,
        source=source,
        source_type=source_type,
        product_name=product_name,
        product_id=product_id,
        user_id=user_id,
        session_id=session_id,
        referrer=referrer or REFERRER,  # Use passed referrer or default
        user_agent=user_agent,
        referer=referer,
        ip_address=ip_address,
    )
    
    return RedirectResponse(url=final_url, status_code=302)


@app.get("/api/clicks/stats")
async def get_click_stats(
    days: int = Query(7, description="Number of days to look back"),
    source: Optional[str] = Query(None, description="Filter by source"),
    environment: Optional[str] = Query(None, description="Filter by environment (development/production)"),
):
    """Get click statistics"""
    if not psycopg_available:
        return JSONResponse({"error": "Database not available"}, status_code=503)
    
    try:
        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            async with conn.cursor() as cur:
                # Build WHERE clause dynamically
                where_clauses = ["clicked_at > NOW() - INTERVAL '%s days'"]
                params = [days]
                
                if source:
                    where_clauses.append("source = %s")
                    params.append(source)
                
                if environment:
                    where_clauses.append("environment = %s")
                    params.append(environment)
                
                where_sql = " AND ".join(where_clauses)
                
                # Total clicks
                query = f"""
                    SELECT 
                        COUNT(*) as total_clicks,
                        COUNT(DISTINCT user_id) as unique_users,
                        COUNT(DISTINCT url) as unique_urls
                    FROM link_clicks 
                    WHERE {where_sql}
                """
                
                await cur.execute(query, params)
                row = await cur.fetchone()
                
                # Top clicked URLs (with same filters)
                top_query = f"""
                    SELECT url, source, product_name, environment, COUNT(*) as clicks
                    FROM link_clicks
                    WHERE {where_sql}
                    GROUP BY url, source, product_name, environment
                    ORDER BY clicks DESC
                    LIMIT 20
                """
                await cur.execute(top_query, params)
                top_urls = await cur.fetchall()
                
                # Clicks by environment breakdown
                env_query = f"""
                    SELECT environment, COUNT(*) as clicks
                    FROM link_clicks
                    WHERE clicked_at > NOW() - INTERVAL '%s days'
                    GROUP BY environment
                """
                await cur.execute(env_query, [days])
                env_breakdown = await cur.fetchall()
                
                return {
                    "period_days": days,
                    "filter_environment": environment,
                    "filter_source": source,
                    "total_clicks": row[0] if row else 0,
                    "unique_users": row[1] if row else 0,
                    "unique_urls": row[2] if row else 0,
                    "clicks_by_environment": {
                        r[0]: r[1] for r in env_breakdown
                    },
                    "top_urls": [
                        {"url": r[0], "source": r[1], "product_name": r[2], "environment": r[3], "clicks": r[4]}
                        for r in top_urls
                    ],
                }
    except Exception as e:
        log.error(f"❌ Failed to get click stats: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "madeinca-backend"}


# Helper function to generate tracked URLs (can be imported by agent)
def create_tracked_url(
    url: str,
    source: Optional[str] = None,
    source_type: Optional[str] = None,
    product_name: Optional[str] = None,
    product_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    base_url: str = "http://localhost:8000",
) -> str:
    """
    Create a tracked URL that goes through the click tracking endpoint.
    
    Usage in agent:
        from backend import create_tracked_url
        tracked = create_tracked_url("https://nrsbrakes.com", source="madeinca", product_name="NRS Brakes")
    """
    from urllib.parse import quote
    
    params = {"url": url}
    if source:
        params["source"] = source
    if source_type:
        params["source_type"] = source_type
    if product_name:
        params["product_name"] = product_name
    if product_id:
        params["product_id"] = product_id
    if user_id:
        params["user_id"] = user_id
    if session_id:
        params["session_id"] = session_id
    
    query = urlencode(params)
    return f"{base_url}/click?{query}"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

