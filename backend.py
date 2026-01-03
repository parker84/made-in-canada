"""
Made in Canada - Backend API

FastAPI backend for:
- Click tracking with UTM parameters
- Agent chat API with streaming
- Product search API

Run with: uvicorn backend:app --port 8000 --reload
"""

from fastapi import FastAPI, Query, Request
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse
from datetime import datetime
from typing import Optional, AsyncGenerator
from pydantic import BaseModel
import logging
import os
import asyncio
import json

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


async def init_pageview_tracking_table():
    """Initialize the pageview tracking table if it doesn't exist"""
    if not psycopg_available:
        return
    
    try:
        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS pageviews (
                        id SERIAL PRIMARY KEY,
                        page TEXT NOT NULL,
                        user_id TEXT,
                        session_id TEXT,
                        referrer TEXT,
                        user_agent TEXT,
                        ip_address TEXT,
                        environment TEXT DEFAULT 'development',
                        metadata JSONB,
                        viewed_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_pageviews_page ON pageviews(page)")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_pageviews_user ON pageviews(user_id)")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_pageviews_time ON pageviews(viewed_at)")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_pageviews_session ON pageviews(session_id)")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_pageviews_environment ON pageviews(environment)")
                await conn.commit()
        log.info("✅ Pageview tracking table initialized")
    except Exception as e:
        log.error(f"❌ Failed to initialize pageview tracking table: {e}")


async def init_feedback_table():
    """Initialize the feedback tracking table if it doesn't exist"""
    if not psycopg_available:
        return
    
    try:
        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id SERIAL PRIMARY KEY,
                        message_id TEXT NOT NULL,
                        user_id TEXT,
                        session_id TEXT,
                        query TEXT,
                        response TEXT,
                        rating TEXT NOT NULL,
                        comment TEXT,
                        environment TEXT DEFAULT 'development',
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id)")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id)")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_time ON feedback(created_at)")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_environment ON feedback(environment)")
                await conn.commit()
        log.info("✅ Feedback table initialized")
    except Exception as e:
        log.error(f"❌ Failed to initialize feedback table: {e}")


@app.on_event("startup")
async def startup():
    """Initialize database on startup"""
    await init_click_tracking_table()
    await init_pageview_tracking_table()
    await init_feedback_table()


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


# ============================================================================
# Pageview Tracking
# ============================================================================

class PageviewRequest(BaseModel):
    """Request body for logging a pageview"""
    page: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    referrer: Optional[str] = None
    metadata: Optional[dict] = None


async def log_pageview(
    page: str,
    user_id: Optional[str],
    session_id: Optional[str],
    referrer: Optional[str],
    user_agent: Optional[str],
    ip_address: Optional[str],
    metadata: Optional[dict],
):
    """Log a pageview to the database"""
    log.info(f"👁️ Pageview [{ENVIRONMENT}]: {page} | user={user_id} | session={session_id}")
    
    if not psycopg_available:
        return
    
    try:
        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO pageviews 
                    (page, user_id, session_id, referrer, user_agent, ip_address, environment, metadata, viewed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    page,
                    user_id,
                    session_id,
                    referrer,
                    user_agent,
                    ip_address,
                    ENVIRONMENT,
                    json.dumps(metadata) if metadata else None,
                    datetime.now(),
                ))
                await conn.commit()
    except Exception as e:
        log.error(f"❌ Failed to log pageview: {e}")


@app.post("/api/pageview")
async def track_pageview(request: Request, body: PageviewRequest):
    """
    Track a pageview event.
    
    Example:
        POST /api/pageview
        {"page": "/", "user_id": "abc123", "session_id": "sess456"}
    """
    user_agent = request.headers.get("user-agent")
    ip_address = request.client.host if request.client else None
    
    await log_pageview(
        page=body.page,
        user_id=body.user_id,
        session_id=body.session_id,
        referrer=body.referrer,
        user_agent=user_agent,
        ip_address=ip_address,
        metadata=body.metadata,
    )
    
    return {"status": "ok"}


@app.get("/api/pageviews/stats")
async def get_pageview_stats(
    days: int = Query(7, description="Number of days to look back"),
    page: Optional[str] = Query(None, description="Filter by page"),
    environment: Optional[str] = Query(None, description="Filter by environment (development/production)"),
):
    """Get pageview statistics"""
    if not psycopg_available:
        return JSONResponse({"error": "Database not available"}, status_code=503)
    
    try:
        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            async with conn.cursor() as cur:
                # Build WHERE clause dynamically
                where_clauses = ["viewed_at > NOW() - INTERVAL '%s days'"]
                params = [days]
                
                if page:
                    where_clauses.append("page = %s")
                    params.append(page)
                
                if environment:
                    where_clauses.append("environment = %s")
                    params.append(environment)
                
                where_sql = " AND ".join(where_clauses)
                
                # Total pageviews
                query = f"""
                    SELECT 
                        COUNT(*) as total_pageviews,
                        COUNT(DISTINCT user_id) as unique_users,
                        COUNT(DISTINCT session_id) as unique_sessions
                    FROM pageviews 
                    WHERE {where_sql}
                """
                
                await cur.execute(query, params)
                row = await cur.fetchone()
                
                # Top pages (with same filters)
                top_query = f"""
                    SELECT page, environment, COUNT(*) as views
                    FROM pageviews
                    WHERE {where_sql}
                    GROUP BY page, environment
                    ORDER BY views DESC
                    LIMIT 20
                """
                await cur.execute(top_query, params)
                top_pages = await cur.fetchall()
                
                # Pageviews by environment breakdown
                env_query = f"""
                    SELECT environment, COUNT(*) as views
                    FROM pageviews
                    WHERE viewed_at > NOW() - INTERVAL '%s days'
                    GROUP BY environment
                """
                await cur.execute(env_query, [days])
                env_breakdown = await cur.fetchall()
                
                # Pageviews by hour (for charts)
                hourly_query = f"""
                    SELECT 
                        date_trunc('hour', viewed_at) as hour,
                        COUNT(*) as views
                    FROM pageviews
                    WHERE {where_sql}
                    GROUP BY hour
                    ORDER BY hour DESC
                    LIMIT 168  -- 7 days of hours
                """
                await cur.execute(hourly_query, params)
                hourly_data = await cur.fetchall()
                
                return {
                    "period_days": days,
                    "filter_environment": environment,
                    "filter_page": page,
                    "total_pageviews": row[0] if row else 0,
                    "unique_users": row[1] if row else 0,
                    "unique_sessions": row[2] if row else 0,
                    "pageviews_by_environment": {
                        r[0]: r[1] for r in env_breakdown
                    },
                    "top_pages": [
                        {"page": r[0], "environment": r[1], "views": r[2]}
                        for r in top_pages
                    ],
                    "hourly_data": [
                        {"hour": r[0].isoformat() if r[0] else None, "views": r[1]}
                        for r in hourly_data
                    ],
                }
    except Exception as e:
        log.error(f"❌ Failed to get pageview stats: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================================
# Feedback Tracking
# ============================================================================

class FeedbackRequest(BaseModel):
    """Request body for submitting feedback"""
    message_id: str
    rating: str  # "up" or "down"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    query: Optional[str] = None
    response: Optional[str] = None
    comment: Optional[str] = None


async def log_feedback(
    message_id: str,
    rating: str,
    user_id: Optional[str],
    session_id: Optional[str],
    query: Optional[str],
    response: Optional[str],
    comment: Optional[str],
):
    """Log feedback to the database"""
    emoji = "👍" if rating == "up" else "👎"
    log.info(f"{emoji} Feedback [{ENVIRONMENT}]: {rating} | user={user_id} | message={message_id}")
    
    if not psycopg_available:
        return
    
    try:
        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO feedback 
                    (message_id, user_id, session_id, query, response, rating, comment, environment, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    message_id,
                    user_id,
                    session_id,
                    query[:2000] if query else None,  # Truncate long queries
                    response[:5000] if response else None,  # Truncate long responses
                    rating,
                    comment,
                    ENVIRONMENT,
                    datetime.now(),
                ))
                await conn.commit()
    except Exception as e:
        log.error(f"❌ Failed to log feedback: {e}")


@app.post("/api/feedback")
async def submit_feedback(body: FeedbackRequest):
    """
    Submit feedback for a chat response.
    
    Example:
        POST /api/feedback
        {"message_id": "abc123", "rating": "up", "query": "winter jackets", "response": "..."}
    """
    if body.rating not in ("up", "down"):
        return JSONResponse({"error": "rating must be 'up' or 'down'"}, status_code=400)
    
    await log_feedback(
        message_id=body.message_id,
        rating=body.rating,
        user_id=body.user_id,
        session_id=body.session_id,
        query=body.query,
        response=body.response,
        comment=body.comment,
    )
    
    return {"status": "ok"}


@app.get("/api/feedback/stats")
async def get_feedback_stats(
    days: int = Query(7, description="Number of days to look back"),
    environment: Optional[str] = Query(None, description="Filter by environment (development/production)"),
):
    """Get feedback statistics"""
    if not psycopg_available:
        return JSONResponse({"error": "Database not available"}, status_code=503)
    
    try:
        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            async with conn.cursor() as cur:
                # Build WHERE clause dynamically
                where_clauses = ["created_at > NOW() - INTERVAL '%s days'"]
                params = [days]
                
                if environment:
                    where_clauses.append("environment = %s")
                    params.append(environment)
                
                where_sql = " AND ".join(where_clauses)
                
                # Total feedback counts
                query = f"""
                    SELECT 
                        COUNT(*) as total_feedback,
                        COUNT(*) FILTER (WHERE rating = 'up') as thumbs_up,
                        COUNT(*) FILTER (WHERE rating = 'down') as thumbs_down,
                        COUNT(DISTINCT user_id) as unique_users
                    FROM feedback 
                    WHERE {where_sql}
                """
                
                await cur.execute(query, params)
                row = await cur.fetchone()
                
                # Recent negative feedback (for review)
                negative_query = f"""
                    SELECT message_id, user_id, query, comment, created_at
                    FROM feedback
                    WHERE {where_sql} AND rating = 'down'
                    ORDER BY created_at DESC
                    LIMIT 20
                """
                await cur.execute(negative_query, params)
                negative_feedback = await cur.fetchall()
                
                # Feedback by environment
                env_query = f"""
                    SELECT environment, rating, COUNT(*) as count
                    FROM feedback
                    WHERE created_at > NOW() - INTERVAL '%s days'
                    GROUP BY environment, rating
                """
                await cur.execute(env_query, [days])
                env_breakdown = await cur.fetchall()
                
                return {
                    "period_days": days,
                    "filter_environment": environment,
                    "total_feedback": row[0] if row else 0,
                    "thumbs_up": row[1] if row else 0,
                    "thumbs_down": row[2] if row else 0,
                    "unique_users": row[3] if row else 0,
                    "satisfaction_rate": round(row[1] / row[0] * 100, 1) if row and row[0] > 0 else None,
                    "feedback_by_environment": [
                        {"environment": r[0], "rating": r[1], "count": r[2]}
                        for r in env_breakdown
                    ],
                    "recent_negative_feedback": [
                        {
                            "message_id": r[0],
                            "user_id": r[1],
                            "query": r[2],
                            "comment": r[3],
                            "created_at": r[4].isoformat() if r[4] else None,
                        }
                        for r in negative_feedback
                    ],
                }
    except Exception as e:
        log.error(f"❌ Failed to get feedback stats: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "madeinca-backend"}


# ============================================================================
# Agent Chat API
# ============================================================================

class ChatRequest(BaseModel):
    """Request body for chat endpoint"""
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    referrer: Optional[str] = "madeincanada.dev"


class ChatResponse(BaseModel):
    """Response body for non-streaming chat"""
    content: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


# Lazy-loaded agent (cached)
_agent_instance = None


def get_agent():
    """Get or create the agent instance (cached)"""
    global _agent_instance
    if _agent_instance is None:
        # Import here to avoid circular imports and speed up startup
        from team import get_agent_team_no_cache
        _agent_instance = get_agent_team_no_cache()
        log.info("🤖 Agent initialized")
    return _agent_instance


async def stream_agent_response(
    message: str,
    user_id: Optional[str],
    session_id: Optional[str],
    referrer: Optional[str],
) -> AsyncGenerator[str, None]:
    """Stream agent response as Server-Sent Events"""
    # Set tracking context
    from team import tracking_context
    tracking_context.set_context(
        user_id=user_id,
        session_id=session_id,
        referrer=referrer or "madeincanada.dev",
    )
    
    agent = get_agent()
    
    try:
        # When stream=True, arun() returns an async generator directly (not a coroutine)
        stream = agent.arun(
            message,
            stream=True,
            stream_events=True,
            user_id=user_id,
            session_id=session_id,
        )
        
        async for chunk in stream:
            if hasattr(chunk, "event"):
                # Send different event types
                if chunk.event == "RunContent" and chunk.content:
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk.content})}\n\n"
                elif chunk.event == "ToolCallStarted":
                    tool_name = chunk.tool.tool_name if hasattr(chunk, 'tool') else "unknown"
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name})}\n\n"
                elif chunk.event == "ToolCallCompleted":
                    yield f"data: {json.dumps({'type': 'tool_complete'})}\n\n"
        
        # Signal end of stream
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        log.error(f"❌ Agent error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Chat with the agent (non-streaming).
    
    For streaming, use /api/chat/stream instead.
    """
    from team import tracking_context
    tracking_context.set_context(
        user_id=request.user_id,
        session_id=request.session_id,
        referrer=request.referrer or "madeincanada.dev",
    )
    
    agent = get_agent()
    
    try:
        response = await agent.arun(
            request.message,
            user_id=request.user_id,
            session_id=request.session_id,
        )
        return ChatResponse(
            content=response.content,
            user_id=request.user_id,
            session_id=request.session_id,
        )
    except Exception as e:
        log.error(f"❌ Agent error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Chat with the agent with streaming response (Server-Sent Events).
    
    Response format (SSE):
        data: {"type": "tool_start", "tool": "search_products_sync"}
        data: {"type": "tool_complete"}
        data: {"type": "content", "content": "Here are some..."}
        data: {"type": "content", "content": " Canadian products"}
        data: {"type": "done"}
    
    Usage with JavaScript:
        const eventSource = new EventSource('/api/chat/stream');
        eventSource.onmessage = (e) => {
            const data = JSON.parse(e.data);
            if (data.type === 'content') console.log(data.content);
        };
    """
    return StreamingResponse(
        stream_agent_response(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            referrer=request.referrer,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.post("/api/search/fast")
async def fast_search(request: ChatRequest):
    """
    Fast product search - bypasses LLM agent entirely.
    
    Returns formatted markdown directly from the search tool.
    This is 10x faster than /api/chat because it skips:
    - LLM reasoning about what tool to call
    - LLM processing of tool output
    - LLM generating response tokens
    """
    import time
    start_time = time.time()
    
    # Set tracking context
    from team import tracking_context, search_products
    tracking_context.set_context(
        user_id=request.user_id,
        session_id=request.session_id,
        referrer=request.referrer or "madeincanada.dev",
    )
    
    log.info(f"⚡ Fast search: '{request.message[:50]}...'")
    
    try:
        # Call search directly (no agent)
        result = await search_products(request.message, for_user=True)
        
        elapsed = time.time() - start_time
        log.info(f"⚡ Fast search completed in {elapsed:.2f}s")
        
        return ChatResponse(
            content=result,
            user_id=request.user_id,
            session_id=request.session_id,
        )
    except Exception as e:
        log.error(f"❌ Fast search error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def stream_fast_search(
    message: str,
    user_id: Optional[str],
    session_id: Optional[str],
    referrer: Optional[str],
) -> AsyncGenerator[str, None]:
    """Stream fast search results as SSE - bypasses LLM entirely"""
    import time
    start_time = time.time()
    
    # Set tracking context
    from team import tracking_context, search_products
    tracking_context.set_context(
        user_id=user_id,
        session_id=session_id,
        referrer=referrer or "madeincanada.dev",
    )
    
    log.info(f"⚡ Fast search stream: '{message[:50]}...'")
    
    # Emit tool start
    yield f"data: {json.dumps({'type': 'tool_start', 'tool': 'search_products'})}\n\n"
    
    try:
        # Call search directly (no agent) with user-friendly formatting
        result = await search_products(message, for_user=True)
        
        # Emit tool complete
        elapsed = time.time() - start_time
        log.info(f"⚡ Fast search completed in {elapsed:.2f}s")
        yield f"data: {json.dumps({'type': 'tool_complete'})}\n\n"
        
        # Emit content directly (no LLM processing!)
        yield f"data: {json.dumps({'type': 'content', 'content': result})}\n\n"
        
        # Done
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        log.error(f"❌ Fast search error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


@app.post("/api/search/fast/stream")
async def fast_search_stream(request: ChatRequest):
    """
    Fast product search with SSE streaming - bypasses LLM agent entirely.
    
    Emits the same events as /api/chat/stream but 10x faster because:
    - No LLM reasoning
    - No LLM output processing  
    - Content returned immediately after search
    """
    return StreamingResponse(
        stream_fast_search(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            referrer=request.referrer,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# Product Search API (direct database search, no agent)
# ============================================================================

class SearchRequest(BaseModel):
    """Request body for search endpoint"""
    query: str
    limit: int = 25
    made_in_canada_only: bool = False


class ProductResult(BaseModel):
    """A single product result"""
    name: str
    brand: Optional[str] = None
    description: Optional[str] = None
    price: Optional[str] = None
    currency: Optional[str] = None
    url: str
    source_site: Optional[str] = None
    made_in_canada: Optional[bool] = None
    made_in_canada_reason: Optional[str] = None
    average_rating: Optional[float] = None
    num_reviews: Optional[int] = None
    images: Optional[list] = None


class SearchResponse(BaseModel):
    """Response body for search endpoint"""
    query: str
    total_results: int
    results: list[ProductResult]


@app.post("/api/search", response_model=SearchResponse)
async def search_products_api(request: SearchRequest):
    """
    Search products directly (without the agent).
    
    This is useful for:
    - Building custom UIs
    - Mobile apps
    - Third-party integrations
    
    Returns raw search results without agent formatting.
    """
    try:
        from team import search_products
        
        # Call the search function
        result_str = await search_products(request.query, limit=request.limit)
        
        # Parse results (search_products returns a formatted string for the agent)
        # For the API, we should return structured data
        # Let's call the underlying function directly
        from team import (
            generate_embedding,
            parse_query,
            dedupe_results,
            rerank_results,
            DB_CONFIG,
            INITIAL_SEARCH_LIMIT,
            RERANK_TOP_N,
            MADE_IN_CANADA_BOOST,
        )
        
        parsed = parse_query(request.query)
        embedding = await generate_embedding(parsed.intent)
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        
        conn_string = (
            f"host={DB_CONFIG['host']} "
            f"dbname={DB_CONFIG['dbname']} "
            f"user={DB_CONFIG['user']} "
            f"password={DB_CONFIG['password']}"
        )
        
        mic_filter = "AND p.made_in_canada = true" if request.made_in_canada_only else ""
        
        async with await psycopg.AsyncConnection.connect(conn_string) as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    --sql
                    WITH q AS (
                        SELECT 
                            %s::vector AS q_embedding,
                            plainto_tsquery('english', %s) AS q_ts
                    ),
                    
                    vec AS (
                        SELECT
                            p.name,
                            p.brand,
                            p.description,
                            p.price,
                            p.currency,
                            p.url,
                            p.source_site,
                            1 - (p.embedding <=> q.q_embedding) AS vector_similarity,
                            NULL::float AS text_rank,
                            p.num_reviews,
                            p.average_rating,
                            p.images,
                            p.made_in_canada,
                            p.made_in_canada_reason
                        FROM products p
                        CROSS JOIN q
                        WHERE p.embedding IS NOT NULL {mic_filter}
                        ORDER BY p.embedding <=> q.q_embedding
                        LIMIT %s
                    ),
                    
                    txt AS (
                        SELECT
                            p.name,
                            p.brand,
                            p.description,
                            p.price,
                            p.currency,
                            p.url,
                            p.source_site,
                            NULL::float AS vector_similarity,
                            ts_rank_cd(
                                to_tsvector('english',
                                    coalesce(p.name,'') || ' ' ||
                                    coalesce(p.brand,'') || ' ' ||
                                    coalesce(p.description,'')
                                ),
                                q.q_ts
                            )::float AS text_rank,
                            p.num_reviews,
                            p.average_rating,
                            p.images,
                            p.made_in_canada,
                            p.made_in_canada_reason
                        FROM products p
                        CROSS JOIN q
                        WHERE q.q_ts @@ to_tsvector('english',
                            coalesce(p.name,'') || ' ' ||
                            coalesce(p.brand,'') || ' ' ||
                            coalesce(p.description,'')
                        ) {mic_filter}
                        ORDER BY text_rank DESC
                        LIMIT %s
                    ),
                    
                    combined AS (
                        SELECT * FROM vec
                        UNION ALL
                        SELECT * FROM txt
                    ),
                    
                    deduped AS (
                        SELECT
                            url,
                            max(name) AS name,
                            max(brand) AS brand,
                            max(description) AS description,
                            max(price) AS price,
                            max(currency) AS currency,
                            max(source_site) AS source_site,
                            max(coalesce(vector_similarity, 0.0)) AS vector_similarity,
                            max(coalesce(text_rank, 0.0)) AS text_rank,
                            max(num_reviews) AS num_reviews,
                            max(average_rating) AS average_rating,
                            max(images::text)::jsonb AS images,
                            bool_or(made_in_canada) AS made_in_canada,
                            max(made_in_canada_reason) AS made_in_canada_reason
                        FROM combined
                        GROUP BY url
                    )
                    
                    SELECT
                        name,
                        brand,
                        description,
                        price,
                        currency,
                        url,
                        source_site,
                        made_in_canada,
                        made_in_canada_reason,
                        average_rating,
                        num_reviews,
                        images
                    FROM deduped
                    ORDER BY 
                        (0.5 * vector_similarity) + (0.5 * text_rank) + 
                        CASE WHEN made_in_canada = true THEN %s ELSE 0.0 END DESC
                    LIMIT %s
                    --end-sql
                """, (
                    embedding_str,
                    parsed.intent,
                    INITIAL_SEARCH_LIMIT,
                    INITIAL_SEARCH_LIMIT,
                    MADE_IN_CANADA_BOOST,
                    request.limit,
                ))
                
                rows = await cur.fetchall()
        
        # Convert to response
        products = []
        for row in rows:
            products.append(ProductResult(
                name=row[0] or "",
                brand=row[1],
                description=row[2],
                price=row[3],
                currency=row[4],
                url=row[5] or "",
                source_site=row[6],
                made_in_canada=row[7],
                made_in_canada_reason=row[8],
                average_rating=row[9],
                num_reviews=row[10],
                images=row[11] if row[11] else None,
            ))
        
        return SearchResponse(
            query=request.query,
            total_results=len(products),
            results=products,
        )
        
    except Exception as e:
        log.error(f"❌ Search error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


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

