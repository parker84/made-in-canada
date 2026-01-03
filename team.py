"""
Made in Canada - Agent Team

This module provides an AI agent that can search the product knowledge base
and help users find Canadian products.
"""

import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.cohere import Cohere
from agno.db.postgres import PostgresDb
from textwrap import dedent
from decouple import config
from typing import List
import os
import logging
import asyncio

import coloredlogs
import psycopg
import cohere
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any
from urllib.parse import urlencode

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level=os.getenv("LOG_LEVEL", "INFO"), logger=logger)

# Click tracking configuration
TRACKING_ENABLED = config("TRACKING_ENABLED", default="true").lower() == "true"
TRACKING_BASE_URL = config("TRACKING_BASE_URL", default="http://localhost:8000")


class TrackingContext:
    """
    Stores tracking context (user_id, session_id) for click tracking.
    
    Set the context from your app before running the agent:
        from team import tracking_context
        tracking_context.set_context(user_id="user@example.com", session_id="abc-123")
    
    The context is then automatically used when creating tracked URLs.
    """
    
    def __init__(self):
        self._user_id: Optional[str] = None
        self._session_id: Optional[str] = None
        self._referrer: Optional[str] = None
    
    def set_context(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        referrer: Optional[str] = None,
    ):
        """Set the tracking context for the current request/session"""
        self._user_id = user_id
        self._session_id = session_id
        self._referrer = referrer
        logger.debug(f"📊 Tracking context set: user={user_id}, session={session_id}")
    
    def clear_context(self):
        """Clear the tracking context"""
        self._user_id = None
        self._session_id = None
        self._referrer = None
    
    @property
    def user_id(self) -> Optional[str]:
        return self._user_id
    
    @property
    def session_id(self) -> Optional[str]:
        return self._session_id
    
    @property
    def referrer(self) -> Optional[str]:
        return self._referrer
    
    def create_tracked_url(
        self,
        url: str,
        source: Optional[str] = None,
        source_type: str = "product",
        product_name: Optional[str] = None,
        product_id: Optional[str] = None,
    ) -> str:
        """
        Create a tracked URL that goes through the click tracking endpoint.
        Automatically includes user_id, session_id, and referrer from context.
        Adds UTM parameters automatically on redirect.
        
        If tracking is disabled, returns the original URL.
        """
        if not TRACKING_ENABLED:
            return url
        
        params = {"url": url}
        if source:
            params["source"] = source
        if source_type:
            params["source_type"] = source_type
        if product_name:
            params["product_name"] = product_name[:100]  # Limit length for URL
        if product_id:
            params["product_id"] = product_id
        if self._user_id:
            params["user_id"] = self._user_id
        if self._session_id:
            params["session_id"] = self._session_id
        if self._referrer:
            params["referrer"] = self._referrer
        
        query = urlencode(params)
        return f"{TRACKING_BASE_URL}/click?{query}"


# Global tracking context instance - import and use from your app
tracking_context = TrackingContext()


def create_tracked_url(
    url: str,
    source: Optional[str] = None,
    source_type: str = "product",
    product_name: Optional[str] = None,
    product_id: Optional[str] = None,
) -> str:
    """
    Convenience function that uses the global tracking context.
    Prefer using tracking_context.create_tracked_url() directly.
    """
    return tracking_context.create_tracked_url(
        url=url,
        source=source,
        source_type=source_type,
        product_name=product_name,
        product_id=product_id,
    )


@dataclass
class ParsedQuery:
    """Parsed search query with extracted filters and intent"""
    original: str  # Original query
    intent: str  # Core search intent (product type, what they're looking for)
    filters: Dict[str, Any]  # Extracted filters
    
    # Filter types
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    made_in_canada_only: bool = False
    min_rating: Optional[float] = None
    
    def to_search_query(self) -> str:
        """Convert back to optimized search query for embedding"""
        return self.intent


def parse_query(query: str) -> ParsedQuery:
    """
    Parse user query to extract filters and core intent.
    
    Examples:
        "hoodies under $100" → intent="hoodies", max_price=100
        "Roots jackets made in canada" → intent="jackets", brand="Roots", made_in_canada_only=True
        "warm winter boots 4+ stars" → intent="warm winter boots", min_rating=4.0
    """
    original = query
    filters = {}
    intent = query
    
    min_price = None
    max_price = None
    brand = None
    category = None
    min_rating = None
    made_in_canada_only = query_wants_made_in_canada(query)
    
    # Extract price filters
    # "under $100", "less than $50", "below 75"
    under_match = re.search(r'(?:under|less than|below|max|<)\s*\$?(\d+(?:\.\d{2})?)', query, re.IGNORECASE)
    if under_match:
        max_price = float(under_match.group(1))
        intent = re.sub(r'(?:under|less than|below|max|<)\s*\$?\d+(?:\.\d{2})?', '', intent, flags=re.IGNORECASE)
        filters['max_price'] = max_price
    
    # "over $50", "more than $100", "above 200", "min $30"
    over_match = re.search(r'(?:over|more than|above|min|>)\s*\$?(\d+(?:\.\d{2})?)', query, re.IGNORECASE)
    if over_match:
        min_price = float(over_match.group(1))
        intent = re.sub(r'(?:over|more than|above|min|>)\s*\$?\d+(?:\.\d{2})?', '', intent, flags=re.IGNORECASE)
        filters['min_price'] = min_price
    
    # Price range: "$50-$100", "$50 to $100"
    range_match = re.search(r'\$?(\d+(?:\.\d{2})?)\s*[-–to]+\s*\$?(\d+(?:\.\d{2})?)', query, re.IGNORECASE)
    if range_match:
        min_price = float(range_match.group(1))
        max_price = float(range_match.group(2))
        intent = re.sub(r'\$?\d+(?:\.\d{2})?\s*[-–to]+\s*\$?\d+(?:\.\d{2})?', '', intent, flags=re.IGNORECASE)
        filters['min_price'] = min_price
        filters['max_price'] = max_price
    
    # Extract rating filter: "4+ stars", "4.5 stars", "highly rated"
    rating_match = re.search(r'(\d+(?:\.\d)?)\+?\s*(?:stars?|rating)', query, re.IGNORECASE)
    if rating_match:
        min_rating = float(rating_match.group(1))
        intent = re.sub(r'\d+(?:\.\d)?\+?\s*(?:stars?|rating)', '', intent, flags=re.IGNORECASE)
        filters['min_rating'] = min_rating
    
    # Extract known brand names
    known_brands = [
        "Roots", "Lululemon", "Canada Goose", "Aritzia", "Province of Canada",
        "Mejuri", "Duer", "Reigning Champ", "Arc'teryx", "Mackage", "Moose Knuckles",
        "Naked & Famous", "Wings+Horns", "Frank And Oak", "Tentree", "Kotn",
        "Baffin", "Kamik", "Sorel", "Native Shoes"
    ]
    for b in known_brands:
        if b.lower() in query.lower():
            brand = b
            filters['brand'] = brand
            # Don't remove brand from intent - it helps with search
            break
    
    # Clean up intent
    # Remove "made in canada" phrases as they're handled by the filter
    intent = re.sub(r'(?:made[- ]?in[- ]?canada|canadian[- ]?made|fabriqué au canada)', '', intent, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    intent = ' '.join(intent.split()).strip()
    
    # If intent is now empty or too short, use original minus filter terms
    if len(intent) < 3:
        intent = original
    
    logger.debug(f"📝 Parsed query: intent='{intent}', filters={filters}")
    
    return ParsedQuery(
        original=original,
        intent=intent,
        filters=filters,
        min_price=min_price,
        max_price=max_price,
        brand=brand,
        category=category,
        made_in_canada_only=made_in_canada_only,
        min_rating=min_rating,
    )

# ------------constants
DEBUG_MODE = os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG"

# LLM Provider Configuration
AGENT_LLM_PROVIDER = config("AGENT_LLM_PROVIDER", default="openai")
AGENT_MODEL_ID = config("AGENT_MODEL_ID", default="gpt-5-nano")

# Knowledge Base Configuration
EMBEDDING_MODEL = "embed-v4.0"
EMBEDDING_DIMENSIONS = 1536
RERANK_MODEL = "rerank-v3.5"

# Funnel: SQL_CANDIDATES → dedupe → INITIAL_SEARCH_LIMIT → rerank → PROMPT_TOP_N
SQL_CANDIDATES = 250  # Raw candidates from SQL (before dedupe)
INITIAL_SEARCH_LIMIT = 100  # After dedupe, before rerank
RERANK_TOP_N = 50  # After re-ranking
PROMPT_TOP_N = 50  # Final results to show

# Lengths
MAX_DESCRIPTION_LENGTH = 500
MAX_MARKDOWN_LENGTH = 500

# Made in Canada ranking boost
MADE_IN_CANADA_BOOST = 0.3  # Add 30% boost to hybrid score for Made in Canada products
MADE_IN_CANADA_KEYWORDS = ["made in canada", "canadian made", "made-in-canada", "fabriqué au canada"]

# Rating smoothing (Empirical Bayes)
# Smooth DOWN ratings with few reviews to avoid over-trusting 5-star with 1 review
RATING_PRIOR = 4.0  # Prior average rating (assume average product is ~4 stars)
RATING_CONFIDENCE = 10  # Number of "virtual" prior reviews to add
MIN_REVIEWS_FOR_TRUST = 10  # Below this, use the prior rating for ranking

# Database Configuration
DB_CONFIG = {
    "host": config("POSTGRES_HOST"),
    "dbname": config("POSTGRES_DB"),
    "user": config("POSTGRES_USER"),
    "password": config("POSTGRES_PASSWORD"),
}

db_url = f"postgresql+psycopg://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['dbname']}"

team_storage = PostgresDb(db_url=db_url)

ADDITIONAL_CONTEXT = dedent("""
    Your outputs will be in markdown format so when using $ for money you need to escape it with a backslash.
    Focus on helping Canadian businesses, artists, creators, and the Canadian economy.
    Spell using Canadian proper grammar (ex: "favor" -> "favour").
    You are an expert at finding Made in Canada products.
""")

MAX_TOOL_CALLS = 5
NUM_HISTORY_RUNS = 3


# ------------Knowledge Base Functions
def query_wants_made_in_canada(query: str) -> bool:
    """Detect if the query is explicitly asking for Made in Canada products."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in MADE_IN_CANADA_KEYWORDS)


def normalize_product_name(name: str) -> str:
    """Normalize product name for deduplication comparison."""
    if not name:
        return ""
    # Lowercase
    name = name.lower()
    
    # Remove size variants (at end or in parentheses)
    name = re.sub(r'\s*[-–]\s*(xs|s|m|l|xl|xxl|xxxl|os|one\s*size|small|medium|large|x-?large|xx-?large|size\s+\w+)\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\(\s*(xs|s|m|l|xl|xxl|xxxl|os|one\s*size|small|medium|large|x-?large|xx-?large|size\s+\w+)\s*\)\s*$', '', name, flags=re.IGNORECASE)
    
    # Remove ALL color words anywhere in the name (not just at end)
    colors = r'\b(black|white|grey|gray|charcoal|navy|blue|red|green|brown|tan|beige|cream|ivory|pink|purple|orange|yellow|olive|burgundy|maroon|teal|coral|salmon|heather|oatmeal|natural|midnight|forest|hunter|sky|royal|slate|stone|sand|camel|cognac|rust|wine|plum|lavender|mint|sage|moss|cedar|walnut|mahogany|espresso|chocolate|mocha|latte|vanilla|snow|arctic|onyx|jet|raven|ink|graphite|pewter|silver|gold|bronze|copper|rose|blush|dusty|faded|washed|vintage)\b'
    name = re.sub(colors, '', name, flags=re.IGNORECASE)
    
    # Remove gendered prefixes (men's, women's, kids, unisex, etc.)
    name = re.sub(r"^(men'?s?|women'?s?|kid'?s?|unisex|ladies'?)\s+", '', name, flags=re.IGNORECASE)
    
    # Remove common modifier words
    name = re.sub(r'\b(classic|original|premium|lightweight|heavyweight|printed|logo|slim|relaxed|oversized|fitted|regular|cropped|long|short)\b', '', name, flags=re.IGNORECASE)
    
    # Remove extra whitespace and clean up
    name = ' '.join(name.split())
    return name.strip()


def extract_core_product_type(name: str) -> str:
    """Extract the core product type for similarity matching."""
    if not name:
        return ""
    name = name.lower()
    # Common product types to extract
    product_types = [
        "hoodie", "hoody", "sweatshirt", "sweater", "jacket", "coat", "parka",
        "t-shirt", "tee", "shirt", "polo", "tank", "top", "blouse",
        "pants", "jeans", "shorts", "joggers", "leggings", "skirt", "dress",
        "boots", "shoes", "sneakers", "sandals", "slippers",
        "hat", "cap", "beanie", "toque", "scarf", "gloves", "mittens",
        "bag", "backpack", "tote", "wallet", "belt",
    ]
    for pt in product_types:
        if pt in name:
            return pt
    return ""


def dedupe_results(results: list, max_per_brand_type: int = 0) -> list:
    """
    Remove duplicate products based on normalized name + brand.
    Keeps the first (highest ranked) occurrence.
    
    Uses multiple strategies:
    1. Normalized name + brand exact match (same product, different color/size)
    2. URL-based dedup (same product URL with size variants)
    3. Optional: Limit products per brand+type (e.g., max 3 toques per brand)
    
    Args:
        results: List of tuples from database (name at index 0, brand at index 1, url at index 5)
        max_per_brand_type: Max products per (brand, product_type) combo. 0 = no limit.
    
    Returns:
        Deduplicated list
    """
    seen_names = set()
    seen_urls = set()
    seen_product_brand = {}  # (brand, product_type) -> count
    deduped = []
    
    for row in results:
        name = row[0] or ""
        brand = row[1] or ""
        url = row[5] if len(row) > 5 else ""
        
        # Strategy 1: Normalized name + brand exact match
        norm_name = normalize_product_name(name)
        name_key = f"{norm_name}|{brand.lower()}"
        
        # Strategy 2: URL-based dedup (remove size variants from URL)
        url_base = re.sub(r'[?#].*$', '', url)  # Remove query params
        url_base = re.sub(r'-?(xs|s|m|l|xl|xxl|xxxl)(?:-|$)', '', url_base, flags=re.IGNORECASE)
        
        # Strategy 3: Limit per brand+type (only if max_per_brand_type > 0)
        product_type = extract_core_product_type(name)
        brand_lower = brand.lower()
        brand_type_key = f"{brand_lower}|{product_type}" if product_type else None
        
        # Check for duplicates
        is_dup = False
        dup_reason = ""
        
        if name_key in seen_names:
            is_dup = True
            dup_reason = "same normalized name"
        elif url_base in seen_urls:
            is_dup = True
            dup_reason = "same URL base"
        elif max_per_brand_type > 0 and brand_type_key:
            count = seen_product_brand.get(brand_type_key, 0)
            if count >= max_per_brand_type:
                is_dup = True
                dup_reason = f"already have {max_per_brand_type} {product_type}(s) from {brand}"
        
        if is_dup:
            logger.debug(f"   🔄 Deduped: {name[:40]}... ({dup_reason})")
            continue
        
        # Add to seen sets
        seen_names.add(name_key)
        seen_urls.add(url_base)
        if brand_type_key:
            seen_product_brand[brand_type_key] = seen_product_brand.get(brand_type_key, 0) + 1
        
        deduped.append(row)
    
    if len(results) != len(deduped):
        logger.info(f"🔄 Deduped {len(results)} → {len(deduped)} products")
    
    return deduped


def smooth_rating_for_ranking(rating: Optional[float], num_reviews: Optional[int]) -> float:
    """
    Apply Empirical Bayes smoothing to ratings for ranking purposes.
    
    Smooths ratings DOWN towards the prior (4.0) when:
    - There are few reviews (less trust in the rating)
    - The rating is above the prior (we're skeptical of high ratings with few reviews)
    
    This prevents products with 5 stars and 1 review from outranking
    products with 4.8 stars and 100 reviews.
    
    Args:
        rating: The product's average rating (1-5 scale)
        num_reviews: Number of reviews
        
    Returns:
        Smoothed rating for ranking (not for display)
    """
    # If no rating data, return the prior
    if rating is None:
        return RATING_PRIOR
    
    # If too few reviews, just use the prior
    if num_reviews is None or num_reviews < MIN_REVIEWS_FOR_TRUST:
        logger.debug(f"   ⭐ Rating {rating:.1f} with {num_reviews or 0} reviews → using prior {RATING_PRIOR} for ranking")
        return RATING_PRIOR
    
    # Only smooth DOWN, not up
    # If the rating is already at or below the prior, keep it
    if rating <= RATING_PRIOR:
        return rating
    
    # Bayesian average: (C * prior + n * rating) / (C + n)
    # This pulls high ratings down towards the prior when reviews are few
    smoothed = (RATING_CONFIDENCE * RATING_PRIOR + num_reviews * rating) / (RATING_CONFIDENCE + num_reviews)
    
    # Don't smooth below the prior (only smooth down towards prior, not past it)
    smoothed = max(smoothed, RATING_PRIOR)
    
    if smoothed < rating - 0.1:  # Only log if significant smoothing occurred
        logger.debug(f"   ⭐ Rating {rating:.1f} ({num_reviews} reviews) → smoothed to {smoothed:.2f} for ranking")
    
    return smoothed


async def generate_embedding(text: str) -> List[float]:
    """Generate embeddings for text using Cohere"""
    cohere_client = cohere.AsyncClientV2(api_key=config("COHERE_API_KEY"))
    response = await cohere_client.embed(
        texts=[text],
        model=EMBEDDING_MODEL,
        input_type="search_query",  # For searching
        embedding_types=["float"],
        output_dimension=int(EMBEDDING_DIMENSIONS),
    )
    return response.embeddings.float_[0]


async def rerank_results(query: str, results: list, top_n: int = RERANK_TOP_N, prioritize_made_in_canada: bool = False) -> list:
    """Re-rank search results using Cohere's rerank model.
    
    Args:
        query: The user's search query
        results: List of tuples from database
        top_n: Number of top results to return after re-ranking
        prioritize_made_in_canada: If True, sort Made in Canada products to the top after reranking
        
    Returns:
        Re-ranked list of tuples with rerank_score appended to each result
    """
    if not results:
        return results
    
    # Enhance query for reranking
    if prioritize_made_in_canada:
        # Strong preference for Made in Canada when explicitly requested
        rerank_query = query + " | CRITICAL: Only show products that are MADE IN CANADA. Products with Made in Canada: YES are extremely relevant. Products with Made in Canada: NO or UNKNOWN are not relevant."
        logger.info("🍁 Prioritizing Made in Canada products (explicit request detected)")
    else:
        # Default preference - ratings are already smoothed in the documents
        rerank_query = query + " | give higher relevance to products that are made in canada and products with higher smoothed ratings. Ratings have been smoothed to account for review count - trust the smoothed rating. Products with many reviews are more trustworthy."
    
    # Create documents for re-ranking (combine name, brand, description, made_in_canada, reviews)
    # Columns: name(0), brand(1), description(2), price(3), currency(4), url(5), source_site(6), 
    #          similarity(7), markdown_content(8), hybrid_score(9), num_reviews(10), average_rating(11), 
    #          images(12), made_in_canada(13), made_in_canada_reason(14)
    documents = []
    for row in results:
        name, brand, description, price, currency = row[0], row[1], row[2], row[3], row[4]
        markdown = row[8] if len(row) > 8 else None
        num_reviews = row[10] if len(row) > 10 else None
        average_rating = row[11] if len(row) > 11 else None
        made_in_canada = row[13] if len(row) > 13 else None
        made_in_canada_reason = row[14] if len(row) > 14 else None
        
        doc_text = f"Product Name: {name or ''} | Brand Name: {brand or ''}"
        if description:
            doc_text += f" | Description: {description[:MAX_DESCRIPTION_LENGTH]}"
        if markdown:
            doc_text += f" | Markdown Content: {markdown[:MAX_MARKDOWN_LENGTH]}"
        if price:
            doc_text += f" | Price: {price} {currency or ''}"
        # Add Made in Canada status and reason - emphasize this more
        if made_in_canada is True:
            doc_text += " | MADE IN CANADA: YES - VERIFIED CANADIAN MANUFACTURED"
            if made_in_canada_reason:
                doc_text += f" (Reason: {made_in_canada_reason})"
        elif made_in_canada is False:
            doc_text += " | Made in Canada: NO - NOT CANADIAN MADE"
            if made_in_canada_reason:
                doc_text += f" (Reason: {made_in_canada_reason})"
        else:
            doc_text += " | Made in Canada: UNKNOWN - UNVERIFIED"
        
        # Add reviews info for reranker using SMOOTHED rating for ranking
        # This prevents products with 5 stars and 1 review from dominating
        smoothed_rating = smooth_rating_for_ranking(average_rating, num_reviews)
        review_count = num_reviews or 0
        
        # Simple format: Trust Score based on smoothed rating + review count
        # Higher score = better quality signal
        trust_score = smoothed_rating * min(1.0, review_count / 20)  # Scale by review count up to 20
        doc_text += f" | Quality Score: {trust_score:.1f}/5 ({review_count} reviews)"
        
        logger.debug(f"Document text: {doc_text}")
        documents.append(doc_text)
    
    try:
        cohere_client = cohere.AsyncClientV2(api_key=config("COHERE_API_KEY"))
        rerank_response = await cohere_client.rerank(
            model=RERANK_MODEL,
            query=rerank_query,
            documents=documents,
            top_n=min(top_n * 2, len(results)),  # Get more results to filter
        )
        
        # Re-order results and append rerank score
        reranked_results = []
        for result in rerank_response.results:
            original_idx = result.index
            rerank_score = result.relevance_score
            # Append rerank_score to the tuple
            reranked_results.append((*results[original_idx], rerank_score))
        
        # If prioritizing Made in Canada, sort them to the top
        if prioritize_made_in_canada:
            # Sort by: (1) made_in_canada=True first, then (2) rerank_score descending
            # made_in_canada is at index 13
            def sort_key(row):
                made_in_canada = row[13] if len(row) > 13 else None
                rerank_score = row[-1] if row[-1] is not None else 0
                # Return tuple: (not made_in_canada, -rerank_score)
                # This puts True first (since not True = False < not False = True)
                return (made_in_canada is not True, -rerank_score)
            
            reranked_results.sort(key=sort_key)
            
            # Count how many Made in Canada products we have
            mic_count = sum(1 for r in reranked_results if r[13] is True)
            logger.info(f"🍁 Found {mic_count} Made in Canada products out of {len(reranked_results)}")
        
        # Limit to top_n after sorting
        reranked_results = reranked_results[:top_n]
        
        logger.info(f"🔄 Re-ranked {len(results)} results → top {len(reranked_results)}")
        return reranked_results
        
    except Exception as e:
        logger.warning(f"⚠️ Re-ranking failed, using original order: {e}")
        # Return with None for rerank_score
        return [(*r, None) for r in results[:top_n]]


async def search_products(query: str, limit: int = RERANK_TOP_N, for_user: bool = False) -> str:
    """
    Hybrid search (vector + lexical/FTS) then rerank.
    
    Args:
        query: User's search query
        limit: Maximum results to return
        for_user: If True, format output for direct user display. 
                  If False, format for agent consumption with selection instructions.
    
    Funnel: INITIAL_SEARCH_LIMIT → RERANK_TOP_N → PROMPT_TOP_N
    """
    import time
    total_start = time.time()
    
    try:
        # Parse query to extract filters and core intent
        parse_start = time.time()
        parsed = parse_query(query)
        logger.info(f"📝 Query: '{parsed.intent}' | Filters: {parsed.filters} | Mode: {'user' if for_user else 'agent'}")
        logger.info(f"⏱️ Query parsing: {time.time() - parse_start:.2f}s ({(time.time() - parse_start)*1000:.0f}ms)")
        
        # Generate embedding for the intent (cleaner than full query)
        embed_start = time.time()
        embedding = await generate_embedding(parsed.intent)
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        logger.info(f"⏱️ Embedding: {time.time() - embed_start:.2f}s ({(time.time() - embed_start)*1000:.0f}ms)")

        # Tuning knobs
        # SQL fetches more candidates than we need, then dedupe reduces them
        VECTOR_CANDIDATES = SQL_CANDIDATES  # Raw candidates per search type
        TEXT_CANDIDATES = SQL_CANDIDATES
        HYBRID_CANDIDATES = SQL_CANDIDATES  # Total before dedupe
        ALPHA = 0.5  # weight for vector similarity (0..1). Higher = more semantic, lower = more keyword

        conn_string = (
            f"host={DB_CONFIG['host']} "
            f"dbname={DB_CONFIG['dbname']} "
            f"user={DB_CONFIG['user']} "
            f"password={DB_CONFIG['password']}"
        )

        # Build filter clauses from parsed query
        wants_mic = parsed.made_in_canada_only
        mic_filter = "AND p.made_in_canada = true" if wants_mic else ""
        
        # Build additional WHERE clauses for parsed filters
        additional_filters = []
        filter_params = []
        
        if parsed.min_price is not None:
            additional_filters.append("AND CAST(NULLIF(regexp_replace(p.price, '[^0-9.]', '', 'g'), '') AS NUMERIC) >= %s")
            filter_params.append(parsed.min_price)
        if parsed.max_price is not None:
            additional_filters.append("AND CAST(NULLIF(regexp_replace(p.price, '[^0-9.]', '', 'g'), '') AS NUMERIC) <= %s")
            filter_params.append(parsed.max_price)
        if parsed.min_rating is not None:
            additional_filters.append("AND p.average_rating >= %s")
            filter_params.append(parsed.min_rating)
        if parsed.brand:
            additional_filters.append("AND p.brand ILIKE %s")
            filter_params.append(f"%{parsed.brand}%")
        
        extra_where = " ".join(additional_filters)
        
        if wants_mic:
            logger.info("🍁 Filtering to Made in Canada products only (explicit request)")
        if filter_params:
            logger.info(f"🔍 Applied filters: {parsed.filters}")

        results = []
        db_start = time.time()
        async with await psycopg.AsyncConnection.connect(conn_string) as conn:
            async with conn.cursor() as cur:
                # Notes:
                # - Vector candidates: fastest with ivfflat index on embedding
                # - Text candidates: uses Postgres FTS (to_tsvector/plainto_tsquery)
                # - We normalize text rank to 0..1 across candidates, then blend with vector sim.
                # - If you already have a precomputed tsvector column, swap the to_tsvector(...) with that column.
                sql_query = f"""
                    --sql
                    WITH
                    q AS (
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
                            (1 - (p.embedding <=> q.q_embedding))::float AS vector_similarity,
                            0.0::float AS text_rank,
                            p.markdown AS markdown_content,
                            p.num_reviews,
                            p.average_rating,
                            p.images,
                            p.made_in_canada,
                            p.made_in_canada_reason
                        FROM products p
                        CROSS JOIN q
                        WHERE p.embedding IS NOT NULL {mic_filter} {extra_where}
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
                            p.markdown AS markdown_content,
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
                        ) {mic_filter} {extra_where}
                        ORDER BY text_rank DESC
                        LIMIT %s
                    ),

                    candidates AS (
                        SELECT * FROM vec
                        UNION ALL
                        SELECT * FROM txt
                    ),

                    -- Clean URLs for deduplication: remove query params and size variants
                    candidates_with_url_base AS (
                        SELECT *,
                            -- Remove query params (?...) and hash (#...)
                            -- Then remove size variants like -xs, -s, -m, -l, -xl, etc.
                            regexp_replace(
                                regexp_replace(url, '[?#].*$', ''),  -- Remove query/hash
                                '-?(xs|s|m|l|xl|xxl|xxxl)(-|$)', '', 'gi'  -- Remove size variants
                            ) AS url_base
                        FROM candidates
                    ),

                    deduped AS (
                        -- Group by cleaned URL to merge size variants
                        SELECT
                            url_base,
                            max(url) AS url,  -- Keep one of the original URLs
                            max(name) AS name,
                            max(brand) AS brand,
                            max(description) AS description,
                            max(price) AS price,
                            max(currency) AS currency,
                            max(source_site) AS source_site,
                            max(markdown_content) AS markdown_content,
                            max(coalesce(vector_similarity, 0.0)) AS vector_similarity,
                            max(coalesce(text_rank, 0.0)) AS text_rank,
                            max(num_reviews) AS num_reviews,
                            max(average_rating) AS average_rating,
                            max(images::text)::jsonb AS images,
                            bool_or(made_in_canada) AS made_in_canada,
                            max(made_in_canada_reason) AS made_in_canada_reason
                        FROM candidates_with_url_base
                        GROUP BY url_base
                    ),

                    scored AS (
                        SELECT
                            name,
                            brand,
                            description,
                            price,
                            currency,
                            url,
                            source_site,
                            vector_similarity,
                            text_rank,
                            -- Normalize vector_similarity across candidates to 0..1 (min-max)
                            CASE
                                WHEN max(vector_similarity) OVER () > min(vector_similarity) OVER ()
                                THEN (vector_similarity - min(vector_similarity) OVER ()) / 
                                     (max(vector_similarity) OVER () - min(vector_similarity) OVER ())
                                ELSE 1.0
                            END AS vector_similarity_norm,
                            -- Normalize text_rank across candidates to 0..1 (min-max)
                            CASE
                                WHEN max(text_rank) OVER () > min(text_rank) OVER ()
                                THEN (text_rank - min(text_rank) OVER ()) / 
                                     (max(text_rank) OVER () - min(text_rank) OVER ())
                                ELSE 0.0
                            END AS text_rank_norm,
                            markdown_content,
                            num_reviews,
                            average_rating,
                            images,
                            made_in_canada,
                            made_in_canada_reason
                        FROM deduped
                    )

                    SELECT
                        name,
                        brand,
                        description,
                        price,
                        currency,
                        url,
                        source_site,
                        vector_similarity AS similarity,
                        markdown_content,
                        -- Hybrid score breakdown:
                        -- 1. Base relevance (max 0.6): weighted vector + text similarity, scaled to max 0.6
                        -- 2. Made in Canada boost: +0.2 if made_in_canada = true
                        -- 3. Rating boost (max 0.2, requires 10+ reviews):
                        --    4.0 = no boost, 4.5 = +0.1, 5.0 = +0.2, <4.0 = negative
                        
                        -- Base relevance: cap at 0.6
                        0.6 * ((%s * vector_similarity_norm) + ((1 - %s) * text_rank_norm)) +
                        
                        -- Made in Canada boost: +0.2
                        CASE WHEN made_in_canada = true THEN 0.2 ELSE 0.0 END +
                        
                        -- Rating boost: (rating - 4.0) * 0.2, only if 10+ reviews
                        -- 4.0 → 0, 4.5 → 0.1, 5.0 → 0.2, 3.5 → -0.1, 3.0 → -0.2
                        CASE 
                            WHEN num_reviews IS NULL OR num_reviews < %s THEN 0.0  -- No boost if too few reviews
                            ELSE (COALESCE(average_rating, 4.0) - 4.0) * 0.2  -- Scale: each 0.5 star above 4 = +0.1
                        END
                        AS hybrid_score,
                        num_reviews,
                        average_rating,
                        images,
                        made_in_canada,
                        made_in_canada_reason
                    FROM scored
                    ORDER BY hybrid_score DESC
                    LIMIT %s
                    --end-sql
                """

                # Build parameters tuple - need to duplicate filter_params for both vec and txt CTEs
                base_params = [embedding_str, parsed.intent]
                vec_params = filter_params.copy()
                txt_params = filter_params.copy()
                
                # Scoring params: ALPHA (x2 for weighted blend), MIN_REVIEWS_FOR_TRUST for rating boost
                scoring_params = [ALPHA, ALPHA, MIN_REVIEWS_FOR_TRUST]
                
                all_params = tuple(
                    base_params + vec_params + [VECTOR_CANDIDATES] + 
                    txt_params + [TEXT_CANDIDATES] +
                    scoring_params + [HYBRID_CANDIDATES]
                )
                
                await cur.execute(sql_query, all_params)
                results = await cur.fetchall()
        
        logger.info(f"⏱️ DB query: {time.time() - db_start:.2f}s ({(time.time() - db_start)*1000:.0f}ms) - {len(results)} results")

        if results:
            # hybrid_score is at index 9
            top_score = results[0][9] if results[0][9] is not None else 0.0
            logger.info(f"🔍 Found {len(results)} hybrid candidates (top hybrid score: {top_score:.2%})")
        else:
            logger.warning(f"No products found. Try a different search query than '{parsed.intent}'")
            return "No products found in the knowledge base. Try a different search query."

        # Step 1: Deduplicate results (remove size/color variants)
        dedupe_start = time.time()
        pre_dedupe = len(results)
        results = dedupe_results(results)
        dedupe_time = time.time() - dedupe_start
        post_dedupe = len(results)
        
        # Step 2: Limit to INITIAL_SEARCH_LIMIT (order preserved from SQL hybrid_score)
        results = results[:INITIAL_SEARCH_LIMIT]
        post_limit = len(results)
        
        # Step 3: Rerank down to RERANK_TOP_N
        rerank_start = time.time()
        results = await rerank_results(parsed.intent, results, top_n=RERANK_TOP_N, prioritize_made_in_canada=wants_mic)
        rerank_time = time.time() - rerank_start
        post_rerank = len(results)
        
        logger.info(f"📊 Funnel: {pre_dedupe} sql → {post_dedupe} de-duped ({dedupe_time:.2f}s) → {post_limit} limited → {post_rerank} re-ranked ({rerank_time:.2f}s)")

        # Format results - tuple now includes rerank_score at the end
        # Columns: name(0), brand(1), description(2), price(3), currency(4), url(5), source_site(6), 
        #          similarity(7), markdown_content(8), hybrid_score(9), num_reviews(10), average_rating(11), 
        #          images(12), made_in_canada(13), made_in_canada_reason(14), rerank_score(15)
        
        format_start = time.time()
        
        # Limit results for user output
        display_results = results[:PROMPT_TOP_N] if for_user else results
        
        # Build header based on audience
        if for_user:
            header = f"""Here are **{len(display_results)} Canadian products** for "{parsed.original}":

---
"""
        else:
            header = f"""## Search Results for: "{parsed.original}"

**Found {len(results)} candidate products** (showing top {min(len(results), RERANK_TOP_N)} by relevance)

**Your task:** Review these {len(results)} products and SELECT THE TOP {PROMPT_TOP_N} to show the user.
Prioritize products that:
1. Best match the user's intent: "{parsed.intent}"
2. Are Made in Canada (🍁) if relevant to the query
3. Have good reviews and ratings
4. Offer good value for price

**Filters applied:** {parsed.filters if parsed.filters else 'None'}

---
"""
        formatted_results = [header]
        for i, row in enumerate(display_results, 1):
            name = row[0] or "Unknown Product"
            brand_raw = row[1]
            description = row[2]
            price = row[3]
            currency = row[4] or "CAD"
            url = row[5]
            source_site = row[6]
            similarity = row[7]
            markdown_content = row[8]
            # hybrid_score = row[9]  # not shown to user
            num_reviews = row[10]
            average_rating = row[11]
            images = row[12]  # JSONB array of image URLs
            made_in_canada = row[13]  # Boolean or None
            made_in_canada_reason = row[14] if len(row) > 14 else None  # AI justification
            rerank_score = row[-1]  # Last element is rerank_score
            
            # Infer brand from source_site if missing
            if brand_raw:
                brand = brand_raw
            elif source_site:
                SITE_TO_BRAND = {
                    "provinceofcanada.com": "Province of Canada",
                    "muttonheadstore.com": "Muttonhead",
                    "www.mtnhead.com": "Muttonhead",
                    "roots.com": "Roots",
                    "www.roots.com": "Roots",
                }
                brand = SITE_TO_BRAND.get(source_site, source_site.replace("www.", "").replace(".com", "").replace(".ca", "").title())
            else:
                brand = "Unknown Brand"
            
            # Clean product name (remove size/gender variants)
            clean_name = re.sub(
                r'\s*[-–]\s*(Unisex|Men\'?s?|Women\'?s?|Kids?|Junior|Jr\.?)?\s*[-–]?\s*'
                r'(XXS|XS|S|M|L|XL|XXL|XXXL|2XL|3XL|Small|Medium|Large|X-Large|XX-Large|One Size).*$',
                '', name, flags=re.IGNORECASE
            )
            clean_name = re.sub(r'\s*[-–]?\s*(XXS|XS|XXL|XXXL|2XL|3XL)\s*$', '', clean_name, flags=re.IGNORECASE)
            clean_name = re.sub(r'\s*[-–]\s*(Unisex|Men\'?s?|Women\'?s?)\s*$', '', clean_name, flags=re.IGNORECASE)
            clean_name = clean_name.strip(' -–')
            
            # Made in Canada emoji for title
            mic_emoji = " 🍁" if made_in_canada else ""
            
            if for_user:
                # User-friendly format with ## headers
                formatted_results.append(f"\n## {i}. {clean_name}{mic_emoji}\n")
            else:
                # Agent format with ### and score
                formatted_results.append(f"### {i}. {clean_name}{mic_emoji}")
                if rerank_score is not None:
                    formatted_results.append(f" (Score: `{rerank_score:.0%}`)")
                formatted_results.append("\n")
            
            # Add product image (first image from the array)
            if images and isinstance(images, list) and len(images) > 0:
                first_image = images[0]
                formatted_results.append(f"\n![{clean_name}]({first_image})\n")
            
            # Brand and price
            if for_user:
                price_str = f"${price} {currency}" if price else "Price N/A"
                rating_str = f"⭐ {average_rating:.1f} ({num_reviews} reviews)" if average_rating and num_reviews else ""
                formatted_results.append(f"\n**{brand}** · {price_str}")
                if rating_str:
                    formatted_results.append(f" · {rating_str}")
                formatted_results.append("\n")
            else:
                if brand:
                    formatted_results.append(f"\n**Brand:** {brand}")
                if price:
                    formatted_results.append(f"\n**Price:** `{price} {currency}`")
            
            # Add Made in Canada status with 🍁
            if made_in_canada is True:
                reason_short = (made_in_canada_reason[:100] + "...") if made_in_canada_reason and len(made_in_canada_reason) > 100 else made_in_canada_reason
                reason_text = f" — {reason_short}" if reason_short else ""
                formatted_results.append(f"\n\n**Made in Canada:** 🍁 Yes{reason_text}")
            elif made_in_canada is False:
                reason_text = f" — {made_in_canada_reason}" if made_in_canada_reason else ""
                formatted_results.append(f"\n\n**Made in Canada:** ❌ No{reason_text}")
            else:
                formatted_results.append(f"\n\n**Made in Canada:** ❓ Unknown")
            
            # Add reviews section (only for agent, user already has inline)
            if not for_user:
                if average_rating is not None or num_reviews is not None:
                    rating_str = f"`{average_rating:.1f}`⭐" if average_rating else "N/A"
                    reviews_str = f"`{num_reviews}` reviews" if num_reviews else "No reviews"
                    formatted_results.append(f"\n\n**Reviews:** {rating_str} ({reviews_str})")
            
            # Description
            if description:
                desc_clean = ' '.join(description.split())  # Clean whitespace
                if for_user:
                    desc_short = desc_clean[:200] + "..." if len(desc_clean) > 200 else desc_clean
                    formatted_results.append(f"\n\n{desc_short}")
                else:
                    desc_short = desc_clean[:MAX_DESCRIPTION_LENGTH] + "..." if len(desc_clean) > MAX_DESCRIPTION_LENGTH else desc_clean
                    formatted_results.append(f"\n\n**Product Description:** {desc_short}")
            
            # Markdown content (only for agent)
            if not for_user and markdown_content:
                markdown_short = markdown_content[:MAX_MARKDOWN_LENGTH] + "..." if len(markdown_content) > MAX_MARKDOWN_LENGTH else markdown_content
                formatted_results.append(f"\n\n**Raw Markdown Content:** {markdown_short}")
            
            # Create tracked URL with UTM parameters
            tracked_url = create_tracked_url(
                url=url,
                source=source_site,
                source_type="product",
                product_name=name,
            )
            
            if for_user:
                formatted_results.append(f"\n\n[View Product →]({tracked_url})")
                formatted_results.append("\n\n---")
            else:
                formatted_results.append(f"\n\n**Link:** [View Product]({tracked_url})")
                formatted_results.append(f"\n\n**Source:** {source_site}")
            formatted_results.append("")
        
        # Add footer for user
        if for_user:
            formatted_results.append("\n*Looking for something else? Try a specific brand, price range, or product type!*")
        
        logger.info(f"⏱️ Formatting: {time.time() - format_start:.2f}s ({(time.time() - format_start)*1000:.0f}ms)")
        logger.info(f"⏱️ Total search: {time.time() - total_start:.2f}s ({(time.time() - total_start)*1000:.0f}ms)")

        return "\n".join(formatted_results)

    except Exception as e:
        logger.error(f"Error searching products: {e}")
        return f"Error accessing product database: {str(e)}"



def search_products_sync(query: str, limit: int = 10) -> str:
    """Synchronous wrapper for search_products to use as an agent tool"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(search_products(query, limit))


# ------------Agent Configuration
def get_product_finding_instructions() -> str:
    """Get the instructions for the product finding agent"""
    return dedent(f"""
        You are an expert at finding Made in Canada products and Canadian-owned businesses.
        
        ALWAYS search the product knowledge base first using search_products_sync.
        
        ## SELECTION PROCESS
        
        The search will return ~{RERANK_TOP_N} candidate products ranked by relevance. 
        YOUR JOB: Select and present only the TOP {PROMPT_TOP_N} BEST matches to the user.
        
        When selecting, prioritize:
        1. **Relevance** - How well does it match what the user asked for?
        2. **Made in Canada** - If user asked for Canadian products, prioritize 🍁 products
        3. **Quality signals** - Good reviews (4+ stars with multiple reviews)
        4. **Value** - Reasonable price for the product type
        5. **Variety** - CRITICAL: Show diverse products, not the same item in different colors!
        
        ## ⚠️ STRICT DEDUPLICATION RULES
        
        NEVER show the same product twice in different colors or sizes!
        
        Examples of DUPLICATES to AVOID:
        - "Flag Fleece Hoodie Black" AND "Flag Fleece Hoodie Navy" ❌ (same hoodie, different colors)
        - "Classic T-Shirt - S" AND "Classic T-Shirt - XL" ❌ (same shirt, different sizes)
        - "Men's Joggers" AND "Women's Joggers" ❌ (same product, different genders)
        - "Heritage Fleece Hoodie" AND "Stanfield's Logo Heritage Fleece Hoodie" ❌ (same product, slightly different name)
        
        Instead: Pick the BEST ONE variant and move on to a DIFFERENT product.
        
        Goal: Show {PROMPT_TOP_N} UNIQUE products from different brands/styles
        
        ## STEPS TO FOLLOW
        
        1. Search the product knowledge base for matching products
        2. ANALYZE the ~{RERANK_TOP_N} results and mentally rank them
        3. SELECT the TOP {PROMPT_TOP_N} best matches (not just the first 10!)
        4. Present these {PROMPT_TOP_N} products with images and details
        5. Ask a follow-up question
        
        ## PRODUCT PRESENTATION
        
        For each of the {PROMPT_TOP_N} products you select, use this EXACT format:
        
        ## 1. Product Name Here 🍁
        
        ![Product Name](image_url)
        
        **Brand:** Brand Name  
        **Price:** `$XX.XX CAD`  
        **Reviews:** `4.5`⭐ (`23` reviews)  
        **Description:** Brief product description here.  
        **Link:** [View Product](url)  
        **Relevance:** `92%`  
        **Made in Canada:** 🍁 Yes
        
        ---
        
        Key points:
        - Use ## (h2) for product name - MUST be a big header!
        - Product name goes FIRST, then image below it
        - REMOVE sizing info from names (e.g., "- XS", "Size Large")
        - Add 🍁 to the header if Made in Canada
        - Use --- between products for visual separation
        - **NEVER show raw URLs** - always use [View Product](url) or [Shop Now](url)
        
        ## PRODUCT NAME HEADERS
        
        ALWAYS use ## (h2) for product names - they should be BIG and prominent:
        
        ## 1. Flag Fleece Hoodie 🍁
        
        NOT:
        - "1. Flag Fleece Hoodie" (no header - too small!)
        - "### 1. Flag Fleece Hoodie" (h3 - still too small!)
        - "#### 1. Flag Fleece Hoodie" (h4 - way too small!)
        
        ## 🍁 MADE IN CANADA FORMATTING
        
        When a product has "Made in Canada: Yes":
        - Add 🍁 next to the product name: "## 1. Classic Hoodie 🍁"
        - Format as: "**Made in Canada:** 🍁 Yes"
        
        For NOT made in Canada: "**Made in Canada:** ❌ No"
        For unknown: "**Made in Canada:** ❓ Unknown"
        
        ## NAME CLEANUP
        
        Remove size/color variants from names:
        - "Classic Hoodie - XS" → "Classic Hoodie"
        - "Cozy Sweater Size Medium" → "Cozy Sweater"
        - "T-Shirt Navy - L/XL" → "T-Shirt Navy"
        
        ## FORMATTING
        
        Wrap numbers in backticks:
        - Price: `\\$78.00 CAD`
        - Score: `92%`  
        - Rating: `4.5`⭐ (`23` reviews)

        Image URL fix: If URL has double domain like "https://www.kamik.com//www.kamik.com/...", 
        remove the duplicate.
        
        ## LINK FORMATTING
        
        ⚠️ NEVER display raw URLs to users! Always use friendly link text:
        
        ✅ GOOD:
        - **Link:** [View Product](url)
        - **Link:** [Shop Now](url)
        - **Link:** [Buy on Roots](url)
        
        ❌ BAD:
        - **Link:** https://www.roots.com/ca/en/some-product.html
        - **Link:** [https://www.roots.com/...](url)
        
        The URLs in the search results are tracking links - use them as-is but with friendly text.
        
        ## KEY REMINDERS
        
        - Select the BEST {PROMPT_TOP_N}, not just the first 10
        - **NEVER show color/size variants of the same product** - pick ONE and move on!
        - If you see "Flag Hoodie Black" and "Flag Hoodie Navy", only show ONE
        - Aim for variety: different brands, different product types
        - NEVER skip Reviews or Description fields
        - If reviews missing, show "No reviews yet"
        - If description missing, write a brief one
        - Sort YOUR selection by relevance (highest first)
        - End with a meaningful follow-up question
        
        Focus on Canadian brands: Roots, Lululemon, Canada Goose, Aritzia, 
        Province of Canada, Mejuri, Duer, Reigning Champ, etc.
    """)


def get_llm_model():
    """Get the configured LLM model based on provider"""
    if AGENT_LLM_PROVIDER == "openai":
        return OpenAIChat(id=AGENT_MODEL_ID)
    elif AGENT_LLM_PROVIDER == "cohere":
        return Cohere(id=AGENT_MODEL_ID)
    else:
        raise ValueError(f"Unsupported LLM provider: {AGENT_LLM_PROVIDER}")


def _create_agent():
    """Create a new agent instance (internal helper)"""
    logger.info(f"🤖 Initializing agent with {AGENT_LLM_PROVIDER}/{AGENT_MODEL_ID}")
    
    return Agent(
        name="Made in Canada Product Finder",
        role="Find and recommend Canadian products",
        model=get_llm_model(),
        tools=[
            search_products_sync,
        ],
        instructions=get_product_finding_instructions(),
        additional_context=ADDITIONAL_CONTEXT,
        debug_mode=DEBUG_MODE,
        markdown=True,
        add_datetime_to_context=True,
        tool_call_limit=MAX_TOOL_CALLS,
        # ----------memory----------
        db=team_storage,
        add_history_to_context=True,
        num_history_runs=NUM_HISTORY_RUNS,
    )


@st.cache_resource
def get_agent_team():
    """Get the product finder agent (Streamlit cached version)"""
    return _create_agent()


def get_agent_team_no_cache():
    """
    Get the product finder agent (no caching - for backend API use).
    
    Use this when running outside of Streamlit (e.g., FastAPI backend).
    The backend should cache the agent instance itself.
    """
    return _create_agent()


async def main():
    """CLI interface for testing the agent"""
    team = get_agent_team()
    print("🍁 Made in Canada Product Finder is ready. Type 'exit' to quit.")
    while True:
        user_input = input("💁 You: ")
        if user_input.strip().lower() == "exit":
            break
        response = await team.arun(user_input)
        print(f"🍁 Agent: {response.content}")


if __name__ == "__main__":
    search_products_sync("hoodie test")

    asyncio.run(main())

