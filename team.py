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

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level=os.getenv("LOG_LEVEL", "INFO"), logger=logger)

# ------------constants
DEBUG_MODE = os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG"

# LLM Provider Configuration
AGENT_LLM_PROVIDER = config("AGENT_LLM_PROVIDER", default="openai")
AGENT_MODEL_ID = config("AGENT_MODEL_ID", default="gpt-5-nano")

# Knowledge Base Configuration
EMBEDDING_MODEL = "embed-v4.0"
EMBEDDING_DIMENSIONS = 1536
RERANK_MODEL = "rerank-v3.5"
INITIAL_SEARCH_LIMIT = 50  # Fetch more results for re-ranking
RERANK_TOP_N = 10  # Return top 10 after re-ranking

# Lengths
MAX_DESCRIPTION_LENGTH = 1000
MAX_MARKDOWN_LENGTH = 1000

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


async def rerank_results(query: str, results: list, top_n: int = RERANK_TOP_N) -> list:
    """Re-rank search results using Cohere's rerank model.
    
    Args:
        query: The user's search query
        results: List of tuples from database
        top_n: Number of top results to return after re-ranking
        
    Returns:
        Re-ranked list of tuples with rerank_score appended to each result
    """
    if not results:
        return results
    
    # TODO: let's rank things better (ex: pass in emp bayes smoothed ratings to the re-ranker, or rank again after the re-ranker based on ratings)
    query = query + " | give higher relevance to products that are made in canada and products with high average ratings - but consider the number of reviews too (ex: 4.8 stars and 100 reviews is better than 5 stars and 1 reviews)"
    
    # Create documents for re-ranking (combine name, brand, description)
    documents = []
    for row in results:
        name, brand, description, price, currency = row[0], row[1], row[2], row[3], row[4]
        markdown = row[8] if len(row) > 8 else None
        
        doc_text = f"Product Name: {name or ''} | Brand Name: {brand or ''}"
        if description:
            doc_text += f" | Description: {description[:MAX_DESCRIPTION_LENGTH]}"
        if markdown:
            doc_text += f" | Markdown Content: {markdown[:MAX_MARKDOWN_LENGTH]}"
        if price:
            doc_text += f" | Price: {price} {currency or ''}"
        logger.debug(f"Document text: {doc_text}")
        documents.append(doc_text)
    
    try:
        cohere_client = cohere.AsyncClientV2(api_key=config("COHERE_API_KEY"))
        rerank_response = await cohere_client.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=min(top_n, len(results)),
        )
        
        # Re-order results and append rerank score
        reranked_results = []
        for result in rerank_response.results:
            original_idx = result.index
            rerank_score = result.relevance_score
            # Append rerank_score to the tuple
            reranked_results.append((*results[original_idx], rerank_score))
        
        logger.info(f"🔄 Re-ranked {len(results)} results → top {len(reranked_results)}")
        return reranked_results
        
    except Exception as e:
        logger.warning(f"⚠️ Re-ranking failed, using original order: {e}")
        # Return with None for rerank_score
        return [(*r, None) for r in results[:top_n]]


async def search_products(query: str, limit: int = RERANK_TOP_N) -> str:
    """Hybrid search (vector + lexical/FTS) then rerank."""
    try:
        # Generate embedding for the query
        embedding = await generate_embedding(query)
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        # Tuning knobs
        VECTOR_CANDIDATES = INITIAL_SEARCH_LIMIT            # e.g. 50-200
        TEXT_CANDIDATES = INITIAL_SEARCH_LIMIT              # e.g. 50-200
        HYBRID_CANDIDATES = max(VECTOR_CANDIDATES, TEXT_CANDIDATES)
        ALPHA = 0.5  # weight for vector similarity (0..1). Higher = more semantic, lower = more keyword

        conn_string = (
            f"host={DB_CONFIG['host']} "
            f"dbname={DB_CONFIG['dbname']} "
            f"user={DB_CONFIG['user']} "
            f"password={DB_CONFIG['password']}"
        )

        results = []
        async with await psycopg.AsyncConnection.connect(conn_string) as conn:
            async with conn.cursor() as cur:
                # Notes:
                # - Vector candidates: fastest with ivfflat index on embedding
                # - Text candidates: uses Postgres FTS (to_tsvector/plainto_tsquery)
                # - We normalize text rank to 0..1 across candidates, then blend with vector sim.
                # - If you already have a precomputed tsvector column, swap the to_tsvector(...) with that column.
                sql_query = """
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
                            p.images
                        FROM products p
                        CROSS JOIN q
                        WHERE p.embedding IS NOT NULL
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
                            p.images
                        FROM products p
                        CROSS JOIN q
                        WHERE q.q_ts @@ to_tsvector('english',
                            coalesce(p.name,'') || ' ' ||
                            coalesce(p.brand,'') || ' ' ||
                            coalesce(p.description,'')
                        )
                        ORDER BY text_rank DESC
                        LIMIT %s
                    ),

                    candidates AS (
                        SELECT * FROM vec
                        UNION ALL
                        SELECT * FROM txt
                    ),

                    deduped AS (
                        -- If the same product appears in both sets, keep the best signals.
                        SELECT
                            url,
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
                            max(images::text)::jsonb AS images
                        FROM candidates
                        GROUP BY url
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
                            images
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
                        (%s * vector_similarity_norm) + ((1 - %s) * text_rank_norm) AS hybrid_score,
                        num_reviews,
                        average_rating,
                        images
                    FROM scored
                    ORDER BY hybrid_score DESC
                    LIMIT %s
                    --end-sql
                """

                # Pull a combined candidate set (bigger than final limit) then rerank down.
                await cur.execute(
                    sql_query,
                    (
                        embedding_str,
                        query,
                        VECTOR_CANDIDATES,
                        TEXT_CANDIDATES,
                        ALPHA,  # for hybrid_score calculation
                        ALPHA,  # for hybrid_score calculation
                        HYBRID_CANDIDATES,
                    ),
                )
                results = await cur.fetchall()

        if results:
            # hybrid_score is at index 9
            top_score = results[0][9] if results[0][9] is not None else 0.0
            logger.info(f"🔍 Found {len(results)} hybrid candidates (top hybrid score: {top_score:.2%})")
        else:
            logger.warning(f"No products found. Try a different search query than '{query}'")
            return "No products found in the knowledge base. Try a different search query."

        # Rerank down to final limit (e.g., Cohere rerank)
        results = await rerank_results(query, results, top_n=limit)

        # Format results - tuple now includes rerank_score at the end
        # Columns: name(0), brand(1), description(2), price(3), currency(4), url(5), source_site(6), 
        #          similarity(7), markdown_content(8), hybrid_score(9), num_reviews(10), average_rating(11), images(12), rerank_score(13)
        formatted_results = [f"Found {len(results)} products matching your query:\n"]
        for i, row in enumerate(results, 1):
            name = row[0]
            brand = row[1]
            description = row[2]
            price = row[3]
            currency = row[4]
            url = row[5]
            source_site = row[6]
            similarity = row[7]
            markdown_content = row[8]
            # hybrid_score = row[9]  # not shown to user
            num_reviews = row[10]
            average_rating = row[11]
            images = row[12]  # JSONB array of image URLs
            rerank_score = row[-1]  # Last element is rerank_score
            
            formatted_results.append(f"### {i}. {name or 'Unknown Product'}")
            if rerank_score is not None:
                formatted_results.append(f" (Score: `{rerank_score:.0%}`)")
            
            # Add product image (first image from the array)
            if images and isinstance(images, list) and len(images) > 0:
                first_image = images[0]
                formatted_results.append(f"\n\n![{name or 'Product Image'}]({first_image})")
            
            if brand:
                formatted_results.append(f"\n\n**Brand:** {brand}")
            if price:
                formatted_results.append(f"\n\n**Price:** `{price} {currency or ''}`")
            # Add reviews section
            if average_rating is not None or num_reviews is not None:
                rating_str = f"`{average_rating:.1f}`⭐" if average_rating else "N/A"
                reviews_str = f"`{num_reviews}` reviews" if num_reviews else "No reviews"
                formatted_results.append(f"\n\n**Reviews:** {rating_str} ({reviews_str})")
            if description:
                desc_short = description[:MAX_DESCRIPTION_LENGTH] + "..." if len(description) > MAX_DESCRIPTION_LENGTH else description
                formatted_results.append(f"\n\n**Product Description:** {desc_short}")
            if markdown_content:
                markdown_short = markdown_content[:MAX_MARKDOWN_LENGTH] + "..." if len(markdown_content) > MAX_MARKDOWN_LENGTH else markdown_content
                formatted_results.append(f"\n\n**Raw Markdown Content:** {markdown_short}")
            formatted_results.append(f"\n\n**Link:** [{url}]({url})")
            formatted_results.append(f"\n\n**Source:** {source_site}")
            formatted_results.append("")

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
    # TODO: we need to improve the data in the database + filtering abilities
    # 1. actually made in canada or just a canadian company?
    return dedent("""
        You are an expert at finding Made in Canada products and Canadian-owned businesses.
        
        ALWAYS search the product knowledge base first using search_products_sync.
        
        Here are the steps you need to follow:
        1. Search the product knowledge base for matching products
        2. Present the results with product images and details
        3. Highlight which products are Made in Canada vs Canadian-owned
        4. Ask a follow-up question to help the user further
        
        When presenting products, ALWAYS include ALL of these fields:
        - Product image (displayed as markdown image)
        - Product name and brand (REMOVE any sizing info like "- XS", "- S/M", "Size Large", "XXL", etc. from product names)
        - Price (if available)
        - Reviews - ALWAYS show rating and review count (e.g. "4.5⭐ (23 reviews)"). If no reviews, show "No reviews yet"
        - Product description - ALWAYS include a brief description of the product
        - Link to the product
        - Relevance score (shown as percentage - this indicates how well the product matches the query)
        - Whether it's Made in Canada
        
        IMPORTANT: Clean up product names by removing size variants. For example:
        - "Classic Hoodie - XS" → "Classic Hoodie"
        - "Cozy Sweater Size Medium" → "Cozy Sweater"
        - "T-Shirt Navy - L/XL" → "T-Shirt Navy"
        
        Focus on Canadian brands like: Roots, Lululemon, Canada Goose, Aritzia, 
        Province of Canada, Mejuri, Duer, etc.
        
        Present each product as a card with its image followed by details. The search results 
        already include markdown-formatted images - preserve and display these images in your response.
        
        IMPORTANT FORMATTING: When displaying numbers to the user, wrap them in backticks for visual emphasis:
        - Price: `\\$78.00 CAD`
        - Score: `92%`
        - Rating: `4.5`⭐ (`23` reviews)
        
        NEVER skip the Reviews or Description fields - they help users make informed decisions.
        If reviews are missing, show "No reviews yet". If description is missing, write a brief one based on the product name.
        
        Sort products by Score descending (highest relevance first).
        
        At the end, ask the user a meaningful follow-up question.
    """)


def get_llm_model():
    """Get the configured LLM model based on provider"""
    if AGENT_LLM_PROVIDER == "openai":
        return OpenAIChat(id=AGENT_MODEL_ID)
    elif AGENT_LLM_PROVIDER == "cohere":
        return Cohere(id=AGENT_MODEL_ID)
    else:
        raise ValueError(f"Unsupported LLM provider: {AGENT_LLM_PROVIDER}")


@st.cache_resource
def get_agent_team():
    """Get the product finder agent"""
    logger.info(f"🤖 Initializing agent with {AGENT_LLM_PROVIDER}/{AGENT_MODEL_ID}")
    
    product_finder_agent = Agent(
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

    return product_finder_agent


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

