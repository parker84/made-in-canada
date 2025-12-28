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
        results: List of tuples from database (name, brand, description, price, currency, url, source_site, similarity, markdown)
        top_n: Number of top results to return after re-ranking
        
    Returns:
        Re-ranked list of results (same format as input)
    """
    if not results:
        return results
    
    # Create documents for re-ranking (combine name, brand, description)
    documents = []
    for name, brand, description, price, currency, url, source_site, similarity, markdown in results:
        doc_text = f"Product Name: {name or ''} | Brand Name: {brand or ''}"
        if description:
            doc_text += f" | Description: {description[:MAX_DESCRIPTION_LENGTH] if description else ''}"
        if markdown:
            doc_text += f" | Markdown Content: {markdown[:MAX_MARKDOWN_LENGTH] if markdown else ''}"
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
        
        # Re-order results based on rerank scores
        reranked_results = []
        for result in rerank_response.results:
            original_idx = result.index
            reranked_results.append(results[original_idx])
        
        logger.info(f"🔄 Re-ranked {len(results)} results → top {len(reranked_results)}")
        return reranked_results
        
    except Exception as e:
        logger.warning(f"⚠️ Re-ranking failed, using original order: {e}")
        return results[:top_n]


async def search_products(query: str, limit: int = RERANK_TOP_N) -> str:
    """Search the product knowledge base for similar products.
    
    Args:
        query: The user's query to search for
        limit: Number of results to return after re-ranking (default: 10)
        
    Returns:
        A formatted string with relevant products
    """
    try:
        # Generate embedding for the query
        embedding = await generate_embedding(query)
        embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
        
        # Connect to database
        conn_string = f"host={DB_CONFIG['host']} dbname={DB_CONFIG['dbname']} user={DB_CONFIG['user']} password={DB_CONFIG['password']}"
        conn = await psycopg.AsyncConnection.connect(conn_string)
        
        async with conn.cursor() as cur:
            # Fetch more results initially for re-ranking
            sql_query = """
                SELECT 
                    name,
                    brand,
                    description,
                    price,
                    currency,
                    url,
                    source_site,
                    1 - (embedding <=> %s::vector) AS similarity,
                    markdown as markdown_content
                FROM products
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            await cur.execute(sql_query, (embedding_str, embedding_str, INITIAL_SEARCH_LIMIT))
            results = await cur.fetchall()
            
            if results:
                logger.info(f"🔍 Found {len(results)} products (top similarity: {results[0][7]:.2%})")
            else:
                logger.warning(f"No products found in the knowledge base. Try a different search query than '{query}'")
        await conn.close()
        
        if not results:
            return "No products found in the knowledge base. Try a different search query."
        
        # Cohere Re-ranking - filter to top N most relevant
        results = await rerank_results(query, results, top_n=limit)
        
        
        # Format results
        formatted_results = [f"Found {len(results)} products matching your query:\n"]
        for i, (name, brand, description, price, currency, url, source_site, similarity, markdown_content) in enumerate(results, 1):
            formatted_results.append(f"### {i}. {name or 'Unknown Product'}")
            if brand:
                formatted_results.append(f"**Brand:** {brand}")
            if price:
                formatted_results.append(f"**Price:** {price} {currency or ''}")
            if description:
                desc_short = description[:MAX_DESCRIPTION_LENGTH] + "..." if len(description) > MAX_DESCRIPTION_LENGTH else description
                formatted_results.append(f"**Description:** {desc_short}")
            if markdown_content:
                markdown_short = markdown_content[:MAX_MARKDOWN_LENGTH] + "..." if len(markdown_content) > MAX_MARKDOWN_LENGTH else markdown_content
                formatted_results.append(f"**Markdown:** {markdown_short}")
            formatted_results.append(f"**Link:** [{url}]({url})")
            formatted_results.append(f"**Source:** {source_site} | **Match:** {similarity:.1%}")
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
    # 2. images?
    # 3. ratings?
    return dedent("""
        You are an expert at finding Made in Canada products and Canadian-owned businesses.
        
        ALWAYS search the product knowledge base first using search_products_sync.
        
        Here are the steps you need to follow:
        1. Search the product knowledge base for matching products
        2. Present the results in a clear, formatted table
        3. Highlight which products are Made in Canada vs Canadian-owned
        4. Ask a follow-up question to help the user further
        
        When presenting products, include:
        - Product name and brand
        - Price (if available)
        - Link to the product
        - Whether it's Made in Canada
        
        Focus on Canadian brands like: Roots, Lululemon, Canada Goose, Aritzia, 
        Province of Canada, Mejuri, Duer, etc.
        
        Format your response into a table with columns:
        - Product Name
        - Brand  
        - Price
        - Link
        - Made in Canada?
        
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

