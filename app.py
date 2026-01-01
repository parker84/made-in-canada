"""
Made in Canada - Streamlit Frontend

A shopping assistant that helps find Canadian products from the knowledge base.

This UI calls the backend API for agent interactions, allowing the backend 
to scale independently.
"""

import streamlit as st
from typing import AsyncGenerator
from random import choice
import uuid
import asyncio
import logging
import os
import time
import json

import httpx
import coloredlogs
from decouple import config

logger = logging.getLogger(__name__)
coloredlogs.install(level=os.getenv("LOG_LEVEL", "INFO"), logger=logger)

# Configuration
SHOW_PROGRESS_STATUS = True
BACKEND_URL = config("BACKEND_URL", default="http://localhost:8000")

# User-friendly tool names
TOOL_DISPLAY_NAMES = {
    "search_products_sync": "Searching Product Database",
}


def get_thinking_message() -> str:
    """Get a random thinking message"""
    messages = [
        "Searching for Canadian products... 🍁",
        "Looking through the catalogue... 📦",
        "Finding made in Canada options... 🇨🇦",
        "Checking the knowledge base... 🔍",
        "Exploring Canadian brands... 🏷️",
        "Hunting for quality products... 🎯",
        "Scouring Canadian retailers... 🛍️",
    ]
    return choice(messages)


def login_screen():
    """Display the login screen"""
    st.header("🍁 Made in Canada")
    st.write("Find Canadian products and support local businesses.")
    st.write("Please log in to continue.")
    if st.button("🔐 Log in with Google", type="primary"):
        st.login("google")
        st.stop()
    st.stop()


def is_logged_in() -> bool:
    """Check if user is logged in"""
    return hasattr(st, 'user') and hasattr(st.user, 'is_logged_in') and st.user.is_logged_in


def get_user_first_name() -> str:
    """Get the user's first name or 'there' as fallback"""
    if hasattr(st, 'user') and hasattr(st.user, 'name') and st.user.name:
        return st.user.name.split(' ')[0]
    return "there"


def get_user_email() -> str:
    """Get the user's email or fallback"""
    if hasattr(st, 'user') and hasattr(st.user, 'email') and st.user.email:
        return st.user.email
    return "anonymous_user"


# Set page config
st.set_page_config(
    page_title="Made in Canada",
    page_icon="🍁",
    initial_sidebar_state="collapsed",
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


async def stream_from_backend(
    prompt: str,
    user_id: str,
    session_id: str,
) -> AsyncGenerator[tuple[str, str], None]:
    """
    Stream response from the backend API.
    
    Yields tuples of (event_type, content):
    - ("tool_start", tool_name)
    - ("tool_complete", "")
    - ("content", text_chunk)
    - ("done", "")
    - ("error", error_message)
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            async with client.stream(
                "POST",
                # f"{BACKEND_URL}/api/chat/stream",
                f"{BACKEND_URL}/api/search/fast/stream",  # Fast search - no LLM overhead
                json={
                    "message": prompt,
                    "user_id": user_id,
                    "session_id": session_id,
                    "referrer": "madeincanada.dev",
                },
            ) as response:
                if response.status_code != 200:
                    logger.error(f"Backend returned {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    logger.debug(f"Backend URL: {BACKEND_URL}")
                    logger.debug(f"Prompt: {prompt}")
                    logger.debug(f"User ID: {user_id}")
                    logger.debug(f"Session ID: {session_id}")
                    yield ("error", f"Backend returned {response.status_code}")
                    return
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])  # Remove "data: " prefix
                            event_type = data.get("type", "unknown")
                            
                            if event_type == "content":
                                yield ("content", data.get("content", ""))
                            elif event_type == "tool_start":
                                yield ("tool_start", data.get("tool", ""))
                            elif event_type == "tool_complete":
                                yield ("tool_complete", "")
                            elif event_type == "done":
                                yield ("done", "")
                            elif event_type == "error":
                                yield ("error", data.get("error", "Unknown error"))
                        except json.JSONDecodeError:
                                pass  # Skip malformed lines
        except httpx.ConnectError:
            yield ("error", f"Cannot connect to backend at {BACKEND_URL}. Is it running?")
        except Exception as e:
            yield ("error", str(e))


async def parse_backend_stream(
    prompt: str,
    user_id: str,
    session_id: str,
) -> AsyncGenerator[tuple[str, str], None]:
    """Parse the backend stream and yield content/status updates for the UI"""
    last_event = "start"
    tool_start_time = None
    current_tool = None
    planning_start_time = time.time()
    
    async for event_type, content in stream_from_backend(prompt, user_id, session_id):
        if event_type == "content":
            if last_event != "content":
                if planning_start_time:
                    elapsed = time.time() - planning_start_time
                    yield ("status_complete", f"✅ ({int(round(elapsed))}s)")
                    planning_start_time = None
                yield ("status_start", "💭 Generating response...")
                last_event = "content"
            yield ("content", content)
            
        elif event_type == "tool_start" and SHOW_PROGRESS_STATUS:
            if last_event in ["analyzing", "start"]:
                elapsed = time.time() - (planning_start_time or time.time())
                yield ("status_complete", f"✅ ({int(round(elapsed))}s)")
                planning_start_time = None
            
            current_tool = content
            tool_display = TOOL_DISPLAY_NAMES.get(
                current_tool, 
                current_tool.replace("_", " ").title()
            )
            
            tool_start_time = time.time()
            last_event = "tool_call"
            yield ("status_start", f"🔍 {tool_display}...")
            
        elif event_type == "tool_complete":
            if tool_start_time and current_tool:
                elapsed = time.time() - tool_start_time
                logger.info(f"✅ {current_tool} completed in {elapsed:.2f}s")
                yield ("status_complete", f"✅ ({int(round(elapsed))}s)")
                planning_start_time = time.time()
                yield ("status_start", "🧠 Analyzing results...")
                last_event = "analyzing"
        
        elif event_type == "error":
            yield ("error", content)
            
        elif event_type == "done":
            pass  # Stream complete


# Sidebar (always visible)
with st.sidebar:
    st.link_button("❤️ Feedback", "https://forms.gle/5dWaY279oFsfwhTw9")
    st.link_button("📧 Contact us", "mailto:parkerbrydon@gmail.com")

# Check login status
if not is_logged_in():
    login_screen()

# User is logged in - show main app
with st.sidebar:
    st.markdown("---")
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.button("🔐 Log out", on_click=st.logout, type="secondary")

# Main content
st.title("🍁 Made in Canada")
st.caption("Find Canadian products and support Canadian businesses")

# Welcome message
first_name = get_user_first_name()
if not st.session_state.messages:
    intro_messages = [
        f"👋 Hey {first_name}! I can help you find products that are **Made in Canada** 🇨🇦",
        f"👋 Welcome {first_name}! Let's find some great Canadian products together 🍁",
        f"👋 Hi {first_name}! Ready to discover Canadian-made products? 🇨🇦",
    ]
    st.markdown(choice(intro_messages))
    st.markdown("""
    Try asking things like:
    - "Find me a warm winter jacket 🧥"
    - "Looking for Canadian-made leather goods 👜"
    - "I'm looking for a new hockey stick for my son 🏒"
    """)

# Display chat messages
for message in st.session_state.messages:
    avatar = "🍁" if message["role"] == "assistant" else "🦫"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


@st.cache_data
def get_placeholder():
    """Get a random placeholder for the chat input"""
    return choice([
        "Find me a warm Canadian-made winter jacket 🧥",
        "Looking for cozy sweaters from Roots 🍁",
        "Help me find Canadian leather goods 👜",
        "What activewear is made in Canada? 🏃",
        "Looking for Canadian-made gifts 🎁",
    ])


# Chat input
if prompt := st.chat_input(placeholder=get_placeholder()):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar="🦫"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant", avatar="🍁"):
        with st.spinner(get_thinking_message()):
            start_time = time.time()
            logger.info(f"🚀 Starting agent run for: {prompt[:50]}...")
            
            async def process_stream():
                response_parts = []
                
                status_container = st.empty()
                response_placeholder = st.empty()
                status_lines = ["🧠 Thinking..."]
                status_container.caption("\n\n".join(status_lines))
                
                async for content_type, content in parse_backend_stream(
                    prompt,
                    get_user_email(),
                    st.session_state.session_id,
                ):
                    if content_type == "status_start":
                        status_lines.append(content)
                        status_container.caption("\n\n".join(status_lines))
                    elif content_type == "status_complete":
                        if status_lines:
                            status_lines[-1] = f"{status_lines[-1]} {content}"
                        status_container.caption("\n\n".join(status_lines))
                    elif content_type == "content":
                        if status_lines:
                            status_container.empty()
                            status_lines = []
                        response_parts.append(content)
                        response_placeholder.markdown("".join(response_parts))
                    elif content_type == "error":
                        response_placeholder.error(f"❌ {content}")
                        return content  # Return error as response
                
                if status_lines:
                    status_container.empty()
                
                return "".join(response_parts)
            
            full_response = asyncio.run(process_stream())
            total_time = time.time() - start_time
            logger.info(f"✨ Total response time: {total_time:.2f}s")
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
