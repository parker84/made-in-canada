"""
Made in Canada - Streamlit Frontend

A shopping assistant that helps find Canadian products from the knowledge base.
"""

import streamlit as st
from typing import AsyncIterator, AsyncGenerator
from agno.agent import RunOutput
from team import get_agent_team
from random import choice
import uuid
import asyncio
import logging
import os
import time

import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level=os.getenv("LOG_LEVEL", "INFO"), logger=logger)

# Configuration
SHOW_PROGRESS_STATUS = True

# User-friendly tool names
TOOL_DISPLAY_NAMES = {
    "search_products_sync": "Searching Product Database",
}


def get_thinking_message() -> str:
    """Get a random thinking message"""
    # TODO: let's make these more fun
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


async def parse_stream(stream: AsyncIterator[RunOutput]) -> AsyncGenerator[tuple[str, str], None]:
    """Parse the agent stream and yield content/status updates"""
    last_event = "start"
    tool_start_time = None
    current_tool = None
    planning_start_time = time.time()
    
    async for chunk in stream:
        logger.debug(f"{chunk.event if hasattr(chunk, 'event') else 'unknown'}")
        if hasattr(chunk, "event"):
            if chunk.event == 'RunContent' and chunk.content:
                if last_event != "content":
                    if planning_start_time:
                        elapsed = time.time() - planning_start_time
                        yield ("status_complete", f"✅ ({int(round(elapsed))}s)")
                        planning_start_time = None
                    yield ("status_start", "💭 Generating response...")
                    last_event = "content"
                yield ("content", chunk.content)
                
            elif SHOW_PROGRESS_STATUS and chunk.event == "ToolCallStarted":
                if last_event in ["analyzing", "start"]:
                    elapsed = time.time() - (planning_start_time or time.time())
                    yield ("status_complete", f"✅ ({int(round(elapsed))}s)")
                    planning_start_time = None
                
                current_tool = chunk.tool.tool_name
                tool_display = TOOL_DISPLAY_NAMES.get(
                    current_tool, 
                    current_tool.replace("_", " ").title()
                )
                
                tool_start_time = time.time()
                last_event = "tool_call"
                yield ("status_start", f"🔍 {tool_display}...")
                
            elif chunk.event == "ToolCallCompleted":
                if tool_start_time and current_tool:
                    elapsed = time.time() - tool_start_time
                    logger.info(f"✅ {current_tool} completed in {elapsed:.2f}s")
                    yield ("status_complete", f"✅ ({int(round(elapsed))}s)")
                    planning_start_time = time.time()
                    yield ("status_start", "🧠 Analyzing results...")
                    last_event = "analyzing"


# Sidebar
with st.sidebar:

    # TODO:
    # 1. previous chats in the sidebar
    # 2. google auth login
    
    st.link_button("❤️ Feedback", "https://forms.gle/5dWaY279oFsfwhTw9")
    st.link_button("📧 Contact us", "mailto:parkerbrydon@gmail.com")
    st.markdown("---")
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main content
st.title("🍁 Made in Canada")
st.caption("Find Canadian products and support Canadian businesses")

# Welcome message
if not st.session_state.messages:
    # TODO: let's make the intro message better
    st.markdown("""
    👋 **Welcome!** I can help you find products that are:
    - **Made in Canada** 🇨🇦
    - **From Canadian-owned businesses** 🍁
    
    Try asking things like:
    - "Find me a warm winter jacket 🧥"
    - "Looking for Canadian-made leather goods 👜"
    - "I'm looking for a new hockey stick for my son 🏒"
    """)

# Display chat messages
for message in st.session_state.messages:
    avatar = "🍁" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


@st.cache_data
def get_placeholder():
    """Get a random placeholder for the chat input"""
    # TODO: let's make these better
    return choice([
        "Find me a warm Canadian-made winter jacket 🧥",
        "Looking for cozy sweaters from Roots 🍁",
        "Help me find Canadian leather goods 👜",
        "What activewear is made in Canada? 🏃",
        "Looking for Canadian-made gifts 🎁",
    ])


async def run_agent(prompt: str):
    """Run the agent with the given prompt"""
    agent = get_agent_team()
    return agent.arun(
        prompt,
        stream=True,
        stream_events=True,
        user_id="streamlit_user", # TODO: make this specific to the user with email login
        session_id=st.session_state.session_id,
    )


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
                stream = await run_agent(prompt)
                
                status_container = st.empty()
                response_placeholder = st.empty()
                status_lines = ["🧠 Thinking..."]
                status_container.caption("\n\n".join(status_lines))
                
                async for content_type, content in parse_stream(stream):
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
                
                if status_lines:
                    status_container.empty()
                
                return "".join(response_parts)
            
            full_response = asyncio.run(process_stream())
            total_time = time.time() - start_time
            logger.info(f"✨ Total response time: {total_time:.2f}s")
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

