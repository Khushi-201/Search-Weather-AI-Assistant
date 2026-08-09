import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent


# ============================================================
# Configuration
# ============================================================

load_dotenv(override=True)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            color: #777;
            margin-bottom: 30px;
        }

        .tool-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #f5f5f5;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API keys
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# ============================================================
# Validate API keys
# ============================================================

missing_keys = []

if not GOOGLE_API_KEY:
    missing_keys.append("GOOGLE_API_KEY")

if not TAVILY_API_KEY:
    missing_keys.append("TAVILY_API_KEY")

if not WEATHERSTACK_API_KEY:
    missing_keys.append("WEATHERSTACK_API_KEY")


if missing_keys:
    st.error(
        "Missing API keys: " + ", ".join(missing_keys)
    )
    st.info(
        "Add the missing keys to your .env file and restart Streamlit."
    )
    st.stop()


# ============================================================
# Weather Tool
# ============================================================

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    try:
        response = requests.get(
            "https://api.weatherstack.com/current",
            params={
                "access_key": WEATHERSTACK_API_KEY,
                "query": city
            },
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        if "current" not in data:
            return f"Unable to fetch weather information for {city}."

        current = data["current"]

        return (
            f"City: {city}\n"
            f"Temperature: {current['temperature']}°C\n"
            f"Weather: {current['weather_descriptions'][0]}\n"
            f"Humidity: {current['humidity']}%\n"
            f"Wind Speed: {current['wind_speed']} km/h"
        )

    except requests.RequestException as e:
        return f"Weather API error: {str(e)}"


# ============================================================
# Tools
# ============================================================

search_tool = TavilySearch(
    max_results=2
)

tools = [
    search_tool,
    get_weather_data
]


# ============================================================
# Gemini LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.5,
    google_api_key=GOOGLE_API_KEY
)


# ============================================================
# Agent
# ============================================================

system_prompt = """
You are a helpful AI assistant.

You have access to two tools:

1. Tavily Search
   - Use this when the user needs current or web-based information.

2. Weather Tool
   - Use this when the user asks about current weather.

Use the appropriate tool when necessary.

If the user asks a question that requires multiple steps,
perform those steps before giving the final answer.

Give clear and concise answers.
"""


agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)


# ============================================================
# UI
# ============================================================

st.markdown(
    '<div class="main-title">🤖 Agentic AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Gemini + LangChain + Tavily + Weatherstack'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("🛠️ Available Tools")

    st.markdown(
        """
        <div class="tool-box">

        🔎 <b>Tavily Search</b>

        Search the web for current information.

        </div>

        <div class="tool-box">

        🌤️ <b>Weather Tool</b>

        Get current weather information for a city.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.caption(
        "Powered by Gemini, LangChain, Tavily and Weatherstack"
    )


# ============================================================
# Chat history
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# Chat Input
# ============================================================

user_input = st.chat_input(
    "Ask me anything..."
)


if user_input:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_input
                            }
                        ]
                    }
                )

                answer = result["messages"][-1].content

                if isinstance(answer, list):
                    answer = "\n".join(
                        item["text"]
                        for item in answer
                        if item.get("type") == "text"
                    )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(
                    f"Something went wrong:\n\n{str(e)}"
                )

