import os 
import certifi 
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchainhub import Client
from langchain.tools import tool
import requests


from langchain.agents import create_agent


os.getenv("GOOGLE_API_KEY")
os.getenv("TAVILY_API_KEY")
os.getenv("WEATHERSTACK_API_KEY")
print(os.getenv("WEATHERSTACK_API_KEY"))


search_tool = TavilySearch(
    max_results=2
)


response= search_tool.invoke("What is the latest political news in india?")
response


@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={os.getenv('WEATHERSTACK_API_KEY')}&query={city}"
    )
    
    response=requests.get(url)
    
    data=response.json()
    
    if "current" not in data:
        return f"Error: Unable to fetch weather data for {city}"
    
    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )
    
    
    
    


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY"))


response = llm.invoke("Tell me a joke about AI")
response


# %%
# To know what models my api could use

# from google import genai
# import os

# client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# for model in client.models.list():
#     print(model.name)

# %%
# hub=Client()
# prompt=hub.pull("hwchase17/react")    #It was a pre-built ReAct prompt

'''Purane ke saath use hota tha , can't use with new version, it expects system instruction not old prompt template'''


prompt = """
You are a helpful assistant.
Use the available tools when necessary to answer the user's question.
"""

# %%
tools=[search_tool, get_weather_data]

# %%
agent=create_agent(model=llm, tools=tools, system_prompt=prompt)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of India and then find its current weather?"
        }
    ]
})

print(result["messages"][-1].content)

# %%
# agent_executor = agent.as_executor(agent_name="Agentic AI", verbose=True, tools=tools)
'''New LangChain me AgentExecutor nhi chahiye hota'''


