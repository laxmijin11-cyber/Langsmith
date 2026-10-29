from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import requests
import os

load_dotenv()

# ---------- Tools ----------
search_tool = DuckDuckGoSearchRun()


@tool
def get_weather_data(city: str) -> str:
    """Fetches the current weather data for a given city."""
    key = os.getenv("WEATHER_API_KEY")
    url = f"https://api.weatherstack.com/current?access_key={key}&query={city}"
    response = requests.get(url)
    return str(response.json())


# ---------- LLM ----------
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=1024,
    temperature=0,
)

# ---------- Build agent (LangGraph way) ----------
agent = create_react_agent(
    model=llm,
    tools=[search_tool, get_weather_data],
)

# ---------- Invoke ----------
response = agent.invoke({"messages": [("user", "What is the current temp of delhi")]})

print(response["messages"][-1].content)


'''
# from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
from langchain_core.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()


@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f"https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}"

    response = requests.get(url)

    return str(response.json())


from langchain_groq import ChatGroq

# llm = ChatOpenAI()
llm = ChatGroq(model="qwen/qwen3.8-27b", max_tokens=45)

# Step 2: Pull the ReAct prompt from LangChain Hub
# prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

try:
    prompt = hub.pull("hwchase17/react")
except Exception:
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    )
    
# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm, tools=[search_tool, get_weather_data], prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent, tools=[search_tool, get_weather_data], verbose=True, max_iterations=5
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# Step 5: Invoke
response = agent_executor.invoke({"input": "What is the current temp of gurgaon"})
print(response)

print(response["output"])
'''
