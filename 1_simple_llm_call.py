from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os

load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

# model = ChatOpenAI()

model = ChatGroq(
    model="qwen/qwen3.8-27b", api_key=os.getenv("GROQ_API_KEY"), max_tokens=43
)
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of Peru?"})
print(result)
