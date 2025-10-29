import os
import json
from dotenv import load_dotenv

load_dotenv()


from langsmith import Client
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


tools = [TavilySearch()]

llm = ChatOpenAI(
    api_key=os.getenv("METIS_API_KEY"),  
    base_url="https://api.metisai.ir/openai/v1",  
    model="gpt-4o-mini",
    temperature=0.0
)

client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
react_prompt = client.pull_prompt("hwchase17/react")

graph = create_agent(model=llm, tools=tools, system_prompt=getattr(react_prompt, "template", None))


def main():
    result = graph.invoke(
        {
            "messages": [
                {"role": "user", "content": "Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their detils."}
            ]
        }
    )
    print(result["results"])

if __name__== "__main__":
    main()