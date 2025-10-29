import os

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(
    api_key=os.getenv("METIS_API_KEY"),
    base_url="https://api.metisai.ir/openai/v1",
    model="gpt-4o-mini",
    temperature=0.0,
)


agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)

output_extractor = RunnableLambda(lambda x: x.get("structured_response", None))
chain = agent | output_extractor


def main():
    result = chain.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
                }
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()

