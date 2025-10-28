import os
from dotenv import load_dotenv

load_dotenv()


from langchain.agents.react.base import REACT_PROMPT
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


tools = [TavilySearch()]

llm = ChatOpenAI(
    api_key=os.getenv("METIS_API_KEY"),  
    base_url="https://api.metisai.ir/openai/v1",  
    model="gpt-4o-mini",
    temperature=0.0
)

react_prompt = REACT_PROMPT

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chain = agent_executor


def main():
    result =  chain.invoke(
        input={
            "input": "Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their detils."
        }
    )

if __name__== "__main__":
    main()