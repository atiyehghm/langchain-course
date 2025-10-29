import json
import os

from dotenv import load_dotenv

load_dotenv()


from langchain.agents import create_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]

llm = ChatOpenAI(
    api_key=os.getenv("METIS_API_KEY"),
    base_url="https://api.metisai.ir/openai/v1",
    model="gpt-4o-mini",
    temperature=0.0,
)
output_parser = PydanticOutputParser[AgentResponse](pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "tool_names", "agent_scratchpad", "tools"],
).partial(format_instructions=output_parser.get_format_instructions())


graph = create_agent(
    model=llm,
    tools=tools,
    system_prompt=getattr(react_prompt_with_format_instructions, "template", None),
)


def main():
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their detils.",
                }
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()
