import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y


if __name__ == "__main__":
    print("Hello Tool Calling")
    
    system_prompt = "you're a helpful assistant"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    tools = [TavilySearch(), multiply]

    llm = ChatOpenAI(
    api_key=os.getenv("METIS_API_KEY"),
    base_url="https://api.metisai.ir/openai/v1",
    model="gpt-4o-mini",
    temperature=0.0,
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    
    # Invoke the agent
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "what is the weather in dubai right now? compare it with San Francisco, output should be in celsius"}
        ]
    })


    print("Agent result:", result['messages'][-1].content)