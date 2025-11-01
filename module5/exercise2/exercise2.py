from dotenv import load_dotenv
import os
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from typing import List, Dict, Tuple
from langchain_core.tools import Tool, tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """the function returns the text length in characters"""
    text = text.strip().replace("'", "").replace('"', '')
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"There is no tool available with name {tool_name}")


def implement_set_api_key(api_key: str) -> None:
    lines = open(".env").readlines()
    filtered = [l for l in lines if not l.startswith("METIS_API_KEY=")]

    with open(".env", "w") as f:
        f.writelines(filtered)
        f.write(f"METIS_API_KEY={api_key}")

def implement_create_model_with_tools(tools: List[Tool]) -> ChatOpenAI:
    llm = ChatOpenAI(
        api_key=os.getenv("METIS_API_KEY"),
        base_url="https://api.metisai.ir/openai/v1",
        model="gpt-4o-mini",
        temperature=0.0
    ).bind_tools(tools=tools)

    return llm


def implement_check_for_tool_calls(response: AIMessage) -> bool:
    tool_calls = getattr(response, "tool_calls", None) or []
    if len(tool_calls) > 0:
        return True
    return False

def implement_execute_tool_call(tool_call: Dict, available_tools: List[Tool]) -> Tuple[str, str]:
    tool_name = tool_call.get('name')
    tool_args = tool_call.get('args', {})
    tool_call_id = tool_call.get('id', None)

    tool = find_tool_by_name(tools=available_tools, tool_name=tool_name)
    result = tool.invoke(tool_args)
    return tool_call_id, str(result)


def implement_run_agent_with_tool_calling(model_with_tools: ChatOpenAI, user_input: str, available_tools: List[Tool]) -> str:

    messages = [HumanMessage(content=user_input)]

    while True:
        # we give the model the whole message history in every invoke.
        ai_message = model_with_tools.invoke(messages)
        if implement_check_for_tool_calls(response=ai_message):
            messages.append(ai_message)
            for tool_call in getattr(ai_message, "tool_calls", []):
                tool_call_id, result = implement_execute_tool_call(tool_call=tool_call, available_tools=available_tools)
                messages.append(
                    ToolMessage(content=result, tool_call_id=tool_call_id)
                )
            continue
        return ai_message.content


def main():
    tools = [get_text_length]

    model_with_tools = implement_create_model_with_tools(tools)

    result = implement_run_agent_with_tool_calling(
        model_with_tools=model_with_tools,
        user_input="What is the length of the word: ATIYEH?",
        available_tools=tools
    )

    print(result)

if __name__ == "__main__":
    main()


