from typing import List, Union
from dotenv import load_dotenv

from langchain.agents import tool
from langchain.schema import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description

load_dotenv()


# -----------------------------
# 1️⃣ Define tools
# -----------------------------
@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"🧰 Running get_text_length with {text=}")
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for t in tools:
        if t.name == tool_name:
            return t
    raise ValueError(f"Tool '{tool_name}' not found!")


# -----------------------------
# 2️⃣ Build prompt
# -----------------------------
template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
"""

tools = [get_text_length]
prompt = PromptTemplate.from_template(template).partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)


# -----------------------------
# 3️⃣ Build our "agent"
# -----------------------------
llm = ChatOpenAI(temperature=0, stop=["\nObservation", "Observation"])

# Chain the components together:
# input -> prompt -> llm -> output parser
agent_chain = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
    }
    | prompt
    | llm
    | ReActSingleInputOutputParser()
)


# -----------------------------
# 4️⃣ Implement custom AgentExecutor
# -----------------------------
def run_agent(question: str):
    """Manually implement the reasoning loop (like AgentExecutor)"""
    intermediate_steps = []  # (AgentAction, observation) tuples

    while True:
        # Pass the history (scratchpad) and input to the LLM
        agent_step: Union[AgentAction, AgentFinish] = agent_chain.invoke(
            {
                "input": question,
                "agent_scratchpad": intermediate_steps,
            }
        )

        print(f"\n🧩 Agent step: {agent_step}")

        # If the LLM is done, return final answer
        if isinstance(agent_step, AgentFinish):
            print("\n✅ Final answer:", agent_step.return_values["output"])
            break

        # Otherwise, execute the tool
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = str(agent_step.tool_input)
        observation = tool_to_use.func(tool_input)

        # Store step for next iteration
        intermediate_steps.append((agent_step, str(observation)))
        print(f"📥 Observation: {observation}")


# -----------------------------
# 5️⃣ Run
# -----------------------------
if __name__ == "__main__":
    run_agent("What is the length of the word: DOG")