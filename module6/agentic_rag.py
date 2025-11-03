import os

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from markdownify import markdownify

load_dotenv()


ALLOWED_DOMAINS = ["https://fastapi.tiangolo.com/", "https://docs.pydantic.dev/"]
DOCS_TXT = (
    "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/index.md"
)


@tool
def fetch_documentation(url: str) -> str:
    """Fetch and convert documentation from a URL"""
    if not any(url.startswith(domain) for domain in ALLOWED_DOMAINS):
        return (
            "Error: URL not allowed. "
            f"Must start with one of: {', '.join(ALLOWED_DOMAINS)}"
        )
    response = requests.get(url, timeout=10.0)
    response.raise_for_status()
    return markdownify(response.text)


# We will fetch the content of llms.txt, so this can
# be done ahead of time without requiring an LLM request.
llms_txt_content = requests.get(DOCS_TXT).text

# System prompt for the agent
system_prompt = f"""
You are an expert Python developer and technical assistant.
Your primary role is to help users with questions about FASTAPI and related tools like Pydantic.

Instructions:

1. If a user asks a question you're unsure about — or one that likely involves API usage,
   behavior, or configuration — you MUST use the `fetch_documentation` tool to consult the relevant docs.
2. When citing documentation, summarize clearly and include relevant context from the content.
3. Do not use any URLs outside of the allowed domain.
4. If a documentation fetch fails, tell the user and proceed with your best expert understanding.

You can access official documentation from the following approved sources:

{llms_txt_content}

You MUST consult the documentation to get up to date documentation
before answering a user's question about FASTAPI.

Your answers should be clear, concise, and technically accurate.
"""

tools = [fetch_documentation]

model = ChatOpenAI(
    api_key=os.getenv("METIS_API_KEY"),
    base_url="https://api.metisai.ir/openai/v1",
    model="gpt-4o-mini",
    temperature=0.0,
)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    name="Agentic RAG",
)

response = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content=("What are the ways of implementing authentication in FASTAPI?")
            )
        ]
    }
)

print(response["messages"][-1].content)
