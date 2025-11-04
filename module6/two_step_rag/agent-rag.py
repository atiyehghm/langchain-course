import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("METIS_API_KEY"),
    base_url="https://api.metisai.ir/openai/v1",
)

vector_store = Chroma(
    collection_name="lili_blog_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_data",
)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


tools = [retrieve_context]

model = ChatOpenAI(
    api_key=os.getenv("METIS_API_KEY"),
    base_url="https://api.metisai.ir/openai/v1",
    model="gpt-4o-mini",
    temperature=0.0,
)

prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)


query = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()


## The Langsmith trace link: https://smith.langchain.com/o/d0b24c19-953f-4707-841f-4a5ee9e320ad/projects/p/23292381-e649-4b77-bd37-8b9cb7a27020/r/006ad0d1-1aad-450a-9e4c-e50e6ad0d6f4?trace_id=006ad0d1-1aad-450a-9e4c-e50e6ad0d6f4&start_time=2025-11-04T07:25:11.541457
