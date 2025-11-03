import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("METIS_API_KEY"),
    base_url="https://api.metisai.ir/openai/v1",
)

vector_store = Chroma(
    collection_name="lili_blog_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_data"
)

@tool(response_format='content_and_artifact')
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs= vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs