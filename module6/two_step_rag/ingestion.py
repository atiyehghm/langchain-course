import os

import bs4
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def main():
    print("Loading...")
    # Only keep post title, headers, and content fornm the full HTML
    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content")
    )
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )

    docs = loader.load()
    print(f"Total Characters: {len(docs[0].page_content)}")

    print("Splitting...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,  # Track index in original document
    )

    all_splits = text_splitter.split_documents(docs)
    print(f"Split blog post into {len(all_splits)} sub-documents.")

    print("Storing...")
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

    document_ids = vector_store.add_documents(documents=all_splits)
    print(document_ids[:3])


if __name__ == "__main__":
    main()
