import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters.character import CharacterTextSplitter

load_dotenv()


def main():
    print("Ingesting...")
    loader = TextLoader(
        "/home/atiyehghm/Desktop/langchain/langchain_udemy_course/langchain-course/module6/mediumblog1.txt"
    )
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks...")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("METIS_API_KEY"),
        base_url="https://api.metisai.ir/openai/v1",
    )

    print("Ingesting...")
    # Initialize ChromaDB (persist locally)
    db = Chroma(
        collection_name="medium-blogs",
        embedding_function=embeddings,
        persist_directory="./chroma_data",  # folder to save data
    )

    db.add_documents(texts)
    print("Finished!")


if __name__ == "__main__":
    main()
