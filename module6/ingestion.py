from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings



load_dotenv()

def main():
    print("Ingesting...")
    loader = TextLoader("/home/atiyehghm/Desktop/langchain/langchain_udemy_course/langchain-course/module6/mediumblog1.txt")
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks...")


if __name__ == "__main__":
    main()