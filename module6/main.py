import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchainhub import pull

load_dotenv()


if __name__ == "__main__":
    print(" Retrieving...")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    query = "what is Pinecone in machine learning?"

    # Use langchain_community for Pinecone in v1
    vectorstore = Pinecone.from_existing_index(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    # Pull the prompt from langchainhub
    retrieval_qa_chat_prompt = pull("langchain-ai/retrieval-qa-chat")

    # Create retriever
    retriever = vectorstore.as_retriever()

    # Build the chain using LCEL (LangChain Expression Language)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retrieval_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | retrieval_qa_chat_prompt
        | llm
    )

    result = retrieval_chain.invoke({"question": query})

    print(result)
