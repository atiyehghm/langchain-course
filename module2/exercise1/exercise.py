from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def implement_set_api_key(api_key: str) -> None:
    lines= open(".env").readlines()
    filtered = [l for l in lines if not l.startswith("GROQ_API_KEY=")]

    with open(".env", "w") as f:
        f.writelines(filtered)
        f.write(f"GROQ_API_KEY={api_key}")

def implement_llama_4_model() -> ChatGroq:
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-4-8b-instant",
        temperature=0.0
    )
    return llm

def implement_llama_3_3_model():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.0
    )
    return llm

def implement_query_model(model: ChatGroq, test_prompt: str) -> str:
    chain = model | StrOutputParser()

    response = chain.invoke(test_prompt)
    return response



if __name__ == "__main__":
    model = implement_llama_3_3_model()
    response = implement_query_model(model, "What is Machine Learning?")
    print(response)