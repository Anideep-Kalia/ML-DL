from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from openai import OpenAI
from schemas import SearchQuery
from schemas import WeatherRequest
from tools import get_weather

# data preprocessing
docs = [
    "LangChain is a framework for building LLM applications.",
    "RAG means Retrieval Augmented Generation.",
    "Pydantic enforces structured outputs from LLMs."
]
documents = [Document(page_content=d) for d in docs]
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.split_documents(documents)
db = FAISS.from_documents(chunks, OpenAIEmbeddings())


# LLM Calling
client = OpenAI()
def generate_search_query(question: str) -> SearchQuery:
    prompt = f"""
        You are an AI that converts questions into structured search queries.
        Question: {question}
        Return ONLY valid JSON that matches:
        {SearchQuery.model_json_schema()}
    """

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}]
    )

    raw = resp.choices[0].message.content
    return SearchQuery.model_validate_json(raw)


# RAG implementing
def run_rag(question: str):
    search = generate_search_query(question)
    docs = db.similarity_search(search.query, k=3)
    return docs


# Tool calling 
def run_weather_agent(user_query: str):
    prompt = f"""
        You are a tool-using AI.
        User said: {user_query}
        If this is a weather request, return JSON matching:
        {WeatherRequest.model_json_schema()}
        Else return null.
    """

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}]
    )

    raw = resp.choices[0].message.content

    if raw.strip() == "null":
        return "Not a weather query"

    parsed = WeatherRequest.model_validate_json(raw)
    return get_weather(parsed)
