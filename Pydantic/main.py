# main.py
from fastapi import FastAPI
from rag_runner import run_rag
from tool_calling import run_weather_agent

app = FastAPI()

@app.post("/ask")
def ask(q: str):
    rag_docs = run_rag(q)
    weather = run_weather_agent(q)

    return {
        "rag": [d.page_content for d in rag_docs],
        "weather": weather
    }


# Script for run: uvicorn main:app --reload

