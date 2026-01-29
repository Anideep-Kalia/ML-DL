from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)
## streamlit framework

st.title('Langchain Demo With LLAMA2 API')
input_text=st.text_input("Search the topic u want")

# ollama LLAma2 LLm 
llama=Ollama(model="llama2")
output_parser_llama=StrOutputParser()
chain_llama=prompt|llama|output_parser_llama

# openAI LLm 
gpt=ChatOpenAI(model="gpt-3.5-turbo")
output_parser_gpt=StrOutputParser()
chain_gpt=prompt|gpt|output_parser_gpt

if input_text:
    st.write(chain_llama.invoke({"question":input_text}))
    st.write(chain_gpt.invoke({"question":input_text}))
