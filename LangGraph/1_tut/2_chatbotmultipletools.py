#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## tools

from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper


# These are the just API fetching, there is no vectordb or vectorisation, here 'top_k_results' is applied based on  classical Information Retrieval (IR) i.e. Both arXiv and Wikipedia rank results using keyword-based relevance signals, metadata, and heuristics.
# 
# | System         | Ranking basis                    |
# | -------------- | -------------------------------- |
# | arXiv API      | Keyword + metadata weighting     |
# | Wikipedia API  | BM25 + title/link boosts         |
# | FAISS / Chroma | Embedding similarity             |
# | Google Search  | Hybrid (keywords + ML + signals) |
# 

# In[ ]:


api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=500)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv,description="Query arxiv papers")
print(arxiv.name)


# In[3]:


arxiv.invoke("What is the latest research on quantum computing?")


# In[ ]:


api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
wiki.name


# #### tavily_search is used for live, web-based information retrieval for LLMs. It’s a search engine optimized specifically for LLM agents.
# 
# - queries the live web
# - ranks results using classical + ML-based relevance
# - returns clean, summarized, LLM-friendly text
# - minimizes noise (ads, nav bars, SEO junk)

# In[ ]:


from dotenv import load_dotenv
load_dotenv()

import os

os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


# In[ ]:


### Tavily Search Tool
from langchain_community.tools.tavily_search import TavilySearchResults
tavily = TavilySearchResults()


# In[ ]:


tavily.invoke("Provide me the recent AI news?")


# In[ ]:


## combine all these tools in the list
tools=[arxiv, wiki, tavily]


# In[ ]:


## Initialize the LLM Model
from langchain_groq import ChatGroq

llm=ChatGroq(model="qwen-qwq-32b")


# In[ ]:


llm.invoke("What is AI")


# In[ ]:


llm_with_tools=llm.bind_tools(tools=tools)


# In[ ]:


## Execute this call
llm_with_tools.invoke("What is the recent news on AI?")


# In[ ]:


## Execute this call
llm_with_tools.invoke("What is the latest research on quantum computing?")


# In[ ]:


## Execute this call
llm_with_tools.invoke("What is machine learning?")


# ## Workflow 

# In[11]:


## State Schema
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage ## Human message or AI message
from typing import Annotated  ## labelling
from langgraph.graph.message import add_messages  ## Reducers in Langgraph


# In[12]:


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]



# In[13]:


### Entire Chatbot With LangGraph
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition


# In[ ]:


### Node definition
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

## Edgess
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)

graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))


# In[26]:


messages=graph.invoke({"messages":"1706.03762"})
for m in messages['messages']:
    m.pretty_print()


# In[27]:


messages=graph.invoke({"messages":"Hi My name is Krish"})
for m in messages['messages']:
    m.pretty_print()


# In[ ]:


### Node definition
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

## Edgess
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "tool_calling_llm")

graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))


# In[19]:


messages=graph.invoke({"messages":"What is the recent AI news and then please tell me the recent research paper on quantum computing?"})
for m in messages['messages']:
    m.pretty_print()


# In[ ]:




