from pydantic import BaseModel, Field
from typing import List, Literal

class SearchQuery(BaseModel):
# this '...' is elipsis operator signiying that this field is required and has no value as default so LLMs will be forced to fill something
# and description is for LLMs to get context of the class and it will be added to the query  
    query: str = Field(..., description="Search query for vector database")
    filters: List[str] = Field(default_factory=list)

class WeatherRequest(BaseModel):
    city: str
    units: Literal["celsius", "fahrenheit"]         # for creating references

class FinalAnswer(BaseModel):
    answer: str
    sources: List[str]