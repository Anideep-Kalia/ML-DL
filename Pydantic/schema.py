from pydantic import BaseModel, Field
from typing import List, Literal

class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query for vector database")
    filters: List[str] = Field(default_factory=list)

class WeatherRequest(BaseModel):
    city: str
    units: Literal["celsius", "fahrenheit"]         # for creating references

class FinalAnswer(BaseModel):
    answer: str
    sources: List[str]