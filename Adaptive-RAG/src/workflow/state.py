from typing import List, TypedDict

class GraphState(TypedDict):
    """State object for workflow containing query, documents, and control flags."""
    question: str # User's Original Query
    generation: str # LLM-generated response
    web_search: bool # Control flag for web-search equipment
    documents: List[str] # Retrieved document context
