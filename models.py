# Pydantic models for Intelligent Search API (add here if needed).
# GET /ask-ai returns a plain dict; no request/response models required for the current flow.
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class AskAiResponse(BaseModel):
    """Optional typed shape for /ask-ai response (for docs/validation)."""
    type: str  # NAVIGATE | AI_RESPONSE | ERROR
    content: Optional[str] = None
    tableData: Optional[List[Dict[str, Any]]] = None
    thought_process: Optional[str] = None
    sql_query: Optional[str] = None
    action_taken: Optional[str] = None
    next_steps: Optional[List[str]] = None
    navigateTo: Optional[str] = None
    route: Optional[str] = None
    source: Optional[str] = None
