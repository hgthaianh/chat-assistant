from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class UserProfile(BaseModel):
    prefs: Dict[str, str] = Field(description="User preferences (e.g., language, style).")
    constraints: List[str] = Field(description="Constraints or rules that must be followed.")

class SessionSummary(BaseModel):
    user_profile: UserProfile
    key_facts: List[str] = Field(description="Key facts extracted from the conversation.")
    decisions: List[str] = Field(description="Decisions that have been made.")
    open_questions: List[str] = Field(description="Questions that remain unanswered.")
    todos: List[str] = Field(description="Action items or next steps.")
    message_range_summarized: Dict[str, int] = Field(description="Range of messages summarized, e.g., {'from': 0, 'to': 42}.")

class QueryUnderstanding(BaseModel):
    is_ambiguous: bool = Field(description="True if the query is ambiguous or lacks context.")
    rewritten_query: Optional[str] = Field(description="The query rewritten to be self-contained and clear.")
    needed_context_from_memory: List[str] = Field(description="Specific information keys needed from memory.")
    clarifying_questions: List[str] = Field(description="Follow-up questions to ask the user if the intent is still unclear.")
    final_augmented_context: str = Field(description="The final context string augmented with memory and history.")

