import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from dotenv import load_dotenv
import uvicorn

from app.services.memory import MemoryService
from app.services.processor import QueryProcessor
from app.schemas import SessionSummary, QueryUnderstanding

load_dotenv()

app = FastAPI(title="Vulcan Labs AI Assistant")

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
memory_service = MemoryService(api_key=GEMINI_KEY, threshold=200)
query_processor = QueryProcessor(api_key=GEMINI_KEY)


db = {
    "history": [],              
    "short_term_memory": None  
}

class ChatRequest(BaseModel):
    message: str

@app.post("/reset")
async def reset_endpoint():
    """Clear conversation history and memory for a fresh start."""
    db["history"] = []
    db["short_term_memory"] = None
    return {"message": "Memory and History reset successfully."}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. Append user message to history
    db["history"].append({"role": "user", "content": request.message})
    
    # 2. Check Memory Threshold
    current_tokens = memory_service.count_tokens(db["history"])
    summary_info = None
    
    # If history exceeds threshold (e.g., 200 tokens), trigger summarization
    if current_tokens > memory_service.threshold:
        # Keep 20% recent messages, summarize the older 80%
        split_idx = max(1, int(len(db["history"]) * 0.8))
        to_summarize = db["history"][:split_idx]
        
        # Generate and store summary
        db["short_term_memory"] = await memory_service.generate_summary(to_summarize)
        
        # REQUIREMENT: message_range_summarized must be accurate. 
        # We manually overwrite it here because the LLM might guess incorrectly.
        db["short_term_memory"].message_range_summarized = {
            "from": 0,
            "to": split_idx - 1  # 0-indexed inclusive range
        }
        
        # Truncate history, keeping only the recent chunk
        db["history"] = db["history"][split_idx:]
        summary_info = db["short_term_memory"]

    # 3. Query Understanding & Context Augmentation
    understanding = await query_processor.understand_query(
        user_query=request.message,
        history=db["history"][:-1], # Exclude current message for context analysis
        summary=db["short_term_memory"]
    )

    # 4. Handle Ambiguity / Clarification
    if understanding.is_ambiguous and understanding.clarifying_questions:
        return {
            "type": "clarification_needed",
            "is_ambiguous": True,
            "clarifying_questions": understanding.clarifying_questions,
            "rewritten_query": understanding.rewritten_query,
            "summary_triggered": summary_info is not None
        }

    # 5. Generate Final Response
    final_response = await query_processor.generate_final_answer(
        understanding.final_augmented_context
    )
    
    db["history"].append({"role": "assistant", "content": final_response})

    return {
        "type": "final_answer",
        "response": final_response,
        "debug_info": {
            "is_ambiguous": understanding.is_ambiguous,
            "rewritten": understanding.rewritten_query,
            "context_augmented": True,
            "summary_triggered": summary_info is not None,
            "current_token_count": current_tokens
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)