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

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    db["history"].append({"role": "user", "content": request.message})
    
    current_tokens = memory_service.count_tokens(db["history"])
    summary_info = None
    
    if current_tokens > memory_service.threshold:
        split_idx = max(1, int(len(db["history"]) * 0.8))
        to_summarize = db["history"][:split_idx]
        
        db["short_term_memory"] = await memory_service.generate_summary(to_summarize)
        
        db["history"] = db["history"][split_idx:]
        summary_info = db["short_term_memory"]

    understanding = await query_processor.understand_query(
        user_query=request.message,
        history=db["history"][:-1],
        summary=db["short_term_memory"]
    )

    if understanding.is_ambiguous and understanding.clarifying_questions:
        return {
            "type": "clarification_needed",
            "is_ambiguous": True,
            "clarifying_questions": understanding.clarifying_questions,
            "rewritten_query": understanding.rewritten_query,
            "summary_triggered": summary_info is not None
        }

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