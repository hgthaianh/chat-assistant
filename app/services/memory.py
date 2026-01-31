from typing import List, Dict
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.schemas import SessionSummary

logger = logging.getLogger("uvicorn.error")

class MemoryService:
    def __init__(self, api_key: str, threshold: int = 10000):
        if not api_key:
            raise ValueError("API Key is missing in .env")
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=api_key,
            temperature=0
        )
        self.threshold = threshold

    def count_tokens(self, messages: List[Dict]) -> int:
        if not messages:
            return 0
            
        content = " ".join([str(m.get("content", "")) for m in messages])
        
        try:
            return self.llm.get_num_tokens(content)
        except Exception as e:
            logger.warning(f"⚠️ Token counting failed ({e}). Using heuristic fallback.")
            return len(content) // 4

    async def generate_summary(self, messages: List[Dict]) -> SessionSummary:
        structured_llm = self.llm.with_structured_output(SessionSummary)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là một trợ lý AI chuyên tóm tắt hội thoại. Nhiệm vụ của bạn là trích xuất thông tin quan trọng vào cấu trúc JSON."),
            ("human", "Hãy tóm tắt đoạn hội thoại sau: {messages}")
        ])

        chain = prompt | structured_llm

        try:
            return await chain.ainvoke({"messages": str(messages)})
        except Exception as e:
            logger.error(f"❌ LangChain Summarization failed: {e}")
            return SessionSummary(
                user_profile={"prefs": {}, "constraints": []},
                key_facts=[],
                decisions=[],
                open_questions=[],
                todos=[],
                message_range_summarized={"from": 0, "to": 0}
            )