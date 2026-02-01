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
            model="gemini-2.5-flash-lite", 
            google_api_key=api_key,
            temperature=0
        )
        self.threshold = threshold

    def count_tokens(self, messages: List[Dict]) -> int:
        """
        Uses LangChain's built-in token counting utility.
        """
        if not messages:
            return 0
            
        # Combine content for counting
        content = " ".join([str(m.get("content", "")) for m in messages])
        
        try:
            # LangChain automatically handles the API call for token counting
            return self.llm.get_num_tokens(content)
        except Exception as e:
            logger.warning(f"⚠️ Token counting failed ({e}). Using heuristic fallback.")
            # Rough estimate: characters / 4
            return len(content) // 4

    async def generate_summary(self, messages: List[Dict]) -> SessionSummary:
        """
        Summarizes the conversation using LCEL (LangChain Expression Language)
        and Structured Output to ensure strict adherence to the schema.
        """
        # 1. Enforce the output schema using Pydantic (SessionSummary)
        # LangChain handles retries if the model output is malformed.
        structured_llm = self.llm.with_structured_output(SessionSummary)

        # 2. Create the Prompt Template
        # We instruct the AI to act as a specialized summarizer.
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert AI assistant specialized in summarizing conversations. Your task is to extract key information into a structured JSON format."),
            ("human", "Please summarize the following conversation: {messages}")
        ])

        # 3. Build the Chain: Prompt -> LLM -> JSON Object
        chain = prompt | structured_llm

        try:
            # Execute the Chain
            return await chain.ainvoke({"messages": str(messages)})
        except Exception as e:
            logger.error(f"❌ LangChain Summarization failed: {e}")
            # return an empty summary as a fallback to prevent app crash
            return SessionSummary(
                user_profile={"prefs": {}, "constraints": []},
                key_facts=[],
                decisions=[],
                open_questions=[],
                todos=[],
                message_range_summarized={"from": 0, "to": 0}
            )