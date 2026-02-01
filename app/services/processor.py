from typing import List, Dict, Optional
import logging

# LangChain Imports 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.schemas import QueryUnderstanding, SessionSummary

logger = logging.getLogger("uvicorn.error")

class QueryProcessor:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API Key is missing in .env")
            
        # FIX: Use ChatGoogleGenerativeAI instead of genai.GenerativeModel for LangChain compatibility
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=api_key,
            temperature=0
        )

    async def understand_query(self, user_query: str, history: List[Dict], summary: Optional[SessionSummary]) -> QueryUnderstanding:
        """
        Flow 2: Query Understanding using LangChain Structured Output.
        """
        # 1. Enforce the output schema to Pydantic Object automatically
        structured_llm = self.llm.with_structured_output(QueryUnderstanding)

        # 2. Prepare Context
        summary_text = summary.model_dump_json() if summary else "No prior memory."
        
        # Truncate history to avoid token overflow (taking the last 6 messages)
        recent_history = history[-6:]
        history_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in recent_history])

        # 3. Create Prompt Template
        system_prompt = """
        You are a language expert. Analyze the user's query based on the provided Memory and History.
        
        [SUMMARY MEMORY]: {summary}
        [RECENT HISTORY]: {history}
        
        Your Task:
        1. is_ambiguous: Set to True if the query lacks context, subject, or predicate (e.g., "what is it?", "how to do that?").
        2. rewritten_query: Rewrite the query to be fully self-contained and unambiguous (Coreference Resolution).
        3. needed_context_from_memory: Identify what information from the Summary Memory is needed to answer.
        4. clarifying_questions: Generate follow-up questions if the user's intent is too vague to be rewritten.
        5. final_augmented_context: Synthesize all necessary information to answer the query.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])

        # 4. Build Chain
        chain = prompt | structured_llm
        
        try:
            return await chain.ainvoke({
                "summary": summary_text,
                "history": history_text,
                "query": user_query
            })
        except Exception as e:
            logger.error(f"❌ Query Understanding failed: {e}")
            # Safe Fallback
            return QueryUnderstanding(
                is_ambiguous=False,
                rewritten_query=user_query,
                needed_context_from_memory=[],
                clarifying_questions=[],
                final_augmented_context=f"Query: {user_query}\nContext: {history_text}"
            )

    async def generate_final_answer(self, augmented_context: str) -> str:
        """
        Flow 3: Generate the final answer.
        """
        prompt = ChatPromptTemplate.from_template(
            "Based on the following context, please answer the user concisely and clearly:\n\n[Context]: {context}"
        )
        
        # StrOutputParser returns the string text directly
        chain = prompt | self.llm | StrOutputParser()
        
        return await chain.ainvoke({"context": augmented_context})