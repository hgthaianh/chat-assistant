import google.generativeai as genai
from typing import List, Dict, Optional
from app.schemas import QueryUnderstanding, SessionSummary

class QueryProcessor:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API Key is missing in .env")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    async def understand_query(self, user_query: str, history: List[Dict], summary: Optional[SessionSummary]) -> QueryUnderstanding:
        """
        Bước 1 & 2: Phân tích câu hỏi, viết lại nếu mơ hồ, và xác định ngữ cảnh cần thiết.
        """
        history_str = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in history])
        summary_str = summary.model_dump_json() if summary else "Không có bộ nhớ trước đó."

        prompt = f"""
        Bạn là một chuyên gia ngôn ngữ. Nhiệm vụ của bạn là phân tích ý định người dùng (Query Understanding).
        
        [BỘ NHỚ TÓM TẮT TỪ PHIÊN TRƯỚC]:
        {summary_str}
        
        [LỊCH SỬ HỘI THOẠI GẦN ĐÂY]:
        {history_str}
        
        [CÂU HỎI MỚI CỦA NGƯỜI DÙNG]:
        "{user_query}"
        
        Hãy thực hiện các bước sau và trả về kết quả JSON (không thêm văn bản thừa):
        1. 'is_ambiguous': True nếu câu hỏi thiếu chủ ngữ/vị ngữ hoặc dùng đại từ thay thế (nó, cái đó, họ) mà không rõ ràng.
        2. 'rewritten_query': Viết lại câu hỏi đầy đủ nghĩa dựa trên lịch sử và bộ nhớ (Coreference Resolution).
        3. 'needed_context_from_memory': Liệt kê các thông tin cần lấy từ bộ nhớ.
        4. 'clarifying_questions': Nếu không thể đoán được ý định, hãy đặt 1-3 câu hỏi để hỏi lại người dùng.
        5. 'final_augmented_context': Tổng hợp toàn bộ thông tin cần thiết (câu hỏi đã viết lại + dữ liệu liên quan) để chuẩn bị cho bước trả lời.
        """

        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": QueryUnderstanding
                }
            )
            return QueryUnderstanding.model_validate_json(response.text)
        except Exception as e:
            print(f"Error in understand_query: {e}")
            return QueryUnderstanding(
                is_ambiguous=False,
                rewritten_query=user_query,
                needed_context_from_memory=[],
                clarifying_questions=[],
                final_augmented_context=f"User Query: {user_query}\nHistory: {history_str}"
            )

    async def generate_final_answer(self, augmented_context: str) -> str:
        prompt = f"""
        Bạn là trợ lý AI hữu ích. Hãy trả lời người dùng dựa trên thông tin sau:
        
        [NGỮ CẢNH ĐÃ BỔ SUNG]:
        {augmented_context}
        
        Yêu cầu: Trả lời tự nhiên, ngắn gọn, đi thẳng vào vấn đề.
        """
        
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            return f"Xin lỗi, tôi gặp sự cố khi sinh câu trả lời. (Error: {str(e)})"