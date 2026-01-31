from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class UserProfile(BaseModel):
    prefs: Dict[str, str] = Field(description="Sở thích người dùng (vđ: ngôn ngữ, phong cách)")
    constraints: List[str] = Field(description="Các hạn chế hoặc quy tắc cần tuân thủ")

class SessionSummary(BaseModel):
    user_profile: UserProfile
    key_facts: List[str] = Field(description="Các sự kiện quan trọng trong hội thoại")
    decisions: List[str] = Field(description="Các quyết định đã được đưa ra")
    open_questions: List[str] = Field(description="Các câu hỏi còn dang dở")
    todos: List[str] = Field(description="Các việc cần làm tiếp theo")
    message_range_summarized: Dict[str, int] = Field(description="Ví dụ: {'from': 0, 'to': 42}")

class QueryUnderstanding(BaseModel):
    is_ambiguous: bool = Field(description="True nếu câu hỏi mơ hồ hoặc thiếu thông tin")
    rewritten_query: Optional[str] = Field(description="Truy vấn được viết lại để rõ ràng hơn")
    needed_context_from_memory: List[str] = Field(description="Các thông tin cần lấy từ bộ nhớ")
    clarifying_questions: List[str] = Field(description="Các câu hỏi làm rõ nếu vẫn chưa hiểu ý người dùng")
    final_augmented_context: str = Field(description="Ngữ cảnh cuối cùng đã được bổ sung")
