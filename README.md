# Chat Assistant with Session Memory

A runnable demo of a chat assistant backend that supports **Session Memory** (via automatic summarization) and **Query Understanding** (rewriting ambiguous queries and asking clarifying questions).

## 🚀 Features

1.  **Session Memory**: Automatically triggers summarization when conversation context exceeds a configurable threshold (default: ~200 tokens for demo purposes). The summary is stored and used to augment future context.
2.  **Query Understanding**:
    *   **Ambiguity Detection**: Identifies if a user's query is vague given the context.
    *   **Query Rewriting**: Rewrites ambiguous queries to be self-contained (Coreference Resolution).
    *   **Clarification**: Generates clarifying questions if the intent remains distinct.

## 🛠️ Tech Stack

*   **Language**: Python 3.10+
*   **API Framework**: FastAPI
*   **LLM Orchestration**: LangChain
*   **Model**: Google Gemini 2.5 Flash Lite
*   **Structured Output**: Pydantic & LangChain Structured Output

## 📦 Setup Instructions

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd chat-assistant
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Configuration**:
    Create a `.env` file in the root directory and add your Gemini API Key:
    ```
    GEMINI_API_KEY=your_api_key_here
    ```

## 🏃 check How to Run

### 1. Start the Backend Server
Run the FastAPI server:
```bash
python -m app.main
```
The server will start at `http://0.0.0.0:8000`.

### 2. Run the Client Demo
Open a new terminal and run the demo script to verify the required flows:
```bash
python client_demo.py
```

This script demonstrates two key flows:
*   **Flow 1**: Sends a long message to trigger the **Session Memory Summarization**.
*   **Flow 2**: Simulates a conversation with ambiguous queries to trigger **Query Understanding** (Rewriting & Clarification).

## 🏗️ High-Level Design

### Pipeline
1.  **Chat Input**: User sends a message.
2.  **Memory Check**:
    *   System checks if current conversation history token count > `threshold`.
    *   **If triggered**: Old messages are summarized into a structured `SessionSummary` object (User Profile, key facts, etc.) and stored in `short_term_memory`.
3.  **Query Understanding**:
    *   The `QueryProcessor` analyzes the new query + recent history + session summary.
    *   It determines `is_ambiguous`.
    *   **If Ambiguous**: It rewrites the query and may generate `clarifying_questions`.
    *   **Context Augmentation**: Relevant info from memory is retrieved.
4.  **Response Generation**:
    *   The final augmented context is sent to the LLM to generate the answer.

### Structured Schemas
The system uses strict Pydantic models for outputs (see `app/schemas.py`):
*   `SessionSummary`: Captures User Profile, Decisions, Todos, etc.
*   `QueryUnderstanding`: Captures Ambiguity status, Rewritten Query, and Clarifying Questions.

## ⚠️ Assumptions & Limitations
*   **Memory**: Currently stored in-memory (Python dictionary). Restarting the server clears memory.
*   **Threshold**: Set low (200 tokens) in the code for easy demonstration. In production, this would be ~10k tokens.
*   **Concurrency**: Simple blocking implementation for demo; not optimized for high concurrent load.