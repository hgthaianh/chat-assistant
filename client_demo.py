import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def print_step(step_name):
    print(f"\n{'='*50}\n▶ {step_name}\n{'='*50}")

def send_chat(message):
    print(f"User: {message}")
    start = time.time()
    try:
        response = requests.post(f"{BASE_URL}/chat", json={"message": message})
        duration = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"Assistant ({duration:.2f}s):")
            
            # Print significant debug info as requested
            if "debug_info" in data:
                debug = data["debug_info"]
                if debug.get("is_ambiguous"):
                    print(f"   [🔍 Detected Ambiguity]")
                    print(f"   [📝 Rewritten Query]: {debug.get('rewritten')}")
                if debug.get("summary_triggered"):
                    print(f"   [💾 Memory Triggered]: New summary created from conversation history!")
            
            print(f"   ► Response: {data.get('response')}")
            
            if data.get('type') == 'clarification_needed':
                print(f"   ❓ Clarifying Questions: {data.get('clarifying_questions')}")
        else:
            print("Error:", response.text)
    except Exception as e:
        print(f"Connection Error: {e}")

def run_demo():
    print_step("FLOW 1: SESSION MEMORY TRIGGER")
    print("Simulating a very long conversation to force the system to summarize...\n")
    
    # Send a massive text block to exceed the token threshold
    long_text = "What is Artificial Intelligence? " * 300 
    send_chat(f"Here is a lot of context for our memory test: {long_text[:50]}... (and so on)")

    print_step("FLOW 2: AMBIGUOUS QUERY HANDLING")
    # Step 1: Establish context
    send_chat("I am applying for the AI Engineer position at Vulcan Labs.")
    
    # Step 2: Ask an ambiguous question (System must resolve 'the test' from context)
    send_chat("How long does the test usually take?") 

    print_step("FLOW 3: CLARIFICATION REQUEST")
    # Ask something completely vague that requires clarification
    send_chat("Can you help me install it?")

if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"Error: Is the server running? ({e})")