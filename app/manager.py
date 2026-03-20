import os
import httpx
from app.memory import Memory


class AIManager:
    def __init__(self):
        self.memory = Memory()

    def handle_prompt(self, user_prompt: str):
        intent = self.classify_intent(user_prompt)
        plan = self.create_plan(intent)

        past_context = self.load_memory_context()
        results = []

        for step in plan:
            result = self.process_step(step, intent, user_prompt, past_context)
            results.append(result)

        final_output = "\n".join(results)
        self.memory.save(user_prompt, final_output)

        return {
            "intent": intent,
            "plan": plan,
            "memory_used": past_context,
            "results": results
        }

    # ------------------------
    # Memory
    # ------------------------
    def load_memory_context(self):
        memories = self.memory.recent(limit=3)
        if not memories:
            return ""

        context = "PAST CONTEXT:\n"
        for user_input, ai_output in memories:
            context += f"- User: {user_input}\n- AI: {ai_output[:200]}\n\n"
        return context

    # ------------------------
    # Intent classification
    # ------------------------
    def classify_intent(self, text: str) -> str:
        t = text.lower()

        if "code" in t or "python" in t:
            return "coding"
        if "plan" in t or "strategy":
            return "planning"
        if "write" in t:
            return "writing"
        if "remember" in t:
            return "memory"

        return "general"

    # ------------------------
    # Planning
    # ------------------------
    def create_plan(self, intent: str):
        return {
            "coding": [
                "Understand the problem",
                "Write code",
                "Explain result"
            ],
            "planning": [
                "Analyze requirements",
                "Create plan",
                "Refine output"
            ],
            "memory": [
                "Acknowledge information",
                "Confirm memory saved"
            ]
        }.get(intent, ["Respond naturally"])

    # ------------------------
    # Task execution
    # ------------------------
    def process_step(self, step, intent, user_prompt, memory_context):
        prompt = f"""
{memory_context}

TASK: {step}
USER REQUEST: {user_prompt}
"""

        return self.call_groq(prompt)

    # ------------------------
    # Groq API
    # ------------------------
    def call_groq(self, user_prompt: str):
        api_key = os.getenv("GROQ_API_KEY")
        url = "https://api.groq.com/openai/v1/chat/completions"

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are an intelligent AI assistant with memory."},
                {"role": "user", "content": user_prompt}
            ]
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        response = httpx.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]
