import os
import httpx
from app.memory import Memory


class AIManager:
    def __init__(self):
        self.memory = Memory()

    def handle_prompt(self, user_prompt: str):
        intent = self.classify_intent(user_prompt)
        plan = self.create_plan(intent)

        memory_context = self.load_memory_context()
        results = []

        for step in plan:
            results.append(
                self.process_step(step, intent, user_prompt, memory_context)
            )

        final_text = "\n".join(results)
        self.memory.save(user_prompt, final_text)

        return {
            "intent": intent,
            "plan": plan,
            "memory_used": memory_context,
            "results": results
        }

    # -------- MEMORY --------
    def load_memory_context(self):
        memories = self.memory.recent(limit=3)
        if not memories:
            return ""

        context = "PAST MEMORY:\n"
        for user, ai in memories:
            context += f"- User: {user}\n- AI: {ai[:150]}\n\n"
        return context

    # -------- INTENT --------
    def classify_intent(self, text: str):
        t = text.lower()

        if "remember" in t:
            return "memory"
        if "code" in t or "python" in t:
            return "coding"
        if "plan" in t or "strategy" in t:
            return "planning"
        if "write" in t or "blog" in t:
            return "writing"

        return "general"

    # -------- PLAN --------
    def create_plan(self, intent: str):
        return {
            "memory": [
                "Acknowledge memory",
                "Confirm it is saved"
            ],
            "coding": [
                "Understand problem",
                "Write code",
                "Explain result"
            ],
            "planning": [
                "Analyze request",
                "Create strategy",
                "Finalize output"
            ]
        }.get(intent, ["Respond helpfully"])

    # -------- EXECUTION --------
    def process_step(self, step, intent, user_prompt, memory_context):
        prompt = f"""
{memory_context}

TASK: {step}
USER REQUEST: {user_prompt}
"""
        return self.call_groq(prompt)

    # -------- GROQ API (SAFE) --------
    def call_groq(self, user_prompt: str):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "[ERROR] GROQ_API_KEY missing"

        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI assistant with memory."
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code != 200:
                return f"[Groq Error {response.status_code}] {response.text}"

            return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            return f"[Groq Exception] {str(e)}"
