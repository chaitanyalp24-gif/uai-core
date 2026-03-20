import os
import httpx
from app.memory import Memory
from app.vector_memory import VectorMemory


class AIManager:
    def __init__(self):
        self.memory = Memory()
        self.vector_memory = VectorMemory()

    def handle_prompt(self, user_prompt: str):
        intent = self.classify_intent(user_prompt)

        # 🔎 Retrieve semantic memories
        semantic_context = self.get_semantic_context(user_prompt)

        plan = self.create_plan(intent)
        results = []

        for step in plan:
            results.append(
                self.process_step(
                    step,
                    intent,
                    user_prompt,
                    semantic_context
                )
            )

        final_output = "\n".join(results)

        # Save to both memories
        self.memory.save(user_prompt, final_output)
        self.vector_memory.add(user_prompt)
        self.vector_memory.add(final_output)

        return {
            "intent": intent,
            "semantic_memory": semantic_context,
            "plan": plan,
            "results": results
        }

    # ---------- VECTOR MEMORY ----------
    def get_semantic_context(self, prompt):
        matches = self.vector_memory.search(prompt, k=3)
        if not matches:
            return ""

        context = "RELEVANT PAST KNOWLEDGE:\n"
        for m in matches:
            context += f"- {m}\n"
        return context

    # ---------- INTENT ----------
    def classify_intent(self, text):
        t = text.lower()
        if "remember" in t:
            return "memory"
        if "code" in t or "python" in t:
            return "coding"
        if "plan" in t or "strategy" in t:
            return "planning"
        return "general"

    # ---------- PLAN ----------
    def create_plan(self, intent):
        return {
            "coding": [
                "Understand problem",
                "Write solution",
                "Explain code"
            ],
            "planning": [
                "Analyze goal",
                "Design approach",
                "Refine result"
            ],
            "memory": [
                "Confirm information",
                "Acknowledge storage"
            ]
        }.get(intent, ["Respond helpfully"])

    # ---------- EXECUTION ----------
    def process_step(self, step, intent, user_prompt, semantic_context):
        prompt = f"""
{semantic_context}

TASK: {step}
USER REQUEST: {user_prompt}
"""
        return self.call_groq(prompt)

    # ---------- GROQ ----------
    def call_groq(self, user_prompt):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "[ERROR] GROQ_API_KEY missing"

        url = "https://api.groq.com/openai/v1/chat/completions"

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are an AI assistant with semantic memory."},
                {"role": "user", "content": user_prompt}
            ]
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            r = httpx.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code != 200:
                return f"[Groq Error {r.status_code}] {r.text}"
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Groq Exception] {str(e)}"
