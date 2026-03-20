import os
import httpx


class AIManager:

    def handle_prompt(self, user_prompt: str):
        intent = self.classify_intent(user_prompt)
        plan = self.create_plan(intent)
        results = []

        for step in plan:
            results.append(self.process_step(step, intent, user_prompt))

        return {
            "intent": intent,
            "plan": plan,
            "results": results
        }

    # ------------------------
    # Intent classification
    # ------------------------
    def classify_intent(self, text: str) -> str:
        t = text.lower()

        if "code" in t or "python" in t or "program" in t:
            return "coding"
        if "plan" in t or "strategy" in t or "business" in t:
            return "planning"
        if "write" in t or "blog" in t or "article" in t:
            return "writing"
        if "research" in t or "find" in t:
            return "research"

        return "general"

    # ------------------------
    # Task planning
    # ------------------------
    def create_plan(self, intent: str):
        return {
            "coding": [
                "Understand the problem",
                "Design the solution",
                "Write the code",
                "Explain the result"
            ],
            "planning": [
                "Define the goal",
                "Analyze requirements",
                "Design a strategy",
                "Finalize the plan"
            ],
            "writing": [
                "Choose topic",
                "Create outline",
                "Write content",
                "Edit and refine"
            ],
            "research": [
                "Collect sources",
                "Extract information",
                "Summarize findings"
            ]
        }.get(intent, ["Respond clearly"])

    # ------------------------
    # Route steps to Groq
    # ------------------------
    def process_step(self, step: str, intent: str, user_prompt: str):
        prompt = f"{step}. User request: {user_prompt}"

        return self.call_groq(prompt)

    # ------------------------
    # ✅ Groq API Call (FIXED)
    # ------------------------
    def call_groq(self, user_prompt: str):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            return "[ERROR] GROQ_API_KEY not loaded"

        url = "https://api.groq.com/openai/v1/chat/completions"

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional and helpful AI assistant."
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

        response = httpx.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            return f"[ERROR] Groq Error {response.status_code}: {response.text}"

        return response.json()["choices"][0]["message"]["content"]
