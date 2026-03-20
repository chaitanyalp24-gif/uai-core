from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.manager import AIManager

app = FastAPI()
manager = AIManager()


@app.get("/")
def home():
    return {"status": "UAI system online"}


@app.post("/uai")
def uai_entry(prompt: str):
    return manager.handle_prompt(prompt)
