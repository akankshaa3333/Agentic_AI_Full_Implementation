from fastapi import FastAPI

from pydantic import BaseModel

from dotenv import load_dotenv

from langchain_groq import ChatGroq

# Load env
load_dotenv()

# ---------------- LLM ---------------- #

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ---------------- FASTAPI ---------------- #

app = FastAPI()

# ---------------- REQUEST MODEL ---------------- #

class ChatRequest(BaseModel):

    question: str

# ---------------- RESPONSE MODEL ---------------- #

class ChatResponse(BaseModel):

    answer: str

# ---------------- ASYNC ENDPOINT ---------------- #

@app.post("/chat", response_model=ChatResponse)

async def chat(request: ChatRequest):

    question = request.question

    # Async LLM call
    response = await llm.ainvoke(question)

    return ChatResponse(
        answer=response.content
    )