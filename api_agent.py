from fastapi import FastAPI

from pydantic import BaseModel

from dotenv import load_dotenv

from langchain_groq import ChatGroq

# Load env
load_dotenv()

# ---------------- LLM ---------------- #

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
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

# ---------------- API ROUTE ---------------- #

@app.post("/chat", response_model=ChatResponse)

def chat_endpoint(request: ChatRequest):

    question = request.question

    response = llm.invoke(question)

    return ChatResponse(
        answer=response.content
    )