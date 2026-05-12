from fastapi import FastAPI

from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from dotenv import load_dotenv

from langchain_groq import ChatGroq

# Load env
load_dotenv()

# ---------------- LLM ---------------- #

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    streaming=True,
    temperature=0
)

# ---------------- FASTAPI ---------------- #

app =FastAPI()

# ---------------- REQUEST MODEL ---------------- #

class ChatRequest(BaseModel):

    question: str

# ---------------- STREAM GENERATOR ---------------- #

def generate_response(question: str):

    for chunk in llm.stream(question):

        if chunk.content:

            yield chunk.content

# ---------------- API ENDPOINT ---------------- #

@app.post("/chat")

def chat(request: ChatRequest):

    return StreamingResponse(

        generate_response(request.question),

        media_type="text/plain"
    )