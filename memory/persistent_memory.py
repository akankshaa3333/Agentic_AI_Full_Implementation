from fastapi import FastAPI

from pydantic import BaseModel

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String
)

from sqlalchemy.orm import (
    declarative_base,
    sessionmaker
)

# Load env
load_dotenv()

# ---------------- DATABASE ---------------- #

DATABASE_URL = "sqlite:///memory.db"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

# ---------------- TABLE ---------------- #

class Message(Base):

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)

    role = Column(String)

    content = Column(String)

# Create tables
Base.metadata.create_all(engine)

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

# ---------------- CHAT ENDPOINT ---------------- #

@app.post("/chat")

async def chat(request: ChatRequest):

    db = SessionLocal()

    # Save user message
    user_message = Message(
        role="user",
        content=request.question
    )

    db.add(user_message)

    db.commit()

    # Load conversation history
    messages = db.query(Message).all()

    conversation = ""

    for msg in messages:

        conversation += f"{msg.role}: {msg.content}\n"

    # Generate AI response
    prompt = f"""
    Conversation History:
    {conversation}

    Reply to latest user message.
    """

    response = await llm.ainvoke(prompt)

    # Save AI response
    ai_message = Message(
        role="ai",
        content=response.content
    )

    db.add(ai_message)

    db.commit()

    db.close()

    return {
        "answer": response.content
    }