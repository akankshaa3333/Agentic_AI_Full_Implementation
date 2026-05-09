from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

# Load environment variables
load_dotenv()

# Create LLM
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7
)

# ---------------- MEMORY ---------------- #

chat_history = []

# ---------------- CHAT LOOP ---------------- #

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Add user message to memory
    chat_history.append(
        HumanMessage(content=question)
    )

    # Send full history to LLM
    response = llm.invoke(chat_history)

    # Print AI response
    print("\nAI:")
    print(response.content)

    # Store AI response in memory
    chat_history.append(
        AIMessage(content=response.content)
    )