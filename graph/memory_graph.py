from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage
)

# Load environment variables
load_dotenv()

# ---------------- LLM ---------------- #

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7
)

# ---------------- STATE ---------------- #

class ChatState(TypedDict):

    messages: List[BaseMessage]

# ---------------- CHAT NODE ---------------- #

def chatbot_node(state: ChatState):

    print("\n--- CHATBOT NODE RUNNING ---\n")

    messages = state["messages"]

    # Send full conversation history
    response = llm.invoke(messages)

    # Add AI response to memory
    messages.append(
        AIMessage(content=response.content)
    )

    return {
        "messages": messages
    }

# ---------------- GRAPH ---------------- #

graph = StateGraph(ChatState)

# Add node
graph.add_node("chatbot", chatbot_node)

# Add edges
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# Compile graph
app = graph.compile()

# ---------------- MEMORY ---------------- #

conversation_history = []

# ---------------- CHAT LOOP ---------------- #

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Add user message
    conversation_history.append(
        HumanMessage(content=question)
    )

    # Run graph
    result = app.invoke({
        "messages": conversation_history
    })

    # Update memory
    conversation_history = result["messages"]

    # Print latest AI message
    print("\nAI:")

    print(conversation_history[-1].content)