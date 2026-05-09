from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Create LLM
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0
)

# ---------------- STATE ---------------- #

class ChatState(TypedDict):

    question: str
    answer: str

# ---------------- NODE ---------------- #

def chatbot_node(state: ChatState):

    print("\n--- CHATBOT NODE RUNNING ---\n")

    question = state["question"]

    response = llm.invoke(question)

    return {
        "answer": response.content
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

# ---------------- RUN GRAPH ---------------- #

while True:

    question = input("\nAsk Anything: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    result = app.invoke({
        "question": question
    })

    print("\nAI Response:\n")

    print(result["answer"])