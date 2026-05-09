from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()

# ---------------- LLM ---------------- #

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0
)

# ---------------- SEARCH TOOL ---------------- #

search_tool = DuckDuckGoSearchRun()

# ---------------- STATE ---------------- #

class AgentState(TypedDict):

    question: str
    search_result: str
    answer: str

# ---------------- ROUTER ---------------- #

def router(state: AgentState):

    question = state["question"].lower()

    search_keywords = [
        "latest",
        "news",
        "current",
        "today",
        "recent"
    ]

    if any(word in question for word in search_keywords):

        return "search"

    return "chat"

# ---------------- SEARCH NODE ---------------- #

def search_node(state: AgentState):

    print("\n--- SEARCH NODE RUNNING ---\n")

    question = state["question"]

    # Search web
    result = search_tool.invoke(question)

    return {
        "search_result": result
    }

# ---------------- CHAT NODE ---------------- #

def chat_node(state: AgentState):

    print("\n--- CHAT NODE RUNNING ---\n")

    question = state["question"]

    response = llm.invoke(question)

    return {
        "answer": response.content
    }

# ---------------- SUMMARIZER NODE ---------------- #

def summarizer_node(state: AgentState):

    print("\n--- SUMMARIZER NODE RUNNING ---\n")

    question = state["question"]

    search_result = state["search_result"]

    prompt = f"""
    User Question:
    {question}

    Search Result:
    {search_result}

    Give a clean helpful answer.
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }

# ---------------- GRAPH ---------------- #

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("search", search_node)
graph.add_node("chat", chat_node)
graph.add_node("summarizer", summarizer_node)

# Conditional routing
graph.add_conditional_edges(
    START,
    router,
    {
        "search": "search",
        "chat": "chat"
    }
)

# Search flow
graph.add_edge("search", "summarizer")

# End flows
graph.add_edge("summarizer", END)
graph.add_edge("chat", END)

# Compile graph
app = graph.compile()

# ---------------- RUN ---------------- #

while True:

    question = input("\nAsk Anything: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    result = app.invoke({
        "question": question
    })

    print("\n============================")
    print("FINAL ANSWER")
    print("============================\n")

    print(result["answer"])