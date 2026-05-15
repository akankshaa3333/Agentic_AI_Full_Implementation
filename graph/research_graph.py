from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Create LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ---------------- STATE ---------------- #

class ResearchState(TypedDict):

    question: str
    research: str
    summary: str

# ---------------- RESEARCH NODE ---------------- #

def research_node(state: ResearchState):

    print("\n--- RESEARCH NODE RUNNING ---\n")

    question = state["question"]

    prompt = f"""
    Do detailed research on the following topic:

    {question}

    Give detailed information.
    """

    response = llm.invoke(prompt)

    return {
        "research": response.content
    }

# ---------------- SUMMARIZER NODE ---------------- #

def summarizer_node(state: ResearchState):

    print("\n--- SUMMARIZER NODE RUNNING ---\n")

    research_content = state["research"]

    prompt = f"""
    Summarize the following research in simple points:

    {research_content}
    """

    response = llm.invoke(prompt)

    return {
        "summary": response.content
    }

# ---------------- GRAPH ---------------- #

graph = StateGraph(ResearchState)

# Add nodes
graph.add_node("researcher", research_node)
graph.add_node("summarizer", summarizer_node)

# Add edges
graph.add_edge(START, "researcher")
graph.add_edge("researcher", "summarizer")
graph.add_edge("summarizer", END)

# Compile graph
app = graph.compile()

# ---------------- RUN ---------------- #

while True:

    question = input("\nAsk Research Question: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    result = app.invoke({
        "question": question
    })

    print("\n============================")
    print("FINAL SUMMARY")
    print("============================\n")

    print(result["summary"])