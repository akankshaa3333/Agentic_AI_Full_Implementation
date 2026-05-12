from typing import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

from langchain_groq import ChatGroq

# Load env
load_dotenv()

# ---------------- LLM ---------------- #

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0
)

# ---------------- STATE ---------------- #

class AgentState(TypedDict):

    question: str

    route: str

    answer: str

# ---------------- SUPERVISOR ---------------- #

def supervisor_agent(state: AgentState):

    print("\n--- SUPERVISOR RUNNING ---\n")

    question = state["question"].lower()

    # Dynamic routing logic

    if "code" in question or "python" in question:

        return {
            "route": "coder"
        }

    elif "research" in question or "ai" in question:

        return {
            "route": "researcher"
        }

    else:

        return {
            "route": "general"
        }

# ---------------- RESEARCH AGENT ---------------- #

def researcher_agent(state: AgentState):

    print("\n--- RESEARCH AGENT RUNNING ---\n")

    question = state["question"]

    prompt = f"""
    You are an AI researcher.

    Answer deeply:

    {question}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }

# ---------------- CODER AGENT ---------------- #

def coder_agent(state: AgentState):

    print("\n--- CODER AGENT RUNNING ---\n")

    question = state["question"]

    prompt = f"""
    You are an expert Python developer.

    Solve this coding task:

    {question}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }

# ---------------- GENERAL AGENT ---------------- #

def general_agent(state: AgentState):

    print("\n--- GENERAL AGENT RUNNING ---\n")

    question = state["question"]

    response = llm.invoke(question)

    return {
        "answer": response.content
    }

# ---------------- ROUTER FUNCTION ---------------- #

def route_decision(state: AgentState):

    return state["route"]

# ---------------- GRAPH ---------------- #

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("supervisor", supervisor_agent)

graph.add_node("researcher", researcher_agent)

graph.add_node("coder", coder_agent)

graph.add_node("general", general_agent)

# Start edge
graph.add_edge(START, "supervisor")

# Conditional routing
graph.add_conditional_edges(

    "supervisor",

    route_decision,

    {
        "researcher": "researcher",

        "coder": "coder",

        "general": "general"
    }
)

# End edges
graph.add_edge("researcher", END)

graph.add_edge("coder", END)

graph.add_edge("general", END)

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