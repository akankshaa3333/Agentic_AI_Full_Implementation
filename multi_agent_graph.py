from typing import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# ---------------- LLM ---------------- #

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7
)

# ---------------- STATE ---------------- #

class AgentState(TypedDict):

    question: str

    research: str

    summary: str

    final_answer: str

# ---------------- RESEARCH AGENT ---------------- #

def researcher_agent(state: AgentState):

    print("\n--- RESEARCH AGENT RUNNING ---\n")

    question = state["question"]

    prompt = f"""
    You are a professional AI researcher.

    Research the following topic deeply:

    {question}

    Give detailed technical information.
    """

    response = llm.invoke(prompt)

    return {
        "research": response.content
    }

# ---------------- SUMMARY AGENT ---------------- #

def summarizer_agent(state: AgentState):

    print("\n--- SUMMARY AGENT RUNNING ---\n")

    research = state["research"]

    prompt = f"""
    You are an expert summarizer.

    Convert the following research into:
    - simple explanation
    - key points
    - beginner-friendly format

    Research:
    {research}
    """

    response = llm.invoke(prompt)

    return {
        "summary": response.content
    }

# ---------------- CRITIC AGENT ---------------- #

def critic_agent(state: AgentState):

    print("\n--- CRITIC AGENT RUNNING ---\n")

    summary = state["summary"]

    prompt = f"""
    You are a critical reviewer.

    Improve the following answer:
    - make it clearer
    - improve structure
    - remove confusion
    - ensure accuracy

    Answer:
    {summary}
    """

    response = llm.invoke(prompt)

    return {
        "final_answer": response.content
    }

# ---------------- GRAPH ---------------- #

graph = StateGraph(AgentState)

# Add agents
graph.add_node("researcher", researcher_agent)

graph.add_node("summarizer", summarizer_agent)

graph.add_node("critic", critic_agent)

# Add workflow edges
graph.add_edge(START, "researcher")

graph.add_edge("researcher", "summarizer")

graph.add_edge("summarizer", "critic")

graph.add_edge("critic", END)

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
    print("FINAL IMPROVED ANSWER")
    print("============================\n")

    print(result["final_answer"])