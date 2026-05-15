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

    plan: str

    research: str

    reflection: str

    final_answer: str

# ---------------- PLANNER AGENT ---------------- #

def planner_agent(state: AgentState):

    print("\n--- PLANNER AGENT RUNNING ---\n")

    question = state["question"]

    prompt = f"""
    You are an expert research planner.

    Create a step-by-step research strategy for:

    {question}

    Include:
    - what to research
    - important concepts
    - investigation approach
    """

    response = llm.invoke(prompt)

    return {
        "plan": response.content
    }

# ---------------- RESEARCH AGENT ---------------- #

def researcher_agent(state: AgentState):

    print("\n--- RESEARCH AGENT RUNNING ---\n")

    question = state["question"]

    plan = state["plan"]

    prompt = f"""
    You are a professional researcher.

    Research the topic deeply.

    Question:
    {question}

    Research Plan:
    {plan}

    Give detailed explanation.
    """

    response = llm.invoke(prompt)

    return {
        "research": response.content
    }

# ---------------- REFLECTION AGENT ---------------- #

def reflection_agent(state: AgentState):

    print("\n--- REFLECTION AGENT RUNNING ---\n")

    research = state["research"]

    prompt = f"""
    You are a critical AI reviewer.

    Analyze the following research answer.

    Improve:
    - clarity
    - accuracy
    - structure
    - completeness

    Research:
    {research}

    Return improved final answer.
    """

    response = llm.invoke(prompt)

    return {
        "final_answer": response.content
    }

# ---------------- GRAPH ---------------- #

graph = StateGraph(AgentState)

# Add agents
graph.add_node("planner", planner_agent)

graph.add_node("researcher", researcher_agent)

graph.add_node("reflector", reflection_agent)

# Workflow edges
graph.add_edge(START, "planner")

graph.add_edge("planner", "researcher")

graph.add_edge("researcher", "reflector")

graph.add_edge("reflector", END)

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
    print("FINAL AUTONOMOUS ANSWER")
    print("============================\n")

    print(result["final_answer"])