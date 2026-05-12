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
    temperature=0.7
)

# ---------------- SEARCH TOOL ---------------- #

search_tool = DuckDuckGoSearchRun()

# ---------------- STATE ---------------- #

class AgentState(TypedDict):

    question: str

    plan: str

    web_search: str

    research: str

    final_answer: str

# ---------------- PLANNER AGENT ---------------- #

def planner_agent(state: AgentState):

    print("\n--- PLANNER AGENT RUNNING ---\n")

    question = state["question"]

    prompt = f"""
    You are an expert research planner.

    Create a research strategy for:

    {question}

    Mention:
    - important topics
    - search direction
    - research goals
    """

    response = llm.invoke(prompt)

    return {
        "plan": response.content
    }

# ---------------- WEB SEARCH AGENT ---------------- #

def web_search_agent(state: AgentState):

    print("\n--- WEB SEARCH AGENT RUNNING ---\n")

    question = state["question"]

    search_result = search_tool.invoke(question)

    return {
        "web_search": search_result
    }

# ---------------- RESEARCH AGENT ---------------- #

def research_agent(state: AgentState):

    print("\n--- RESEARCH AGENT RUNNING ---\n")

    question = state["question"]

    plan = state["plan"]

    web_search = state["web_search"]

    prompt = f"""
    You are a professional AI researcher.

    User Question:
    {question}

    Research Plan:
    {plan}

    Web Search Results:
    {web_search}

    Create a detailed research answer.
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
    You are a critical reviewer.

    Improve the following answer:
    - improve clarity
    - improve structure
    - ensure correctness
    - make answer polished

    Research:
    {research}
    """

    response = llm.invoke(prompt)

    return {
        "final_answer": response.content
    }

# ---------------- GRAPH ---------------- #

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("planner", planner_agent)

graph.add_node("web_search", web_search_agent)

graph.add_node("researcher", research_agent)

graph.add_node("reflector", reflection_agent)

# Add edges
graph.add_edge(START, "planner")

graph.add_edge("planner", "web_search")

graph.add_edge("web_search", "researcher")

graph.add_edge("researcher", "reflector")

graph.add_edge("reflector", END)

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
    print("FINAL AGENT ANSWER")
    print("============================\n")

    print(result["final_answer"])