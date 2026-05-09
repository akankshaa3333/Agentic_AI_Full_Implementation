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

class GraphState(TypedDict):

    question: str
    answer: str

# ---------------- ROUTER FUNCTION ---------------- #

def router(state: GraphState):

    question = state["question"]

    # Simple math detection
    math_symbols = ["+", "-", "*", "/", "%"]

    if any(symbol in question for symbol in math_symbols):

        return "calculator"

    else:

        return "researcher"

# ---------------- CALCULATOR NODE ---------------- #

def calculator_node(state: GraphState):

    print("\n--- CALCULATOR NODE RUNNING ---\n")

    question = state["question"]

    try:

        result = eval(question)

        return {
            "answer": f"Calculation Result: {result}"
        }

    except Exception as e:

        return {
            "answer": f"Calculation Error: {str(e)}"
        }

# ---------------- RESEARCH NODE ---------------- #

def researcher_node(state: GraphState):

    print("\n--- RESEARCH NODE RUNNING ---\n")

    question = state["question"]

    prompt = f"""
    Answer this question in simple detail:

    {question}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }

# ---------------- GRAPH ---------------- #

graph = StateGraph(GraphState)

# Add nodes
graph.add_node("calculator", calculator_node)
graph.add_node("researcher", researcher_node)

# Conditional routing
graph.add_conditional_edges(
    START,
    router
)

# Connect nodes to END
graph.add_edge("calculator", END)
graph.add_edge("researcher", END)

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