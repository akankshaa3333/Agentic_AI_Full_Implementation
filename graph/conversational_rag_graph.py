from typing import TypedDict, List

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage
)

# Load environment variables
load_dotenv()

# ---------------- LOAD PDF ---------------- #

loader = PyPDFLoader("documents/Akanksha_Resumee.pdf")

documents = loader.load()

# ---------------- EXTRACT TEXT ---------------- #

full_text = ""

for doc in documents:

    full_text += doc.page_content + "\n"

# ---------------- CHUNKING ---------------- #

chunk_size = 1000

chunks = []

for i in range(0, len(full_text), chunk_size):

    chunk = full_text[i:i + chunk_size]

    chunks.append(chunk)

print(f"\nTotal Chunks Loaded: {len(chunks)}")

# ---------------- LLM ---------------- #

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7
)

# ---------------- STATE ---------------- #

class GraphState(TypedDict):

    messages: List[BaseMessage]

    context: str

# ---------------- RETRIEVER NODE ---------------- #

def retriever_node(state: GraphState):

    print("\n--- RETRIEVER NODE RUNNING ---\n")

    # Fake retrieval
    relevant_chunks = chunks[:3]

    context = "\n\n".join(relevant_chunks)

    return {
        "context": context
    }

# ---------------- CHATBOT NODE ---------------- #

def chatbot_node(state: GraphState):

    print("\n--- CHATBOT NODE RUNNING ---\n")

    messages = state["messages"]

    context = state["context"]

    # Latest user question
    latest_question = messages[-1].content

    prompt = f"""
    Answer the user's question using the PDF context.

    PDF Context:
    {context}

    Conversation History:
    {messages}

    Latest User Question:
    {latest_question}
    """

    response = llm.invoke(prompt)

    # Store AI response in memory
    messages.append(
        AIMessage(content=response.content)
    )

    return {
        "messages": messages
    }

# ---------------- GRAPH ---------------- #

graph = StateGraph(GraphState)

# Add nodes
graph.add_node("retriever", retriever_node)

graph.add_node("chatbot", chatbot_node)

# Add edges
graph.add_edge(START, "retriever")

graph.add_edge("retriever", "chatbot")

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

    # Store user message
    conversation_history.append(
        HumanMessage(content=question)
    )

    # Run graph
    result = app.invoke({

        "messages": conversation_history

    })

    # Update memory
    conversation_history = result["messages"]

    # Print latest AI response
    print("\nAI:\n")

    print(conversation_history[-1].content)