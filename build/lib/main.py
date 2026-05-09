from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Create LLM object
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)
@tool
def calculator(expression: str) -> str:
    """
    Useful for solving mathematical expressions.
    """
    return str(eval(expression))   

#Bind tools To LLM
llm_with_tools = llm.bind_tools([calculator])

# Infinite chat loop
while True:

    # User input
    question = input("\nAsk Anything: ")

    # Exit condition
    if question.lower() == "exit":
        print("Goodbye!")
        break


        human_prompt = HumanMessage(content=question)

    # Send prompt to LLM
    response = llm_with_tools.invoke([human_prompt])

    # Print AI response
    print("\nAI Response:\n")
    print(response.content)