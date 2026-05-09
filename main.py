from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# Load environment variables
load_dotenv()

# Create LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Create calculator tool
@tool
def calculator(expression: str) -> str:
    """
    Useful for solving mathematical expressions.
    """

    return str(eval(expression))

# Bind tools
llm_with_tools = llm.bind_tools([calculator])

# Chat loop
while True:

    question = input("\nAsk Anything: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Step 1: Send question to LLM
    response = llm_with_tools.invoke([
        HumanMessage(content=question)
    ])

    # Step 2: Check if tool call exists
    if response.tool_calls:

        # Get first tool call
        tool_call = response.tool_calls[0]

        # Tool name
        tool_name = tool_call["name"]

        # Tool arguments
        tool_args = tool_call["args"]

        print("\nTool Selected:", tool_name)
        print("Tool Args:", tool_args)

        # Execute calculator tool
        result = calculator.invoke(tool_args)

        print("Tool Result:", result)

        # Step 3: Send tool result back to LLM
        final_response = llm_with_tools.invoke([
            HumanMessage(content=question),
            response,
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
        ])

        print("\nFinal Answer:\n")
        print(final_response.content)

    else:
        print("\nAI Response:\n")
        print(response.content)