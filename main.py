from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()

# Create LLM
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0
)

# ---------------- CALCULATOR TOOL ---------------- #

@tool
def calculator(expression: str) -> str:
    """
    Useful for solving mathematical calculations.
    Input should be a valid mathematical expression.
    """

    try:
        result = eval(expression)
        return str(result)

    except Exception as e:
        return f"Error: {str(e)}"


# ---------------- SEARCH TOOL ---------------- #

search_tool = DuckDuckGoSearchRun()

# ---------------- TOOLS LIST ---------------- #

tools = {
    "calculator": calculator,
    "duckduckgo_search": search_tool
}

# Bind tools to LLM
llm_with_tools = llm.bind_tools(
    [calculator, search_tool]
)

# ---------------- CHAT LOOP ---------------- #

while True:

    question = input("\nAsk Anything: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Send question to AI
    response = llm_with_tools.invoke([
        HumanMessage(content=question)
    ])

    # Print raw response
    print("\n============================")
    print("RAW AI RESPONSE")
    print("============================")
    print(response)

    # Check tool calls
    if response.tool_calls:

        print("\n============================")
        print("TOOL CALL DETECTED")
        print("============================")

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"\nTool Name: {tool_name}")
            print(f"Tool Args: {tool_args}")

            try:

                # Calculator Tool
                if tool_name == "calculator":

                    result = calculator.invoke(tool_args)

                # Search Tool
                elif tool_name == "duckduckgo_search":

                    query = tool_args.get("query", "")
                    result = search_tool.invoke(query)

                else:
                    result = "Unknown Tool"

                print("\n============================")
                print("TOOL RESULT")
                print("============================")
                print(result)

            except Exception as e:

                print("\nTool Execution Error:")
                print(str(e))

    else:

        print("\n============================")
        print("FINAL ANSWER")
        print("============================")
        print(response.content)