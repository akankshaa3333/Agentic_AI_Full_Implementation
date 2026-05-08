from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Create LLM object
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# Infinite chat loop
while True:

    # User input
    question = input("\nAsk Anything: ")

    # Exit condition
    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Send prompt to LLM
    response = llm.invoke(question)

    # Print AI response
    print("\nAI Response:\n")
    print(response.content)