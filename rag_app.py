from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# ---------------- LOAD PDF ---------------- #

loader = PyPDFLoader("documents/Akanksha_Resumee.pdf")

documents = loader.load()

print("\nPDF Loaded Successfully!\n")

# ---------------- EXTRACT TEXT ---------------- #

full_text = ""

for doc in documents:

    full_text += doc.page_content + "\n"

# ---------------- MANUAL CHUNKING ---------------- #

chunk_size = 1000

chunks = []

for i in range(0, len(full_text), chunk_size):

    chunk = full_text[i:i + chunk_size]

    chunks.append(chunk)

print(f"Total Chunks Created: {len(chunks)}")

# ---------------- LLM ---------------- #

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0
)

# ---------------- CHAT LOOP ---------------- #

while True:

    question = input("\nAsk Question From PDF: ")

    if question.lower() == "exit":

        print("Goodbye!")

        break

    # Simple retrieval
    relevant_chunks = chunks[:3]

    context = "\n\n".join(relevant_chunks)

    # Prompt
    prompt = f"""
    Answer the question using the PDF context below.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    print("\n============================")
    print("AI ANSWER")
    print("============================\n")

    print(response.content)
