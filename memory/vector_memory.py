from chromadb import Client
from chromadb.config import Settings

# ---------------- CHROMA CLIENT ---------------- #

client = Client(Settings(
    persist_directory="chroma_db"
))

# ---------------- COLLECTION ---------------- #

collection = client.get_or_create_collection(
    name="ai_knowledge"
)

# ---------------- STORE DOCUMENTS ---------------- #

documents = [

    "Agentic AI uses planning and reasoning.",

    "LangGraph is used for AI workflows.",

    "RAG combines retrieval and generation.",

    "FastAPI is used for AI backends."
]

# Add docs
collection.add(

    documents=documents,

    ids=[
        "1",
        "2",
        "3",
        "4"
    ]
)

print("\nDocuments stored successfully!\n")

# ---------------- SEARCH ---------------- #

while True:

    query = input("\nSearch Query: ")

    if query.lower() == "exit":

        break

    results = collection.query(

        query_texts=[query],

        n_results=2
    )

    print("\nRESULTS:\n")

    print(results["documents"])