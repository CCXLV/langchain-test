from dotenv import load_dotenv
from typing import TypedDict, List

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import START, StateGraph

from qdrant_client import QdrantClient
from qdrant_client.http import models


load_dotenv()

llm = init_chat_model("gemini-2.0-flash-lite", model_provider="google_genai")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

client = QdrantClient(url="http://localhost:6333")

# Create collection if it doesn't exist
try:
    client.get_collection("test")
except Exception:
    client.create_collection(
        collection_name="test",
        vectors_config=models.VectorParams(
            size=768, distance=models.Distance.COSINE  # Size of the embedding vectors
        ),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name="test",
    embedding=embeddings,
)

with open("data/data.txt", "r") as file:
    text = file.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = splitter.split_text(text)

# Index chunks
_ = vector_store.add_texts(texts=all_splits)

# Define prompt for question-answering
prompt = """
You are a helpful assistant that can answer questions about the text provided.

Text:
{context}

Question:
{question}
"""


# Define state for application
class State(TypedDict):
    question: str
    context: List[str]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    response = llm.invoke(
        prompt.format(context=state["context"], question=state["question"])
    )
    return {"answer": response.content}


# Compile application
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Test application
question = """
Explain how the relationship between input features and output variables differs 
across all three types of machine learning algorithms, and provide a specific 
example of how this difference affects their practical applications in real-world scenarios.
"""
response = graph.invoke({"question": question})
print(response)
