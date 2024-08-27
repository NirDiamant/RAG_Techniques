import os
import sys
from dotenv import load_dotenv
from langchain.docstore.document import Document
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path sicnce we work with notebooks
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Define the document's path
path = "../data/Understanding_Climate_Change.pdf"

# Create a vector store
vectorstore = encode_pdf(path)


# ## Method 1: LLM based function to rerank the retrieved documents

# Create a custom reranking function

class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of a document to a query.")


def rerank_documents(query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    prompt_template = PromptTemplate(
        input_variables=["query", "doc"],
        template="""On a scale of 1-10, rate the relevance of the following document to the query. Consider the specific context and intent of the query, not just keyword matches.
        Query: {query}
        Document: {doc}
        Relevance Score:"""
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
    llm_chain = prompt_template | llm.with_structured_output(RatingScore)

    scored_docs = []
    for doc in docs:
        input_data = {"query": query, "doc": doc.page_content}
        score = llm_chain.invoke(input_data).relevance_score
        try:
            score = float(score)
        except ValueError:
            score = 0  # Default score if parsing fails
        scored_docs.append((doc, score))

    reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_docs[:top_n]]


# Example usage of the reranking function with a sample query relevant to the document

query = "What are the impacts of climate change on biodiversity?"
initial_docs = vectorstore.similarity_search(query, k=15)
reranked_docs = rerank_documents(query, initial_docs)

# print first 3 initial documents
print("Top initial documents:")
for i, doc in enumerate(initial_docs[:3]):
    print(f"\nDocument {i + 1}:")
    print(doc.page_content[:200] + "...")  # Print first 200 characters of each document

# Print results
print(f"Query: {query}\n")
print("Top reranked documents:")
for i, doc in enumerate(reranked_docs):
    print(f"\nDocument {i + 1}:")
    print(doc.page_content[:200] + "...")  # Print first 200 characters of each document


# Create a custom retriever based on our reranker
# Create a custom retriever class
class CustomRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str, num_docs=2) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query, k=30)
        return rerank_documents(query, initial_docs, top_n=num_docs)


# Create the custom retriever
custom_retriever = CustomRetriever(vectorstore=vectorstore)

# Create an LLM for answering questions
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# Create the RetrievalQA chain with the custom retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=custom_retriever,
    return_source_documents=True
)

# Example query

result = qa_chain({"query": query})

print(f"\nQuestion: {query}")
print(f"Answer: {result['result']}")
print("\nRelevant source documents:")
for i, doc in enumerate(result["source_documents"]):
    print(f"\nDocument {i + 1}:")
    print(doc.page_content[:200] + "...")  # Print first 200 characters of each document

# Example that demonstrates why we should use reranking
chunks = [
    "The capital of France is great.",
    "The capital of France is huge.",
    "The capital of France is beautiful.",
    """Have you ever visited Paris? It is a beautiful city where you can eat delicious food and see the Eiffel Tower. 
    I really enjoyed all the cities in france, but its capital with the Eiffel Tower is my favorite city.""",
    "I really enjoyed my trip to Paris, France. The city is beautiful and the food is delicious. I would love to visit again. Such a great capital city."
]
docs = [Document(page_content=sentence) for sentence in chunks]


def compare_rag_techniques(query: str, docs: List[Document] = docs) -> None:
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    print("Comparison of Retrieval Techniques")
    print("==================================")
    print(f"Query: {query}\n")

    print("Baseline Retrieval Result:")
    baseline_docs = vectorstore.similarity_search(query, k=2)
    for i, doc in enumerate(baseline_docs):
        print(f"\nDocument {i + 1}:")
        print(doc.page_content)

    print("\nAdvanced Retrieval Result:")
    custom_retriever = CustomRetriever(vectorstore=vectorstore)
    advanced_docs = custom_retriever.get_relevant_documents(query)
    for i, doc in enumerate(advanced_docs):
        print(f"\nDocument {i + 1}:")
        print(doc.page_content)


query = "what is the capital of france?"
compare_rag_techniques(query, docs)

# ## Method 2: Cross Encoder models

# <div style="text-align: center;">
# 
# <img src="../images/rerank_cross_encoder.svg" alt="rerank cross encoder" style="width:40%; height:auto;">
# </div>

# Define the cross encoder class
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


class CrossEncoderRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")
    cross_encoder: Any = Field(description="Cross-encoder model for reranking")
    k: int = Field(default=5, description="Number of documents to retrieve initially")
    rerank_top_k: int = Field(default=3, description="Number of documents to return after reranking")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Initial retrieval
        initial_docs = self.vectorstore.similarity_search(query, k=self.k)

        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in initial_docs]

        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)

        # Sort documents by score
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)

        # Return top reranked documents
        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")


# Create an instance and showcase over an example
# Create the cross-encoder retriever
cross_encoder_retriever = CrossEncoderRetriever(
    vectorstore=vectorstore,
    cross_encoder=cross_encoder,
    k=10,  # Retrieve 10 documents initially
    rerank_top_k=5  # Return top 5 after reranking
)

# Set up the LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# Create the RetrievalQA chain with the cross-encoder retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=cross_encoder_retriever,
    return_source_documents=True
)

# Example query
query = "What are the impacts of climate change on biodiversity?"
result = qa_chain({"query": query})

print(f"\nQuestion: {query}")
print(f"Answer: {result['result']}")
print("\nRelevant source documents:")
for i, doc in enumerate(result["source_documents"]):
    print(f"\nDocument {i + 1}:")
    print(doc.page_content[:200] + "...")  # Print first 200 characters of each document
