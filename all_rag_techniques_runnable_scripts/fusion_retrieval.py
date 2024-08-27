import os
import sys
from dotenv import load_dotenv
from langchain.docstore.document import Document

from typing import List
from rank_bm25 import BM25Okapi
import numpy as np

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path sicnce we work with notebooks
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Define document path
path = "../data/Understanding_Climate_Change.pdf"


# Encode the pdf to vector store and return split document from the step before to create BM25 instance
def encode_pdf_and_get_split_documents(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """

    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore, cleaned_texts


# Create vectorstore and get the chunked documents
vectorstore, cleaned_texts = encode_pdf_and_get_split_documents(path)


# Create a bm25 index for retrieving documents by keywords
def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    """
    Create a BM25 index from the given documents.

    BM25 (Best Matching 25) is a ranking function used in information retrieval.
    It's based on the probabilistic retrieval framework and is an improvement over TF-IDF.

    Args:
    documents (List[Document]): List of documents to index.

    Returns:
    BM25Okapi: An index that can be used for BM25 scoring.
    """
    # Tokenize each document by splitting on whitespace
    # This is a simple approach and could be improved with more sophisticated tokenization
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)


bm25 = create_bm25_index(cleaned_texts)  # Create BM25 index from the cleaned texts (chunks)


# Define a function that retrieves both semantically and by keyword, normalizes the scores and gets the top k documents
def fusion_retrieval(vectorstore, bm25, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
    """
    Perform fusion retrieval combining keyword-based (BM25) and vector-based search.

    Args:
    vectorstore (VectorStore): The vectorstore containing the documents.
    bm25 (BM25Okapi): Pre-computed BM25 index.
    query (str): The query string.
    k (int): The number of documents to retrieve.
    alpha (float): The weight for vector search scores (1-alpha will be the weight for BM25 scores).

    Returns:
    List[Document]: The top k documents based on the combined scores.
    """
    # Step 1: Get all documents from the vectorstore
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)

    # Step 2: Perform BM25 search
    bm25_scores = bm25.get_scores(query.split())

    # Step 3: Perform vector search
    vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))

    # Step 4: Normalize scores
    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))

    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

    # Step 5: Combine scores
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores

    # Step 6: Rank documents
    sorted_indices = np.argsort(combined_scores)[::-1]

    # Step 7: Return top k documents
    return [all_docs[i] for i in sorted_indices[:k]]


# Use Case example
# Query
query = "What are the impacts of climate change on the environment?"

# Perform fusion retrieval
top_docs = fusion_retrieval(vectorstore, bm25, query, k=5, alpha=0.5)
docs_content = [doc.page_content for doc in top_docs]
show_context(docs_content)
