import os
import sys
from dotenv import load_dotenv
from langchain.docstore.document import Document
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# Function to encode the PDF to a vector store and return split documents
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
    loader = PyPDFLoader(path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore, cleaned_texts


# Function to create BM25 index for keyword retrieval
def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    """
    Create a BM25 index from the given documents.

    Args:
        documents (List[Document]): List of documents to index.

    Returns:
        BM25Okapi: An index that can be used for BM25 scoring.
    """
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)


# Function for fusion retrieval combining keyword-based (BM25) and vector-based search
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
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
    bm25_scores = bm25.get_scores(query.split())
    vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))

    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
    sorted_indices = np.argsort(combined_scores)[::-1]

    return [all_docs[i] for i in sorted_indices[:k]]


class FusionRetrievalRAG:
    def __init__(self, path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initializes the FusionRetrievalRAG class by setting up the vector store and BM25 index.

        Args:
        path (str): Path to the PDF file.
        chunk_size (int): The size of each text chunk.
        chunk_overlap (int): The overlap between consecutive chunks.
        """
        self.vectorstore, self.cleaned_texts = encode_pdf_and_get_split_documents(path, chunk_size, chunk_overlap)
        self.bm25 = create_bm25_index(self.cleaned_texts)

    def run(self, query: str, k: int = 5, alpha: float = 0.5):
        """
        Executes the fusion retrieval for the given query.

        Args:
        query (str): The search query.
        k (int): The number of documents to retrieve.
        alpha (float): The weight of vector search vs. BM25 search.

        Returns:
        List[Document]: The top k retrieved documents.
        """
        top_docs = fusion_retrieval(self.vectorstore, self.bm25, query, k, alpha)
        docs_content = [doc.page_content for doc in top_docs]
        show_context(docs_content)


def parse_args():
    """
    Parses command-line arguments.

    Returns:
    args: The parsed arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Fusion Retrieval RAG Script")
    parser.add_argument('--path', type=str, default="../data/Understanding_Climate_Change.pdf",
                        help='Path to the PDF file.')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Size of each chunk.')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='Overlap between consecutive chunks.')
    parser.add_argument('--query', type=str, default='What are the impacts of climate change on the environment?',
                        help='Query to retrieve documents.')
    parser.add_argument('--k', type=int, default=5, help='Number of documents to retrieve.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for vector search vs. BM25.')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    retriever = FusionRetrievalRAG(path=args.path, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    retriever.run(query=args.query, k=args.k, alpha=args.alpha)
