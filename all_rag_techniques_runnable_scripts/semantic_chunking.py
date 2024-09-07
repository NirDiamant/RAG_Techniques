import time
import os
import sys
import argparse
from dotenv import load_dotenv
from helper_functions import *
from langchain_experimental.text_splitter import SemanticChunker, BreakpointThresholdType
from langchain_openai.embeddings import OpenAIEmbeddings

# Add the parent directory to the path since we work with notebooks
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Load environment variables from a .env file (e.g., OpenAI API key)
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# Function to run semantic chunking and return chunking and retrieval times
class SemanticChunkingRAG:
    """
    A class to handle the Semantic Chunking RAG process for document chunking and query retrieval.
    """

    def __init__(self, path, n_retrieved=2, embeddings=None, breakpoint_type: BreakpointThresholdType = "percentile",
                 breakpoint_amount=90):
        """
        Initializes the SemanticChunkingRAG by encoding the content using a semantic chunker.

        Args:
            path (str): Path to the PDF file to encode.
            n_retrieved (int): Number of chunks to retrieve for each query (default: 2).
            embeddings: Embedding model to use.
            breakpoint_type (str): Type of semantic breakpoint threshold.
            breakpoint_amount (float): Amount for the semantic breakpoint threshold.
        """
        print("\n--- Initializing Semantic Chunking RAG ---")
        # Read PDF to string
        content = read_pdf_to_string(path)

        # Use provided embeddings or initialize OpenAI embeddings
        self.embeddings = embeddings if embeddings else OpenAIEmbeddings()

        # Initialize the semantic chunker
        self.semantic_chunker = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_amount
        )

        # Measure time for semantic chunking
        start_time = time.time()
        self.semantic_docs = self.semantic_chunker.create_documents([content])
        self.time_records = {'Chunking': time.time() - start_time}
        print(f"Semantic Chunking Time: {self.time_records['Chunking']:.2f} seconds")

        # Create a vector store and retriever from the semantic chunks
        self.semantic_vectorstore = FAISS.from_documents(self.semantic_docs, self.embeddings)
        self.semantic_retriever = self.semantic_vectorstore.as_retriever(search_kwargs={"k": n_retrieved})

    def run(self, query):
        """
        Retrieves and displays the context for the given query.

        Args:
            query (str): The query to retrieve context for.

        Returns:
            tuple: The retrieval time.
        """
        # Measure time for semantic retrieval
        start_time = time.time()
        semantic_context = retrieve_context_per_question(query, self.semantic_retriever)
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"Semantic Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")

        # Display the retrieved context
        show_context(semantic_context)
        return self.time_records


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a PDF document with semantic chunking RAG.")
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to encode.")
    parser.add_argument("--n_retrieved", type=int, default=2,
                        help="Number of chunks to retrieve for each query (default: 2).")
    parser.add_argument("--breakpoint_threshold_type", type=str,
                        choices=["percentile", "standard_deviation", "interquartile", "gradient"],
                        default="percentile",
                        help="Type of breakpoint threshold to use for chunking (default: percentile).")
    parser.add_argument("--breakpoint_threshold_amount", type=float, default=90,
                        help="Amount of the breakpoint threshold to use (default: 90).")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Size of each text chunk in simple chunking (default: 1000).")
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="Overlap between consecutive chunks in simple chunking (default: 200).")
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?",
                        help="Query to test the retriever (default: 'What is the main cause of climate change?').")
    parser.add_argument("--experiment", action="store_true",
                        help="Run the experiment to compare performance between semantic chunking and simple chunking.")

    return parser.parse_args()


# Main function to process PDF, chunk text, and test retriever
def main(args):
    # Initialize SemanticChunkingRAG
    semantic_rag = SemanticChunkingRAG(
        path=args.path,
        n_retrieved=args.n_retrieved,
        breakpoint_type=args.breakpoint_threshold_type,
        breakpoint_amount=args.breakpoint_threshold_amount
    )

    # Run a query
    semantic_rag.run(args.query)


if __name__ == '__main__':
    # Call the main function with parsed arguments
    main(parse_args())
