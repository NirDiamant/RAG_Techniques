from helper_functions import *
from evaluation.evalute_rag import *

import os
import sys
import argparse
from dotenv import load_dotenv

# Add the parent directory to the path since we work with notebooks
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Load environment variables from a .env file (e.g., OpenAI API key)
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# Function to validate command line inputs
def validate_args(args):
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")
    if args.n_retrieved <= 0:
        raise ValueError("n_retrieved must be a positive integer.")
    return args


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Encode a PDF document and test a retriever.")
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to encode.")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Size of each text chunk (default: 1000).")
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="Overlap between consecutive chunks (default: 200).")
    parser.add_argument("--n_retrieved", type=int, default=2,
                        help="Number of chunks to retrieve for each query (default: 2).")
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?",
                        help="Query to test the retriever (default: 'What is the main cause of climate change?').")
    parser.add_argument("--evaluate", action="store_true",
                        help="Whether to evaluate the retriever's performance (default: False).")

    # Parse and validate arguments
    return validate_args(parser.parse_args())


# Main function to encode PDF, retrieve context, and optionally evaluate retriever
def main(args):
    # Encode the PDF document into a vector store using OpenAI embeddings
    chunks_vector_store = encode_pdf(args.path, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    # Create a retriever from the vector store, specifying how many chunks to retrieve
    chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": args.n_retrieved})

    # Test the retriever with the user's query
    context = retrieve_context_per_question(args.query, chunks_query_retriever)
    show_context(context)  # Display the context retrieved for the query

    # Evaluate the retriever's performance on the query (if requested)
    if args.evaluate:
        evaluate_rag(chunks_query_retriever)


if __name__ == '__main__':
    # Call the main function with parsed arguments
    main(parse_args())
