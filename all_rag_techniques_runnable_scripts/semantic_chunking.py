import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path sicnce we work with notebooks
from helper_functions import *
from evaluation.evalute_rag import *

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Define file path
path = "../data/Understanding_Climate_Change.pdf"

# Read PDF to string
content = read_pdf_to_string(path)

# Breakpoint types:
# * 'interquartile': the interquartile distance is used to split chunks.


text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type='percentile',
                                breakpoint_threshold_amount=90)  # chose which embeddings and breakpoint type and threshold to use

# Split original text to semantic chunks
docs = text_splitter.create_documents([content])

# Create vector store and retriever
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
chunks_query_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Test the retriever
test_query = "What is the main cause of climate change?"
context = retrieve_context_per_question(test_query, chunks_query_retriever)
show_context(context)
