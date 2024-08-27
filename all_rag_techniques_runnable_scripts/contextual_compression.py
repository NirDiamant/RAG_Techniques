import os
import sys
from dotenv import load_dotenv
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path sicnce we work with notebooks
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Define document's path
path = "../data/Understanding_Climate_Change.pdf"

# Create a vector store
vector_store = encode_pdf(path)

# Create a retriever + contexual compressor + combine them
# Create a retriever
retriever = vector_store.as_retriever()

# Create a contextual compressor
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
compressor = LLMChainExtractor.from_llm(llm)

# Combine the retriever with the compressor
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# Create a QA chain with the compressed retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    return_source_documents=True
)

# Example usage
query = "What is the main topic of the document?"
result = qa_chain.invoke({"query": query})
print(result["result"])
print("Source documents:", result["source_documents"])
