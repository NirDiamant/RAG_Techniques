import os
import sys
import argparse
from dotenv import load_dotenv

# Add the parent directory to the path since we work with notebooks
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Define the HyDe retriever class - creating vector store, generating hypothetical document, and retrieving
class HyDERetriever:
    def __init__(self, files_path, chunk_size=500, chunk_overlap=100):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
        self.embeddings = OpenAIEmbeddings()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = encode_pdf(files_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        self.hyde_prompt = PromptTemplate(
            input_variables=["query", "chunk_size"],
            template="""Given the question '{query}', generate a hypothetical document that directly answers this question. The document should be detailed and in-depth.
            The document size has to be exactly {chunk_size} characters.""",
        )
        self.hyde_chain = self.hyde_prompt | self.llm

    def generate_hypothetical_document(self, query):
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        return self.hyde_chain.invoke(input_variables).content

    def retrieve(self, query, k=3):
        hypothetical_doc = self.generate_hypothetical_document(query)
        similar_docs = self.vectorstore.similarity_search(hypothetical_doc, k=k)
        return similar_docs, hypothetical_doc


# Main class for running the retrieval process
class ClimateChangeRAG:
    def __init__(self, path, query):
        self.retriever = HyDERetriever(path)
        self.query = query

    def run(self):
        # Retrieve results and hypothetical document
        results, hypothetical_doc = self.retriever.retrieve(self.query)

        # Plot the hypothetical document and the retrieved documents
        docs_content = [doc.page_content for doc in results]

        print("Hypothetical document:\n")
        print(text_wrap(hypothetical_doc) + "\n")
        show_context(docs_content)


# Argument parsing function
def parse_args():
    parser = argparse.ArgumentParser(description="Run the Climate Change RAG method.")
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to process.")
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?",
                        help="Query to test the retriever (default: 'What is the main topic of the document?').")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Create and run the RAG method instance
    rag_runner = ClimateChangeRAG(args.path, args.query)
    rag_runner.run()
