import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path sicnce we work with notebooks
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Define document(s) path
path = "../data/Understanding_Climate_Change.pdf"


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
            the document size has be exactly {chunk_size} characters.""",
        )
        self.hyde_chain = self.hyde_prompt | self.llm

    def generate_hypothetical_document(self, query):
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        return self.hyde_chain.invoke(input_variables).content

    def retrieve(self, query, k=3):
        hypothetical_doc = self.generate_hypothetical_document(query)
        similar_docs = self.vectorstore.similarity_search(hypothetical_doc, k=k)
        return similar_docs, hypothetical_doc


# Create a HyDe retriever instance
retriever = HyDERetriever(path)

# Demonstrate on a use case
test_query = "What is the main cause of climate change?"
results, hypothetical_doc = retriever.retrieve(test_query)

# Plot the hypothetical document and the retrieved documents
docs_content = [doc.page_content for doc in results]

print("hypothetical_doc:\n")
print(text_wrap(hypothetical_doc) + "\n")
show_context(docs_content)
