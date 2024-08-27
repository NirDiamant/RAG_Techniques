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


# Define the explainable retriever class 
class ExplainableRetriever:
    def __init__(self, texts):
        self.embeddings = OpenAIEmbeddings()

        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)

        # Create a base retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        # Create an explanation chain
        explain_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            Analyze the relationship between the following query and the retrieved context.
            Explain why this context is relevant to the query and how it might help answer the query.
            
            Query: {query}
            
            Context: {context}
            
            Explanation:
            """
        )
        self.explain_chain = explain_prompt | self.llm

    def retrieve_and_explain(self, query):
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(query)

        explained_results = []

        for doc in docs:
            # Generate explanation
            input_data = {"query": query, "context": doc.page_content}
            explanation = self.explain_chain.invoke(input_data).content

            explained_results.append({
                "content": doc.page_content,
                "explanation": explanation
            })

        return explained_results


# Create a mock example and explainable retriever instance
# Usage
texts = [
    "The sky is blue because of the way sunlight interacts with the atmosphere.",
    "Photosynthesis is the process by which plants use sunlight to produce energy.",
    "Global warming is caused by the increase of greenhouse gases in Earth's atmosphere."
]

explainable_retriever = ExplainableRetriever(texts)

# Show the results
query = "Why is the sky blue?"
results = explainable_retriever.retrieve_and_explain(query)

for i, result in enumerate(results, 1):
    print(f"Result {i}:")
    print(f"Content: {result['content']}")
    print(f"Explanation: {result['explanation']}")
    print()
