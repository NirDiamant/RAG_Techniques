import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path sicnce we work with notebooks
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Define files path
path = "../data/Understanding_Climate_Change.pdf"

# Create a vector store
vectorstore = encode_pdf(path)

# Initialize the language model

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)


# Defining prompt templates
class RetrievalResponse(BaseModel):
    response: str = Field(..., title="Determines if retrieval is necessary", description="Output only 'Yes' or 'No'.")


retrieval_prompt = PromptTemplate(
    input_variables=["query"],
    template="Given the query '{query}', determine if retrieval is necessary. Output only 'Yes' or 'No'."
)


class RelevanceResponse(BaseModel):
    response: str = Field(..., title="Determines if context is relevant",
                          description="Output only 'Relevant' or 'Irrelevant'.")


relevance_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', determine if the context is relevant. Output only 'Relevant' or 'Irrelevant'."
)


class GenerationResponse(BaseModel):
    response: str = Field(..., title="Generated response", description="The generated response.")


generation_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', generate a response."
)


class SupportResponse(BaseModel):
    response: str = Field(..., title="Determines if response is supported",
                          description="Output 'Fully supported', 'Partially supported', or 'No support'.")


support_prompt = PromptTemplate(
    input_variables=["response", "context"],
    template="Given the response '{response}' and the context '{context}', determine if the response is supported by the context. Output 'Fully supported', 'Partially supported', or 'No support'."
)


class UtilityResponse(BaseModel):
    response: int = Field(..., title="Utility rating", description="Rate the utility of the response from 1 to 5.")


utility_prompt = PromptTemplate(
    input_variables=["query", "response"],
    template="Given the query '{query}' and the response '{response}', rate the utility of the response from 1 to 5."
)

# Create LLMChains for each step
retrieval_chain = retrieval_prompt | llm.with_structured_output(RetrievalResponse)
relevance_chain = relevance_prompt | llm.with_structured_output(RelevanceResponse)
generation_chain = generation_prompt | llm.with_structured_output(GenerationResponse)
support_chain = support_prompt | llm.with_structured_output(SupportResponse)
utility_chain = utility_prompt | llm.with_structured_output(UtilityResponse)


# Defining the self RAG logic flow
def self_rag(query, vectorstore, top_k=3):
    print(f"\nProcessing query: {query}")

    # Step 1: Determine if retrieval is necessary
    print("Step 1: Determining if retrieval is necessary...")
    input_data = {"query": query}
    retrieval_decision = retrieval_chain.invoke(input_data).response.strip().lower()
    print(f"Retrieval decision: {retrieval_decision}")

    if retrieval_decision == 'yes':
        # Step 2: Retrieve relevant documents
        print("Step 2: Retrieving relevant documents...")
        docs = vectorstore.similarity_search(query, k=top_k)
        contexts = [doc.page_content for doc in docs]
        print(f"Retrieved {len(contexts)} documents")

        # Step 3: Evaluate relevance of retrieved documents
        print("Step 3: Evaluating relevance of retrieved documents...")
        relevant_contexts = []
        for i, context in enumerate(contexts):
            input_data = {"query": query, "context": context}
            relevance = relevance_chain.invoke(input_data).response.strip().lower()
            print(f"Document {i + 1} relevance: {relevance}")
            if relevance == 'relevant':
                relevant_contexts.append(context)

        print(f"Number of relevant contexts: {len(relevant_contexts)}")

        # If no relevant contexts found, generate without retrieval
        if not relevant_contexts:
            print("No relevant contexts found. Generating without retrieval...")
            input_data = {"query": query, "context": "No relevant context found."}
            return generation_chain.invoke(input_data).response

        # Step 4: Generate response using relevant contexts
        print("Step 4: Generating responses using relevant contexts...")
        responses = []
        for i, context in enumerate(relevant_contexts):
            print(f"Generating response for context {i + 1}...")
            input_data = {"query": query, "context": context}
            response = generation_chain.invoke(input_data).response

            # Step 5: Assess support
            print(f"Step 5: Assessing support for response {i + 1}...")
            input_data = {"response": response, "context": context}
            support = support_chain.invoke(input_data).response.strip().lower()
            print(f"Support assessment: {support}")

            # Step 6: Evaluate utility
            print(f"Step 6: Evaluating utility for response {i + 1}...")
            input_data = {"query": query, "response": response}
            utility = int(utility_chain.invoke(input_data).response)
            print(f"Utility score: {utility}")

            responses.append((response, support, utility))

        # Select the best response based on support and utility
        print("Selecting the best response...")
        best_response = max(responses, key=lambda x: (x[1] == 'fully supported', x[2]))
        print(f"Best response support: {best_response[1]}, utility: {best_response[2]}")
        return best_response[0]
    else:
        # Generate without retrieval
        print("Generating without retrieval...")
        input_data = {"query": query, "context": "No retrieval necessary."}
        return generation_chain.invoke(input_data).response


# Test the self-RAG function easy query with high relevance

query = "What is the impact of climate change on the environment?"
response = self_rag(query, vectorstore)

print("\nFinal response:")
print(response)

# Test the self-RAG function with a more challenging query with low relevance

query = "how did harry beat quirrell?"
response = self_rag(query, vectorstore)

print("\nFinal response:")
print(response)
