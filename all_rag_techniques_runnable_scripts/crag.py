import os
import sys
import argparse
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import DuckDuckGoSearchResults
from helper_functions import encode_pdf
import json

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path since we work with notebooks

# Load environment variables from a .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


class RetrievalEvaluatorInput(BaseModel):
    """
    Model for capturing the relevance score of a document to a query.
    """
    relevance_score: float = Field(..., description="Relevance score between 0 and 1, "
                                                    "indicating the document's relevance to the query.")


class QueryRewriterInput(BaseModel):
    """
    Model for capturing a rewritten query suitable for web search.
    """
    query: str = Field(..., description="The query rewritten for better web search results.")


class KnowledgeRefinementInput(BaseModel):
    """
    Model for extracting key points from a document.
    """
    key_points: str = Field(..., description="Key information extracted from the document in bullet-point form.")


class CRAG:
    """
    A class to handle the CRAG process for document retrieval, evaluation, and knowledge refinement.
    """

    def __init__(self, path, model="gpt-4o-mini", max_tokens=1000, temperature=0, lower_threshold=0.3,
                 upper_threshold=0.7):
        """
        Initializes the CRAG Retriever by encoding the PDF document and creating the necessary models and search tools.

        Args:
            path (str): Path to the PDF file to encode.
            model (str): The language model to use for the CRAG process.
            max_tokens (int): Maximum tokens to use in LLM responses (default: 1000).
            temperature (float): The temperature to use for LLM responses (default: 0).
            lower_threshold (float): Lower threshold for document evaluation scores (default: 0.3).
            upper_threshold (float): Upper threshold for document evaluation scores (default: 0.7).
        """
        print("\n--- Initializing CRAG Process ---")

        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

        # Encode the PDF document into a vector store
        self.vectorstore = encode_pdf(path)

        # Initialize OpenAI language model
        self.llm = ChatOpenAI(model=model, max_tokens=max_tokens, temperature=temperature)

        # Initialize search tool
        self.search = DuckDuckGoSearchResults()

    @staticmethod
    def retrieve_documents(query, faiss_index, k=3):
        docs = faiss_index.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def evaluate_documents(self, query, documents):
        return [self.retrieval_evaluator(query, doc) for doc in documents]

    def retrieval_evaluator(self, query, document):
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="On a scale from 0 to 1, how relevant is the following document to the query? "
                     "Query: {query}\nDocument: {document}\nRelevance score:"
        )
        chain = prompt | self.llm.with_structured_output(RetrievalEvaluatorInput)
        input_variables = {"query": query, "document": document}
        result = chain.invoke(input_variables).relevance_score
        return result

    def knowledge_refinement(self, document):
        prompt = PromptTemplate(
            input_variables=["document"],
            template="Extract the key information from the following document in bullet points:"
                     "\n{document}\nKey points:"
        )
        chain = prompt | self.llm.with_structured_output(KnowledgeRefinementInput)
        input_variables = {"document": document}
        result = chain.invoke(input_variables).key_points
        return [point.strip() for point in result.split('\n') if point.strip()]

    def rewrite_query(self, query):
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Rewrite the following query to make it more suitable for a web search:\n{query}\nRewritten query:"
        )
        chain = prompt | self.llm.with_structured_output(QueryRewriterInput)
        input_variables = {"query": query}
        return chain.invoke(input_variables).query.strip()

    @staticmethod
    def parse_search_results(results_string):
        try:
            results = json.loads(results_string)
            return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results]
        except json.JSONDecodeError:
            print("Error parsing search results. Returning empty list.")
            return []

    def perform_web_search(self, query):
        rewritten_query = self.rewrite_query(query)
        web_results = self.search.run(rewritten_query)
        web_knowledge = self.knowledge_refinement(web_results)
        sources = self.parse_search_results(web_results)
        return web_knowledge, sources

    def generate_response(self, query, knowledge, sources):
        response_prompt = PromptTemplate(
            input_variables=["query", "knowledge", "sources"],
            template="Based on the following knowledge, answer the query. "
                     "Include the sources with their links (if available) at the end of your answer:"
                     "\nQuery: {query}\nKnowledge: {knowledge}\nSources: {sources}\nAnswer:"
        )
        input_variables = {
            "query": query,
            "knowledge": knowledge,
            "sources": "\n".join([f"{title}: {link}" if link else title for title, link in sources])
        }
        response_chain = response_prompt | self.llm
        return response_chain.invoke(input_variables).content

    def run(self, query):
        print(f"\nProcessing query: {query}")

        # Retrieve and evaluate documents
        retrieved_docs = self.retrieve_documents(query, self.vectorstore)
        eval_scores = self.evaluate_documents(query, retrieved_docs)

        print(f"\nRetrieved {len(retrieved_docs)} documents")
        print(f"Evaluation scores: {eval_scores}")

        # Determine action based on evaluation scores
        max_score = max(eval_scores)
        sources = []

        if max_score > 0.7:
            print("\nAction: Correct - Using retrieved document")
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            final_knowledge = best_doc
            sources.append(("Retrieved document", ""))
        elif max_score < 0.3:
            print("\nAction: Incorrect - Performing web search")
            final_knowledge, sources = self.perform_web_search(query)
        else:
            print("\nAction: Ambiguous - Combining retrieved document and web search")
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            retrieved_knowledge = self.knowledge_refinement(best_doc)
            web_knowledge, web_sources = self.perform_web_search(query)
            final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
            sources = [("Retrieved document", "")] + web_sources

        print("\nFinal knowledge:")
        print(final_knowledge)

        print("\nSources:")
        for title, link in sources:
            print(f"{title}: {link}" if link else title)

        print("\nGenerating response...")
        response = self.generate_response(query, final_knowledge, sources)
        print("\nResponse generated")
        return response


# Function to validate command line inputs
def validate_args(args):
    if args.max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer.")
    if args.temperature < 0 or args.temperature > 1:
        raise ValueError("temperature must be between 0 and 1.")
    return args


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="CRAG Process for Document Retrieval and Query Answering.")
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to encode.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Language model to use (default: gpt-4o-mini).")
    parser.add_argument("--max_tokens", type=int, default=1000,
                        help="Maximum tokens to use in LLM responses (default: 1000).")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Temperature to use for LLM responses (default: 0).")
    parser.add_argument("--query", type=str, default="What are the main causes of climate change?",
                        help="Query to test the CRAG process.")
    parser.add_argument("--lower_threshold", type=float, default=0.3,
                        help="Lower threshold for score evaluation (default: 0.3).")
    parser.add_argument("--upper_threshold", type=float, default=0.7,
                        help="Upper threshold for score evaluation (default: 0.7).")

    return validate_args(parser.parse_args())


# Main function to handle argument parsing and call the CRAG class
def main(args):
    # Initialize the CRAG process
    crag = CRAG(
        path=args.path,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        lower_threshold=args.lower_threshold,
        upper_threshold=args.upper_threshold
    )

    # Process the query
    response = crag.run(args.query)
    print(f"Query: {args.query}")
    print(f"Answer: {response}")


if __name__ == '__main__':
    main(parse_args())
