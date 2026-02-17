import os
import sys
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Define the Response class
class Response(BaseModel):
    answer: str = Field(..., title="The answer to the question. The options can be only 'Yes' or 'No'")


# Define utility functions
def get_user_feedback(query, response, relevance, quality, comments=""):
    return {
        "query": query,
        "response": response,
        "relevance": int(relevance),
        "quality": int(quality),
        "comments": comments
    }


def store_feedback(feedback):
    with open("../data/feedback_data.json", "a") as f:
        json.dump(feedback, f)
        f.write("\n")


def load_feedback_data():
    feedback_data = []
    try:
        with open("../data/feedback_data.json", "r") as f:
            for line in f:
                feedback_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print("No feedback data file found. Starting with empty feedback.")
    return feedback_data


def adjust_relevance_scores(query: str, docs: List[Any], feedback_data: List[Dict[str, Any]]) -> List[Any]:
    relevance_prompt = PromptTemplate(
        input_variables=["query", "feedback_query", "doc_content", "feedback_response"],
        template="""
        Determine if the following feedback response is relevant to the current query and document content.
        You are also provided with the Feedback original query that was used to generate the feedback response.
        Current query: {query}
        Feedback query: {feedback_query}
        Document content: {doc_content}
        Feedback response: {feedback_response}

        Is this feedback relevant? Respond with only 'Yes' or 'No'.
        """
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
    relevance_chain = relevance_prompt | llm.with_structured_output(Response)

    for doc in docs:
        relevant_feedback = []
        for feedback in feedback_data:
            input_data = {
                "query": query,
                "feedback_query": feedback['query'],
                "doc_content": doc.page_content[:1000],
                "feedback_response": feedback['response']
            }
            result = relevance_chain.invoke(input_data).answer

            if result == 'yes':
                relevant_feedback.append(feedback)

        if relevant_feedback:
            avg_relevance = sum(f['relevance'] for f in relevant_feedback) / len(relevant_feedback)
            doc.metadata['relevance_score'] *= (avg_relevance / 3)

    return sorted(docs, key=lambda x: x.metadata['relevance_score'], reverse=True)


def fine_tune_index(feedback_data: List[Dict[str, Any]], texts: List[str]) -> Any:
    good_responses = [f for f in feedback_data if f['relevance'] >= 4 and f['quality'] >= 4]
    additional_texts = " ".join([f['query'] + " " + f['response'] for f in good_responses])
    all_texts = texts + additional_texts
    new_vectorstore = encode_from_string(all_texts)
    return new_vectorstore


# Define the main RAG class
class RetrievalAugmentedGeneration:
    def __init__(self, path: str):
        self.path = path
        self.content = read_pdf_to_string(self.path)
        self.vectorstore = encode_from_string(self.content)
        self.retriever = self.vectorstore.as_retriever()
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        self.qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=self.retriever)

    def run(self, query: str, relevance: int, quality: int):
        response = self.qa_chain(query)["result"]
        feedback = get_user_feedback(query, response, relevance, quality)
        store_feedback(feedback)

        docs = self.retriever.get_relevant_documents(query)
        adjusted_docs = adjust_relevance_scores(query, docs, load_feedback_data())
        self.retriever.search_kwargs['k'] = len(adjusted_docs)
        self.retriever.search_kwargs['docs'] = adjusted_docs

        return response


# Argument parsing
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run the RAG system with feedback integration.")
    parser.add_argument('--path', type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="Path to the document.")
    parser.add_argument('--query', type=str, default='What is the greenhouse effect?',
                        help="Query to ask the RAG system.")
    parser.add_argument('--relevance', type=int, default=5, help="Relevance score for the feedback.")
    parser.add_argument('--quality', type=int, default=5, help="Quality score for the feedback.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rag = RetrievalAugmentedGeneration(args.path)
    result = rag.run(args.query, args.relevance, args.quality)
    print(f"Response: {result}")

    # Fine-tune the vectorstore periodically
    new_vectorstore = fine_tune_index(load_feedback_data(), rag.content)
    rag.retriever = new_vectorstore.as_retriever()
