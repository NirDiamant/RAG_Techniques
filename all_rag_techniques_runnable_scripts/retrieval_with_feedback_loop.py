import os
import sys
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import json
from typing import List, Dict, Any

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path sicnce we work with notebooks
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define documents path
path = "../data/Understanding_Climate_Change.pdf"

# Create vector store and retrieval QA chain
content = read_pdf_to_string(path)
vectorstore = encode_from_string(content)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)


# Function to format user feedback in a dictionary
def get_user_feedback(query, response, relevance, quality, comments=""):
    return {
        "query": query,
        "response": response,
        "relevance": int(relevance),
        "quality": int(quality),
        "comments": comments
    }


# Function to store the feedback in a json file
def store_feedback(feedback):
    with open("../data/feedback_data.json", "a") as f:
        json.dump(feedback, f)
        f.write("\n")


# Function to read the feedback file
def load_feedback_data():
    feedback_data = []
    try:
        with open("../data/feedback_data.json", "r") as f:
            for line in f:
                feedback_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print("No feedback data file found. Starting with empty feedback.")
    return feedback_data


# Function to adjust files relevancy based on the feedbacks file
class Response(BaseModel):
    answer: str = Field(..., title="The answer to the question. The options can be only 'Yes' or 'No'")


def adjust_relevance_scores(query: str, docs: List[Any], feedback_data: List[Dict[str, Any]]) -> List[Any]:
    # Create a prompt template for relevance checking
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

    # Create an LLMChain for relevance checking
    relevance_chain = relevance_prompt | llm.with_structured_output(Response)

    for doc in docs:
        relevant_feedback = []

        for feedback in feedback_data:
            # Use LLM to check relevance
            input_data = {
                "query": query,
                "feedback_query": feedback['query'],
                "doc_content": doc.page_content[:1000],
                "feedback_response": feedback['response']
            }
            result = relevance_chain.invoke(input_data).answer

            if result == 'yes':
                relevant_feedback.append(feedback)

        # Adjust the relevance score based on feedback
        if relevant_feedback:
            avg_relevance = sum(f['relevance'] for f in relevant_feedback) / len(relevant_feedback)
            doc.metadata['relevance_score'] *= (avg_relevance / 3)  # Assuming a 1-5 scale, 3 is neutral

    # Re-rank documents based on adjusted scores
    return sorted(docs, key=lambda x: x.metadata['relevance_score'], reverse=True)


# Function to fine tune the vector index to include also queries + answers that received good feedbacks
def fine_tune_index(feedback_data: List[Dict[str, Any]], texts: List[str]) -> Any:
    # Filter high-quality responses
    good_responses = [f for f in feedback_data if f['relevance'] >= 4 and f['quality'] >= 4]

    # Extract queries and responses, and create new documents
    additional_texts = []
    for f in good_responses:
        combined_text = f['query'] + " " + f['response']
        additional_texts.append(combined_text)

    # make the list a string
    additional_texts = " ".join(additional_texts)

    # Create a new index with original and high-quality texts
    all_texts = texts + additional_texts
    new_vectorstore = encode_from_string(all_texts)

    return new_vectorstore


# Demonstration of how to retrieve answers with respect to user feedbacks
query = "What is the greenhouse effect?"

# Get response from RAG system
response = qa_chain(query)["result"]

relevance = 5
quality = 5

# Collect feedback
feedback = get_user_feedback(query, response, relevance, quality)

# Store feedback
store_feedback(feedback)

# Adjust relevance scores for future retrievals
docs = retriever.get_relevant_documents(query)
adjusted_docs = adjust_relevance_scores(query, docs, load_feedback_data())

# Update the retriever with adjusted docs
retriever.search_kwargs['k'] = len(adjusted_docs)
retriever.search_kwargs['docs'] = adjusted_docs

# Finetune the vectorstore periodicly
# Periodically (e.g., daily or weekly), fine-tune the index
new_vectorstore = fine_tune_index(load_feedback_data(), content)
retriever = new_vectorstore.as_retriever()
