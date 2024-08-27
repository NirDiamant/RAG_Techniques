import sys
import os
import re
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from enum import Enum
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path sicnce we work with notebooks

from helper_functions import *


class QuestionGeneration(Enum):
    """
    Enum class to specify the level of question generation for document processing.

    Attributes:
        DOCUMENT_LEVEL (int): Represents question generation at the entire document level.
        FRAGMENT_LEVEL (int): Represents question generation at the individual text fragment level.
    """
    DOCUMENT_LEVEL = 1
    FRAGMENT_LEVEL = 2


# Depending on the model, for Mitral 7B it can be max 8000, for Llama 3.1 8B 128k
DOCUMENT_MAX_TOKENS = 4000
DOCUMENT_OVERLAP_TOKENS = 100

# Embeddings and text similarity calculated on shorter texts
FRAGMENT_MAX_TOKENS = 128
FRAGMENT_OVERLAP_TOKENS = 16

# Questions generated on document or fragment level
QUESTION_GENERATION = QuestionGeneration.DOCUMENT_LEVEL
# how many questions will be generated for specific document or fragment
QUESTIONS_PER_DOCUMENT = 40


# Define classes and functions used by this pipeline
class QuestionList(BaseModel):
    question_list: List[str] = Field(..., title="List of questions generated for the document or fragment")


class OpenAIEmbeddingsWrapper(OpenAIEmbeddings):
    """
    A wrapper class for OpenAI embeddings, providing a similar interface to the original OllamaEmbeddings.
    """

    def __call__(self, query: str) -> List[float]:
        """
        Allows the instance to be used as a callable to generate an embedding for a query.

        Args:
            query (str): The query string to be embedded.

        Returns:
            List[float]: The embedding for the query as a list of floats.
        """
        return self.embed_query(query)


def clean_and_filter_questions(questions: List[str]) -> List[str]:
    """
    Cleans and filters a list of questions.

    Args:
        questions (List[str]): A list of questions to be cleaned and filtered.

    Returns:
        List[str]: A list of cleaned and filtered questions that end with a question mark.
    """
    cleaned_questions = []
    for question in questions:
        cleaned_question = re.sub(r'^\d+\.\s*', '', question.strip())
        if cleaned_question.endswith('?'):
            cleaned_questions.append(cleaned_question)
    return cleaned_questions


def generate_questions(text: str) -> List[str]:
    """
    Generates a list of questions based on the provided text using OpenAI.

    Args:
        text (str): The context data from which questions are generated.

    Returns:
        List[str]: A list of unique, filtered questions.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "num_questions"],
        template="Using the context data: {context}\n\nGenerate a list of at least {num_questions} "
                 "possible questions that can be asked about this context. Ensure the questions are "
                 "directly answerable within the context and do not include any answers or headers. "
                 "Separate the questions with a new line character."
    )
    chain = prompt | llm.with_structured_output(QuestionList)
    input_data = {"context": text, "num_questions": QUESTIONS_PER_DOCUMENT}
    result = chain.invoke(input_data)

    # Extract the list of questions from the QuestionList object
    questions = result.question_list

    filtered_questions = clean_and_filter_questions(questions)
    return list(set(filtered_questions))


def generate_answer(content: str, question: str) -> str:
    """
    Generates an answer to a given question based on the provided context using OpenAI.

    Args:
        content (str): The context data used to generate the answer.
        question (str): The question for which the answer is generated.

    Returns:
        str: The precise answer to the question based on the provided context.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Using the context data: {context}\n\nProvide a brief and precise answer to the question: {question}"
    )
    chain = prompt | llm
    input_data = {"context": content, "question": question}
    return chain.invoke(input_data)


def split_document(document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Splits a document into smaller chunks of text.

    Args:
        document (str): The text of the document to be split.
        chunk_size (int): The size of each chunk in terms of the number of tokens.
        chunk_overlap (int): The number of overlapping tokens between consecutive chunks.

    Returns:
        List[str]: A list of text chunks, where each chunk is a string of the document content.
    """
    tokens = re.findall(r'\b\w+\b', document)
    chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(chunk_tokens)
        if i + chunk_size >= len(tokens):
            break
    return [" ".join(chunk) for chunk in chunks]


def print_document(comment: str, document: Any) -> None:
    """
    Prints a comment followed by the content of a document.

    Args:
        comment (str): The comment or description to print before the document details.
        document (Any): The document whose content is to be printed.

    Returns:
        None
    """
    print(
        f'{comment} (type: {document.metadata["type"]}, index: {document.metadata["index"]}): {document.page_content}')


# Example usage

# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddingsWrapper()

# Example document
example_text = "This is an example document. It contains information about various topics."

# Generate questions
questions = generate_questions(example_text)
print("Generated Questions:")
for q in questions:
    print(f"- {q}")

# Generate an answer
sample_question = questions[0] if questions else "What is this document about?"
answer = generate_answer(example_text, sample_question)
print(f"\nQuestion: {sample_question}")
print(f"Answer: {answer}")

# Split document
chunks = split_document(example_text, chunk_size=10, chunk_overlap=2)
print("\nDocument Chunks:")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}: {chunk}")

# Example of using OpenAIEmbeddings
doc_embedding = embeddings.embed_documents([example_text])
query_embedding = embeddings.embed_query("What is the main topic?")
print("\nDocument Embedding (first 5 elements):", doc_embedding[0][:5])
print("Query Embedding (first 5 elements):", query_embedding[:5])


# Main pipeline
def process_documents(content: str, embedding_model: OpenAIEmbeddings):
    """
    Process the document content, split it into fragments, generate questions,
    create a FAISS vector store, and return a retriever.

    Args:
        content (str): The content of the document to process.
        embedding_model (OpenAIEmbeddings): The embedding model to use for vectorization.

    Returns:
        VectorStoreRetriever: A retriever for the most relevant FAISS document.
    """
    # Split the whole text content into text documents
    text_documents = split_document(content, DOCUMENT_MAX_TOKENS, DOCUMENT_OVERLAP_TOKENS)
    print(f'Text content split into: {len(text_documents)} documents')

    documents = []
    counter = 0
    for i, text_document in enumerate(text_documents):
        text_fragments = split_document(text_document, FRAGMENT_MAX_TOKENS, FRAGMENT_OVERLAP_TOKENS)
        print(f'Text document {i} - split into: {len(text_fragments)} fragments')

        for j, text_fragment in enumerate(text_fragments):
            documents.append(Document(
                page_content=text_fragment,
                metadata={"type": "ORIGINAL", "index": counter, "text": text_document}
            ))
            counter += 1

            if QUESTION_GENERATION == QuestionGeneration.FRAGMENT_LEVEL:
                questions = generate_questions(text_fragment)
                documents.extend([
                    Document(page_content=question,
                             metadata={"type": "AUGMENTED", "index": counter + idx, "text": text_document})
                    for idx, question in enumerate(questions)
                ])
                counter += len(questions)
                print(f'Text document {i} Text fragment {j} - generated: {len(questions)} questions')

        if QUESTION_GENERATION == QuestionGeneration.DOCUMENT_LEVEL:
            questions = generate_questions(text_document)
            documents.extend([
                Document(page_content=question,
                         metadata={"type": "AUGMENTED", "index": counter + idx, "text": text_document})
                for idx, question in enumerate(questions)
            ])
            counter += len(questions)
            print(f'Text document {i} - generated: {len(questions)} questions')

    for document in documents:
        print_document("Dataset", document)

    print(f'Creating store, calculating embeddings for {len(documents)} FAISS documents')
    vectorstore = FAISS.from_documents(documents, embedding_model)

    print("Creating retriever returning the most relevant FAISS document")
    return vectorstore.as_retriever(search_kwargs={"k": 1})


# Example
# Load sample PDF document to string variable
path = "../data/Understanding_Climate_Change.pdf"
content = read_pdf_to_string(path)

# Instantiate OpenAI Embeddings class that will be used by FAISS
embedding_model = OpenAIEmbeddings()

# Process documents and create retriever
document_query_retriever = process_documents(content, embedding_model)

# Example usage of the retriever
query = "What is climate change?"
retrieved_docs = document_query_retriever.get_relevant_documents(query)
print(f"\nQuery: {query}")
print(f"Retrieved document: {retrieved_docs[0].page_content}")

# Find the most relevant FAISS document in the store. In most cases, this will be an augmented question rather than the original text document.
query = "How do freshwater ecosystems change due to alterations in climatic factors?"
print(f'Question:{os.linesep}{query}{os.linesep}')
retrieved_documents = document_query_retriever.invoke(query)

for doc in retrieved_documents:
    print_document("Relevant fragment retrieved", doc)

# Find the parent text document and use it as context for the generative model to generate an answer to the question.
context = doc.metadata['text']
print(f'{os.linesep}Context:{os.linesep}{context}')
answer = generate_answer(context, query)
print(f'{os.linesep}Answer:{os.linesep}{answer}')
