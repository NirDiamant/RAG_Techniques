import sys
import os
import re
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from enum import Enum
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, Field
import argparse

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path

from helper_functions import *


class QuestionGeneration(Enum):
    """
    Enum class to specify the level of question generation for document processing.
    """
    DOCUMENT_LEVEL = 1
    FRAGMENT_LEVEL = 2


DOCUMENT_MAX_TOKENS = 4000
DOCUMENT_OVERLAP_TOKENS = 100
FRAGMENT_MAX_TOKENS = 128
FRAGMENT_OVERLAP_TOKENS = 16
QUESTION_GENERATION = QuestionGeneration.DOCUMENT_LEVEL
QUESTIONS_PER_DOCUMENT = 40


class QuestionList(BaseModel):
    question_list: List[str] = Field(..., title="List of questions generated for the document or fragment")


class OpenAIEmbeddingsWrapper(OpenAIEmbeddings):
    """
    A wrapper class for OpenAI embeddings, providing a similar interface to the original OllamaEmbeddings.
    """
    def __call__(self, query: str) -> List[float]:
        return self.embed_query(query)


def clean_and_filter_questions(questions: List[str]) -> List[str]:
    cleaned_questions = []
    for question in questions:
        cleaned_question = re.sub(r'^\d+\.\s*', '', question.strip())
        if cleaned_question.endswith('?'):
            cleaned_questions.append(cleaned_question)
    return cleaned_questions


def generate_questions(text: str) -> List[str]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "num_questions"],
        template="Using the context data: {context}\n\nGenerate a list of at least {num_questions} "
                 "possible questions that can be asked about this context."
    )
    chain = prompt | llm.with_structured_output(QuestionList)
    input_data = {"context": text, "num_questions": QUESTIONS_PER_DOCUMENT}
    result = chain.invoke(input_data)
    questions = result.question_list
    return list(set(clean_and_filter_questions(questions)))


def generate_answer(content: str, question: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Using the context data: {context}\n\nProvide a brief and precise answer to the question: {question}"
    )
    chain = prompt | llm
    input_data = {"context": content, "question": question}
    return chain.invoke(input_data)


def split_document(document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    tokens = re.findall(r'\b\w+\b', document)
    chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(chunk_tokens)
        if i + chunk_size >= len(tokens):
            break
    return [" ".join(chunk) for chunk in chunks]


def print_document(comment: str, document: Any) -> None:
    print(f'{comment} (type: {document.metadata["type"]}, index: {document.metadata["index"]}): {document.page_content}')


class DocumentProcessor:
    def __init__(self, content: str, embedding_model: OpenAIEmbeddings):
        self.content = content
        self.embedding_model = embedding_model

    def run(self):
        text_documents = split_document(self.content, DOCUMENT_MAX_TOKENS, DOCUMENT_OVERLAP_TOKENS)
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
        vectorstore = FAISS.from_documents(documents, self.embedding_model)

        print("Creating retriever returning the most relevant FAISS document")
        return vectorstore.as_retriever(search_kwargs={"k": 1})


def parse_args():
    parser = argparse.ArgumentParser(description="Process a document and create a retriever.")
    parser.add_argument('--path', type=str, default='../data/Understanding_Climate_Change.pdf',
                        help="Path to the PDF document to process")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load sample PDF document to string variable
    content = read_pdf_to_string(args.path)

    # Instantiate OpenAI Embeddings class that will be used by FAISS
    embedding_model = OpenAIEmbeddings()

    # Process documents and create retriever
    processor = DocumentProcessor(content, embedding_model)
    document_query_retriever = processor.run()

    # Example usage of the retriever
    query = "What is climate change?"
    retrieved_docs = document_query_retriever.get_relevant_documents(query)
    print(f"\nQuery: {query}")
    print(f"Retrieved document: {retrieved_docs[0].page_content}")

    # Further query example
    query = "How do freshwater ecosystems change due to alterations in climatic factors?"
    retrieved_documents = document_query_retriever.get_relevant_documents(query)
    for doc in retrieved_documents:
        print_document("Relevant fragment retrieved", doc)

    context = doc.metadata['text']
    answer = generate_answer(context, query)
    print(f'{os.linesep}Answer:{os.linesep}{answer}')
