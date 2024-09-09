import asyncio
import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.summarize.chain import load_summarize_chain
from langchain.docstore.document import Document
from helper_functions import encode_pdf, encode_from_string

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# Function to encode to both summary and chunk levels, sharing the page metadata
async def encode_pdf_hierarchical(path, chunk_size=1000, chunk_overlap=200, is_string=False):
    """
    Asynchronously encodes a PDF book into a hierarchical vector store using OpenAI embeddings.
    Includes rate limit handling with exponential backoff.
    """
    if not is_string:
        loader = PyPDFLoader(path)
        documents = await asyncio.to_thread(loader.load)
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, is_separator_regex=False
        )
        documents = text_splitter.create_documents([path])

    summary_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
    summary_chain = load_summarize_chain(summary_llm, chain_type="map_reduce")

    async def summarize_doc(doc):
        summary_output = await retry_with_exponential_backoff(summary_chain.ainvoke([doc]))
        summary = summary_output['output_text']
        return Document(page_content=summary, metadata={"source": path, "page": doc.metadata["page"], "summary": True})

    summaries = []
    batch_size = 5
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_summaries = await asyncio.gather(*[summarize_doc(doc) for doc in batch])
        summaries.extend(batch_summaries)
        await asyncio.sleep(1)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    detailed_chunks = await asyncio.to_thread(text_splitter.split_documents, documents)

    for i, chunk in enumerate(detailed_chunks):
        chunk.metadata.update({"chunk_id": i, "summary": False, "page": int(chunk.metadata.get("page", 0))})

    embeddings = OpenAIEmbeddings()

    async def create_vectorstore(docs):
        return await retry_with_exponential_backoff(asyncio.to_thread(FAISS.from_documents, docs, embeddings))

    summary_vectorstore, detailed_vectorstore = await asyncio.gather(
        create_vectorstore(summaries),
        create_vectorstore(detailed_chunks)
    )

    return summary_vectorstore, detailed_vectorstore


def retrieve_hierarchical(query, summary_vectorstore, detailed_vectorstore, k_summaries=3, k_chunks=5):
    """
    Performs a hierarchical retrieval using the query.
    """
    top_summaries = summary_vectorstore.similarity_search(query, k=k_summaries)
    relevant_chunks = []
    for summary in top_summaries:
        page_number = summary.metadata["page"]
        page_filter = lambda metadata: metadata["page"] == page_number
        page_chunks = detailed_vectorstore.similarity_search(query, k=k_chunks, filter=page_filter)
        relevant_chunks.extend(page_chunks)
    return relevant_chunks


class HierarchicalRAG:
    def __init__(self, pdf_path, chunk_size=1000, chunk_overlap=200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.summary_store = None
        self.detailed_store = None

    async def run(self, query):
        if os.path.exists("../vector_stores/summary_store") and os.path.exists("../vector_stores/detailed_store"):
            embeddings = OpenAIEmbeddings()
            self.summary_store = FAISS.load_local("../vector_stores/summary_store", embeddings, allow_dangerous_deserialization=True)
            self.detailed_store = FAISS.load_local("../vector_stores/detailed_store", embeddings, allow_dangerous_deserialization=True)
        else:
            self.summary_store, self.detailed_store = await encode_pdf_hierarchical(self.pdf_path, self.chunk_size, self.chunk_overlap)
            self.summary_store.save_local("../vector_stores/summary_store")
            self.detailed_store.save_local("../vector_stores/detailed_store")

        results = retrieve_hierarchical(query, self.summary_store, self.detailed_store)
        for chunk in results:
            print(f"Page: {chunk.metadata['page']}")
            print(f"Content: {chunk.page_content}...")
            print("---")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run Hierarchical RAG on a given PDF.")
    parser.add_argument("--pdf_path", type=str, default="../data/Understanding_Climate_Change.pdf", help="Path to the PDF document.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of each text chunk.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap between consecutive chunks.")
    parser.add_argument("--query", type=str, default='What is the greenhouse effect',
                        help="Query to search in the document.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rag = HierarchicalRAG(args.pdf_path, args.chunk_size, args.chunk_overlap)
    asyncio.run(rag.run(args.query))
