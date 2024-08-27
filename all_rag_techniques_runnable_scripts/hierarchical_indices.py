import asyncio
import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.summarize.chain import load_summarize_chain
from langchain.docstore.document import Document
from helper_functions import encode_pdf, encode_from_string

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path sicnce we work with notebooks
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Define document path
path = "../data/Understanding_Climate_Change.pdf"


# Function to encode to both summary and chunk levels, sharing the page metadata
async def encode_pdf_hierarchical(path, chunk_size=1000, chunk_overlap=200, is_string=False):
    """
    Asynchronously encodes a PDF book into a hierarchical vector store using OpenAI embeddings.
    Includes rate limit handling with exponential backoff.
    
    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.
        
    Returns:
        A tuple containing two FAISS vector stores:
        1. Document-level summaries
        2. Detailed chunks
    """

    # Load PDF documents
    if not is_string:
        loader = PyPDFLoader(path)
        documents = await asyncio.to_thread(loader.load)
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        documents = text_splitter.create_documents([path])

    # Create document-level summaries
    summary_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
    summary_chain = load_summarize_chain(summary_llm, chain_type="map_reduce")

    async def summarize_doc(doc):
        """
        Summarizes a single document with rate limit handling.
        
        Args:
            doc: The document to be summarized.
            
        Returns:
            A summarized Document object.
        """
        # Retry the summarization with exponential backoff
        summary_output = await retry_with_exponential_backoff(summary_chain.ainvoke([doc]))
        summary = summary_output['output_text']
        return Document(
            page_content=summary,
            metadata={"source": path, "page": doc.metadata["page"], "summary": True}
        )

    # Process documents in smaller batches to avoid rate limits
    batch_size = 5  # Adjust this based on your rate limits
    summaries = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_summaries = await asyncio.gather(*[summarize_doc(doc) for doc in batch])
        summaries.extend(batch_summaries)
        await asyncio.sleep(1)  # Short pause between batches

    # Split documents into detailed chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    detailed_chunks = await asyncio.to_thread(text_splitter.split_documents, documents)

    # Update metadata for detailed chunks
    for i, chunk in enumerate(detailed_chunks):
        chunk.metadata.update({
            "chunk_id": i,
            "summary": False,
            "page": int(chunk.metadata.get("page", 0))
        })

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create vector stores asynchronously with rate limit handling
    async def create_vectorstore(docs):
        """
        Creates a vector store from a list of documents with rate limit handling.
        
        Args:
            docs: The list of documents to be embedded.
            
        Returns:
            A FAISS vector store containing the embedded documents.
        """
        return await retry_with_exponential_backoff(
            asyncio.to_thread(FAISS.from_documents, docs, embeddings)
        )

    # Generate vector stores for summaries and detailed chunks concurrently
    summary_vectorstore, detailed_vectorstore = await asyncio.gather(
        create_vectorstore(summaries),
        create_vectorstore(detailed_chunks)
    )

    return summary_vectorstore, detailed_vectorstore


# Retrieve information according to summary level, and then retrieve information from the chunk level vector store and filter according to the summary level pages
def retrieve_hierarchical(query, summary_vectorstore, detailed_vectorstore, k_summaries=3, k_chunks=5):
    """
    Performs a hierarchical retrieval using the query.

    Args:
        query: The search query.
        summary_vectorstore: The vector store containing document summaries.
        detailed_vectorstore: The vector store containing detailed chunks.
        k_summaries: The number of top summaries to retrieve.
        k_chunks: The number of detailed chunks to retrieve per summary.

    Returns:
        A list of relevant detailed chunks.
    """

    # Retrieve top summaries
    top_summaries = summary_vectorstore.similarity_search(query, k=k_summaries)

    relevant_chunks = []
    for summary in top_summaries:
        # For each summary, retrieve relevant detailed chunks
        page_number = summary.metadata["page"]
        page_filter = lambda metadata: metadata["page"] == page_number
        page_chunks = detailed_vectorstore.similarity_search(
            query,
            k=k_chunks,
            filter=page_filter
        )
        relevant_chunks.extend(page_chunks)

    return relevant_chunks


async def main():
    # Encode the PDF book to both document-level summaries and detailed chunks if the vector stores do not exist

    if os.path.exists("../vector_stores/summary_store") and os.path.exists("../vector_stores/detailed_store"):
        embeddings = OpenAIEmbeddings()
        summary_store = FAISS.load_local("../vector_stores/summary_store", embeddings, allow_dangerous_deserialization=True)
        detailed_store = FAISS.load_local("../vector_stores/detailed_store", embeddings,
                                          allow_dangerous_deserialization=True)

    else:
        summary_store, detailed_store = await encode_pdf_hierarchical(path)
        summary_store.save_local("../vector_stores/summary_store")
        detailed_store.save_local("../vector_stores/detailed_store")

    # Demonstrate on a use case
    query = "What is the greenhouse effect?"
    results = retrieve_hierarchical(query, summary_store, detailed_store)

    # Print results
    for chunk in results:
        print(f"Page: {chunk.metadata['page']}")
        print(f"Content: {chunk.page_content}...")  # Print first 100 characters
        print("---")


if __name__ == "__main__":
    asyncio.run(main())
