import os
import sys
from dotenv import load_dotenv
from langchain.docstore.document import Document

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path sicnce we work with notebooks
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Define path to PDF
path = "../data/Understanding_Climate_Change.pdf"

# Read PDF to string
content = read_pdf_to_string(path)


# Function to split text into chunks with metadata of the chunk chronological index
def split_text_to_chunks_with_indices(text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(Document(page_content=chunk, metadata={"index": len(chunks), "text": text}))
        start += chunk_size - chunk_overlap
    return chunks


# Split our document accordingly
chunks_size = 400
chunk_overlap = 200
docs = split_text_to_chunks_with_indices(content, chunks_size, chunk_overlap)

# Create vector store and retriever
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
chunks_query_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


# Function to draw the k<sup>th</sup> chunk (in the original order) from the vector store 

def get_chunk_by_index(vectorstore, target_index: int) -> Document:
    """
    Retrieve a chunk from the vectorstore based on its index in the metadata.
    
    Args:
    vectorstore (VectorStore): The vectorstore containing the chunks.
    target_index (int): The index of the chunk to retrieve.
    
    Returns:
    Optional[Document]: The retrieved chunk as a Document object, or None if not found.
    """
    # This is a simplified version. In practice, you might need a more efficient method
    # to retrieve chunks by index, depending on your vectorstore implementation.
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
    for doc in all_docs:
        if doc.metadata.get('index') == target_index:
            return doc
    return None


# Check the function
chunk = get_chunk_by_index(vectorstore, 0)
print(chunk.page_content)


# Function that retrieves from the vector stroe based on semantic similarity and then pads each retrieved chunk with its num_neighbors before and after, taking into account the chunk overlap to construct a meaningful wide window arround it
def retrieve_with_context_overlap(vectorstore, retriever, query: str, num_neighbors: int = 1, chunk_size: int = 200,
                                  chunk_overlap: int = 20) -> List[str]:
    """
    Retrieve chunks based on a query, then fetch neighboring chunks and concatenate them, 
    accounting for overlap and correct indexing.

    Args:
    vectorstore (VectorStore): The vectorstore containing the chunks.
    retriever: The retriever object to get relevant documents.
    query (str): The query to search for relevant chunks.
    num_neighbors (int): The number of chunks to retrieve before and after each relevant chunk.
    chunk_size (int): The size of each chunk when originally split.
    chunk_overlap (int): The overlap between chunks when originally split.

    Returns:
    List[str]: List of concatenated chunk sequences, each centered on a relevant chunk.
    """
    relevant_chunks = retriever.get_relevant_documents(query)
    result_sequences = []

    for chunk in relevant_chunks:
        current_index = chunk.metadata.get('index')
        if current_index is None:
            continue

        # Determine the range of chunks to retrieve
        start_index = max(0, current_index - num_neighbors)
        end_index = current_index + num_neighbors + 1  # +1 because range is exclusive at the end

        # Retrieve all chunks in the range
        neighbor_chunks = []
        for i in range(start_index, end_index):
            neighbor_chunk = get_chunk_by_index(vectorstore, i)
            if neighbor_chunk:
                neighbor_chunks.append(neighbor_chunk)

        # Sort chunks by their index to ensure correct order
        neighbor_chunks.sort(key=lambda x: x.metadata.get('index', 0))

        # Concatenate chunks, accounting for overlap
        concatenated_text = neighbor_chunks[0].page_content
        for i in range(1, len(neighbor_chunks)):
            current_chunk = neighbor_chunks[i].page_content
            overlap_start = max(0, len(concatenated_text) - chunk_overlap)
            concatenated_text = concatenated_text[:overlap_start] + current_chunk

        result_sequences.append(concatenated_text)

    return result_sequences


# Comparing regular retrival and retrival with context window
# Baseline approach
query = "Explain the role of deforestation and fossil fuels in climate change."
baseline_chunk = chunks_query_retriever.get_relevant_documents(query
                                                               ,
                                                               k=1
                                                               )
# Focused context enrichment approach
enriched_chunks = retrieve_with_context_overlap(
    vectorstore,
    chunks_query_retriever,
    query,
    num_neighbors=1,
    chunk_size=400,
    chunk_overlap=200
)

print("Baseline Chunk:")
print(baseline_chunk[0].page_content)
print("\nEnriched Chunks:")
print(enriched_chunks[0])

# An example that showcases the superiority of additional context window
document_content = """
Artificial Intelligence (AI) has a rich history dating back to the mid-20th century. The term "Artificial Intelligence" was coined in 1956 at the Dartmouth Conference, marking the field's official beginning.

In the 1950s and 1960s, AI research focused on symbolic methods and problem-solving. The Logic Theorist, created in 1955 by Allen Newell and Herbert A. Simon, is often considered the first AI program.

The 1960s saw the development of expert systems, which used predefined rules to solve complex problems. DENDRAL, created in 1965, was one of the first expert systems, designed to analyze chemical compounds.

However, the 1970s brought the first "AI Winter," a period of reduced funding and interest in AI research, largely due to overpromised capabilities and underdelivered results.

The 1980s saw a resurgence with the popularization of expert systems in corporations. The Japanese government's Fifth Generation Computer Project also spurred increased investment in AI research globally.

Neural networks gained prominence in the 1980s and 1990s. The backpropagation algorithm, although discovered earlier, became widely used for training multi-layer networks during this time.

The late 1990s and 2000s marked the rise of machine learning approaches. Support Vector Machines (SVMs) and Random Forests became popular for various classification and regression tasks.

Deep Learning, a subset of machine learning using neural networks with many layers, began to show promising results in the early 2010s. The breakthrough came in 2012 when a deep neural network significantly outperformed other machine learning methods in the ImageNet competition.

Since then, deep learning has revolutionized many AI applications, including image and speech recognition, natural language processing, and game playing. In 2016, Google's AlphaGo defeated a world champion Go player, a landmark achievement in AI.

The current era of AI is characterized by the integration of deep learning with other AI techniques, the development of more efficient and powerful hardware, and the ethical considerations surrounding AI deployment.

Transformers, introduced in 2017, have become a dominant architecture in natural language processing, enabling models like GPT (Generative Pre-trained Transformer) to generate human-like text.

As AI continues to evolve, new challenges and opportunities arise. Explainable AI, robust and fair machine learning, and artificial general intelligence (AGI) are among the key areas of current and future research in the field.
"""

chunks_size = 250
chunk_overlap = 20
document_chunks = split_text_to_chunks_with_indices(document_content, chunks_size, chunk_overlap)
document_vectorstore = FAISS.from_documents(document_chunks, embeddings)
document_retriever = document_vectorstore.as_retriever(search_kwargs={"k": 1})

query = "When did deep learning become prominent in AI?"
context = document_retriever.get_relevant_documents(query)
context_pages_content = [doc.page_content for doc in context]

print("Regular retrieval:\n")
show_context(context_pages_content)

sequences = retrieve_with_context_overlap(document_vectorstore, document_retriever, query, num_neighbors=1)
print("\nRetrieval with context enrichment:\n")
show_context(sequences)
