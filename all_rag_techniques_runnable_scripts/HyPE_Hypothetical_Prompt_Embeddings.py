import os
import sys
import argparse
import time
import faiss
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.docstore.in_memory import InMemoryDocstore

# Add the parent directory to the path since we work with notebooks
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file (e.g., OpenAI API key)
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

class HyPE:
    """
    A class to handle the HyPE RAG process, which enhances document chunking by 
    generating hypothetical questions as proxies for retrieval.
    """

    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=3):
        """
        Initializes the HyPE-based RAG retriever by encoding the PDF document with 
        hypothetical prompt embeddings.

        Args:
            path (str): Path to the PDF file to encode.
            chunk_size (int): Size of each text chunk (default: 1000).
            chunk_overlap (int): Overlap between consecutive chunks (default: 200).
            n_retrieved (int): Number of chunks to retrieve for each query (default: 3).
        """
        print("\n--- Initializing HyPE RAG Retriever ---")

        # Encode the PDF document into a FAISS vector store using hypothetical prompt embeddings
        start_time = time.time()
        self.vector_store = self.encode_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.time_records = {'Chunking': time.time() - start_time}
        print(f"Chunking Time: {self.time_records['Chunking']:.2f} seconds")

        # Create a retriever from the vector store
        self.chunks_query_retriever = self.vector_store.as_retriever(search_kwargs={"k": n_retrieved})

    def generate_hypothetical_prompt_embeddings(self, chunk_text):
        """
        Uses an LLM to generate multiple hypothetical questions for a single chunk.
        These questions act as 'proxies' for the chunk during retrieval.

        Parameters:
        chunk_text (str): Text contents of the chunk.

        Returns:
        tuple: (Original chunk text, List of embedding vectors generated from the questions)
        """
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

        question_gen_prompt = PromptTemplate.from_template(
            "Analyze the input text and generate essential questions that, when answered, \
            capture the main points of the text. Each question should be one line, \
            without numbering or prefixes.\n\n \
            Text:\n{chunk_text}\n\nQuestions:\n"
        )
        question_chain = question_gen_prompt | llm | StrOutputParser()

        # Parse questions from response
        questions = question_chain.invoke({"chunk_text": chunk_text}).replace("\n\n", "\n").split("\n")

        return chunk_text, embedding_model.embed_documents(questions)

    def prepare_vector_store(self, chunks):
        """
        Creates and populates a FAISS vector store using hypothetical prompt embeddings.

        Parameters:
        chunks (List[str]): A list of text chunks to be embedded and stored.

        Returns:
        FAISS: A FAISS vector store containing the embedded text chunks.
        """
        vector_store = None  # Wait to initialize to determine vector size

        with ThreadPoolExecutor() as pool:
            # Parallelized embedding generation
            futures = [pool.submit(self.generate_hypothetical_prompt_embeddings, c) for c in chunks]

            for f in tqdm(as_completed(futures), total=len(chunks)):  
                chunk, vectors = f.result()  # Retrieve processed chunk and embeddings

                # Initialize FAISS store once vector size is known
                if vector_store is None:
                    vector_store = FAISS(
                        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
                        index=faiss.IndexFlatL2(len(vectors[0])),
                        docstore=InMemoryDocstore(),
                        index_to_docstore_id={}
                    )

                # Store multiple vector representations per chunk
                chunks_with_embedding_vectors = [(chunk.page_content, vec) for vec in vectors]
                vector_store.add_embeddings(chunks_with_embedding_vectors)

        return vector_store

    def encode_pdf(self, path, chunk_size=1000, chunk_overlap=200):
        """
        Encodes a PDF document into a vector store using hypothetical prompt embeddings.

        Args:
            path: The path to the PDF file.
            chunk_size: The size of each text chunk.
            chunk_overlap: The overlap between consecutive chunks.

        Returns:
            A FAISS vector store containing the encoded book content.
        """
        # Load PDF documents
        loader = PyPDFLoader(path)
        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        texts = text_splitter.split_documents(documents)
        cleaned_texts = replace_t_with_space(texts)

        return self.prepare_vector_store(cleaned_texts)

    def run(self, query):
        """
        Retrieves and displays the context for the given query.

        Args:
            query (str): The query to retrieve context for.

        Returns:
            None
        """
        # Measure retrieval time
        start_time = time.time()
        context = retrieve_context_per_question(query, self.chunks_query_retriever)
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")

        # Deduplicate context and display results
        context = list(set(context))
        show_context(context)


def validate_args(args):
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")
    if args.n_retrieved <= 0:
        raise ValueError("n_retrieved must be a positive integer.")
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Encode a PDF document and test a HyPE-based RAG system.")
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to encode.")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Size of each text chunk (default: 1000).")
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="Overlap between consecutive chunks (default: 200).")
    parser.add_argument("--n_retrieved", type=int, default=3,
                        help="Number of chunks to retrieve for each query (default: 3).")
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?",
                        help="Query to test the retriever (default: 'What is the main cause of climate change?').")
    parser.add_argument("--evaluate", action="store_true",
                        help="Whether to evaluate the retriever's performance (default: False).")

    return validate_args(parser.parse_args())


def main(args):
    # Initialize the HyPE-based RAG Retriever
    hyperag = HyPE(
        path=args.path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_retrieved=args.n_retrieved
    )

    # Retrieve context based on the query
    hyperag.run(args.query)

    # Evaluate the retriever's performance on the query (if requested)
    if args.evaluate:
        evaluate_rag(hyperag.chunks_query_retriever)


if __name__ == '__main__':
    # Call the main function with parsed arguments
    main(parse_args())
