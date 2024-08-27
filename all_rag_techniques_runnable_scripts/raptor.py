import numpy as np
import pandas as pd
from typing import List, Dict, Any

from langchain.chains.llm import LLMChain
from sklearn.mixture import GaussianMixture
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import AIMessage
from langchain.docstore.document import Document
import matplotlib.pyplot as plt
import logging
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path sicnce we work with notebooks
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Define logging, llm and embeddings
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4o-mini")


# Helper Functions

def extract_text(item):
    """Extract text content from either a string or an AIMessage object."""
    if isinstance(item, AIMessage):
        return item.content
    return item


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts using OpenAIEmbeddings."""
    logging.info(f"Embedding {len(texts)} texts")
    return embeddings.embed_documents([extract_text(text) for text in texts])


def perform_clustering(embeddings: np.ndarray, n_clusters: int = 10) -> np.ndarray:
    """Perform clustering on embeddings using Gaussian Mixture Model."""
    logging.info(f"Performing clustering with {n_clusters} clusters")
    gm = GaussianMixture(n_components=n_clusters, random_state=42)
    return gm.fit_predict(embeddings)


def summarize_texts(texts: List[str]) -> str:
    """Summarize a list of texts using OpenAI."""
    logging.info(f"Summarizing {len(texts)} texts")
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text concisely:\n\n{text}"
    )
    chain = prompt | llm
    input_data = {"text": texts}
    return chain.invoke(input_data)


def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, level: int):
    """Visualize clusters using PCA."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'Cluster Visualization - Level {level}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()


# RAPTOR Core Function

def build_raptor_tree(texts: List[str], max_levels: int = 3) -> Dict[int, pd.DataFrame]:
    """Build the RAPTOR tree structure with level metadata and parent-child relationships."""
    results = {}
    current_texts = [extract_text(text) for text in texts]
    current_metadata = [{"level": 0, "origin": "original", "parent_id": None} for _ in texts]

    for level in range(1, max_levels + 1):
        logging.info(f"Processing level {level}")

        embeddings = embed_texts(current_texts)
        n_clusters = min(10, len(current_texts) // 2)
        cluster_labels = perform_clustering(np.array(embeddings), n_clusters)

        df = pd.DataFrame({
            'text': current_texts,
            'embedding': embeddings,
            'cluster': cluster_labels,
            'metadata': current_metadata
        })

        results[level - 1] = df

        summaries = []
        new_metadata = []
        for cluster in df['cluster'].unique():
            cluster_docs = df[df['cluster'] == cluster]
            cluster_texts = cluster_docs['text'].tolist()
            cluster_metadata = cluster_docs['metadata'].tolist()
            summary = summarize_texts(cluster_texts)
            summaries.append(summary)
            new_metadata.append({
                "level": level,
                "origin": f"summary_of_cluster_{cluster}_level_{level - 1}",
                "child_ids": [meta.get('id') for meta in cluster_metadata],
                "id": f"summary_{level}_{cluster}"
            })

        current_texts = summaries
        current_metadata = new_metadata

        if len(current_texts) <= 1:
            results[level] = pd.DataFrame({
                'text': current_texts,
                'embedding': embed_texts(current_texts),
                'cluster': [0],
                'metadata': current_metadata
            })
            logging.info(f"Stopping at level {level} as we have only one summary")
            break

    return results


# Vectorstore Function

def build_vectorstore(tree_results: Dict[int, pd.DataFrame]) -> FAISS:
    """Build a FAISS vectorstore from all texts in the RAPTOR tree."""
    all_texts = []
    all_embeddings = []
    all_metadatas = []

    for level, df in tree_results.items():
        all_texts.extend([str(text) for text in df['text'].tolist()])
        all_embeddings.extend([embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in
                               df['embedding'].tolist()])
        all_metadatas.extend(df['metadata'].tolist())

    logging.info(f"Building vectorstore with {len(all_texts)} texts")

    # Create Document objects manually to ensure correct types
    documents = [Document(page_content=str(text), metadata=metadata)
                 for text, metadata in zip(all_texts, all_metadatas)]

    return FAISS.from_documents(documents, embeddings)


# Define tree traversal retrieval
def tree_traversal_retrieval(query: str, vectorstore: FAISS, k: int = 3) -> List[Document]:
    """Perform tree traversal retrieval."""
    query_embedding = embeddings.embed_query(query)

    def retrieve_level(level: int, parent_ids: List[str] = None) -> List[Document]:
        if parent_ids:
            docs = vectorstore.similarity_search_by_vector_with_relevance_scores(
                query_embedding,
                k=k,
                filter=lambda meta: meta['level'] == level and meta['id'] in parent_ids
            )
        else:
            docs = vectorstore.similarity_search_by_vector_with_relevance_scores(
                query_embedding,
                k=k,
                filter=lambda meta: meta['level'] == level
            )

        if not docs or level == 0:
            return docs

        child_ids = [doc.metadata.get('child_ids', []) for doc, _ in docs]
        child_ids = [item for sublist in child_ids for item in sublist]  # Flatten the list

        child_docs = retrieve_level(level - 1, child_ids)
        return docs + child_docs

    max_level = max(doc.metadata['level'] for doc in vectorstore.docstore.values())
    return retrieve_level(max_level)


# Create Retriever

def create_retriever(vectorstore: FAISS) -> ContextualCompressionRetriever:
    """Create a retriever with contextual compression."""
    logging.info("Creating contextual compression retriever")
    base_retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        "Given the following context and question, extract only the relevant information for answering the question:\n\n"
        "Context: {context}\n"
        "Question: {question}\n\n"
        "Relevant Information:"
    )

    extractor = LLMChainExtractor.from_llm(llm, prompt=prompt)

    return ContextualCompressionRetriever(
        base_compressor=extractor,
        base_retriever=base_retriever
    )


# Define hierarchical retrieval
def hierarchical_retrieval(query: str, retriever: ContextualCompressionRetriever, max_level: int) -> List[Document]:
    """Perform hierarchical retrieval starting from the highest level, handling potential None values."""
    all_retrieved_docs = []

    for level in range(max_level, -1, -1):
        # Retrieve documents from the current level
        level_docs = retriever.get_relevant_documents(
            query,
            filter=lambda meta: meta['level'] == level
        )
        all_retrieved_docs.extend(level_docs)

        # If we've found documents, retrieve their children from the next level down
        if level_docs and level > 0:
            child_ids = [doc.metadata.get('child_ids', []) for doc in level_docs]
            child_ids = [item for sublist in child_ids for item in sublist if
                         item is not None]  # Flatten and filter None

            if child_ids:  # Only modify query if there are valid child IDs
                child_query = f" AND id:({' OR '.join(str(id) for id in child_ids)})"
                query += child_query

    return all_retrieved_docs


# RAPTOR Query Process (Online Process)
def raptor_query(query: str, retriever: ContextualCompressionRetriever, max_level: int) -> Dict[str, Any]:
    """Process a query using the RAPTOR system with hierarchical retrieval."""
    logging.info(f"Processing query: {query}")

    relevant_docs = hierarchical_retrieval(query, retriever, max_level)

    doc_details = []
    for i, doc in enumerate(relevant_docs, 1):
        doc_details.append({
            "index": i,
            "content": doc.page_content,
            "metadata": doc.metadata,
            "level": doc.metadata.get('level', 'Unknown'),
            "similarity_score": doc.metadata.get('score', 'N/A')
        })

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = ChatPromptTemplate.from_template(
        "Given the following context, please answer the question:\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run(context=context, question=query)

    logging.info("Query processing completed")

    result = {
        "query": query,
        "retrieved_documents": doc_details,
        "num_docs_retrieved": len(relevant_docs),
        "context_used": context,
        "answer": answer,
        "model_used": llm.model_name,
    }

    return result


def print_query_details(result: Dict[str, Any]):
    """Print detailed information about the query process, including tree level metadata."""
    print(f"Query: {result['query']}")
    print(f"\nNumber of documents retrieved: {result['num_docs_retrieved']}")
    print(f"\nRetrieved Documents:")
    for doc in result['retrieved_documents']:
        print(f"  Document {doc['index']}:")
        print(f"    Content: {doc['content'][:100]}...")  # Show first 100 characters
        print(f"    Similarity Score: {doc['similarity_score']}")
        print(f"    Tree Level: {doc['metadata'].get('level', 'Unknown')}")
        print(f"    Origin: {doc['metadata'].get('origin', 'Unknown')}")
        if 'child_docs' in doc['metadata']:
            print(f"    Number of Child Documents: {len(doc['metadata']['child_docs'])}")
        print()

    print(f"\nContext used for answer generation:")
    print(result['context_used'])

    print(f"\nGenerated Answer:")
    print(result['answer'])

    print(f"\nModel Used: {result['model_used']}")


# ## Example Usage and Visualization
# 

# ## Define data folder


path = "../data/Understanding_Climate_Change.pdf"

# Process texts
loader = PyPDFLoader(path)
documents = loader.load()
texts = [doc.page_content for doc in documents]

# Create RAPTOR components instances
# Build the RAPTOR tree
tree_results = build_raptor_tree(texts)

# Build vectorstore
vectorstore = build_vectorstore(tree_results)

# Create retriever
retriever = create_retriever(vectorstore)

# Run a query and observe where it got the data from + results
# Run the pipeline
max_level = 3  # Adjust based on your tree depth
query = "What is the greenhouse effect?"
result = raptor_query(query, retriever, max_level)
print_query_details(result)
