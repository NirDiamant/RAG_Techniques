import numpy as np
import pandas as pd
from typing import List, Dict, Any

from langchain.chains import LLMChain
from sklearn.mixture import GaussianMixture
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.messages import AIMessage
from langchain_core.documents import Document
import matplotlib.pyplot as plt
import logging
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# Helper functions

def extract_text(item):
    """Extract text content from either a string or an AIMessage object."""
    if isinstance(item, AIMessage):
        return item.content
    return item


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts using OpenAIEmbeddings."""
    embeddings = OpenAIEmbeddings()
    logging.info(f"Embedding {len(texts)} texts")
    return embeddings.embed_documents([extract_text(text) for text in texts])


def perform_clustering(embeddings: np.ndarray, n_clusters: int = 10) -> np.ndarray:
    """Perform clustering on embeddings using Gaussian Mixture Model."""
    logging.info(f"Performing clustering with {n_clusters} clusters")
    gm = GaussianMixture(n_components=n_clusters, random_state=42)
    return gm.fit_predict(embeddings)


def summarize_texts(texts: List[str], llm: ChatOpenAI) -> str:
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


def build_vectorstore(tree_results: Dict[int, pd.DataFrame], embeddings) -> FAISS:
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
    documents = [Document(page_content=str(text), metadata=metadata)
                 for text, metadata in zip(all_texts, all_metadatas)]
    return FAISS.from_documents(documents, embeddings)


def create_retriever(vectorstore: FAISS, llm: ChatOpenAI) -> ContextualCompressionRetriever:
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


# Main class RAPTORMethod
class RAPTORMethod:
    def __init__(self, texts: List[str], max_levels: int = 3):
        self.texts = texts
        self.max_levels = max_levels
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-4o-mini")
        self.tree_results = self.build_raptor_tree()

    def build_raptor_tree(self) -> Dict[int, pd.DataFrame]:
        """Build the RAPTOR tree structure with level metadata and parent-child relationships."""
        results = {}
        current_texts = [extract_text(text) for text in self.texts]
        current_metadata = [{"level": 0, "origin": "original", "parent_id": None} for _ in self.texts]

        for level in range(1, self.max_levels + 1):
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
                summary = summarize_texts(cluster_texts, self.llm)
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

    def run(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Run the RAPTOR query pipeline."""
        vectorstore = build_vectorstore(self.tree_results, self.embeddings)
        retriever = create_retriever(vectorstore, self.llm)

        logging.info(f"Processing query: {query}")
        relevant_docs = retriever.get_relevant_documents(query)

        doc_details = [{"content": doc.page_content, "metadata": doc.metadata} for doc in relevant_docs]

        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = ChatPromptTemplate.from_template(
            "Given the following context, please answer the question:\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(context=context, question=query)

        return {
            "query": query,
            "retrieved_documents": doc_details,
            "context_used": context,
            "answer": answer,
            "model_used": self.llm.model_name,
        }


# Argument Parsing and Validation
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run RAPTORMethod")
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to process.")
    parser.add_argument("--query", type=str, default="What is the greenhouse effect?",
                        help="Query to test the retriever (default: 'What is the main topic of the document?').")
    parser.add_argument('--max_levels', type=int, default=3, help="Max levels for RAPTOR tree")
    return parser.parse_args()


# Main Execution
if __name__ == "__main__":
    args = parse_args()
    loader = PyPDFLoader(args.path)
    documents = loader.load()
    texts = [doc.page_content for doc in documents]

    raptor_method = RAPTORMethod(texts, max_levels=args.max_levels)
    result = raptor_method.run(args.query)

    print(f"Query: {result['query']}")
    print(f"Context Used: {result['context_used']}")
    print(f"Answer: {result['answer']}")
    print(f"Model Used: {result['model_used']}")
