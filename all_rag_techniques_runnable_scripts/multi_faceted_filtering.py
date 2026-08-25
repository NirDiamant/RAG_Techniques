import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from helper_functions import text_wrap

# Load environment variables from a .env file
load_dotenv()


def load_customer_documents(csv_path: str) -> List[Document]:
    """Convert each row of the customer dataset into a Document with rich metadata."""
    docs = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            content = (
                f"{row['First Name']} {row['Last Name']} from {row['Company']} "
                f"in {row['City']}, {row['Country']}. "
                f"Subscribed on {row['Subscription Date']}. "
                f"Website: {row['Website']}"
            )
            docs.append(Document(
                page_content=content,
                metadata={
                    "country": row["Country"],
                    "company": row["Company"],
                    "city": row["City"],
                    "subscription_date": row["Subscription Date"],
                    "website": row["Website"],
                },
            ))
    return docs


def apply_metadata_filter(
    doc_score_pairs: List[Tuple[Document, float]],
    filters: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Document, float]]:
    """Keep only documents whose metadata matches the given key-value filters."""
    if not filters:
        return doc_score_pairs
    return [
        (doc, score) for doc, score in doc_score_pairs
        if all(doc.metadata.get(key) == value for key, value in filters.items())
    ]


def apply_similarity_threshold(
    doc_score_pairs: List[Tuple[Document, float]],
    threshold: float = 0.35,
) -> List[Tuple[Document, float]]:
    """Keep only documents whose relevance score is above the threshold."""
    return [(doc, score) for doc, score in doc_score_pairs if score >= threshold]


def apply_content_filter(
    doc_score_pairs: List[Tuple[Document, float]],
    keywords: Optional[List[str]] = None,
    require_all: bool = True,
) -> List[Tuple[Document, float]]:
    """Keep only documents whose content contains the required keywords.

    With ``require_all=True`` every keyword must appear in the document;
    otherwise a single match is enough.
    """
    if not keywords:
        return doc_score_pairs
    keywords_lower = [k.lower() for k in keywords]
    filtered = []
    for doc, score in doc_score_pairs:
        content = doc.page_content.lower()
        matches = [k in content for k in keywords_lower]
        if (all(matches) if require_all else any(matches)):
            filtered.append((doc, score))
    return filtered


def apply_diversity_filter(
    doc_score_pairs: List[Tuple[Document, float]],
    embeddings: HuggingFaceEmbeddings,
    similarity_threshold: float = 0.9,
) -> List[Tuple[Document, float]]:
    """Remove near-duplicate documents greedily using embedding cosine similarity."""
    kept: List[Tuple[Document, float]] = []
    kept_vectors: List[np.ndarray] = []
    for doc, score in doc_score_pairs:
        vector = np.asarray(embeddings.embed_query(doc.page_content))
        is_duplicate = any(
            float(np.dot(vector, other) / (np.linalg.norm(vector) * np.linalg.norm(other)))
            >= similarity_threshold
            for other in kept_vectors
        )
        if not is_duplicate:
            kept.append((doc, score))
            kept_vectors.append(vector)
    return kept


def multi_faceted_filter(
    doc_score_pairs: List[Tuple[Document, float]],
    metadata_filters: Optional[Dict[str, Any]] = None,
    score_threshold: Optional[float] = None,
    required_keywords: Optional[List[str]] = None,
    diversity_threshold: Optional[float] = None,
    embeddings: Optional[HuggingFaceEmbeddings] = None,
) -> List[Tuple[Document, float]]:
    """Apply metadata, similarity, content and diversity filters in sequence."""
    filtered = doc_score_pairs
    if metadata_filters:
        filtered = apply_metadata_filter(filtered, metadata_filters)
    if score_threshold is not None:
        filtered = apply_similarity_threshold(filtered, score_threshold)
    if required_keywords:
        filtered = apply_content_filter(filtered, required_keywords)
    if diversity_threshold is not None:
        filtered = apply_diversity_filter(filtered, embeddings, diversity_threshold)
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Run the multi-faceted filtering pipeline over the customer dataset."
    )
    parser.add_argument(
        "--query", type=str, default="Which customers work for companies in Chile?",
        help="Query to retrieve documents for.",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of documents to retrieve.")
    parser.add_argument(
        "--country", type=str, default=None,
        help="Metadata filter: keep only documents from this country.",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=None,
        help="Drop documents with a relevance score below this threshold.",
    )
    parser.add_argument(
        "--keywords", type=str, nargs="+", default=None,
        help="Content filter: keep only documents containing these keywords.",
    )
    parser.add_argument(
        "--diversity-threshold", type=float, default=None,
        help="Remove documents whose cosine similarity to a kept document is above this threshold.",
    )
    parser.add_argument(
        "--data-path", type=str, default="data/customers-100.csv",
        help="Path to the customer dataset (relative to the repository root).",
    )
    args = parser.parse_args()

    # Build the vector store over the customer dataset
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    customers = load_customer_documents(args.data_path)
    vectorstore = Chroma.from_documents(customers, embedding=embeddings)
    print(f"Indexed {len(customers)} customer documents\n")

    # Retrieve candidates and apply the filtering pipeline
    results = vectorstore.similarity_search_with_relevance_scores(args.query, k=args.k)

    metadata_filters = {"country": args.country} if args.country else None
    filtered = multi_faceted_filter(
        results,
        metadata_filters=metadata_filters,
        score_threshold=args.score_threshold,
        required_keywords=args.keywords,
        diversity_threshold=args.diversity_threshold,
        embeddings=embeddings,
    )

    print(f"Query: {args.query}")
    print(f"Before filtering: {len(results)} documents")
    print(f"After filtering:  {len(filtered)} documents\n")
    for doc, score in filtered:
        print(f"{score:.3f} | {doc.metadata['country']:<10} | {doc.metadata['company']}")
        print(text_wrap(doc.page_content, width=120))
        print()


if __name__ == "__main__":
    main()
