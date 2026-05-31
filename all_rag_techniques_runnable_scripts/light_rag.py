import os
import sys
import time
import argparse
from typing import List
from collections import defaultdict

import networkx as nx
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field

from langchain_openai import (
    ChatOpenAI, OpenAIEmbeddings,
    AzureChatOpenAI, AzureOpenAIEmbeddings,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# Add the parent directory to the path since we work with notebooks
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Load environment variables from a .env file (OpenAI or Azure OpenAI credentials)
load_dotenv()


# ----------------------------- Extraction schema -----------------------------
class Entity(BaseModel):
    name: str = Field(description="Canonical name of the entity, capitalized so it can be merged across chunks")
    type: str = Field(description="Entity type, e.g. concept, organization, person, location, event")
    description: str = Field(description="Concise description of the entity based ONLY on the text")


class Relationship(BaseModel):
    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    description: str = Field(description="How the two entities are related, based on the text")
    keywords: str = Field(description="High-level keywords summarizing the nature of the relationship")


class GraphExtraction(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)


class QueryKeywords(BaseModel):
    low_level: List[str] = Field(description="Specific entities, names, or concrete details in the question")
    high_level: List[str] = Field(description="Broad themes or overarching topics the question is about")


EXTRACTION_PROMPT = """You are an information-extraction system building a knowledge graph.

From the text below extract:
- entities: the key named things/concepts. For each give a name, a type, and a short description grounded in the text.
- relationships: meaningful connections between two extracted entities. For each give source, target, a description of the relation, and high-level keywords.

Rules:
- Only use information present in the text.
- Keep entity names consistent and capitalized so the same entity can be merged across chunks.
- A relationship's source and target must be entities you also list in `entities`.

Text:
\"\"\"{text}\"\"\""""

KEYWORD_PROMPT = """Extract two kinds of keywords from the user question to drive a dual-level graph retrieval.
- low_level: specific entities / concrete details.
- high_level: broad themes / overarching topics.

Question: {question}"""

ANSWER_PROMPT = """Answer the question using ONLY the context below. The context comes from a knowledge graph
(entities and their relationships) plus the original source passages.

### Entities
{entities}

### Relationships
{relations}

### Source passages
{chunks}

Question: {question}

Answer:"""


def _norm(name: str) -> str:
    """Normalize an entity name so the same entity merges across chunks."""
    return name.strip().lower()


class LightRAG:
    """
    A from-scratch implementation of LightRAG: graph-based indexing plus dual-level
    (entity + relationship) retrieval over a corpus.
    """

    def __init__(self, path, chunk_size=1200, chunk_overlap=150, max_chunks=20,
                 model="gpt-4o-mini", embedding_model="text-embedding-3-small", azure=False):
        if azure:
            # Azure reads the AZURE_* deployment variables from the environment.
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_deployment=os.getenv("GPT4O_MODEL_NAME"),
                temperature=0,
            )
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_deployment=os.getenv("TEXT_EMBEDDING_3_LARGE_DEPLOYMENT_NAME"),
            )
        else:
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
            self.llm = ChatOpenAI(model=model, temperature=0)
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.extraction_chain = (
            ChatPromptTemplate.from_template(EXTRACTION_PROMPT)
            | self.llm.with_structured_output(GraphExtraction)
        )
        self.keyword_chain = (
            ChatPromptTemplate.from_template(KEYWORD_PROMPT)
            | self.llm.with_structured_output(QueryKeywords)
        )
        self.answer_chain = ChatPromptTemplate.from_template(ANSWER_PROMPT) | self.llm

        self.graph = nx.Graph()
        self.entity_chunks = defaultdict(set)
        self.time_records = {}

        self._load_and_chunk(path, chunk_size, chunk_overlap, max_chunks)
        self._build_graph()
        self._build_indexes()

    def _load_and_chunk(self, path, chunk_size, chunk_overlap, max_chunks):
        print("\n--- Loading and chunking document ---")
        documents = PyPDFLoader(path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunks = splitter.split_documents(documents)[:max_chunks]
        for i, chunk in enumerate(self.chunks):
            chunk.metadata["chunk_id"] = i
        print(f"{len(self.chunks)} chunks ready for graph extraction")

    def _build_graph(self):
        print("\n--- Extracting entities and relationships (one LLM call per chunk) ---")
        start = time.time()
        for chunk in self.chunks:
            cid = chunk.metadata["chunk_id"]
            try:
                extraction = self.extraction_chain.invoke({"text": chunk.page_content})
            except Exception as e:
                print(f"  skipped chunk {cid}: {e}")
                continue

            for ent in extraction.entities:
                key = _norm(ent.name)
                if not key:
                    continue
                if self.graph.has_node(key):
                    self.graph.nodes[key]["descriptions"].append(ent.description)
                else:
                    self.graph.add_node(key, name=ent.name, type=ent.type, descriptions=[ent.description])
                self.entity_chunks[key].add(cid)

            for rel in extraction.relationships:
                s, t = _norm(rel.source), _norm(rel.target)
                if not s or not t or s == t:
                    continue
                for key, raw in [(s, rel.source), (t, rel.target)]:
                    if not self.graph.has_node(key):
                        self.graph.add_node(key, name=raw, type="unknown", descriptions=[])
                    self.entity_chunks[key].add(cid)
                if self.graph.has_edge(s, t):
                    self.graph.edges[s, t]["descriptions"].append(rel.description)
                    self.graph.edges[s, t]["keywords"].append(rel.keywords)
                    self.graph.edges[s, t]["chunk_ids"].add(cid)
                else:
                    self.graph.add_edge(s, t, descriptions=[rel.description],
                                        keywords=[rel.keywords], chunk_ids={cid})

        # Collapse merged descriptions into a single profile string.
        for _, data in self.graph.nodes(data=True):
            data["description"] = " ".join(dict.fromkeys(data["descriptions"]))
        for _, _, data in self.graph.edges(data=True):
            data["description"] = " ".join(dict.fromkeys(data["descriptions"]))
            data["keywords"] = ", ".join(dict.fromkeys(data["keywords"]))

        self.time_records["Graph build"] = time.time() - start
        print(f"Knowledge graph: {self.graph.number_of_nodes()} entities, "
              f"{self.graph.number_of_edges()} relationships "
              f"({self.time_records['Graph build']:.2f}s)")

    def _build_indexes(self):
        print("\n--- Building dual-level vector indexes ---")
        entity_docs = [
            Document(page_content=f"{d['name']} ({d['type']}): {d['description']}", metadata={"entity": n})
            for n, d in self.graph.nodes(data=True)
        ]
        relation_docs = [
            Document(
                page_content=(f"{self.graph.nodes[u]['name']} -- {self.graph.nodes[v]['name']} | "
                              f"keywords: {d['keywords']} | {d['description']}"),
                metadata={"source": u, "target": v},
            )
            for u, v, d in self.graph.edges(data=True)
        ]
        self.entity_index = FAISS.from_documents(entity_docs, self.embeddings)
        self.relation_index = FAISS.from_documents(relation_docs, self.embeddings)
        print(f"Indexed {len(entity_docs)} entities and {len(relation_docs)} relations")

    def retrieve(self, question, k_entities=4, k_relations=4):
        kws = self.keyword_chain.invoke({"question": question})
        low_query = ", ".join(kws.low_level) or question
        high_query = ", ".join(kws.high_level) or question

        selected_entities, selected_edges, relevant_chunks = set(), set(), set()

        # LOW LEVEL: specific entities + their direct relations.
        for hit in self.entity_index.similarity_search(low_query, k=k_entities):
            e = hit.metadata["entity"]
            if not self.graph.has_node(e):
                continue
            selected_entities.add(e)
            relevant_chunks |= self.entity_chunks.get(e, set())
            for neighbour in self.graph.neighbors(e):
                selected_edges.add(tuple(sorted((e, neighbour))))

        # HIGH LEVEL: thematic relations + their endpoint entities.
        for hit in self.relation_index.similarity_search(high_query, k=k_relations):
            s, t = hit.metadata["source"], hit.metadata["target"]
            if not self.graph.has_edge(s, t):
                continue
            selected_edges.add(tuple(sorted((s, t))))
            selected_entities.update([s, t])
            relevant_chunks |= self.graph.edges[s, t].get("chunk_ids", set())

        entity_block = "\n".join(
            f"- {self.graph.nodes[e]['name']} ({self.graph.nodes[e]['type']}): {self.graph.nodes[e]['description']}"
            for e in selected_entities if self.graph.has_node(e)
        )
        relation_block = "\n".join(
            f"- {self.graph.nodes[u]['name']} -- {self.graph.nodes[v]['name']}: {self.graph.edges[u, v]['description']}"
            for u, v in selected_edges if self.graph.has_edge(u, v)
        )
        chunk_block = "\n---\n".join(self.chunks[c].page_content for c in sorted(relevant_chunks))
        return {"keywords": kws, "entities": entity_block, "relations": relation_block, "chunks": chunk_block}

    def run(self, query, k_entities=4, k_relations=4):
        print("\n--- Dual-level retrieval ---")
        start = time.time()
        ctx = self.retrieve(query, k_entities=k_entities, k_relations=k_relations)
        self.time_records["Retrieval"] = time.time() - start
        print(f"Low-level keywords:  {ctx['keywords'].low_level}")
        print(f"High-level keywords: {ctx['keywords'].high_level}")
        print(f"Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")

        response = self.answer_chain.invoke({
            "entities": ctx["entities"],
            "relations": ctx["relations"],
            "chunks": ctx["chunks"],
            "question": query,
        })
        print("\n=== Answer ===\n", response.content)
        return response.content


def validate_args(args):
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")
    if args.max_chunks <= 0:
        raise ValueError("max_chunks must be a positive integer.")
    if args.k_entities <= 0 or args.k_relations <= 0:
        raise ValueError("k_entities and k_relations must be positive integers.")
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Build a LightRAG graph from a PDF and answer a query.")
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to encode.")
    parser.add_argument("--chunk_size", type=int, default=1200,
                        help="Size of each text chunk (default: 1200).")
    parser.add_argument("--chunk_overlap", type=int, default=150,
                        help="Overlap between consecutive chunks (default: 150).")
    parser.add_argument("--max_chunks", type=int, default=20,
                        help="Max number of chunks to index, to keep the demo fast/cheap (default: 20).")
    parser.add_argument("--k_entities", type=int, default=4,
                        help="Number of entities to retrieve at the low level (default: 4).")
    parser.add_argument("--k_relations", type=int, default=4,
                        help="Number of relations to retrieve at the high level (default: 4).")
    parser.add_argument("--query", type=str,
                        default="How do greenhouse gas emissions contribute to climate change?",
                        help="Query to answer.")
    parser.add_argument("--azure", action="store_true",
                        help="Use Azure OpenAI (reads AZURE_* env vars) instead of plain OpenAI.")
    return validate_args(parser.parse_args())


def main(args):
    light_rag = LightRAG(
        path=args.path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_chunks=args.max_chunks,
        azure=args.azure,
    )
    light_rag.run(args.query, k_entities=args.k_entities, k_relations=args.k_relations)


if __name__ == '__main__':
    main(parse_args())
