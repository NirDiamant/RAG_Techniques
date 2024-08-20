[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/nir-diamant-759323134/)
[![Twitter](https://img.shields.io/twitter/follow/NirDiamantAI?label=Follow%20@NirDiamantAI&style=social)](https://twitter.com/NirDiamantAI)
<a href="https://app.commanddash.io/agent/github_NirDiamant_RAG_Techniques"><img src="https://img.shields.io/badge/AI-Code%20Agent-EB9FDA"></a>

# Advanced RAG Techniques: Elevating Your Retrieval-Augmented Generation Systems 🚀

Welcome to one of the most comprehensive and dynamic collections of Retrieval-Augmented Generation (RAG) tutorials available today. This repository serves as a hub for cutting-edge techniques aimed at enhancing the accuracy, efficiency, and contextual richness of RAG systems.

## Introduction

Retrieval-Augmented Generation (RAG) is revolutionizing the way we combine information retrieval with generative AI. This repository showcases a curated collection of advanced techniques designed to supercharge your RAG systems, enabling them to deliver more accurate, contextually relevant, and comprehensive responses.

Our goal is to provide a valuable resource for researchers and practitioners looking to push the boundaries of what's possible with RAG. By fostering a collaborative environment, we aim to accelerate innovation in this exciting field.

## A Community-Driven Knowledge Hub

This repository thrives on community contributions! Join our Slack community — the central hub for discussing and managing contributions to this project:

[RAG Techniques Slack Community](https://join.slack.com/t/ragtechniques/shared_invite/zt-2oycgwuvp-5WveAcpJtSrkUSyDi8VD1Q)

Whether you're an expert or just starting out, your insights can shape the future of RAG. Join us to propose ideas, get feedback, and collaborate on innovative techniques. For contribution guidelines, please refer to our [CONTRIBUTING.md](https://github.com/NirDiamant/RAG_Techniques/blob/main/CONTRIBUTING.md) file. Let's advance RAG technology together!

🔗 For discussions on GenAI, RAG, or custom agents, or to explore knowledge-sharing opportunities, feel free to [connect on LinkedIn](https://www.linkedin.com/in/nir-diamant-759323134/).

## Key Features

- 🧠 State-of-the-art RAG enhancements
- 📚 Comprehensive documentation for each technique
- 🛠️ Practical implementation guidelines
- 🌟 Regular updates with the latest advancements

## Advanced Techniques

Explore the extensive list of cutting-edge RAG techniques:

### 1. [Simple RAG 🌱](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_rag.ipynb)

#### Overview 🔎
Introducing basic RAG techniques ideal for newcomers.

#### Implementation 🛠️
Start with basic retrieval queries and integrate incremental learning mechanisms.

### 2. [Context Enrichment Techniques 📝](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/context_enrichment_window_around_chunk.ipynb)

#### Overview 🔎
Enhancing retrieval accuracy by embedding individual sentences and extending context to neighboring sentences.

#### Implementation 🛠️
Retrieve the most relevant sentence while also accessing the sentences before and after it in the original text.

### 3. Multi-faceted Filtering 🔍

#### Overview 🔎
Applying various filtering techniques to refine and improve the quality of retrieved results.

#### Implementation 🛠️
- 🏷️ **Metadata Filtering:** Apply filters based on attributes like date, source, author, or document type.
- 📊 **Similarity Thresholds:** Set thresholds for relevance scores to keep only the most pertinent results.
- 📄 **Content Filtering:** Remove results that don't match specific content criteria or essential keywords.
- 🌈 **Diversity Filtering:** Ensure result diversity by filtering out near-duplicate entries.

### 4. [Fusion Retrieval 🔗](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/fusion_retrieval.ipynb)

#### Overview 🔎
Optimizing search results by combining different retrieval methods.

#### Implementation 🛠️
Combine keyword-based search with vector-based search for more comprehensive and accurate retrieval.

### 5. [Intelligent Reranking 📈](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/reranking.ipynb)

#### Overview 🔎
Applying advanced scoring mechanisms to improve the relevance ranking of retrieved results.

#### Implementation 🛠️
- 🧠 **LLM-based Scoring:** Use a language model to score the relevance of each retrieved chunk.
- 🔀 **Cross-Encoder Models:** Re-encode both the query and retrieved documents jointly for similarity scoring.
- 🏆 **Metadata-enhanced Ranking:** Incorporate metadata into the scoring process for more nuanced ranking.

### 6.[Query Transformations 🔄](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/query_transformations.ipynb)

#### Overview 🔎
Modifying and expanding queries to improve retrieval effectiveness.

#### Implementation 🛠️
- ✍️ **Query Rewriting:** Reformulate queries to improve retrieval.
- 🔙 **Step-back Prompting:** Generate broader queries for better context retrieval.
- 🧩 **Sub-query Decomposition:** Break complex queries into simpler sub-queries.

### 7. [Hierarchical Indices 🗂️](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/hierarchical_indices.ipynb)

#### Overview 🔎
Creating a multi-tiered system for efficient information navigation and retrieval.

#### Implementation 🛠️
Implement a two-tiered system for document summaries and detailed chunks, both containing metadata pointing to the same location in the data.

### 8. [Hypothetical Questions (HyDE Approach) ❓](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/HyDe_Hypothetical_Document_Embedding.ipynb)

#### Overview 🔎
Generating hypothetical questions to improve alignment between queries and data.

#### Implementation 🛠️
Create hypothetical questions that point to relevant locations in the data, enhancing query-data matching.

### 9. [Choose Chunk Size 📏](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/choose_chunk_size.ipynb)

#### Overview 🔎
Selecting an appropriate fixed size for text chunks to balance context preservation and retrieval efficiency.

#### Implementation 🛠️
Experiment with different chunk sizes to find the optimal balance between preserving context and maintaining retrieval speed for your specific use case.

### 10. [Semantic Chunking 🧠](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/semantic_chunking.ipynb)

#### Overview 🔎
Dividing documents based on semantic coherence rather than fixed sizes.

#### Implementation 🛠️
Use NLP techniques to identify topic boundaries or coherent sections within documents for more meaningful retrieval units.

### 11. [Contextual Compression 🗜️](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/contextual_compression.ipynb)

#### Overview 🔎
Compressing retrieved information while preserving query-relevant content.

#### Implementation 🛠️
Use an LLM to compress or summarize retrieved chunks, preserving key information relevant to the query.

### 12. [Explainable Retrieval 🔍](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/explainable_retrieval.ipynb)

#### Overview 🔎
Providing transparency in the retrieval process to enhance user trust and system refinement.

#### Implementation 🛠️
Explain why certain pieces of information were retrieved and how they relate to the query.

### 13. [Retrieval with Feedback Loops 🔁](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/retrieval_with_feedback_loop.ipynb)

#### Overview 🔎
Implementing mechanisms to learn from user interactions and improve future retrievals.

#### Implementation 🛠️
Collect and utilize user feedback on the relevance and quality of retrieved documents and generated responses to fine-tune retrieval and ranking models.

### 14. [Adaptive Retrieval 🎯](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/adaptive_retrieval.ipynb)

#### Overview 🔎
Dynamically adjusting retrieval strategies based on query types and user contexts.

#### Implementation 🛠️
Classify queries into different categories and use tailored retrieval strategies for each, considering user context and preferences.

### 15. Iterative Retrieval 🔄

#### Overview 🔎
Performing multiple rounds of retrieval to refine and enhance result quality.

#### Implementation 🛠️
Use the LLM to analyze initial results and generate follow-up queries to fill in gaps or clarify information.

### 16. Ensemble Retrieval 🎭

#### Overview 🔎
Combining multiple retrieval models or techniques for more robust and accurate results.

#### Implementation 🛠️
Apply different embedding models or retrieval algorithms and use voting or weighting mechanisms to determine the final set of retrieved documents.

### 17. [Knowledge Graph Integration (Graph RAG)🕸️](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/graph_rag.ipynb)

#### Overview 🔎
Incorporating structured data from knowledge graphs to enrich context and improve retrieval.

#### Implementation 🛠️
Retrieve entities and their relationships from a knowledge graph relevant to the query, combining this structured data with unstructured text for more informative responses.

### 18. Multi-modal Retrieval 📽️

#### Overview 🔎
Extending RAG capabilities to handle diverse data types for richer responses.

#### Implementation 🛠️
Integrate models that can retrieve and understand different data modalities, combining insights from text, images, and videos.

### 19. [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval 🌳](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/raptor.ipynb)

#### Overview 🔎
Implementing a recursive approach to process and organize retrieved information in a tree structure.

#### Implementation 🛠️
Use abstractive summarization to recursively process and summarize retrieved documents, organizing the information in a tree structure for hierarchical context.

### 20. [Self RAG 🔁](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/self_rag.ipynb)

#### Overview 🔎
A dynamic approach that combines retrieval-based and generation-based methods, adaptively deciding whether to use retrieved information and how to best utilize it in generating responses.

#### Implementation 🛠️
• Implement a multi-step process including retrieval decision, document retrieval, relevance evaluation, response generation, support assessment, and utility evaluation to produce accurate, relevant, and useful outputs.

### 21. [Corrective RAG 🔧](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/crag.ipynb)

#### Overview 🔎
A sophisticated RAG approach that dynamically evaluates and corrects the retrieval process, combining vector databases, web search, and language models for highly accurate and context-aware responses.

#### Implementation 🛠️
• Integrate Retrieval Evaluator, Knowledge Refinement, Web Search Query Rewriter, and Response Generator components to create a system that adapts its information sourcing strategy based on relevance scores and combines multiple sources when necessary.

## 🌟 Special Advanced Technique 🌟

### 22. [Sophisticated Controllable Agent for Complex RAG Tasks 🤖](https://github.com/NirDiamant/Controllable-RAG-Agent)

#### Overview 🔎
An advanced RAG solution designed to tackle complex questions that simple semantic similarity-based retrieval cannot solve. This approach uses a sophisticated deterministic graph as the "brain" 🧠 of a highly controllable autonomous agent, capable of answering non-trivial questions from your own data.

#### Implementation 🛠️
• Implement a multi-step process involving question anonymization, high-level planning, task breakdown, adaptive information retrieval and question answering, continuous re-planning, and rigorous answer verification to ensure grounded and accurate responses.

## Getting Started

To begin implementing these advanced RAG techniques in your projects:

1. Clone this repository:
   ```
   git clone https://github.com/NirDiamant/RAG_Techniques.git
   ```
2. Navigate to the technique you're interested in:
   ```
   cd all_rag_techniques/technique-name
   ```
3. Follow the detailed implementation guide in each technique's directory.

## Contributing

We welcome contributions from the community! If you have a new technique or improvement to suggest:

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

⭐️ If you find this repository helpful, please consider giving it a star!

Keywords: RAG, Retrieval-Augmented Generation, NLP, AI, Machine Learning, Information Retrieval, Natural Language Processing, LLM, Embeddings, Semantic Search