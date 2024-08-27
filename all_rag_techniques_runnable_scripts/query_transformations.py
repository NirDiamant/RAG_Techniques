from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# 1 - Query Rewriting: Reformulating queries to improve retrieval.
re_write_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

# Create a prompt template for query rewriting
query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

Original query: {original_query}

Rewritten query:"""

query_rewrite_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=query_rewrite_template
)

# Create an LLMChain for query rewriting
query_rewriter = query_rewrite_prompt | re_write_llm


def rewrite_query(original_query):
    """
    Rewrite the original query to improve retrieval.
    
    Args:
    original_query (str): The original user query
    
    Returns:
    str: The rewritten query
    """
    response = query_rewriter.invoke(original_query)
    return response.content


# Demonstrate on a use case
# example query over the understanding climate change dataset
original_query = "What are the impacts of climate change on the environment?"
rewritten_query = rewrite_query(original_query)
print("Original query:", original_query)
print("\nRewritten query:", rewritten_query)

# 2 - Step-back Prompting: Generating broader queries for better context retrieval.


step_back_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

# Create a prompt template for step-back prompting
step_back_template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.

Original query: {original_query}

Step-back query:"""

step_back_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=step_back_template
)

# Create an LLMChain for step-back prompting
step_back_chain = step_back_prompt | step_back_llm


def generate_step_back_query(original_query):
    """
    Generate a step-back query to retrieve broader context.
    
    Args:
    original_query (str): The original user query
    
    Returns:
    str: The step-back query
    """
    response = step_back_chain.invoke(original_query)
    return response.content


# Demonstrate on a use case
# example query over the understanding climate change dataset
original_query = "What are the impacts of climate change on the environment?"
step_back_query = generate_step_back_query(original_query)
print("Original query:", original_query)
print("\nStep-back query:", step_back_query)

# 3- Sub-query Decomposition: Breaking complex queries into simpler sub-queries.
sub_query_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

# Create a prompt template for sub-query decomposition
subquery_decomposition_template = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

Original query: {original_query}

example: What are the impacts of climate change on the environment?

Sub-queries:
1. What are the impacts of climate change on biodiversity?
2. How does climate change affect the oceans?
3. What are the effects of climate change on agriculture?
4. What are the impacts of climate change on human health?"""

subquery_decomposition_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=subquery_decomposition_template
)

# Create an LLMChain for sub-query decomposition
subquery_decomposer_chain = subquery_decomposition_prompt | sub_query_llm


def decompose_query(original_query: str):
    """
    Decompose the original query into simpler sub-queries.
    
    Args:
    original_query (str): The original complex query
    
    Returns:
    List[str]: A list of simpler sub-queries
    """
    response = subquery_decomposer_chain.invoke(original_query).content
    sub_queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('Sub-queries:')]
    return sub_queries


# Demonstrate on a use case
# example query over the understanding climate change dataset
original_query = "What are the impacts of climate change on the environment?"
sub_queries = decompose_query(original_query)
print("\nSub-queries:")
for i, sub_query in enumerate(sub_queries, 1):
    print(sub_query)
