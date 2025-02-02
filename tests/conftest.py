import pytest
import os
import sys
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

# Add the main folder to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../"))

# Load environment variables
load_dotenv()

def pytest_addoption(parser):
    parser.addoption(
        "--exclude", action="store", help="Comma-separated list of notebook or script files' paths to exclude"
    )

@pytest.fixture
def notebook_paths(request):
    exclude = request.config.getoption("--exclude")
    folder = 'all_rag_techniques/'
    notebook_paths = os.listdir(folder)

    if exclude:
        exclude_notebooks = set(s for s in exclude.split(',') if s.endswith('.ipynb'))
        include_notebooks = [n for n in notebook_paths if n not in  exclude_notebooks]
    else:
        include_notebooks = notebook_paths
        
    path_with_full_address = [folder + n for n in include_notebooks]
    
    return path_with_full_address

@pytest.fixture
def script_paths(request):
    exclude = request.config.getoption("--exclude")
    folder = 'all_rag_techniques_runnable_scripts/'
    script_paths = os.listdir(folder)

    if exclude:
        exclude_scripts = set(s for s in exclude.split(',') if s.endswith('.py'))
        include_scripts = [n for n in script_paths if n not in  exclude_scripts]
    else:
        include_scripts = script_paths
    
    path_with_full_address = [folder + s for s in include_scripts]
    
    return path_with_full_address

@pytest.fixture(scope="session")
def llm():
    """Fixture for ChatOpenAI model."""
    return ChatOpenAI(
        temperature=0,
        model_name="gpt-4-turbo-preview",
        max_tokens=4000
    )

@pytest.fixture(scope="session")
def embeddings():
    """Fixture for OpenAI embeddings."""
    return OpenAIEmbeddings()

@pytest.fixture(scope="session")
def text_splitter():
    """Fixture for text splitter."""
    return CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

@pytest.fixture(scope="session")
def sample_texts():
    """Fixture for sample test data."""
    return [
        "The Earth is the third planet from the Sun.",
        "Climate change is a significant global challenge.",
        "Renewable energy sources include solar and wind power."
    ]

@pytest.fixture(scope="session")
def vector_store(embeddings, sample_texts, text_splitter):
    """Fixture for vector store."""
    docs = text_splitter.create_documents(sample_texts)
    return FAISS.from_documents(docs, embeddings)

@pytest.fixture(scope="session")
def retriever(vector_store):
    """Fixture for retriever."""
    return vector_store.as_retriever(search_kwargs={"k": 2})

@pytest.fixture(scope="session")
def basic_prompt():
    """Fixture for basic prompt template."""
    return PromptTemplate.from_template("""
    Answer the following question based on the context provided:
    
    Context: {context}
    Question: {question}
    
    Answer:
    """)