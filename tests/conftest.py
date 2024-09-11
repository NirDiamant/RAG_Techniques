import pytest
import os
import sys

# Add the main folder to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../"))

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