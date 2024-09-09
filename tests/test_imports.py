import nbformat
import sys
import os
import re
import pytest

# Add the parent directory to the path to not fail on relative imports 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) 

# Function to extract and execute import statements from a notebook
def execute_imports_from_notebook(notebook_path) -> None:
    # Assert that the file exists
    assert os.path.exists(notebook_path), f"File not found: {notebook_path}"
    # Assert that the file is not empty
    assert os.stat(notebook_path).st_size > 0, f"Notebook is empty: {notebook_path}"
    
    # Try to load the notebook
    try:
        with open(notebook_path, 'r') as f:
            notebook = nbformat.read(f, as_version=4)
    except nbformat.reader.NotJSONError as e:
        pytest.fail(f"Error reading notebook: {e}")
    
    # Regular expression to match import lines
    import_pattern = re.compile(r'^\s*(import\s+[\w.]+|from\s+[\w.]+\s+import\s+(\([\w\s,\n()]+\)|[\w\s,]+))')
    
    errors = []
    # Extract and execute import statements
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            lines = cell['source'].split('\n')
            for line_num, line in enumerate(lines):
                if import_pattern.match(line):
                    try:
                        exec(line)
                    except Exception as e:
                        error_message = (
                            f"Notebook: {notebook_path}, Cell: {i+1}, Line: {line_num+1} - "
                            f"Failed to execute import: {line}\n"
                            f"Exception: {e}\n"
                        )
                        errors.append(error_message)
    return errors

def execute_imports_from_script_files(script_path) -> None:
    # Assert that the file exists
    assert os.path.exists(script_path), f"File not found: {script_path}"
    # Assert that the file is not empty
    assert os.stat(script_path).st_size > 0, f"Script is empty: {script_path}"
    
    # Try to load the script file
    try:
        with open(script_path, 'r') as f:
            script_lines = [line.strip() for line in f.readlines()]
    except nbformat.reader.NotJSONError as e:
        pytest.fail(f"Error reading script: {e}")
    
    # Regular expression to match import lines
    import_pattern = re.compile(r'^\s*(import\s+[\w.]+|from\s+[\w.]+\s+import\s+(\([\w\s,\n()]+\)|[\w\s,]+))')
    
    errors = []
    # Extract and execute import statements
    for line_num, line in enumerate(script_lines):
        if import_pattern.match(line):
            try:
                exec(line)
            except Exception as e:
                error_message = (
                    f"Script: {script_path}, Line: {line_num+1} - "
                    f"Failed to execute import: {line}\n"
                    f"Exception: {e}\n"
                )
                errors.append(error_message)
    return errors

def test_notebook_imports(notebook_paths):
    all_errors = []
    
    for path in notebook_paths:
        errors = execute_imports_from_notebook(path)
        if errors:
            all_errors.extend(errors)
    
    if all_errors:
        pytest.fail("\n".join(all_errors))
        
def test_script_imports(script_paths):
    all_errors = []
    
    for path in script_paths:
        errors = execute_imports_from_script_files(path)
        if errors:
            all_errors.extend(errors)
    
    if all_errors:
        pytest.fail("\n".join(all_errors))