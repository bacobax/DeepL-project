import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell

def collect_python_files(root_dir):
    py_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                py_files.append(full_path)
    # Optional: sort for consistent order
    return sorted(py_files)

def read_and_clean_code(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    # Optional cleanup: remove relative imports
    lines = code.splitlines()
    cleaned = [line for line in lines if not line.strip().startswith("from .") and not line.strip().startswith("import .")]
    return "\n".join(cleaned)

def convert_to_notebook(py_files):
    nb = new_notebook()
    for path in py_files:
        code = read_and_clean_code(path)
        nb.cells.append(new_code_cell(f"# From: {path}\n{code}"))
    return nb

def save_notebook(nb, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

# === Main ===
project_root = "./"  # <- Change this
output_ipynb = "./merged_notebook.ipynb"

files = collect_python_files(project_root)
notebook = convert_to_notebook(files)
save_notebook(notebook, output_ipynb)
print(f"Notebook saved to {output_ipynb}")