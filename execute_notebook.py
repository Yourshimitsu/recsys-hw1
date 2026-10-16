from pathlib import Path
import os
import sys
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent


def execute():
    for key, value in {'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'OMP_NUM_THREADS': '4', 'NUMBA_NUM_THREADS': '4'}.items():
        os.environ.setdefault(key, value)
    notebook_path = ROOT / 'experiments.ipynb'
    notebook = nbformat.read(notebook_path, as_version=4)
    def cell_start(cell, cell_index, **kwargs):
        if cell.cell_type == 'code':
            print(f'Executing notebook cell {cell_index+1}/{len(notebook.cells)}', flush=True)
    client = NotebookClient(notebook, timeout=7200, kernel_name='python3',
                            resources={'metadata': {'path': str(ROOT)}},
                            on_cell_start=cell_start)
    from jupyter_client import KernelManager
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client.km = manager
    try:
        client.execute()
    finally:
        nbformat.write(notebook, notebook_path)
    print('Notebook executed and saved.', flush=True)


if __name__ == '__main__':
    execute()
