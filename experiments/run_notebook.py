#!/usr/bin/env python3
"""
Execute a Jupyter notebook's code cells headless — no Jupyter kernel required.

Used by `make figures` to regenerate the paper figures from the plotter
notebooks in a fresh environment, where no `python3` kernelspec is registered.

Approach (deliberately simple): read the .ipynb JSON, concatenate the code
cells, strip IPython magics (lines starting with % or !) and bare display()
calls, then exec the result. We chdir into the notebook's own folder first so
that relative reads (e.g. modules.csv) and savefig() outputs resolve exactly as
they would when the notebook is opened in Jupyter.

Usage:
    python experiments/run_notebook.py experiments/notebooks/Simulate.ipynb
"""

import json
import os
import sys


def load_code_source(nb_path):
    """Return the notebook's code cells concatenated into one Python string."""
    with open(nb_path) as f:
        notebook = json.load(f)

    lines = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for line in cell.get("source", []):
            stripped = line.lstrip()
            # IPython magics and shell escapes are not valid Python — skip them.
            if stripped.startswith("%") or stripped.startswith("!"):
                continue
            lines.append(line.rstrip("\n"))
        lines.append("")  # blank line marks the cell boundary
    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        print("usage: run_notebook.py <notebook.ipynb>")
        sys.exit(2)

    nb_path = os.path.abspath(sys.argv[1])
    nb_dir = os.path.dirname(nb_path)

    # Run as if the notebook were opened in its own folder.
    os.chdir(nb_dir)
    sys.path.insert(0, nb_dir)

    source = load_code_source(nb_path)
    namespace = {
        "__name__": "__main__",
        "__file__": nb_path,
        "get_ipython": lambda: None,    # some cells guard on this
        "display": lambda *args, **kwargs: None,
    }
    exec(compile(source, nb_path, "exec"), namespace)
    print(f"[run_notebook] OK: {nb_path}")


if __name__ == "__main__":
    main()
