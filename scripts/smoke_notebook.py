"""Execute notebook code cells sequentially for a local smoke check."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "notebooks" / "Full_Code_SuperKart_Model_Deployment_Notebook.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"display": lambda *args, **kwargs: None}
    for cell_number, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        try:
            exec(compile(source, f"notebook_cell_{cell_number}.py", "exec"), namespace)
        except Exception:
            print(f"Notebook code cell {cell_number} failed:")
            traceback.print_exc()
            raise
        print(f"Notebook code cell {cell_number}: OK")
    print("NOTEBOOK_EXECUTION_OK")


if __name__ == "__main__":
    main()
