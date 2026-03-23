import ast
from pathlib import Path
from datetime import datetime

src = Path("../src")
lines = ["# API Documentation\n"]

def generate_doc():

    for file in sorted(src.glob("*.py")):
        lines.append(f"\n## `{file.name}`\n")
        tree = ast.parse(file.read_text())

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                args = [a.arg for a in node.args.args]
                doc = ast.get_docstring(node) or ""
                lines.append(f"### `{node.name}({', '.join(args)})`")
                if doc:
                    lines.append(doc.splitlines()[0])
                lines.append("")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    lines.append(f"API.md last updated at {timestamp}.")

    Path("../API.md").write_text("\n".join(lines))
    
    print("API.md  last updated at", timestamp, ".")