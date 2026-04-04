import ast
import re
from pathlib import Path
from datetime import datetime


HERE = Path(__file__).resolve().parent  # directory where generate_doc.py lives
src = HERE / "src"

STDLIB = {
    "ast",
    "os",
    "sys",
    "json",
    "re",
    "math",
    "time",
    "datetime",
    "pathlib",
    "collections",
    "itertools",
    "functools",
    "typing",
    "shutil",
    "copy",
    "io",
    "abc",
    "enum",
    "dataclasses",
    "warnings",
}


def build_dag():
    all_files = [f for f in src.rglob("*.py") if f.name != "__init__.py"]
    # stem → filename relative to src, e.g. "math_utils" → "l0/math_utils.py"
    src_modules = {f.stem: str(f.relative_to(src)) for f in all_files}

    def resolve(short):
        if short in src_modules:
            return src_modules[short]
        matches = [
            fname
            for stem, fname in src_modules.items()
            if stem.endswith(f"_{short}") or stem == short
        ]
        return matches[0] if len(matches) == 1 else None

    dag = {}
    ext_deps = {}

    for file in sorted(all_files):
        tree = ast.parse(file.read_text())
        deps = []
        external = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                parts = module.split(".")
                # from src.l0.math_utils → short = "math_utils"
                short = parts[-1] if parts[0] == "src" else parts[0]
                resolved = resolve(short)
                rel = str(file.relative_to(src))
                if resolved and resolved != rel:
                    deps.append(resolved)
                elif short and short not in STDLIB:
                    external.append(short)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    short = parts[-1] if parts[0] == "src" else parts[0]
                    resolved = resolve(short)
                    rel = str(file.relative_to(src))
                    if resolved and resolved != rel:
                        deps.append(resolved)
                    elif parts[0] not in STDLIB:
                        external.append(parts[0])

        rel = str(file.relative_to(src))
        dag[rel] = sorted(set(deps))
        ext_deps[rel] = sorted(set(external))

    return dag, ext_deps


def get_level(fpath):
    m = re.match(r"l(\d+)", Path(fpath).parts[0])
    return int(m.group(1)) if m else -1


def render_dag(dag, ext_deps):
    section = ["## Dependency DAG\n"]

    levels = {}
    for fname in dag:
        lvl = get_level(fname)
        levels.setdefault(lvl, []).append(fname)

    section.append("```mermaid")
    section.append("graph TD")
    section.append("")

    for lvl in sorted(levels.keys(), reverse=True):
        label = f"Layer {lvl}" if lvl >= 0 else "Other"
        section.append(f"    subgraph {label}")
        for f in levels[lvl]:
            node_id = f.replace("/", "_").replace(".", "_")
            short = Path(f).stem  # just "math_utils" not "l0/math_utils.py"
            section.append(f'        {node_id}["{short}"]')
        section.append("    end")
        section.append("")

    for file, deps in sorted(dag.items()):
        src_id = file.replace("/", "_").replace(".", "_")
        for dep in deps:
            dep_id = dep.replace("/", "_").replace(".", "_")
            section.append(f"    {src_id} --> {dep_id}")

    section.append("")
    section.append("    classDef l0 fill:#2d6a4f,stroke:#95d5b2,color:#d8f3dc")
    section.append("    classDef l1 fill:#1d4e89,stroke:#90e0ef,color:#caf0f8")
    section.append("    classDef l2 fill:#6a0572,stroke:#c77dff,color:#f3e0ff")
    section.append("    classDef l3 fill:#b5451b,stroke:#f4a261,color:#fde8d8")
    section.append("    classDef l4 fill:#7b3f00,stroke:#e9c46a,color:#fff3cd")
    section.append("    classDef other fill:#3d3d3d,stroke:#adb5bd,color:#f8f9fa")
    section.append("")

    for fname in dag:
        lvl = get_level(fname)
        node_id = fname.replace("/", "_").replace(".", "_")
        cls = f"l{lvl}" if lvl >= 0 else "other"
        section.append(f"    class {node_id} {cls}")

    section.append("```\n")
    return "\n".join(section)


def generate_doc():
    lines = ["# API Documentation\n"]
    dag, ext_deps = build_dag()

    lines.append(render_dag(dag, ext_deps))
    lines.append("")

    all_files = [f for f in src.rglob("*.py") if f.name != "__init__.py"]
    for file in sorted(all_files):
        rel = str(file.relative_to(src))
        lines.append(f"\n## `{rel}`\n")
        tree = ast.parse(file.read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                args = [a.arg for a in node.args.args]
                doc = ast.get_docstring(node) or ""
                lines.append(f"### `{node.name}({', '.join(args)})`")
                if doc:
                    lines.append(doc.splitlines()[0])
                lines.append("")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    lines.append(f"API.md last updated at {timestamp}.")
    (HERE / "API.md").write_text("\n".join(lines))
    print("API.md last updated at", timestamp, ".")


generate_doc()
