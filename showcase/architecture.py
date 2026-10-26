"""The architecture diagram, built from the repository itself.

Code dependencies are computed by parsing the import statements of every file listed for a component (including
imports inside functions); nothing about them is written by hand. Runtime connections (HTTP, processes, files) come
from content.RUNTIME_EDGES, each backed by a marker string in a named source file. Only the standard library is used.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import content

ROOT = Path(__file__).resolve().parent.parent
LAYER_COLOR = {"Entry points": "#2bb3a3", "Studio server": "#4aa3df", "Research pipeline": "#9b8cf0",
               "Sources": "#d6a24e", "Evidence": "#3fb68b", "Verification": "#e5646e", "Models": "#c38bd8",
               "Research backend": "#8b9ba3", "Evaluation": "#6f8fa6", "External": "#5c6870"}


def _module_name(path: str) -> str:
    parts = list(Path(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imports(path: Path, module: str) -> Set[str]:
    """Module names imported anywhere in a Python file (absolute and relative imports, also inside functions)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = module.split(".")[:-1] if path.name != "__init__.py" else module.split(".")
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - (node.level - 1)] if node.level > 1 else package
                prefix = ".".join(base)
                target = f"{prefix}.{node.module}" if node.module else prefix
            else:
                target = node.module or ""
            found.add(target)
            found.update(f"{target}.{alias.name}" for alias in node.names)
    return found


@lru_cache(maxsize=1)
def code_edges() -> Tuple[Tuple[str, str, Tuple[str, ...]], ...]:
    """(from component, to component, evidence) for every import between files of two different components.
    Evidence lists 'file imports module' pairs found in the source."""
    owner: Dict[str, str] = {}
    for component in content.COMPONENTS:
        for path in component["files"]:
            if path.endswith(".py"):
                owner[_module_name(path)] = component["id"]
    edges: Dict[Tuple[str, str], List[str]] = {}
    for component in content.COMPONENTS:
        for path in component["files"]:
            if not path.endswith(".py"):
                continue
            for target in _imports(ROOT / path, _module_name(path)):
                other = owner.get(target)
                if other and other != component["id"]:
                    edges.setdefault((component["id"], other), []).append(f"{path} imports {target}")
    return tuple((a, b, tuple(sorted(set(ev)))) for (a, b), ev in sorted(edges.items()))


def runtime_edges() -> List[Tuple[str, str, str, str]]:
    """(from, to, label, evidence) for connections that are not imports; only those whose marker is found."""
    out = []
    for a, b, label, path, marker in content.RUNTIME_EDGES:
        file = ROOT / path
        if file.is_file() and marker in file.read_text(encoding="utf-8", errors="replace"):
            out.append((a, b, label, f"{path} contains {marker!r}"))
    return out


def nodes() -> Dict[str, dict]:
    table = {c["id"]: dict(c, external=False) for c in content.COMPONENTS}
    table.update({e["id"]: dict(e, layer="External", external=True) for e in content.EXTERNAL})
    return table


def dot(view: Optional[List[str]] = None, selected: Optional[str] = None, show_code: bool = True,
        show_runtime: bool = True) -> str:
    """Graphviz DOT for a view (component ids; None = all). The selected node and its connections are highlighted.
    Node colour shows the layer; solid edges are imports, dashed edges runtime connections."""
    table = nodes()
    keep = set(view) if view else set(table)
    edges = []
    if show_code:
        edges += [(a, b, "code") for a, b, _ in code_edges() if a in keep and b in keep]
    if show_runtime:
        edges += [(a, b, "runtime") for a, b, _, _ in runtime_edges() if a in keep and b in keep]
    touched = {x for a, b, _ in edges for x in (a, b) if selected in (a, b)} if selected else set()
    lines = ['digraph G {', 'graph [rankdir=TB, bgcolor="transparent", pad=0.2, nodesep=0.3, ranksep=0.5, '
             'concentrate=true, splines=true, fontname="Helvetica"];',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=15, penwidth=1.4, '
             'fontcolor="#dde5e8", margin="0.16,0.07"];',
             'edge [color="#3a474e", arrowsize=0.7, penwidth=1.1];']
    for node_id, node in table.items():
        if node_id not in keep:
            continue
        color = LAYER_COLOR.get(node["layer"], "#5c6870")
        dim = bool(selected) and node_id != selected and node_id not in touched
        fill = "#10161a" if dim else ("#1d3b38" if node_id == selected else "#161c1f")
        border = "#2a3439" if dim else color
        shape = ", shape=cylinder" if node.get("kind") == "store" else (", shape=ellipse" if node.get("external") else "")
        font = ', fontcolor="#5c6870"' if dim else ""
        lines.append(f'"{node_id}" [label="{node["name"]}", fillcolor="{fill}", color="{border}", '
                     f'penwidth={3 if node_id == selected else 1.4}{shape}{font}];')
    for a, b, kind in edges:
        active = selected in (a, b)
        color = "#2bb3a3" if active else ("#3a474e" if kind == "code" else "#6b5a3a")
        attrs = [f'color="{color}"', f"style={'solid' if kind == 'code' else 'dashed'}"]
        if active:
            attrs.append("penwidth=2.2")
        lines.append(f'"{a}" -> "{b}" [{", ".join(attrs)}];')
    lines.append("}")
    return "\n".join(lines)


def connections(component_id: str) -> Dict[str, list]:
    """Incoming and outgoing connections of one component, with their evidence."""
    out = {"uses": [], "used_by": [], "runtime_out": [], "runtime_in": []}
    for a, b, evidence in code_edges():
        if a == component_id:
            out["uses"].append((b, evidence))
        if b == component_id:
            out["used_by"].append((a, evidence))
    for a, b, label, evidence in runtime_edges():
        if a == component_id:
            out["runtime_out"].append((b, label, evidence))
        if b == component_id:
            out["runtime_in"].append((a, label, evidence))
    return out
