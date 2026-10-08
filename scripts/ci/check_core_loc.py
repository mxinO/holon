"""Invariant 8: core (holon.api + holon.core) stays within 6000 code lines.

Code lines are lines holding a token other than comments and docstrings (module, class and function docstrings,
found with the AST), so a docstring's closing line or a trailing comment never hides code.
"""
from __future__ import annotations

import ast
import io
import tokenize

from _common import CORE, fail, py_files

CAP = 6000
SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER,
        tokenize.ENCODING}


def code_lines(source: str) -> int:
    docs: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first.value, "value", None), str):
                docs.update(range(first.lineno, first.end_lineno + 1))
    lines: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type not in SKIP:
            lines.update(range(tok.start[0], tok.end[0] + 1))
    return len(lines - docs)


total = sum(code_lines(p.read_text()) for p in py_files(CORE))
print(f"     core code lines: {total} / {CAP}")
fail([f"core has {total} code lines > {CAP}"] if total > CAP else [], "core LOC cap")
