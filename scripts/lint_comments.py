"""Comment format linter.

Inputs: Python files under tailcyclenet/ and scripts/, TOML files under configs/.
Outputs: diagnostic lines on stdout (path:line: message), exit code 1 if any found.
Side effects: none (read-only).

Rules
-----
Python (.py):
  1. Every def/async def must have a docstring specifying inputs, outputs, and side effects.
     One-liner docstrings are allowed for trivially obvious functions.
  2. No comments inside code. No inline comments (code + # on same line) and no standalone
     comment lines inside function/method bodies. Comments are only allowed at module level
     (outside any function). Exceptions: type: ignore, noqa, fmt: skip, pragma directives.

  3. Docstrings and module-level comment blocks must be at most 30 lines.

TOML (.toml) under configs/:
  4. At most 1 comment line per parameter. Multi-line comment blocks are violations except
     for the file-level header.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


_INLINE_COMMENT_EXCEPTIONS = re.compile(
    r'#\s*(?:type:\s*ignore|noqa|fmt:\s*skip|pragma|coding[:=])'
)
_SHEBANG = re.compile(r'^#!')


def _has_inline_comment(line: str) -> bool:
    """Detect a # comment after code on the same line, respecting string literals.

    Inputs: line -- a single source line.
    Outputs: True if an inline comment exists (not inside a string).
    """
    stripped = line.lstrip()
    if not stripped or stripped.startswith('#'):
        return False
    if _SHEBANG.match(stripped):
        return False
    in_single = False
    in_double = False
    in_triple_single = False
    in_triple_double = False
    i = 0
    while i < len(line):
        c = line[i]
        if not in_single and not in_double:
            if line[i:i+3] == '"""':
                in_triple_double = not in_triple_double
                i += 3
                continue
            if line[i:i+3] == "'''":
                in_triple_single = not in_triple_single
                i += 3
                continue
        if in_triple_double or in_triple_single:
            i += 1
            continue
        if c == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if c == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if in_single or in_double:
            if c == '\\':
                i += 2
                continue
            i += 1
            continue
        if c == '#':
            rest = line[i:]
            if _INLINE_COMMENT_EXCEPTIONS.match(rest):
                return False
            return True
        i += 1
    return False


def _is_exempt_comment(line: str) -> bool:
    """Whether a standalone comment line contains an exempt directive.

    Inputs: line -- a source line starting with #.
    Outputs: True if the comment is a pragma/noqa/type-ignore exempt.
    """
    stripped = line.lstrip()
    if not stripped.startswith('#'):
        return False
    return bool(_INLINE_COMMENT_EXCEPTIONS.match(stripped))


def _function_body_lines(source: str) -> set[int]:
    """Collect all line numbers that are inside function bodies (after docstrings).

    Inputs: source -- full Python source text.
    Outputs: set of 1-indexed line numbers inside function/method bodies.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    body_lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body_start = node.lineno + 1
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(getattr(node.body[0], 'value', None), ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            body_start = node.body[0].end_lineno + 1
        body_end = node.end_lineno
        for ln in range(body_start, body_end + 1):
            body_lines.add(ln)
    return body_lines


def _check_python(path: Path) -> list[str]:
    """Lint a Python file for comment-format violations.

    Inputs: path -- Path to a .py file.
    Outputs: list of diagnostic strings.
    """
    source = path.read_text()
    lines = source.splitlines()
    diagnostics: list[str] = []

    body_lines = _function_body_lines(source)

    in_triple = False
    triple_char = None
    for lineno, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if not in_triple:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                triple_char = stripped[:3]
                rest = stripped[3:]
                if triple_char not in rest:
                    in_triple = True
                continue
        else:
            if triple_char in line:
                in_triple = False
            continue

        if _has_inline_comment(line):
            diagnostics.append(
                f'{path}:{lineno}: inline comment (move to docstring or remove)')
        elif stripped.startswith('#') and lineno in body_lines:
            if not _is_exempt_comment(line):
                diagnostics.append(
                    f'{path}:{lineno}: comment inside function body '
                    '(move to docstring or remove)')

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return diagnostics

    for node in ast.walk(tree):
        if isinstance(node, ast.Module):
            doc = ast.get_docstring(node)
            if doc and len(doc.splitlines()) > 30:
                diagnostics.append(
                    f'{path}:1: module docstring is {len(doc.splitlines())} lines (max 30)')
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = node.name
        doc = ast.get_docstring(node)
        if doc is None:
            diagnostics.append(
                f'{path}:{node.lineno}: function `{name}` has no docstring')
        elif len(doc.splitlines()) > 30:
            diagnostics.append(
                f'{path}:{node.lineno}: `{name}` docstring is '
                f'{len(doc.splitlines())} lines (max 30)')

    lines_list = source.splitlines()
    func_line_set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for ln in range(node.lineno, node.end_lineno + 1):
                func_line_set.add(ln)
    i = 0
    while i < len(lines_list):
        lineno = i + 1
        if lineno in func_line_set:
            i += 1
            continue
        if lines_list[i].lstrip().startswith('#'):
            run_start = i
            while (i < len(lines_list)
                   and (i + 1) not in func_line_set
                   and lines_list[i].lstrip().startswith('#')):
                i += 1
            run_len = i - run_start
            if run_len > 30:
                diagnostics.append(
                    f'{path}:{run_start + 1}: module-level comment block is '
                    f'{run_len} lines (max 30)')
        else:
            i += 1
    return diagnostics


def _check_toml(path: Path) -> list[str]:
    """Lint a TOML config for multi-line comment blocks.

    Inputs: path -- Path to a .toml file.
    Outputs: list of diagnostic strings.
    """
    lines = path.read_text().splitlines()
    diagnostics: list[str] = []
    header_end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#') or stripped == '':
            header_end = i + 1
        else:
            break
    i = header_end
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith('#'):
            run_start = i
            while i < len(lines) and lines[i].strip().startswith('#'):
                i += 1
            run_len = i - run_start
            if run_len > 1:
                diagnostics.append(
                    f'{path}:{run_start + 1}: multi-line comment block '
                    f'({run_len} lines); max 1 comment line per parameter')
        else:
            i += 1
    return diagnostics


def main() -> int:
    """Entry point.

    Inputs: none (discovers files from the repo tree).
    Outputs: exit code (0 = clean, 1 = violations found).
    Side effects: prints diagnostics to stdout.
    """
    root = Path(__file__).resolve().parent.parent
    diagnostics: list[str] = []

    py_dirs = [root / 'tailcyclenet', root / 'scripts']
    for d in py_dirs:
        for p in sorted(d.rglob('*.py')):
            if p.name == '__init__.py' and p.stat().st_size == 0:
                continue
            diagnostics.extend(_check_python(p))

    configs_dir = root / 'configs'
    if configs_dir.is_dir():
        for p in sorted(configs_dir.rglob('*.toml')):
            diagnostics.extend(_check_toml(p))

    for d in diagnostics:
        print(d)

    return 1 if diagnostics else 0


if __name__ == '__main__':
    sys.exit(main())
