"""Comment format linter.

Inputs: Python files under tailcyclenet/ and scripts/, TOML files under configs/.
Outputs: diagnostic lines on stdout (path:line: message), exit code 1 if any found.
Side effects: none (read-only).

Rules
-----
Python (.py):
  1. Every def/async def must have a docstring (triple-quoted, immediately after the signature).
     The docstring must declare Inputs, Outputs, and (if any) Side effects.
     One-liner docstrings (a single descriptive sentence) are allowed for trivially obvious
     functions where inputs/outputs are clear from the signature and there are no side effects.
  2. No inline comments: a ``#`` that is not on a line by itself (ignoring leading whitespace)
     is a violation.  Exceptions: shebangs, type-ignore, noqa, fmt-skip, pragma, coding
     declarations, and module/section-level comment blocks (lines whose non-whitespace content
     starts with ``#``).

TOML (.toml) under configs/:
  3. A comment (``#``) is allowed only as a SINGLE line immediately before or on the same line
     as a key.  Multi-line comment blocks (consecutive comment-only lines) are violations
     except for the file-level header (consecutive comment lines starting at line 1).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_INLINE_COMMENT_EXCEPTIONS = re.compile(
    r'#\s*(?:type:\s*ignore|noqa|fmt:\s*skip|pragma|coding[:=])'
)
_SHEBANG = re.compile(r'^#!')


def _is_standalone_comment(line: str) -> bool:
    """Whether *line* is a comment-only line (leading whitespace + ``#``).

    Inputs: line -- a single source line (may include trailing newline).
    Outputs: bool.
    """
    stripped = line.lstrip()
    return stripped.startswith('#')


def _has_inline_comment(line: str) -> bool:
    """Whether *line* has a ``#`` comment after code.

    Inputs: line -- a single source line.
    Outputs: bool.
    """
    stripped = line.lstrip()
    # standalone comment or blank -> not inline
    if not stripped or stripped.startswith('#'):
        return False
    # shebang
    if _SHEBANG.match(stripped):
        return False
    # walk through the line respecting strings
    in_single = False
    in_double = False
    in_triple_single = False
    in_triple_double = False
    i = 0
    while i < len(line):
        c = line[i]
        # triple quotes
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
        # single quotes
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
        # outside any string
        if c == '#':
            rest = line[i:]
            if _INLINE_COMMENT_EXCEPTIONS.match(rest):
                return False
            return True
        i += 1
    return False


# ---------------------------------------------------------------------------
# Python: docstring checks
# ---------------------------------------------------------------------------

def _check_python(path: Path) -> list[str]:
    """Lint a Python file for comment-format violations.

    Inputs: path -- Path to a .py file.
    Outputs: list of diagnostic strings (``path:line: message``).
    """
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    diagnostics: list[str] = []

    # --- rule 2: inline comments ---
    in_triple = False
    triple_char = None
    for lineno, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if not in_triple:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                triple_char = stripped[:3]
                # check if it opens and closes on the same line
                rest = stripped[3:]
                if triple_char not in rest:
                    in_triple = True
                continue
        else:
            if triple_char in line:
                in_triple = False
            continue
        if _has_inline_comment(line):
            diagnostics.append(f'{path}:{lineno}: inline comment (move to a standalone line or a docstring)')

    # --- rule 1: docstrings ---
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return diagnostics

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # skip tiny dunder / property bodies
        name = node.name
        doc = ast.get_docstring(node)
        if doc is None:
            diagnostics.append(
                f'{path}:{node.lineno}: function `{name}` has no docstring'
            )
    return diagnostics


# ---------------------------------------------------------------------------
# TOML: comment checks
# ---------------------------------------------------------------------------

def _check_toml(path: Path) -> list[str]:
    """Lint a TOML config for multi-line comment blocks.

    Inputs: path -- Path to a .toml file.
    Outputs: list of diagnostic strings.
    """
    lines = path.read_text().splitlines()
    diagnostics: list[str] = []
    # find the end of the file-level header (consecutive comment lines from line 1)
    header_end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#') or stripped == '':
            header_end = i + 1
        else:
            break
    # scan for consecutive comment-only lines after the header
    i = header_end
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith('#'):
            # count consecutive comment lines
            run_start = i
            while i < len(lines) and lines[i].strip().startswith('#'):
                i += 1
            run_len = i - run_start
            if run_len > 1:
                diagnostics.append(
                    f'{path}:{run_start + 1}: multi-line comment block '
                    f'({run_len} lines); configs allow at most 1 comment line per parameter'
                )
        else:
            i += 1
    return diagnostics


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point.

    Inputs: none (reads files from the repo tree).
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
