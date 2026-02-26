#!/usr/bin/env python3
"""Run Python code blocks in a markdown file and insert/replace output blocks.

Usage: python run_md_code.py <file.md>

Semantics:
- Non-test blocks: executed cumulatively in a shared namespace.
  stdout is captured and placed in a ```output block immediately after.
- Test blocks (containing `def test_`): a .py file is built from all
  previous successful non-test blocks + this block, then run with
  pytest --tb=short. Output is placed in a ```output block.
- Existing ```output blocks after ```python blocks are replaced.
- No output block is added if there was no output.
"""

import sys
import re
import io
import os
import subprocess
import tempfile
from contextlib import redirect_stdout, redirect_stderr


def parse_chunks(text):
    """Parse markdown into chunks: ('text', content) or ('code', lang, content, raw)."""
    pattern = re.compile(r"(^```(\w*)\n(.*?)^```$)", re.MULTILINE | re.DOTALL)
    chunks = []
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            chunks.append(("text", text[last : m.start()]))
        lang = m.group(2)
        code = m.group(3)
        if code.endswith("\n"):
            code = code[:-1]
        chunks.append(("code", lang, code, m.group(0)))
        last = m.end()
    if last < len(text):
        chunks.append(("text", text[last:]))
    return chunks


def execute_block(code, namespace):
    """Execute code in namespace, return (stdout_str, success_bool)."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            exec(compile(code, "<block>", "exec"), namespace)
        return buf.getvalue(), True
    except Exception as e:
        output = buf.getvalue()
        output += f"{type(e).__name__}: {e}\n"
        return output, False


def run_pytest(code):
    """Write code to a temp file and run pytest on it. Return output string."""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="md_test_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", path, "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = (result.stdout + result.stderr).strip()
        # Replace temp file paths with something readable
        basename = os.path.basename(path).replace(".py", "")
        output = output.replace(path, "<block>")
        output = re.sub(re.escape(basename) + r"\.", "", output)
        return output
    finally:
        os.unlink(path)


def process(filepath):
    with open(filepath) as f:
        text = f.read()

    chunks = parse_chunks(text)

    # Identify output blocks (and whitespace gaps) that follow python blocks, to skip them
    skip = set()
    for i, chunk in enumerate(chunks):
        if chunk[0] == "code" and chunk[1] == "python":
            j = i + 1
            # Check for: optional whitespace-only text gap, then output block
            if j < len(chunks) and chunks[j][0] == "text" and chunks[j][1].strip() == "":
                k = j + 1
                if (
                    k < len(chunks)
                    and chunks[k][0] == "code"
                    and chunks[k][1] == "output"
                ):
                    skip.add(j)  # whitespace gap
                    skip.add(k)  # output block
            elif (
                j < len(chunks)
                and chunks[j][0] == "code"
                and chunks[j][1] == "output"
            ):
                skip.add(j)

    # Process chunks
    namespace = {}
    successful_blocks = []
    result = []

    for i, chunk in enumerate(chunks):
        if i in skip:
            continue

        if chunk[0] == "text":
            result.append(chunk[1])
            continue

        _, lang, code, raw = chunk

        if lang != "python":
            result.append(raw)
            continue

        # Python code block
        result.append(raw)

        print(f"--- block {i} ---", file=sys.stderr)
        print(code, file=sys.stderr)

        has_test = bool(re.search(r"^def test_", code, re.MULTILINE))

        if has_test:
            test_code = "\n\n".join(successful_blocks + [code])
            output = run_pytest(test_code)
        else:
            output, success = execute_block(code, namespace)
            if success:
                successful_blocks.append(code)
            output = output.rstrip()

        if output:
            print(output, file=sys.stderr)
            result.append(f"\n\n```output\n{output}\n```")
        else:
            print("(no output)", file=sys.stderr)
        print(file=sys.stderr)

    new_text = "".join(result)
    with open(filepath, "w") as f:
        f.write(new_text)

    print(f"Done: {filepath}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file.md>", file=sys.stderr)
        sys.exit(1)
    process(sys.argv[1])
