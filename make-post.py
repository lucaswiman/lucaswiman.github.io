#!/usr/bin/env python
"""
Convert a Jupyter notebook to a Jekyll blog post.

Note that the file must already have the right YYYY-MM-DD- prefix.
An error will be raised if not.
"""

import argparse
import re
import subprocess
from pathlib import Path
import sys


def run_cmd(*args, verbose=True):
    """
    Runs the given command arguments and exits with its exit status if nonzero.
    """
    if verbose:
        print(f"Running command `{' '.join(map(str, args))}`")
    cmd_output = subprocess.run(args)
    if cmd_output.returncode != 0:
        if verbose:
            print(f"Received error {cmd_output.returncode}")
        raise Exception(f"Exit status: {cmd_output.returncode}")
    return cmd_output


def extract_title_and_strip(contents):
    """Extract first H1 heading and remove it from the markdown body."""
    match = re.search(r'^# (.+)$', contents, re.MULTILINE)
    title = None
    if match:
        title = match.group(1).strip()
        # Remove the heading line plus any immediately following blank lines
        after = contents[match.end():]
        contents = contents[:match.start()] + after.lstrip('\n')
    return title, contents


def wrap_display_math(contents):
    """Wrap $$...$$ blocks in Liquid raw tags to prevent template processing issues."""
    return re.sub(
        r'(\$\$(?:.|\n)*?\$\$)',
        lambda m: '{% raw %}\n' + m.group(1) + '\n{% endraw %}',
        contents,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('post_path', type=str, help='path to the ipynb post')
    args = parser.parse_args()
    filepath = Path(args.post_path)
    filename = filepath.parts[-1]

    posts_re = re.compile(r'^(\d{4}-\d{2}-\d{2}-[\w-]+)\.ipynb$')
    if not posts_re.match(filename):
        print(f'filename should match {posts_re.pattern!r}.')
        sys.exit(1)
    elif not filepath.exists():
        print(f'filename={filename} does not exist!')
        sys.exit(1)

    [filename_no_ext] = posts_re.findall(filename)
    images_directory = f"{filename_no_ext}_files"

    # Convert notebook to markdown in the project root directory
    run_cmd("jupyter", "nbconvert", "--to", "markdown", str(filepath), "--output-dir", ".")

    # Move the image files into place
    src_images = Path(images_directory)
    dst_images = Path("images") / images_directory
    if src_images.exists():
        run_cmd("rm", "-rf", str(dst_images))
        run_cmd("mv", str(src_images), str(dst_images))

    # Read the generated markdown
    md_path = Path(f"{filename_no_ext}.md")
    with open(md_path, 'r') as f:
        contents = f.read()

    # Fix image paths from relative to absolute Jekyll paths
    contents, n0 = re.subn(
        rf'(!.*?\]\()notebooks/({re.escape(images_directory)}/)',
        r'\1/images/\2',
        contents,
    )
    contents, n1 = re.subn(
        rf'(!.*?\]\()({re.escape(images_directory)}/)',
        r'\1/images/\2',
        contents,
    )
    print(f"Fixed {n0 + n1} image links.")

    # Extract title for Jekyll front matter
    title, contents = extract_title_and_strip(contents)

    # Build Jekyll front matter
    fm_lines = ['---']
    if title:
        safe_title = title.replace('"', '\\"')
        fm_lines.append(f'title: "{safe_title}"')
    fm_lines.append('---')
    front_matter = '\n'.join(fm_lines)

    # Wrap display math in Liquid raw tags to avoid Jekyll template conflicts
    contents = wrap_display_math(contents)

    # Prepend front matter
    contents = front_matter + '\n\n' + contents

    # Write to _posts/
    post_path = Path("_posts") / f"{filename_no_ext}.md"
    print(f"Writing post to {post_path}")
    with open(post_path, 'w') as f:
        f.write(contents)

    # Remove the intermediate markdown file
    md_path.unlink()
