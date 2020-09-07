#!/usr/bin/env python
"""
Move a post ipython notebook to it place etc.

Note that the file must already have the right YYYY-MM-DD- prefix.
An error will be raised if not.
"""


import argparse
import re
import fileinput
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
        sys.exit(cmd_output.returncode)
    return cmd_output


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
    
    run_cmd("nbdev_nb2md", filepath)

    # Move the image files into place
    images_directory = f"{filename_no_ext}_files/"
    run_cmd("rm", "-rf", f"images/{images_directory}")
    run_cmd("mv", images_directory, "images/")

    print("Fixing image links")
    with open(f"./{filename_no_ext}.md", 'r') as f:
        contents = f.read()
    fixed, n = re.subn(rf'(!.*]\()({re.escape(filename_no_ext)}_files/)', r'\1/images/\2', contents)
    print(f"Fixed {n} links.")

    loc = f"./_posts/{filename_no_ext}.md"
    print(f"Writing file to {loc}")
    with open(loc, 'w') as f:
        f.write(fixed)
    run_cmd("rm", f"./{filename_no_ext}.md")
