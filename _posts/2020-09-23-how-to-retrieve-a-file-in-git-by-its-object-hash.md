---
title: How to retrieve a file in git by its object hash





---


In certain circumstances, you may want to retrieve a file in [git](https://en.wikipedia.org/wiki/Git) by some short identifier.

For example, you have some relevant file which is useful in reproducing a computation, say for debugging or regression testing. Or, for compliance, you might want to record and easily retrieve which version of a contract template a user saw, but have the version field only change when the _contents_ of the html file changes. (And not when unrelated files change as with a commit hash.)

One simple way to achieve this is to use the Git _object hash_. This is the hash of file contents that git uses to store the underlying file.

Now let's create an empty git repository, with one file `foo` with contents `foo\n`:
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```python
!git init
```

</div>
<div class="output_area" markdown="1">

    Initialized empty Git repository in /private/var/folders/0x/v2lbxd814bv46ngy5zhtsf700000gn/T/tmppll0ayzo/.git/


</div>

</div>
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```python
!echo foo > foo
!git add .
!git commit -m 'init'
```

</div>
<div class="output_area" markdown="1">

    [master (root-commit) cd0d56c] init
     1 file changed, 1 insertion(+)
     create mode 100644 foo


</div>

</div>

How are these files stored internally in Git? The files are content-addressable by the object hash. We can copute the object hash using the [`git-hash-object` command](https://git-scm.com/docs/git-hash-object):
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```python
!cat foo | git hash-object -w --stdin
```

</div>
<div class="output_area" markdown="1">

    257cc5642cb1a054f08cc83f2d943e56fd3ebe99


</div>

</div>

We can then retrieve the contents in a single command with only that hash with the `git cat-file` command:
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```python
!git cat-file -p 257cc5642cb1a054f08cc83f2d943e56fd3ebe99
```

</div>
<div class="output_area" markdown="1">

    foo


</div>

</div>

Note that other file contents get added to the git tree when you call `hash-object`:
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```python
!git cat-file -p `echo aslkfmlaskm | git hash-object -w --stdin`
```

</div>
<div class="output_area" markdown="1">

    aslkfmlaskm


</div>

</div>

We can then update `foo` to some other contents, and still retrieve the original content using the same object hash:
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```python
!echo bar > foo
!git add .
!git commit -m 'Update foo to bar, as one does.'
!git cat-file -p 257cc5642cb1a054f08cc83f2d943e56fd3ebe99
```

</div>
<div class="output_area" markdown="1">

    [master 9ca289a] Update foo to bar, as one does.
     1 file changed, 1 insertion(+), 1 deletion(-)
    foo


</div>

</div>

The object hash is computed as the sha1 hash of a particular string constructed from the file's contents (`blob `, followed by the content length as a decimal integer, followed by a zero byte, followed by the file contents).

This can be computed in python as follows:
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```python
from hashlib import sha1
from typing import Union
def git_hash_object(s: Union[str, bytes]) -> str:
    if isinstance(s, str):
        b = s.encode('utf-8')
    else:
        b = s
    return sha1(b'blob %d\0%s' % (len(b), b)).hexdigest()

git_hash_object('foo\n')
```

</div>
<div class="output_area" markdown="1">




    '257cc5642cb1a054f08cc83f2d943e56fd3ebe99'



</div>

</div>

The loading code can just call [`git-cat-file`](https://git-scm.com/docs/git-cat-file) as a subprocess, or find some [python implementation](https://gist.github.com/leonidessaguisagjr/594cd8fbbc9b18a1dde5084d981b8028).
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```python
import subprocess
def load_git_object_hash(object_hash: str) -> bytes:
    result = subprocess.run(["git", "cat-file", "-p", object_hash], capture_output=True)
    result.check_returncode()
    return result.stdout

load_git_object_hash(git_hash_object('foo\n'))
```

</div>
<div class="output_area" markdown="1">




    b'foo\n'



</div>

</div>

Now, in our contract template example, we might have some code like:
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```python
def agree_to_contract(user_id):
    with open('contract-template.html', 'r') as f:
        template = f.read()
    terms = get_specific_terms_for_users(user_id)
    record_contract(user_id, git_hash_object(template), terms)
    return render(template, terms)
```

</div>

</div>
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```python
def display_exact_contract(user_id):
    object_hash, terms = get_contract_data(user_id)
    template = load_git_object_hash(object_hash).decode()
    return render(template, terms)
```

</div>

</div>

This allows us to easily group templates by version, retrieve old versions, and not waste a bunch of space storing the same template version again and again. You could apply the same principle to lockfiles, configuration files, docker files, self-contained algorithms, etc.

For auditing or debugging, this approach has some advantages over recording the commit hash, in that you can retrieve the file of interest in constant time without doing any writes to disk. It has the disadvantage that you need to record every file of interest to completely reproduce the relevant computation. In any case, it's a useful technique in some circumstances, and hopefully helps you to learn a bit about how git operates.
