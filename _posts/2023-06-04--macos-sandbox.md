---
title: Sandboxing code on MacOS





---


This post covers some experimentation I did with the MacOS `sandbox-exec` command. The goal is to be able to run and evaluate Python libraries while significantly reducing the risk of a supply chain attack installing malware on my computer or exfiltrating data from my computer.

## Bash Script

The bash script I wrote that inspired this post can be found [here](https://gist.github.com/lucaswiman/1cec6584015149f0df1bb24c875a0709). Please comment with improvements/suggestions!

## Basics


The DSL used by `sandbox-exec` must start with `(version 1)`, seemingly the only version in existence as of 2023.

The rules consist of parenthesis-enclosed rules of the form `([deny/allow] [permissions] [predicates])`. Later rules have higher precedence.

### Permissions

The most important permissions are:
* `default`: Matches any permission, e.g. `(allow default)` or `(deny default)`. The latter is probably useful for running untrusted code (e.g. a new pypi library).
* `file*`:
  * `file-read-metadata`, `file-read-data`, ...
  * `file-write-data`, ...
* `network*`:
  * `network-outbound`, e.g. `(deny network-outbound (remote ip "*:80"))` disallows outbound connections to port 80.
  * `network-bind`. You probably want to leave this one `deny` unless you know you need it, so granting `network-*` is probably overly broad.
* `sysctl-read`. This is needed for code intended to run on multiple OSes, e.g. Python's `os.uname()` method fails without this permission. I don't know how to make this more restricted. You probably don't want to grant `sysctl-write` if you can avoid it.
* `mach*` and `ipc-posix-shm`. I needed to grant these to allow audio (see bash script below), though I'm ignorant of what they do.


Permissions support globbing, e.g. `file*` grants `file-read-data` and `file-write-data` permissions. 

The most complete listing can be found in [this reverse engineered guide](https://reverse.put.as/wp-content/uploads/2011/09/Apple-Sandbox-Guide-v1.0.pdf)


### Predicates

These are enclosed in parentheses:
* `(literal "some literal")`
* `(subpath "/path/to/dir")`
* `(regex #"^/usr/lib/*")`
* `([remote/local] ip "host:port")`, e.g. `(remote ip "*:80")`. Annoyingly, the host must be either `*` or `localhost`, which makes it impossible to only allow particular hosts. There exist firewalling applications which can do that for particular processes, but alas not supported here.
* `(require-any [predicates])` or `(require-all [predicates])` disjunction/conjunction of predicates.

## Examples:





### Deny network access
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```bash
sandbox-exec -p '
(version 1)
(allow default)
(deny network-outbound (remote ip "*:80"))
' curl http://example.com  # fails
```

</div>
<div class="output_area" markdown="1">

    curl: (7) Failed to connect to example.com port 80 after 7 ms: Couldn't connect to server




</div>

</div>
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```bash
sandbox-exec -p '
(version 1)
(allow default)
(deny network-outbound (remote ip "*:80"))
' curl https://example.com > /dev/null # succeeds
```

</div>
<div class="output_area" markdown="1">

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  1256  100  1256    0     0  17414      0 --:--:-- --:--:-- --:--:-- 19323


</div>

</div>

### Deny file writes
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```bash
sandbox-exec -p '
(version 1)
(allow default)
(deny file-write*)
' touch foo
```

</div>
<div class="output_area" markdown="1">

    touch: foo: Operation not permitted




</div>

</div>

### Only allow writes to CWD
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```bash
sandbox-exec -p "
(version 1)
(allow default)
(deny file-write*)
(allow file-write*
  (subpath \"$PWD\")
)
" touch foo  # succeeds

sandbox-exec -p "
(version 1)
(allow default)
(deny file-write*)
(allow file-write*
  (subpath \"$PWD\")
)
" touch /tmp/foo  # fails
```

</div>
<div class="output_area" markdown="1">

    touch: /tmp/foo: Operation not permitted




</div>

</div>

Note that because of how /tmp and /var are mapped to /private, naively granting permissions to PWD will not work in those cases:
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```bash
mkdir -p /tmp/asdf
sandbox-exec -p "
(version 1)
(allow default)
(deny file-write*)
(allow file-write*
  (subpath \"/private/tmp/asdf\")
)
" touch /tmp/asdf/foo && echo "success: /private"

sandbox-exec -p "
(version 1)
(allow default)
(deny file-write*)
(allow file-write*
  (subpath \"/tmp/asdf\")
)
" touch /tmp/asdf/foo || echo "failure: /tmp"
```

</div>
<div class="output_area" markdown="1">

    success: /private
    touch: /tmp/asdf/foo: Operation not permitted
    failure: /tmp


</div>

</div>

Overly broad rules can cause unexpected problems. In particular, many UNIX command do not work if they cannot read parent directories of the current directory, especially if they cannot read `/`:
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```bash
sandbox-exec -p "
(version 1)
(allow default)
(deny file*)
(allow file-read*
  (subpath \"/bin\")
  (subpath \"/private/tmp/asdf\")
)
" /bin/ls /private/tmp/asdf || echo 'failure'


sandbox-exec -p '
(version 1)
(allow default)
(deny file*)
(allow file-read*
  (subpath "/bin")
  (subpath "/private/tmp/asdf")
  (literal "/")  ; required to do ~anything related to reading directories
)
' /bin/ls /private/tmp/asdf > /dev/null && echo 'success'
```

</div>
<div class="output_area" markdown="1">

    Abort trap: 6
    failure
    success


</div>

</div>

### Running Python

Python is particularly obnoxious to sandbox because it scatters files over so much of the filesystem. You basically have to ask python what files it needs, then grant the subprocess at least read access to those files.

You can grant `file-read*` access to `sys.base_prefix`, maybe with some extra futzing for virtualenvs:
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```bash
sandbox-exec -p "
(version 1)
(allow default)
(deny file*)
(allow file-read*
  (subpath \"$(python -c 'import sys; print(sys.base_prefix)')\")
  (literal \"/\")
)
" python -c 'import sys; print("hello!")'
```

</div>
<div class="output_area" markdown="1">

    hello!


</div>

</div>

Then there is the issue of installing dependencies while still disallowing most network access. The best solution I've found to this is run `devpi`, a caching/proxying server for pypi, then set:

```bash
PIP_INDEX_URL="http://127.0.0.1:$PYPI_PROXY_PORT/root/pypi/+simple/"
```
And add `(allow network-outbound (remote ip "localhost:'$PYPI_PROXY_PORT'"))` to your rule set.

## Isn't `sandbox-exec` deprecated?

That is indeed what the man page says. The code in `/usr/share/sandbox` shows that the sandboxing DSL is widely used for running MacOS services, so it's unlikely to be removed soon. That said, while it may _stop working_ in a future OS update, I don't think it will silently fail.

Note that the `trace` feature mentioned in the 2011 reverse-engineered guide appears to have been removed. I couldn't get it working in Ventura 13.4. There is further discussion in [this stackoverflow thread](https://stackoverflow.com/a/61880980/303931).

## Debugging

Unfortunately, Apple has made this harder with the apparent removal of the `trace` command, but sandboxd does still send logs about what it blocks. You can see these logs with:
```bash
log stream --style syslog | grep -i sandbox
```

## Caution about environment variables

By default `sandbox-exec` passes all environment variables to the subprocess, so be careful if you store any secrets in environment variables. Blocking network should reduce the attack surface of any API tokens that may be in the environment.

## Why not use Docker for Mac?

Docker for Mac works well. I would recommend using it for many tasks, and I do use it. However, it will not allow you to run MacOS code natively, which is important for many command line tools. 

It requires overhead (Docker must be running; uses up memory; I might forget to kill a container, etc.) It's also a little frustrating that the easiest way to sandbox code in MacOS is to effectively use a different OS.

Most importantly, I cannot use it for work without obtaining a commercial license, which I'm sure my employer would pay for, but I hate filling out expense reports.

## References:
* Mozilla has a useful reference [rule set they use for nightly builds of Firefox](https://wiki.mozilla.org/Sandbox/OS_X_Rule_Set).
* The most useful reference I found was [this PDF reverse-engineering the DSL](https://reverse.put.as/wp-content/uploads/2011/09/Apple-Sandbox-Guide-v1.0.pdf).
* Playing sound was from [this post](https://mybyways.com/blog/creating-a-macos-sandbox-to-run-kodi).
* See also the existing rulesets in `/usr/share/sandbox`, which includes some useful utility methods.

## Environment

All of the above commands were run on a 2019 Intel Macbook pro running Macos Ventura 13.4. I don't know if they work on ARM Macs, though I presume so.
