# Building Frontrun with Claude Code

This is a companion post to [Announcing Frontrun](/blog/2026-02-25-frontrun-race-conditions), a concurrency testing library for Python. This post is about how I built it with Claude Code.

Modern coding agents behave a bit like a coding genie, in that you can get whatever code you want, but you have to be careful how you ask for it.
As I started using Claude code more and more in December and January, I realized that I could just implement _all_ of my ideas I'd had for interesting projects over the year.
So it's not just that I could implement my ideas, it's that I could make them _much_ better and _much_ more thorough.

And so the initial idea was something like:


<!-- noexec -->
```python
class AccountBalance:
    def __init__(self, balance=0):
        self._balance = balance

    def get_balance(self):
        before_read()
        return self._balance

    def set_balance(self, value):
        before_write()
        self._balance = value

    def deposit(self, amount):
        current = self.get_balance()
        after_read()
        self.set_balance(current + amount)
```

Then have some syntax for defining a schedule, like:
<!-- noexec -->
```python
[(t1, "after read"), (t2, "after read")]
```
## Claude Workflow.

I asked claude to implement basically that, and ended up with it suggesting a very heavyweight decorator-based approach that involved rewriting all your concurrent code in the new framework just to test it.
Then I suggested using `set_trace()`, a python debugging method, and we were off to the races.

The train of ideas flowed naturally after that:
* I asked claude if there was some way to trace bytecodes, so that single line of python where races could occur could be identified and fuzzed like in [hypothesis](https://hypothesis.readthedocs.io/en/latest/).
* Claude mentioned we could have users use different lock-primitives as with the rust library [`loom`](https://github.com/tokio-rs/loom).
  Loom uses a very clever technique I had heard-of-but-not-learned called dependent partial order reduction, that identifies causal relationships between different parts of the code.
* I suggested that Claude should monkey-patch things.
* I suggested that Claude should monkey-patch things.
* I _told_ claude to monkey patch things.
* No really, all the things.
* And so on.

After a while, we settled into a rhythm:
* I would suggest new ways Claude could write tests to break the code (i.e. not find race conditions).
* Claude would write tests that broke.
* I would have claude fix the tests.

It was a bit of an odd experience, where many of the conceptual breakthroughs weren't so much in actually figuring out how to do something, but figuring out the right way to ask Claude to do something I knew _must_ be possible to do.
The key breakthrough was when we had an exchange that went something like this:

> Me: What are some absolutely terrible ideas for how we can intercept io attribute access and other concurrency-relevant ideas. Vile perversions of the intention of how computers should work. Write them up in BAD_IDEAS.md.
>
> [...]
>
> Me: OK, which of those are actually good ideas?
>
> [...]
>
> Me: Cool. Now implement those.

Claude Opus is like a savant boyscout who is infinitely well-read/knowledgeable, but weirdly very uncreative.
It was hard to convince Claude to monkey-patch python threading internals, which is admittedly a very bad idea under _most_ circumstances.
It was even harder to convince Claude that intercepting libc io method calls was a good idea, even though Claude came up with the idea and put it in `BAD_IDEAS.md`.

Like this could've easily been a paper or a pycon talk 3 years ago.
It would've easily taken me _years_ to implement this thing I built in two weeks.
This thing where I did a substantial amount of development _on my phone_.

!["You made this" / "I made this?" meme with Claude Code](/images/2026-02-25-frontrun-race-conditions_files/claude-code-meme.png)

And of course, I had Claude move the meme to the right directory so it would render on the site, which prompted this beautiful exchange:
```
⏺ Now let me view the image to write a reasonable alt text, and add it to the end of the
  post.

⏺ Read(public/images/2026-02-25-frontrun-race-conditions_files/claude-code-meme.png)
  ⎿  Read image (215.8KB)

⏺ Great meme. Now let me add it to the end of the post.
```

So, in the interest of giving credit where it's due:

Claude:
  * Wrote ~100% of the code.
    I don't know rust very well at all, and I'm certainly not an expert on all the different bytecodes used in different versions of python.
  * Came up with the idea of using DPOR and implemented the algorithm.
  * Came up with the approach for detecting IO in C code (using `LD_PRELOAD`), though it advised against actually doing this.

I:
  * Came up with the idea for comment-based trace markers and random bytecode shuffling.
    This seems to be able to find race conditions almost as well as DPOR, at the cost of much less interpretable error traces.
  * Came up with the approach of monkey patching and tracing, then kept pushing Claude to see that through to the logical conclusion.
  * Came up with the overall API and made it significantly more ergonomic than Claude's initial attempts.

Naming the library was one of the most important and difficult parts, and was definitely a team effort.
I've had so many ideas fizzle over the years because of a kind of perfectionism where the first thing I need to do on a software project is give the repo a name, but then I search existing packages and find the good names are often taken.
I had Claude search through pypi for dozens of names with themes around weaving, shuffling, cheating, rigging, ordering, etc., very quickly eliminating ones that were already taken.
Eventually it was down to frontrun and a few others, when we came up with the idea of intercepting libc IO calls with `LD_PRELOAD` and frontrun fit perfectly. You now execute the test suite with:

```
frontrun pytest path/to/test_concurrency.py
```
