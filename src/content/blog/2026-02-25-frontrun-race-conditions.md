# Announcing Frontrun

For the past few weeks, I've been building a concurrency testing library called `frontrun`.
The genesis of the idea was a useful-but-clunky category of concurrency test I've been using for the past several years involving using `threading.Event` or `threading.Barrier` to reproduce race conditions by forcing a particular execution ordering in test.
This can be accomplished e.g. by mock-wrapping a method and having it either wait for an event or trigger an event.
Consider a class like this:

```python
class AccountBalance:
    def __init__(self, balance=0):
        self._balance = balance

    def get_balance(self):
        return self._balance

    def set_balance(self, value):
        self._balance = value

    def deposit(self, amount):
        current = self.get_balance()
        self.set_balance(current + amount)
```

The `deposit` method has a classic time-of-check-to-time-of-use (TOCTOU) bug: it reads the balance, then writes a new value, but those two operations aren't atomic. If two threads deposit concurrently, one update can be lost. But in a naive test, the race window is so small that the test almost always passes:

```python
from threading import Thread
from collections import Counter

def test_deposit_not_threadsafe():
    """This test SHOULD fail, but almost never does."""
    account = AccountBalance(balance=0)

    balances = Counter()

    for _ in range(100):
        account = AccountBalance(balance=0)
        t1 = Thread(target=account.deposit, args=(100,))
        t2 = Thread(target=account.deposit, args=(100,))
        t1.start(); t2.start()
        t1.join(); t2.join()
        balances[account._balance] += 1

    # Both threads read balance=0, then both write 0+100=100.
    # The second deposit is lost!
    assert balances == {200: 100}
```

```output
.                                                                        [100%]
1 passed in 0.03s
```

And some changes may reduce the likelihood of races without actually eliminating them, so how can you be _sure_ you eliminated the race condition you found?
The key idea is to use `threading.Event` or `threading.Barrier` objects to _force_ a certain order of execution:

```python
from threading import Barrier, Thread
from unittest.mock import patch
from collections import Counter

def test_deposit_race_barrier():
    original_get = AccountBalance.get_balance

    def synced_get(self):
        result = original_get(self)
        barrier.wait()  # Both threads must read before either proceeds to write
        return result
    balances = Counter()

    for _ in range(100):
        account = AccountBalance(balance=0)
        barrier = Barrier(2)

        with patch.object(AccountBalance, 'get_balance', synced_get):
            t1 = Thread(target=account.deposit, args=(100,))
            t2 = Thread(target=account.deposit, args=(100,))
            t1.start(); t2.start()
            t1.join(); t2.join()
        balances[account._balance] += 1

    # Both threads read balance=0, then both write 0+100=100.
    # The second deposit is lost!
    assert balances == {200: 100}
```

```output
F                                                                        [100%]
=================================== FAILURES ===================================
__________________________ test_deposit_race_barrier ___________________________
<block>:41: in test_deposit_race_barrier
    assert balances == {200: 100}
E   assert Counter({100: 100}) == {200: 100}
E     
E     Left contains 1 more item:
E     {100: 100}
E     Right contains 1 more item:
E     {200: 100}
E     Use -v to get more diff
=========================== short test summary info ============================
FAILED ../../../..<block>::test_deposit_race_barrier
1 failed in 0.04s
```

By placing a `Barrier(2)` after the read, we guarantee that both threads have read the stale value before either proceeds to write.
This makes the race condition happen 100% of the time instead of ~never.
But setting it up often required intricate mocking or adding test-only hooks, so I wanted a very lightweight way of specifying code markers.

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
* Claude mentioned we could have users use different lock-primitives as with the the rust library [`loom`](https://github.com/tokio-rs/loom).
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

> Me: What are some absolutely terrible ideas for how we can intercept io attribute acces and other concurrency-relevant ideas. Vile perversions of the intention of how computers should work. Write them up in BAD_IDEAS.md.
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

## The End Result

I know this will probably come off as cocky or arrogant, but the end result is somewhere between _brilliant_ and just _unbelievably cool_.

I can point it at buggy code with something much like an ordinary unit test assertion, ending up with:

```python
from frontrun.dpor import explore_dpor

def test_test_balance():
    result = explore_dpor(
        setup=lambda: AccountBalance(),
        threads=[
            lambda bal: bal.deposit(100),
            lambda bal: bal.deposit(100),
        ],
        invariant=lambda bal: bal.get_balance() == 200,
    )
    assert result.property_holds, result.explanation
```

```output
F                                                                        [100%]
=================================== FAILURES ===================================
______________________________ test_test_balance _______________________________
<block>:26: in test_test_balance
    assert result.property_holds, result.explanation
E   AssertionError: Race condition found after 2 interleavings.
E     
E       Lost update: threads 0 and 1 both read _balance before either wrote it back.
E     
E     
E       Thread 0 | py:6     return self._balance
E                | [read AccountBalance._balance]
E       Thread 1 | py:6     return self._balance
E                | [read AccountBalance._balance]
E       Thread 1 | py:9     self._balance = value
E                | [write AccountBalance._balance]
E       Thread 0 | py:9     self._balance = value
E                | [write AccountBalance._balance]
E     
E       Reproduced 10/10 times (100%)
E     
E   assert False
E    +  where False = InterleavingResult(property_holds=False, counterexample=[0, 0, 0, 0, 0, ...(50 steps)], num_explored=2).property_holds
=========================== short test summary info ============================
FAILED ../../../..<block>::test_test_balance
1 failed in 0.05s
```


It can do this with database connections, files, setting variables, etc.
You can write test cases to either reproduce these race conditions or find them.
Like this could've easily been a paper or a pycon talk 3 years ago.
It would've easily taken me _years_ to implement this thing I built in two weeks.
This thing I where I did a substantial amount of development _on my phone_.

!["You made this" / "I made this?" meme with Claude Code](/images/2026-02-25-frontrun-race-conditions_files/claude-code-meme.png)

And of course, I had Claude move the meme to the right directory so it would render on the site, which prompted this beautiful exchange:
```
⏺ Now let me view the image to write a reasonable alt text, and add it to the end of the
  post.

⏺ Read(public/images/2026-02-25-frontrun-race-conditions_files/claude-code-meme.png)
  ⎿  Read image (215.8KB)

⏺ Great meme. Now let me add it to the end of the post.
```

So, in the interest of intellectual giving credit where it's due:

Claude:
  * Wrote ~100% of the code was written by claude.
    I don't know rust very well at all, and I'm certainly not an expert on all the different bytecodes used in different versions of python.
  * Came up with the idea of using DPOR and implemented the aglorithm.
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
