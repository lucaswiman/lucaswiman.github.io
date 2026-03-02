# Announcing Frontrun

## tl;dr


Check out the [documentation](https://lucaswiman.github.io/frontrun), the [GitHub repo](https://github.com/lucaswiman/frontrun), or install it from [PyPI](https://pypi.org/project/frontrun/):

```
pip install frontrun
```

I also wrote a [follow-up post about building frontrun with Claude Code](/blog/2026-02-26-building-frontrun-with-claude).

## Concurrency testing

For the past few weeks, I've been using Claude Code to build a concurrency testing library called `frontrun`.
The genesis of the idea was a useful-but-clunky category of concurrency test I've been using for the past several years involving `threading.Event` or `threading.Barrier` to reproduce race conditions by forcing a particular execution ordering in test.
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
    # The second deposit (could be) lost!
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
But setting it up often required intricate mocking or adding test-only hooks, so I wanted a lightweight way to specify the interleaving to reproduce.

## Testing under `frontrun`

After a few weeks of building with Claude Code (more on that [in a follow-up post](/blog/2026-02-26-building-frontrun-with-claude)), I ended up with something that I think is pretty promising.

If you define an _invariant_, you can point `frontrun` at buggy code with something much like an ordinary unit test assertion.
In our case, the invariant is that both of the deposits should be counted, so no matter how you order the test, you should end up with a balance of $200.
Frontrun can then often output an exact sequence of events required to trigger the race condition.

```python
from frontrun.dpor import explore_dpor

def test_balance():
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
______________________________ test_balance _______________________________
<block>:26: in test_balance
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
FAILED ../../../..<block>::test_balance
1 failed in 0.05s
```

It can do this with database connections, files, setting variables, etc.
You can write test cases to either reproduce these race conditions or find them.

## How it works

`frontrun` supports three modes of execution detailed below.
These are supported by several layers of monkey-patching:
1. Execution tracing to control ordering of events.
2. Patching python threading and IO primitives to prevent deadlocks and other concurrency bugs freezing your test suite.
   We use "spin locks", which keep track of which thread is waiting on what lock, allowing us to identify deadlocks without actually deadlocking.
   The patching of these objects is done when the pytest plugin loads, which allows us to force your application to use the special lock without needing to mock.patch every `from threading import Lock` statement.
3. We also patch libc interaction with file descriptors using `LD_PRELOAD`.
   This means we can detect that conflicts may occur against resources like a filename or database connection and use them in the DPOR analysis below.

Because of (3), you need to run frontrun tests in a separate invocation (`frontrun pytest ...`), and the frontrun tests will automatically skip themselves if monkeypatching is not already set up.

### Trace Markers

These are comment annotations in the code like `# frontrun: after_write` which can then run in a particular schedule you specify in your test.
This mode is useful for reproducing race conditions you already know are there and want to verify using test-driven development. It's basically a nicer syntax to the `Barrier` approach above that doesn't require directly mocking your code or altering its runtime behavior.

### ["DPOR"](https://en.wikipedia.org/wiki/Partial_order_reduction)

This is a method from formal verification of concurrent systems (see Flanagan & Godefroid, 2005; cited in the wikipedia page) where certain events (variable reads, writes, locks, etc.) are annotated and their "causal" structure is kept track of.
In the example above, there are four events: two reads across the two threads (R1 and R2) and two writes (W1 and W2).

* One possible ordering is `R1W1R2W2`.
  In this case, R2 causally depends on W1.
* Another is `R1R2W1W2`.
  In this case, R2 _does not_ causally depend on W1.

Concurrency control primitives like locks disallow certain event orderings.
Then the causal structure also means that some orderings are _equivalent_, so we only need to run one from each equivalence class to verify if a race condition occurs, e.g. `R1R2W1W2` is causally equivalent to `R2R1W1W2`.
This can exponentially reduce the size of the search space and also guarantee completeness (you've explored all possible orderings of events up to causal equivalence.)

### Byte Code Shuffling

We use python tracing operations like `set_trace` to interleave python bytecode operations according to a randomly assigned schedule.
Here the idea is similar to ordinary property-based testing, but the random data under test is the _order of events_ rather than the input data.
(Though it should be possible to compose data and event ordering using `hypothesis` strategies.)
While it sounds like this should hit some kind of exponential blowup and almost never trigger race conditions, many race conditions are _very_ easy to trigger if two events take place around the same time (e.g. starting two threads in succession in a unit test).
They might rarely show up in production because most events against a particular record take place at different times until a user starts rage clicking or whatever.

## Conclusion

While this is still alpha-level software (I find bugs in it almost every time I test it), I think it's now mature enough for others to start kicking the tires and hopefully end up with higher quality software as a result.
I'd be very interested in hearing about bugs in the implementation or success stories.
