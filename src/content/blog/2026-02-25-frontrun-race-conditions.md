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
But setting it up often required intricate mocking or adding test-only hooks, so I wanted a lightweight way to specify the interleaving to reproduce. After a few weeks of building with Claude Code (more on that [in a follow-up post](/blog/2026-02-26-building-frontrun-with-claude)), I ended up with something I'm very happy with.

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

Check out the [documentation](https://lucaswiman.github.io/frontrun), the [GitHub repo](https://github.com/lucaswiman/frontrun), or install it from [PyPI](https://pypi.org/project/frontrun/):

```
pip install frontrun
```

I also wrote a [follow-up post about building frontrun with Claude Code](/blog/2026-02-26-building-frontrun-with-claude).
