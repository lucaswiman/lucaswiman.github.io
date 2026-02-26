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

def test_deposit_not_threadsafe():
    """This test SHOULD fail, but almost never does."""
    account = AccountBalance(balance=0)
    t1 = Thread(target=account.deposit, args=(100,))
    t2 = Thread(target=account.deposit, args=(100,))
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert account._balance == 200  # Almost always passes!
```

```output
/Users/lucaswiman/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none/bin/python: No module named pytest
```

And some changes may reduce the likelihood of races without actually eliminating them, so how can you be _sure_ you eliminated the race condition you found?
The key idea is to use `threading.Event` or `threading.Barrier` objects to force a certain order of execution:

```python
from threading import Barrier, Thread
from unittest.mock import patch

def test_deposit_race():
    account = AccountBalance(balance=0)
    barrier = Barrier(2)

    original_get = AccountBalance.get_balance

    def synced_get(self):
        result = original_get(self)
        barrier.wait()  # Both threads must read before either proceeds to write
        return result

    with patch.object(AccountBalance, 'get_balance', synced_get):
        t1 = Thread(target=account.deposit, args=(100,))
        t2 = Thread(target=account.deposit, args=(100,))
        t1.start(); t2.start()
        t1.join(); t2.join()

    # Both threads read balance=0, then both write 0+100=100.
    # The second deposit is lost!
    assert account._balance == 100  # 200 expected, but only 100!
```

```output
/Users/lucaswiman/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none/bin/python: No module named pytest
```

By placing a `Barrier(2)` after the read, we guarantee that both threads have read the stale value before either proceeds to write. This makes the race condition happen 100% of the time instead of once in a million runs.
