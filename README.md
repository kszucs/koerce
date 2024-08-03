Python Pattern Matching
-----------------------

Reusable pattern matching for Python, implemented in Cython.
I originally developed this system for the Ibis Project in pure python,
but it can be useful in many other contexts as well.

Examples
========

```python
from koerce import match, NoMatch
from koerce.sugar import Namespace
from koerce.patterns import SomeOf, ListOf

assert match([1, 2, 3, SomeOf(int, at_least=1)], four) == four
assert match([1, 2, 3, SomeOf(int, at_least=1)], three) is NoMatch

assert match(int, 1) == 1
assert match(ListOf(int), [1, 2, 3]) == [1, 2, 3]
```

```python
from dataclasses import dataclass
from koerce.sugar import match, Namespace, var
from koerce.patterns import pattern
from koerce.builder import builder

@dataclass
class A:
    x: int
    y: int

@dataclass
class B:
    x: int
    y: int
    z: float


p = Namespace(pattern, __name__)
d = Namespace(builder, __name__)

x = var("x")
y = var("y")

assert match(p.A(~x, ~y) >> d.B(x=x, y=1, z=y), A(1, 2)) == B(x=1, y=1, z=2)
```

More examples and a comprehensive readme are on the way.

Packages are not published to PyPI yet.

Python support follows https://numpy.org/neps/nep-0029-deprecation_policy.html
