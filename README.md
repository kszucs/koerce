# Python Pattern Matching and Object Validation


Reusable pattern matching for Python, implemented in Cython.
I originally developed this system for the Ibis Project but
hopefully it can be useful for others as well.


The implementation aims to be as quick as possible, the pure
python implementation is already quite fast but taking advantage
of Cython allows to mitigate the overhead of the Python
interpreter.
I have also tried to use PyO3 but it had higher overhead than
Cython. The current implementation uses the pure python mode
of cython allowing quick iteration and testing, and then it
can be cythonized and compiled to an extension module giving
a significant speedup.


## Library components

The library contains three main components which can be used
independently or together:

### 1. Deferred object builders

These allow delayed evaluation of python expressions given a
context:

```py
In [1]: from koerce import var, resolve

In [2]: a, b = var("a"), var("b")

In [3]: expr = (a + 1) * b["field"]

In [4]: expr
Out[4]: (($a + 1) * $b['field'])

In [5]: resolve(expr, {"a": 2, "b": {"field": 3}})
Out[5]: 9
```

The syntax sugar provided by the deferred objects allows the
definition of complex object transformations in a concise and
natural way.


### 2. Pattern matchers which operate on various Python objects

Patterns are the heart of the library, they allow searching for
specific structures in Python objects. The library provides an
extensible yet simple way to define patterns and match values
against them.

```py
In [1]: from koerce import match, NoMatch, Anything

In [2]: context = {}

In [3]: match([1, 2, 3, int, "a" @ Anything()], [1, 2, 3, 4, 5], context)
Out[3]: [1, 2, 3, 4, 5]

In [4]: context
Out[4]: {'a': 5}
```

```py
In [1]: from dataclasses import dataclass

In [2]: @dataclass
   ...: class B:
   ...:     x: int
   ...:     y: int
   ...:     z: float
   ...:

In [3]: match(Object(B, y=1, z=2), B(1, 1, 2))
Out[3]: B(x=1, y=1, z=2)

In [4]: Object(B, y=1, z=2)
Out[4]: ObjectOf2(<class '__main__.B'>, 'y'=EqValue(1), 'z'=EqValue(2))
```

where the `Object` pattern checks whether the passed object is
an instance of `B` and `value.y == 1` and `value.z == 2` ignoring
the `x` field.

Patterns can also be constructed from python typehints:

```py
In [1]: from koerce import match, CoercionError

In [2]: class Ordinary:
   ...:     def __init__(self, x, y):
   ...:         self.x = x
   ...:         self.y = y
   ...:
   ...:
   ...: class Coercible(Ordinary):
   ...:
   ...:     @classmethod
   ...:     def __coerce__(cls, value):
   ...:         if isinstance(value, tuple):
   ...:             return Coercible(value[0], value[1])
   ...:         else:
   ...:             raise CoercionError("Cannot coerce value to Coercible")
   ...:

In [3]: match(Ordinary, Ordinary(1, 2))
Out[3]: <__main__.Ordinary at 0x105194fe0>

In [4]: match(Ordinary, (1, 2))
Out[4]: koerce.patterns.NoMatch

In [5]: match(Coercible, (1, 2))
Out[5]: <__main__.Coercible at 0x109ebb320>
```

The pattern creation logic also handles generic types by doing
lightweight type parameter inference. The implementation is quite
compact, available under `Pattern.from_typehint()`.

### 3. A high-level validation system for dataclass-like objects

This abstraction is similar to what attrs or pydantic provide but
there are some differences (TODO listing them).

```py
In [1]: from typing import Optional
   ...: from koerce import Annotable
   ...:
   ...:
   ...: class MyClass(Annotable):
   ...:     x: int
   ...:     y: float
   ...:     z: Optional[list[str]] = None
   ...:

In [2]: MyClass(1, 2.0, ["a", "b"])
Out[2]: MyClass(x=1, y=2.0, z=['a', 'b'])

In [3]: MyClass(1, 2, ["a", "b"])
Out[3]: MyClass(x=1, y=2.0, z=['a', 'b'])

In [4]: MyClass("invalid", 2, ["a", "b"])
Out[4]: # raises validation error
```

Annotable object are mutable by default, but can be made immutable
by passing `immutable=True` to the `Annotable` base class. Often
it is useful to make immutable objects hashable as well, which can
be done by passing `hashable=True` to the `Annotable` base class,
in this case the hash is precomputed during initialization and
stored in the object making the dictionary lookups cheap.

```py
In [1]: from typing import Optional
   ...: from koerce import Annotable
   ...:
   ...:
   ...: class MyClass(Annotable, immutable=True, hashable=True):
   ...:     x: int
   ...:     y: float
   ...:     z: Optional[tuple[str, ...]] = None
   ...:

In [2]: a = MyClass(1, 2.0, ["a", "b"])

In [3]: a
Out[3]: MyClass(x=1, y=2.0, z=('a', 'b'))

In [4]: a.x = 2
AttributeError: Attribute 'x' cannot be assigned to immutable instance of type <class '__main__.MyClass'>

In [5]: {a: 1}
Out[5]: {MyClass(x=1, y=2.0, z=('a', 'b')): 1}
```

## TODO:

The README is under construction, planning to improve it:
- [ ] More advanced matching examples
- [ ] Add benchmarks against pydantic
- [ ] Show variable capturing
- [ ] Show match and replace in nested structures
- [ ] Example of validating functions by using @annotated decorator
- [ ] Explain `allow_coercible` flag
- [ ] Mention other relevant libraries

## Other examples


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
