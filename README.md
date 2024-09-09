# Performant Python Pattern Matching and Object Validation

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
a significant speedup. Benchmarks shows more than 2x speedup
over pydantic's model validation which is written in Rust.

## Installation

The package is published to PyPI, so it can be installed using
pip:

```sh
pip install koerce
```

## Library components

The library contains three main components which can be used
independently or together:

### 1. Deferred object builders

These allow delayed evaluation of python expressions given a
context:

```pycon
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

Patterns are the heart of the library, they allow **searching**
and **replacing** specific structures in Python objects. The
library provides an extensible yet simple way to define patterns
and match values against them.

```pycon
In [1]: from koerce import match, NoMatch, Anything

In [2]: context = {}

In [3]: match([1, 2, 3, int, "a" @ Anything()], [1, 2, 3, 4, 5], context)
Out[3]: [1, 2, 3, 4, 5]

In [4]: context
Out[4]: {'a': 5}
```

Note that `from koerce import koerce` function can be used instead
of `match()` to avoid confusion with the built-in python `match`.

```py
from dataclasses import dataclass
from koerce import Object, match

@dataclass
class B:
    x: int
    y: int
    z: float

match(Object(B, y=1, z=2), B(1, 1, 2))
# B(x=1, y=1, z=2)
```

where the `Object` pattern checks whether the passed object is
an instance of `B` and `value.y == 1` and `value.z == 2` ignoring
the `x` field.

Patterns are also able to capture values as variables making the
matching process more flexible:

```py
from koerce import var

x = var("x")

# `+x` means to capture that object argument as variable `x`
# then the `z` argument must match that captured value
match(Object(B, +x, z=x), B(1, 2, 1))
# it is a match because x and z are equal: B(x=1, y=2, z=1)

match(Object(B, +x, z=x), B(1, 2, 0))
# is is a NoMatch because x and z are unequal
```

Patterns also suitable for match and replace tasks because they
can produce new values:

```py
# >> operator constructs a `Replace` pattern where the right
# hand side is a deferred object
match(Object(B, +x, z=x) >> (x, x + 1), B(1, 2, 1))
# result: (1, 2)
```

Patterns are also composable and can be freely combined using
overloaded operators:

```pycon
In [1]: from koerce import match, Is, Eq, NoMatch

In [2]: pattern = Is(int) | Is(str)
   ...: assert match(pattern, 1) == 1
   ...: assert match(pattern, "1") == "1"
   ...: assert match(pattern, 3.14) is NoMatch

In [3]: pattern = Is(int) | Eq(1)
   ...: assert match(pattern, 1) == 1
   ...: assert match(pattern, None) is NoMatch
```

Patterns can also be constructed from python typehints:

```pycon
In [1]: from koerce import match

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
   ...:             raise ValueError("Cannot coerce value to Coercible")
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

```pycon
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

```pycon
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

## Available Pattern matchers

It is an incompletee list of the matchers, for more details and
examples see `koerce/patterns.py` and `koerce/tests/test_patterns.py`.

### `Anything` and `Nothing`

```pycon
In [1]: from koerce import match, Anything, Nothing

In [2]: match(Anything(), "a")
Out[2]: 'a'

In [3]: match(Anything(), 1)
Out[3]: 1

In [4]: match(Nothing(), 1)
Out[4]: koerce._internal.NoMatch
```

### `Eq` for equality matching

```pycon
In [1]: from koerce import Eq, match, var

In [2]: x = var("x")

In [3]: match(Eq(1), 1)
Out[3]: 1

In [4]: match(Eq(1), 2)
Out[4]: koerce._internal.NoMatch

In [5]: match(Eq(x), 2, context={"x": 2})
Out[5]: 2

In [6]: match(Eq(x), 2, context={"x": 3})
Out[6]: koerce._internal.NoMatch
```

### `Is` for instance matching

Couple simple cases are below:

```pycon
In [1]: from koerce import match, Is

In [2]: class A: pass

In [3]: match(Is(A), A())
Out[3]: <__main__.A at 0x1061070e0>

In [4]: match(Is(A), "A")
Out[4]: koerce._internal.NoMatch

In [5]: match(Is(int), 1)
Out[5]: 1

In [6]: match(Is(int), 3.14)
Out[6]: koerce._internal.NoMatch

In [7]: from typing import Optional

In [8]: match(Is(Optional[int]), 1)
Out[8]: 1

In [9]: match(Is(Optional[int]), None)
```

Generic types are also supported by checking types of attributes / properties:

```py
from koerce import match, Is, NoMatch
from typing import Generic, TypeVar, Any
from dataclasses import dataclass


T = TypeVar("T", covariant=True)
S = TypeVar("S", covariant=True)

@dataclass
class My(Generic[T, S]):
    a: T
    b: S
    c: str


MyAlias = My[T, str]

b_int = My(1, 2, "3")
b_float = My(1, 2.0, "3")
b_str = My("1", "2", "3")

# b_int.a must be an instance of int
# b_int.b must be an instance of Any
assert match(My[int, Any], b_int) is b_int

# both b_int.a and b_int.b must be an instance of int
assert match(My[int, int], b_int) is b_int

# b_int.b should be an instance of a float but it isn't
assert match(My[int, float], b_int) is NoMatch

# now b_float.b is actually a float so it is a match
assert match(My[int, float], b_float) is b_float

# type aliases are also supported
assert match(MyAlias[str], b_str) is b_str
```

### `As` patterns attempting to coerce the value as the given type

```py
from koerce import match, As, NoMatch
from typing import Generic, TypeVar, Any
from dataclasses import dataclass

class MyClass:
    pass

class MyInt(int):
    @classmethod
    def __coerce__(cls, other):
        return MyInt(int(other))


class MyNumber(Generic[T]):
    value: T

    def __init__(self, value):
        self.value = value

    @classmethod
    def __coerce__(cls, other, T):
        return cls(T(other))


assert match(As(int), 1.0) == 1
assert match(As(str), 1.0) == "1.0"
assert match(As(float), 1.0) == 1.0
assert match(As(MyClass), "myclass") is NoMatch

# by implementing the coercible protocol objects can be transparently
# coerced to the given type
assert match(As(MyInt), 3.14) == MyInt(3)

# coercible protocol also supports generic types where the `__coerce__`
# method should be implemented on one of the base classes and the
# type parameters are passed as keyword arguments to `cls.__coerce__()`
assert match(As(MyNumber[float]), 8).value == 8.0
```

`As` and `Is` can be omitted because `match()` tries to convert its
first argument to a pattern using the `koerce.pattern()` function:

```py
from koerce import pattern, As, Is

assert pattern(int, allow_coercion=False) == Is(int)
assert pattern(int, allow_coercion=True) == As(int)

assert match(int, 1, allow_coercion=False) == 1
assert match(int, 1.1, allow_coercion=False) is NoMatch
# lossy coercion is not allowed
assert match(int, 1.1, allow_coercion=True) is NoMatch

# default is allow_coercion=False
assert match(int, 1.1) is NoMatch
```

`As[typehint]` and `Is[typehint]` can be used to create patterns:

```py
from koerce import Pattern, As, Is

assert match(As[int], '1') == 1
assert match(Is[int], 1) == 1
assert match(Is[int], '1') is NoMatch
```

### `If` patterns for conditionals

Allows conditional matching based on the value of the object,
or other variables in the context:

```py
from koerce import match, If, Is, var, NoMatch, Capture

x = var("x")

pattern = Capture(x) & If(x > 0)
assert match(pattern, 1) == 1
assert match(pattern, -1) is NoMatch
```

### `Custom` for user defined matching logic

A function passed to either `match()` or `pattern()` is treated
as a `Custom` pattern:

```py
from koerce import match, Custom, NoMatch, NoMatchError

def is_even(value):
    if value % 2:
        raise NoMatchError("Value is not even")
    else:
        return value

assert match(is_even, 2) == 2
assert match(is_even, 3) is NoMatch
```

### `Capture` to record values in the context

A capture pattern can be defined several ways:

```py
from koerce import Capture, Is, var

x = var("x")

Capture("x")  # captures anything as "x" in the context
Capture(x)  # same as above but using a variable
Capture("x", Is(int))  # captures only integers as "x" in the context
Capture("x", Is(int) | Is(float))  # captures integers and floats as "x" in the context
"x" @ Is(int)  # syntax sugar for Capture("x", Is(int))
+x  # syntax sugar for Capture(x, Anything())
```

```py
from koerce import match, Capture, var

# context is a mutable dictionary passed along the matching process
context = {}
assert match("x" @ Is(int), 1, context) == 1
assert context["x"] == 1
```

### `Replace` for replacing matched values

Allows replacing matched values with new ones:

```py
from koerce import match, Replace, var

x = var("x")

pattern = Replace(Capture(x), x + 1)
assert match(pattern, 1) == 2
assert match(pattern, 2) == 3
```

there is a syntax sugar for `Replace` patterns, the example above
can be written as:

```py
from koerce import match, Replace, var

x = var("x")

assert match(+x >> x + 1, 1) == 2
assert match(+x >> x + 1, 2) == 3
```

replace patterns are especially useful when matching objects:

```py
from dataclasses import dataclass
from koerce import match, Replace, var, namespace

x = var("x")

@dataclass
class A:
    x: int
    y: int

@dataclass
class B:
    x: int
    y: int
    z: float


p, d = namespace(__name__)
x, y = var("x"), var("y")

# if value is an instance of A then capture A.0 as x and A.1 as y
# then construct a new B object with arguments x=x, y=1, z=y
pattern = p.A(+x, +y) >> d.B(x=x, y=1, z=y)
value = A(1, 2)
expected = B(x=1, y=1, z=2)
assert match(pattern, value) == expected
```

replacemenets can also be used in nested structures:

```py
from koerce import match, Replace, var, namespace, NoMatch

@dataclass
class Foo:
    value: str

@dataclass
class Bar:
    foo: Foo
    value: int

p, d = namespace(__name__)

pattern = p.Bar(p.Foo("a") >> d.Foo("b"))
value = Bar(Foo("a"), 123)
expected = Bar(Foo("b"), 123)

assert match(pattern, value) == expected
assert match(pattern, Bar(Foo("c"), 123)) is NoMatch
```

### `SequenceOf` / `ListOf` / `TupleOf`

```py
from koerce import Is, NoMatch, match, ListOf, TupleOf

pattern = ListOf(str)
assert match(pattern, ["foo", "bar"]) == ["foo", "bar"]
assert match(pattern, [1, 2]) is NoMatch
assert match(pattern, 1) is NoMatch
```

### `MappingOf` / `DictOf` / `FrozenDictOf`

```py
from koerce import DictOf, Is, match

pattern = DictOf(Is(str), Is(int))
assert match(pattern, {"a": 1, "b": 2}) == {"a": 1, "b": 2}
assert match(pattern, {"a": 1, "b": "2"}) is NoMatch
```

### `PatternList`

```py
from koerce import match, NoMatch, SomeOf, ListOf, pattern

four = [1, 2, 3, 4]
three = [1, 2, 3]

assert match([1, 2, 3, SomeOf(int, at_least=1)], four) == four
assert match([1, 2, 3, SomeOf(int, at_least=1)], three) is NoMatch

integer = pattern(int, allow_coercion=False)
floating = pattern(float, allow_coercion=False)

assert match([1, 2, *floating], [1, 2, 3]) is NoMatch
assert match([1, 2, *floating], [1, 2, 3.0]) == [1, 2, 3.0]
assert match([1, 2, *floating], [1, 2, 3.0, 4.0]) == [1, 2, 3.0, 4.0]
```

### `PatternMap`

```py
from koerce import match, NoMatch, Is, As

pattern = {
    "a": Is(int),
    "b": As(int),
    "c": Is(str),
    "d": ListOf(As(int)),
}
value = {
    "a": 1,
    "b": 2.0,
    "c": "three",
    "d": (4.0, 5.0, 6.0),
}
assert match(pattern, value) == {
    "a": 1,
    "b": 2,
    "c": "three",
    "d": [4, 5, 6],
}
assert match(pattern, {"a": 1, "b": 2, "c": "three"}) is NoMatch
```

## Annotable objects

Annotable objects are similar to dataclasses but with some differences:
- Annotable objects are mutable by default, but can be made immutable
  by passing `immutable=True` to the `Annotable` base class.
- Annotable objects can be made hashable by passing `hashable=True` to
  the `Annotable` base class, in this case the hash is precomputed during
  initialization and stored in the object making the dictionary lookups
  cheap.
- Validation strictness can be controlled by passing `allow_coercion=False`.
  When `allow_coercion=True` the annotations are treated as `As` patterns
  allowing the values to be coerced to the given type. When
  `allow_coercion=False` the annotations are treated as `Is` patterns and
  the values must be exactly of the given type. The default is
  `allow_coercion=True`.
- Annotable objects support inheritance, the annotations are inherited
  from the base classes and the signatures are merged providing a
  seamless experience.
- Annotable objects can be called with either or both positional and
  keyword arguments, the positional arguments are matched to the
  annotations in order and the keyword arguments are matched to the
  annotations by name.

```py
from typing import Optional
from koerce import Annotable

class MyBase(Annotable):
    x: int
    y: float
    z: Optional[str] = None

class MyClass(MyBase):
    a: str
    b: bytes
    c: tuple[str, ...] = ("a", "b")
    x: int = 1


print(MyClass.__signature__)
# (y: float, a: str, b: bytes, c: tuple = ('a', 'b'), x: int = 1, z: Optional[str] = None)

print(MyClass(2.0, "a", b"b"))
# MyClass(y=2.0, a='a', b=b'b', c=('a', 'b'), x=1, z=None)

print(MyClass(2.0, "a", b"b", c=("c", "d")))
# MyClass(y=2.0, a='a', b=b'b', c=('c', 'd'), x=1, z=None)

print(MyClass(2.0, "a", b"b", c=("c", "d"), x=2))
# MyClass(y=2.0, a='a', b=b'b', c=('c', 'd'), x=2, z=None)

print(MyClass(2.0, "a", b"b", c=("c", "d"), x=2, z="z"))
# MyClass(y=2.0, a='a', b=b'b', c=('c', 'd'), x=2, z='z')

MyClass()
# TypeError: missing a required argument: 'y'

MyClass(2.0, "a", b"b", c=("c", "d"), x=2, z="z", invalid="invalid")
# TypeError: got an unexpected keyword argument 'invalid'

MyClass(2.0, "a", b"b", c=("c", "d"), x=2, z="z", y=3.0)
# TypeError: multiple values for argument 'y'

MyClass("asd", "a", b"b")
# ValidationError
```

## Performance

`koerce`'s performance is at least comparable to `pydantic`'s performance.
`pydantic-core` is written in rust using the `PyO3` bindings making it
a pretty performant library. There is a quicker validation / serialization
library from `Jim Crist-Harif` called [msgspec](https://github.com/jcrist/msgspec)
implemented in hand-crafted C directly using python's C API.

`koerce` is not exactly like `pydantic` or `msgpec` but they are good
candidates to benchmark against:

```
koerce/tests/test_y.py::test_pydantic PASSED
koerce/tests/test_y.py::test_msgspec PASSED
koerce/tests/test_y.py::test_annotated PASSED


------------------------------------------------------------------------------------------- benchmark: 3 tests ------------------------------------------------------------------------------------------
Name (time in ns)            Min                   Max                  Mean              StdDev                Median                IQR            Outliers  OPS (Kops/s)            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_msgspec            230.2801 (1.0)      6,481.4200 (1.60)       252.1706 (1.0)       97.0572 (1.0)        238.1600 (1.0)       5.0002 (1.0)      485;1616    3,965.5694 (1.0)       20000          50
test_annotated          525.6401 (2.28)     4,038.5600 (1.0)        577.7090 (2.29)     132.9966 (1.37)       553.9799 (2.33)     34.9300 (6.99)      662;671    1,730.9752 (0.44)      20000          50
test_pydantic         1,185.0201 (5.15)     6,027.9400 (1.49)     1,349.1259 (5.35)     320.3790 (3.30)     1,278.5601 (5.37)     75.5100 (15.10)   1071;1424      741.2206 (0.19)      20000          50
```

I tried to used the most performant API of both `msgspec` and `pydantic`
receiving the arguments as a dictionary.

I am planning to make more thorough comparisons, but the model-like
annotation API of `koerce` is roughly twice as fast as `pydantic` but
half as fast as `msgspec`. Considering the implementations it also
makes sense, `PyO3` possible has a higher overhead than `Cython` has
but neither of those can match the performance of hand crafted python
`C-API` code.

This performance result could be slightly improved but has two huge
advantage of the other two libraries:
1. It is implemented in pure python with cython decorators, so it
   can be used even without compiling it. It could also enable
   JIT compilers like PyPy or the new copy and patch JIT compiler
   coming with CPython 3.13 to optimize hot paths better.
2. Development an be done in pure python make it much easier to
   contribute to. No one needs to learn Rust or python's C API
   in order to fix bugs or contribute new features.

## TODO:

The README is under construction, planning to improve it:
- [ ] Example of validating functions by using @annotated decorator
- [ ] Explain `allow_coercible` flag
- [ ] Proper error messages for each pattern

## Development

- The project uses `poetry` for dependency management and packaging.
- Python version support follows https://numpy.org/neps/nep-0029-deprecation_policy.html
- The wheels are built using `cibuildwheel` project.
- The implementation is in pure python with cython annotations.
- The project uses `ruff` for code formatting.
- The project uses `pytest` for testing.

More detailed developer guide is coming soon.

## References

The project was mostly inspired by the following projects:
- https://github.com/scravy/awesome-pattern-matching
- https://github.com/HPAC/matchpy
