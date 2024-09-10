from __future__ import annotations

import functools
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import pytest
from typing_extensions import Self

from koerce import NoMatch, match
from koerce._internal import (
    AllOf,
    AnyOf,
    Anything,
    As,
    AsBool,
    AsBuiltin,
    AsCoercible,
    AsCoercibleGeneric,
    AsInt,
    AsType,
    Call,
    CallableWith,
    Capture,
    Custom,
    Deferred,
    DictOf,
    EqValue,
    FrozenDictOf,
    IdenticalTo,
    If,
    Is,
    IsGeneric,
    IsGeneric1,
    IsGeneric2,
    IsGenericN,
    IsIn,
    IsType,
    IsTypeLazy,
    ListOf,
    MappingOf,
    MatchError,
    NoneOf,
    Not,
    Nothing,
    Object,
    ObjectOf1,
    ObjectOf2,
    ObjectOf3,
    ObjectOfN,
    ObjectOfX,
    Option,
    Pattern,
    PatternList,
    PatternMap,
    Replace,
    SequenceOf,
    SomeChunksOf,
    SomeOf,
    TupleOf,
    TypeOf,
    Var,
    pattern,
)
from koerce.utils import FrozenDict


@AsType.register(datetime)
def as_datetime(cls, value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    elif isinstance(value, str):
        return datetime.fromisoformat(value)
    else:
        raise ValueError("Cannot coerce to datetime")


class Min:
    def __init__(self, min):
        self.min = min

    def __eq__(self, other):
        return isinstance(other, Min) and self.min == other.min

    def __call__(self, value, **context):
        if value >= self.min:
            return value
        else:
            raise ValueError(f"{value} is less than {self.min}")


various_values = [1, "1", 1.0, object, False, None]


@pytest.mark.parametrize("value", various_values)
def test_anything(value):
    assert Anything().apply(value) == value


@pytest.mark.parametrize("value", various_values)
def test_nothing(value):
    with pytest.raises(MatchError, match="never matches"):
        Nothing().apply(value)


@pytest.mark.parametrize(
    ("inner", "default", "value", "expected"),
    [
        (Anything(), None, None, None),
        (Anything(), None, "three", "three"),
        (Anything(), 1, None, 1),
        (As(int), 11, None, 11),
        (As(int), None, None, None),
        (As(int), None, 18, 18),
    ],
)
def test_option(inner, default, value, expected):
    p = Option(inner, default=default)
    assert p.apply(value) == expected


@pytest.mark.parametrize("value", various_values)
def test_identical_to(value):
    assert IdenticalTo(value).apply(value) == value

    with pytest.raises(MatchError, match="is not identical to"):
        IdenticalTo(value).apply(2)

    with pytest.raises(MatchError, match="is not identical to"):
        IdenticalTo(value).apply("2")


@pytest.mark.parametrize(
    ["a", "b"],
    [
        (1, 1),
        ("1", "1"),
        (1.0, 1.0),
        (1.0, 1),
        ("", ""),
        (False, False),
        (set(), set()),
        (set(), frozenset()),
        (object, object),
    ],
)
def test_equal_to(a, b):
    pattern = EqValue(a)
    assert pattern.apply(b) is b


@pytest.mark.parametrize(
    ["a", "b"],
    [
        (1, 2),
        ("1", "2"),
        (1.0, 2.0),
        (1.0, 2),
        (set(), []),
    ],
)
def test_equal_to_failing(a, b):
    pattern = EqValue(a)
    with pytest.raises(MatchError, match="is not equal to"):
        assert pattern.apply(b)


@pytest.mark.parametrize(
    ["typ", "value"],
    [
        (int, 1),
        (str, "1"),
        (int, True),
        (int, False),
        (bool, True),
        (bool, False),
        (list, [1, 2]),
    ],
)
def test_instance_of(typ, value):
    pattern = IsType(typ)
    assert pattern.apply(value) is value


@pytest.mark.parametrize(
    ["typ", "value"],
    [
        (int, 1.0),
        (str, 1),
        (bool, 1),
        (bool, 0),
        (list, (1, 2)),
    ],
)
def test_instance_of_failing(typ, value):
    pattern = IsType(typ)
    with pytest.raises(MatchError, match="is not an instance of"):
        pattern.apply(value)


@pytest.mark.parametrize(
    ("typ", "value"),
    [
        (int, 1),
        (str, "1"),
        (bool, True),
        (bool, False),
        (list, [1, 2]),
    ],
)
def test_type_of(typ, value):
    pattern = TypeOf(typ)
    assert pattern.apply(value) is value


@pytest.mark.parametrize(
    ("typ", "value"),
    [
        (int, 1.0),
        (str, 1),
        (int, True),
        (int, False),
        (bool, 1),
        (bool, 0),
        (list, (1, 2)),
    ],
)
def test_type_of_failing(typ, value):
    pattern = TypeOf(int)
    with pytest.raises(MatchError, match="doesn't have the exact type of"):
        pattern.apply(1.0)


class MyMeta(type): ...


class MyClass(metaclass=MyMeta): ...


class MyOtherClass: ...


def test_type_of_with_metaclass():
    p = TypeOf(MyClass)
    v = MyClass()
    assert p.apply(v) is v
    with pytest.raises(MatchError, match="doesn't have the exact type of"):
        p.apply(MyOtherClass())


def test_lazy_instance_of():
    # pick a rarely used stdlib module
    assert "graphlib" not in sys.modules
    p = IsTypeLazy("graphlib.TopologicalSorter")
    assert "graphlib" not in sys.modules

    with pytest.raises(
        MatchError, match="is not an instance of 'graphlib.TopologicalSorter'"
    ):
        p.apply(1)
    assert "graphlib" not in sys.modules

    import graphlib

    sorter = graphlib.TopologicalSorter()
    assert p.apply(sorter) is sorter
    with pytest.raises(
        MatchError, match="is not an instance of 'graphlib.TopologicalSorter'"
    ):
        p.apply("foo")


# covariance is reqired at the moment
T = TypeVar("T", covariant=True)
S = TypeVar("S", covariant=True)
U = TypeVar("U", covariant=True)


@dataclass
class My(Generic[T, S]):
    a: T
    b: S
    c: str


@dataclass
class My1(Generic[T]):
    a: T


@dataclass
class My2(Generic[T, S]):
    a: T
    b: S


@dataclass
class My3(Generic[T, S, U]):
    a: T
    b: S
    c: U


class MyCoercible(Generic[T, S]):
    a: T
    b: S

    def __init__(self, a: T, b: S):
        self.a = a
        self.b = b

    @classmethod
    def __coerce__(cls, value, T, S):
        return cls(T(value), S(value))


def test_is_instance():
    assert Is(int) == IsType(int)
    assert Is(My1[int]) == IsGeneric1(My1[int])
    assert Is(My2[int, float]) == IsGeneric2(My2[int, float])
    assert Is(My3[int, float, str]) == IsGenericN(My3[int, float, str])
    assert Is("pandas.DataFrame") == IsTypeLazy("pandas.DataFrame")
    assert Is(MyCoercible[int, float]) == IsGeneric(MyCoercible[int, float])


def test_as_instance():
    assert As(int) == AsInt()
    assert As(MyCoercible[int, float]) == AsCoercibleGeneric(MyCoercible[int, float])


def test_generic_instance_of():
    p = IsGeneric(My1[int])
    assert isinstance(p, IsGeneric1)

    p = IsGeneric(My2[int, float])
    assert isinstance(p, IsGeneric2)

    p = IsGeneric(My3[int, float, str])
    assert isinstance(p, IsGenericN)


def test_generic_instance_of_n():
    p = IsGenericN(My[int, Any])
    assert p.apply(v := My(1, 2, "3")) is v

    p = IsGenericN(My[int, int])
    assert p.apply(v := My(1, 2, "3")) is v

    p = IsGenericN(My[int, float])
    with pytest.raises(MatchError, match="is not an instance of"):
        p.apply(My(1, 2, "3"))
    assert p.apply(v := My(1, 2.0, "3")) is v

    MyAlias = My[T, str]
    p = IsGenericN(MyAlias[str])
    assert p.apply(v := My("1", "2", "3")) is v


def test_generic_instance_of_1():
    p = IsGeneric1(My1[int])
    assert p.apply(v := My1(1)) is v
    with pytest.raises(MatchError, match="is not an instance of"):
        p.apply(My1(1.0))


def test_generic_instance_of_2():
    p = IsGeneric2(My2[int, float])
    with pytest.raises(MatchError, match="is not an instance of"):
        p.apply(My2(1, 2))
    assert p.apply(v := My2(1, 2.0)) is v


def test_as():
    class MyInt(int):
        @classmethod
        def __coerce__(cls, other):
            return MyInt(int(other))

    class MyGenericInt(Generic[T]):
        value: T

        @classmethod
        def __coerce__(cls, other, T):
            return cls(T(other))

    assert isinstance(As(int), AsInt)
    assert isinstance(As(MyInt), AsCoercible)
    assert isinstance(As(MyGenericInt[int]), AsCoercibleGeneric)


def test_as_int():
    p = AsInt()
    assert p.apply(1) == 1
    assert p.apply(1.0) == 1
    assert p.apply("1") == 1

    with pytest.raises(
        MatchError, match=re.escape("Cannot losslessly convert 'foo' to an integer.")
    ):
        p.apply("foo")
    with pytest.raises(
        MatchError, match="Cannot losslessly convert 1.1 to an integer."
    ):
        p.apply(1.1)


def test_as_bool():
    p = AsBool()
    assert p.apply(True) is True
    assert p.apply(False) is False
    assert p.apply(1) is True
    assert p.apply(0) is False
    assert p.apply("1") is True
    assert p.apply("0") is False
    assert p.apply("True") is True
    assert p.apply("False") is False
    assert p.apply("true") is True
    assert p.apply("false") is False

    with pytest.raises(
        MatchError, match="Cannot losslessly convert 'foo' to a boolean."
    ):
        p.apply("foo")


def test_is_as_typehints():
    p = Pattern.from_typehint(Is[int], allow_coercion=False)
    assert p.apply(1) == 1
    with pytest.raises(MatchError, match="is not an instance of"):
        p.apply(1.0)

    p = Pattern.from_typehint(Is[int], allow_coercion=True)
    assert p.apply(1) == 1
    with pytest.raises(MatchError, match="is not an instance of"):
        p.apply(1.0)

    p = Pattern.from_typehint(As[int], allow_coercion=False)
    assert p.apply(1) == 1
    assert p.apply("1") == 1
    with pytest.raises(MatchError):
        p.apply("foo")

    p = Pattern.from_typehint(As[int], allow_coercion=True)
    assert p.apply(1) == 1
    assert p.apply("1") == 1
    assert p.apply(1.0) == 1
    with pytest.raises(MatchError):
        p.apply("foo")
    with pytest.raises(MatchError):
        p.apply("1.11")


def test_coerced_to():
    class MyInt(int):
        @classmethod
        def __coerce__(cls, other):
            return MyInt(MyInt(other) + 1)

    p = AsCoercible(MyInt)
    assert p.apply(1, context={}) == 2
    assert p.apply("1", context={}) == 2
    with pytest.raises(MatchError, match="`'foo'` cannot be coerced to"):
        p.apply("foo")


def test_generic_coerced_to():
    class DataType:
        def __eq__(self, other):
            return type(self) is type(other)

    class Integer(DataType):
        pass

    class String(DataType):
        pass

    class DataShape:
        def __eq__(self, other):
            return type(self) is type(other)

    class Scalar(DataShape):
        pass

    class Array(DataShape):
        pass

    class Value(Generic[T, S]):
        @classmethod
        def __coerce__(cls, value, T, S):
            if T is String:
                value = str(value)
            elif T is Integer:
                value = int(value)
            else:
                raise ValueError(f"Cannot coerce {value} to {T}")
            return cls(value, T())

        @property
        def dtype(self) -> T: ...

        @property
        def shape(self) -> S: ...

    class Literal(Value[T, Scalar]):
        __slots__ = ("_value", "_dtype")

        def __init__(self, value, dtype):
            self._value = value
            self._dtype = dtype

        @property
        def dtype(self) -> T:
            return self._dtype

        def __eq__(self, other):
            return (
                type(self) is type(other)
                and self._value == other._value
                and self._dtype == other._dtype
            )

    s = Literal("foo", String())
    i = Literal(1, Integer())

    p = IsGeneric(Literal[String])
    assert p.apply(s) is s
    with pytest.raises(MatchError, match="is not an instance of"):
        p.apply(i)

    p = AsCoercibleGeneric(Literal[String])
    assert p.apply("foo") == s
    assert p.apply(1) == Literal("1", String())


def test_not():
    p = Not(IsType(int))
    p1 = ~IsType(int)

    assert p == p1
    with pytest.raises(
        MatchError,
        match=re.escape("`1` is matching IsType(<class 'int'>) whereas it should not"),
    ):
        p.apply(1)
    assert p.apply("foo", context={}) == "foo"


def test_any_of():
    p = AnyOf(IsType(int), IsType(str))
    p1 = IsType(int) | IsType(str)

    assert p == p1
    assert p.apply(1) == 1
    assert p.apply("foo") == "foo"
    msg = re.escape(
        "`1.0` does not match any of [IsType(<class 'int'>), IsType(<class 'str'>)]"
    )
    with pytest.raises(MatchError, match=msg):
        p.apply(1.0)


def test_any_all_of_operator_overloading():
    is_int = IsType(int)
    is_str = IsType(str)
    is_float = IsType(float)

    assert (is_int | is_str) == AnyOf(is_int, is_str)
    assert (is_int & is_str) == AllOf(is_int, is_str)
    assert (is_int & is_str & is_float) == AllOf(is_int, is_str, is_float)
    assert (is_int | is_str | is_float) == AnyOf(is_int, is_str, is_float)
    assert (is_int | is_str & is_float) == AnyOf(is_int, AllOf(is_str, is_float))
    assert ((is_int | is_str) | (is_float | is_int)) == AnyOf(
        is_int, is_str, is_float, is_int
    )
    assert ((is_int & is_str) & (is_float & is_int)) == AllOf(
        is_int, is_str, is_float, is_int
    )


def test_all_of():
    def negative(_):
        return _ < 0

    p = AllOf(IsType(int), If(negative))
    p1 = IsType(int) & If(negative)

    assert p == p1
    with pytest.raises(MatchError, match="`1` does not satisfy the condition"):
        p.apply(1)
    assert p.apply(-1) == -1
    with pytest.raises(MatchError, match="`1.0` is not an instance of <class 'int'>"):
        p.apply(1.0)

    p = AllOf(IsType(int), AsBuiltin(float))
    assert p.apply(1) == 1.0
    with pytest.raises(MatchError, match="`1.0` is not an instance of <class 'int'>"):
        p.apply(1.0)
    with pytest.raises(MatchError, match="`'1'` is not an instance of <class 'int'>"):
        p.apply("1")


def test_if_function():
    def checker(_):
        return _ == 10

    p = If(checker)
    assert p.apply(10) == 10
    with pytest.raises(MatchError, match="does not satisfy the condition"):
        p.apply(11)


def test_isin():
    p = IsIn([1, 2, 3])
    assert p.apply(1) == 1
    with pytest.raises(MatchError, match="is not in"):
        assert p.apply(4)


def test_sequence_of():
    p = SequenceOf(IsType(str), list)
    assert p.apply(["foo", "bar"]) == ["foo", "bar"]
    with pytest.raises(MatchError, match="`1` is not an instance of <class 'str'>"):
        p.apply([1, 2])
    with pytest.raises(MatchError, match="`1` is not iterable"):
        p.apply(1)
    with pytest.raises(
        MatchError, match="`'string'` is a string or bytes, not a sequence"
    ):
        p.apply("string")

    p = SequenceOf(IsType(int), tuple, allow_coercion=True)
    assert p.apply((1, 2)) == (1, 2)
    assert p.apply([1, 2]) == (1, 2)
    with pytest.raises(MatchError, match="not an instance of"):
        p.apply([1, 2, "3"])


class CustomDict(dict):
    pass


def test_mapping_of():
    p = MappingOf(IsType(str), IsType(int))
    assert p.apply({"foo": 1, "bar": 2}) == {"foo": 1, "bar": 2}
    with pytest.raises(MatchError, match="`'baz'` is not an instance of <class 'int'>"):
        p.apply({"foo": 1, "bar": "baz"})
    with pytest.raises(MatchError, match="`1` is not a mapping"):
        assert p.apply(1) is NoMatch

    p = MappingOf(IsType(str), IsType(str), CustomDict, allow_coercion=True)
    assert p.apply({"foo": "bar"}) == CustomDict({"foo": "bar"})
    with pytest.raises(MatchError, match="`1` is not an instance of <class 'str'>"):
        p.apply({"foo": 1})


def test_frozendict_of():
    p = FrozenDictOf(IsType(str), IsType(float))  # , allow_coercion=True)
    assert p == MappingOf(IsType(str), IsType(float), FrozenDict)
    assert p.apply(FrozenDict({"foo": 1.0, "bar": 2.0})) == FrozenDict(
        {"foo": 1.0, "bar": 2.0}
    )
    assert isinstance(p.apply(FrozenDict({"foo": 1.0, "bar": 2.0})), FrozenDict)


def test_capture():
    ctx = {}

    p = Capture("result", IsType(int))
    with pytest.raises(MatchError, match="`'10'` is not an instance of <class 'int'>"):
        p.apply("10", context=ctx)
    assert ctx == {}

    assert p.apply(12, context=ctx) == 12
    assert ctx == {"result": 12}


@pytest.mark.parametrize(
    "x", [Deferred(Var("x")), Var("x")], ids=["deferred", "builder"]
)
def test_capture_with_deferred_and_builder(x):
    ctx = {}

    p = Capture(x, IsType(float))

    with pytest.raises(MatchError, match="`1` is not an instance of <class 'float'>"):
        p.apply(1, context=ctx)
    assert ctx == {}

    assert p.apply(1.0, ctx) == 1.0
    assert ctx == {"x": 1.0}


def test_none_of():
    def negative(x):
        return x < 0

    p = NoneOf(IsType(int), If(negative))
    assert p.apply(1.0) == 1.0
    with pytest.raises(MatchError):
        p.apply(-1.0)
    with pytest.raises(MatchError):
        p.apply(1)


def test_generic_sequence_of():
    class MyList(list):
        @classmethod
        def __coerce__(cls, value, T=...):
            return cls(value)

    p = SequenceOf(IsType(str), MyList)
    assert p.apply(["foo", "bar"], context={}) == MyList(["foo", "bar"])
    with pytest.raises(
        MatchError, match="`'string'` is a string or bytes, not a sequence"
    ):
        p.apply("string")

    p = SequenceOf(IsType(str), tuple)
    assert p == SequenceOf(IsType(str), tuple)
    assert p.apply(("foo", "bar"), context={}) == ("foo", "bar")
    assert p.apply([], context={}) == ()


def test_list_of():
    p = ListOf(IsType(str))
    assert isinstance(p, SequenceOf)
    assert p.apply(["foo", "bar"], context={}) == ["foo", "bar"]
    with pytest.raises(
        MatchError, match=re.escape("`1` is not an instance of <class 'str'>")
    ):
        p.apply(["foo", 1])
    with pytest.raises(
        MatchError, match=re.escape("`1` is not an instance of <class 'str'>")
    ):
        p.apply([1, "foo"])
    with pytest.raises(MatchError, match=re.escape("`1` is not iterable")):
        p.apply(1)


def test_pattern_sequence():
    p = PatternList((IsType(str), IsType(int), IsType(float)))
    assert p.apply(("foo", 1, 1.0)) == ["foo", 1, 1.0]
    assert p.apply(["foo", 1, 1.0]) == ["foo", 1, 1.0]
    with pytest.raises(MatchError, match="is not iterable"):
        p.apply(1)

    p = PatternList((IsType(str),))
    assert p.apply(("foo",)) == ["foo"]
    with pytest.raises(
        MatchError, match="does not have the same length as the pattern"
    ):
        assert p.apply(("foo", "bar"))


class Foo:
    __match_args__ = ("a", "b")

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        return type(self) is type(other) and self.a == other.a and self.b == other.b


class Bar:
    __match_args__ = ("c", "d")

    def __init__(self, c, d):
        self.c = c
        self.d = d

    def __eq__(self, other):
        return type(self) is type(other) and self.c == other.c and self.d == other.d


class Baz:
    __match_args__ = ("e", "f", "g", "h", "i")

    def __init__(self, e, f, g, h, i):
        self.e = e
        self.f = f
        self.g = g
        self.h = h
        self.i = i

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.e == other.e
            and self.f == other.f
            and self.g == other.g
            and self.h == other.h
            and self.i == other.i
        )


def test_object_pattern():
    p = Object(Foo, 1, b=2)
    o = Foo(1, 2)
    assert p.apply(o) is o
    with pytest.raises(MatchError, match="`3` is not equal to the expected `2`"):
        p.apply(Foo(1, 3))

    p = Object(Foo, 1, lambda x: x * 2)
    assert p.apply(Foo(1, 2)) == Foo(1, 4)


def test_object_of_pattern_unrolling():
    p = Object(Foo, 1)
    assert isinstance(p, ObjectOf1)
    assert p.apply(Foo(1, 2)) == Foo(1, 2)

    p = Object(Foo, 1, 2)
    assert isinstance(p, ObjectOf2)
    assert p.apply(Foo(1, 2)) == Foo(1, 2)

    p = Object(Baz, 1, 2, 3)
    assert isinstance(p, ObjectOf3)
    assert p.apply(Baz(1, 2, 3, 4, 5)) == Baz(1, 2, 3, 4, 5)

    p = Object(Baz, 1, 2, 3, 4)
    assert isinstance(p, ObjectOfN)
    assert p.apply(Baz(1, 2, 3, 4, 5)) == Baz(1, 2, 3, 4, 5)


def test_object_of_partial_replacement():
    p = Object(Baz, Replace(1, 11))
    assert isinstance(p, ObjectOf1)
    assert p.apply(Baz(1, 2, 3, 4, 5)) == Baz(11, 2, 3, 4, 5)

    p = Object(Baz, Replace(1, 11), Replace(2, 22))
    assert isinstance(p, ObjectOf2)
    assert p.apply(Baz(1, 2, 3, 4, 5)) == Baz(11, 22, 3, 4, 5)

    p = Object(Baz, Replace(1, 11), Replace(2, 22), Replace(3, 33))
    assert isinstance(p, ObjectOf3)
    assert p.apply(Baz(1, 2, 3, 4, 5)) == Baz(11, 22, 33, 4, 5)

    p = Object(Baz, Replace(1, 11), Replace(2, 22), Replace(3, 33), Replace(4, 44))
    assert isinstance(p, ObjectOfN)
    assert p.apply(Baz(1, 2, 3, 4, 5)) == Baz(11, 22, 33, 44, 5)

    p = Object(Foo | Baz, Replace(1, 11))
    assert isinstance(p, ObjectOfX)
    assert p.apply(Foo(1, 2)) == Foo(11, 2)
    assert p.apply(Baz(1, 2, 3, 4, 5)) == Baz(11, 2, 3, 4, 5)


def test_object_pattern_complex_type():
    p = Object(Not(Foo), 1, 2)
    o = Bar(1, 2)

    # test that the pattern isn't changing the input object if none of
    # its arguments are changed by subpatterns
    assert p.apply(o) is o
    with pytest.raises(MatchError):
        p.apply(Foo(1, 2))
    with pytest.raises(MatchError):
        p.apply(Bar(1, 3))

    p = Object(Not(Foo), 1, b=2)
    with pytest.raises(MatchError):
        p.apply(Bar(1, 2))


def test_object_pattern_from_instance_of():
    class MyType:
        __match_args__ = ("a", "b")

        def __init__(self, a, b):
            self.a = a
            self.b = b

    p = pattern(MyType)
    assert p == IsType(MyType)

    p_call = p(1, 2)
    assert p_call == Object(MyType, 1, 2)


def test_object_pattern_from_coerced_to():
    class MyCoercibleType:
        __match_args__ = ("a", "b")

        def __init__(self, a, b):
            self.a = a
            self.b = b

        @classmethod
        def __coerce__(cls, other):
            a, b = other
            return cls(a, b)

    p = AsCoercible(MyCoercibleType)
    p_call = p(1, 2)
    assert p_call == ObjectOfX(p, (1, 2), {})


def test_object_pattern_matching_order():
    class Foo:
        __match_args__ = ("a", "b", "c")

        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

        def __eq__(self, other):
            return (
                type(self) is type(other)
                and self.a == other.a
                and self.b == other.b
                and self.c == other.c
            )

    a = Var("a")
    p = Object(Foo, Capture(a, IsType(int)), c=a)

    with pytest.raises(MatchError, match=re.escape("`3` is not equal to deferred $a")):
        p.apply(Foo(1, 2, 3))

    assert p.apply(Foo(1, 2, 1)) == Foo(1, 2, 1)


def test_object_pattern_matching_dictionary_field():
    a = Bar(1, dict())
    b = Bar(1, {})
    c = Bar(1, None)
    d = Bar(1, {"foo": 1})

    pattern = Object(Bar, 1, d={})
    assert pattern.apply(a) is a
    assert pattern.apply(b) is b
    with pytest.raises(MatchError):
        pattern.apply(c)

    pattern = Object(Bar, 1, d=None)
    with pytest.raises(MatchError):
        pattern.apply(a)
    assert pattern.apply(c) is c

    pattern = Object(Bar, 1, d={"foo": 1})
    with pytest.raises(MatchError):
        pattern.apply(a)
    assert pattern.apply(d) is d


def test_object_pattern_requires_its_arguments_to_match():
    class Empty:
        __match_args__ = ()

    msg = "The type to match has fewer `__match_args__`"
    with pytest.raises(ValueError, match=msg):
        Object(Empty, 1)

    # if the type matcher (first argument of Object) receives a generic pattern
    # instead of an explicit type, the validation above cannot occur, so test
    # the the pattern still doesn't match when it requires more positional
    # arguments than the object `__match_args__` has
    pattern = Object(IsType(Empty), "a")
    with pytest.raises(
        MatchError, match="has fewer 0 positional arguments than required 1"
    ):
        pattern.apply(Empty())

    pattern = Object(IsType(Empty), a="a")
    with pytest.raises(MatchError, match="does not have the attribute `a`"):
        pattern.apply(Empty())


def test_pattern_list():
    p = PatternList([1, 2, IsType(int), SomeOf(...)])
    assert p.apply([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]
    assert p.apply([1, 2, 3, 4, 5, 6]) == [1, 2, 3, 4, 5, 6]
    assert p.apply([1, 2, 3, 4]) == [1, 2, 3, 4]
    with pytest.raises(MatchError, match="`'3'` is not an instance of <class 'int'>"):
        p.apply([1, 2, "3", 4])

    # subpattern is a simple pattern
    p = PatternList([1, 2, AsInt(), SomeOf(...)])
    assert p.apply([1, 2, 3.0, 4.0, 5.0]) == [1, 2, 3, 4.0, 5.0]

    # subpattern is a sequence
    p = PatternList([1, 2, 3, SomeOf(AsInt(), at_least=1)])
    assert p.apply([1, 2, 3, 4.0, 5.0]) == [1, 2, 3, 4, 5]


def test_pattern_list_from_tuple_typehint():
    p = Pattern.from_typehint(tuple[str, int, float], allow_coercion=False)
    assert p == PatternList([IsType(str), IsType(int), IsType(float)], type_=tuple)
    assert p.apply(["foo", 1, 2.0]) == ("foo", 1, 2.0)
    assert p.apply(("foo", 1, 2.0)) == ("foo", 1, 2.0)
    with pytest.raises(MatchError):
        p.apply(["foo", 1])
    with pytest.raises(MatchError):
        p.apply(["foo", 1, 2.0, 3.0])

    class MyTuple(tuple):
        pass

    p = Pattern.from_typehint(MyTuple[int, bool], allow_coercion=False)
    assert p == PatternList([IsType(int), IsType(bool)], type_=MyTuple)
    assert p.apply([1, True]) == MyTuple([1, True])
    assert p.apply(MyTuple([1, True])) == MyTuple([1, True])
    with pytest.raises(MatchError):
        p.apply([1, 2])

    p = Pattern.from_typehint(tuple[str, int, float], allow_coercion=True)
    assert p.apply(["foo", 1, 2.0]) == ("foo", 1, 2.0)
    assert p.apply(["foo", 1, 2]) == ("foo", 1, 2.0)
    with pytest.raises(MatchError):
        p.apply(["foo", 1])
    with pytest.raises(MatchError):
        p.apply(["foo", 1, 2.0, 3.0])


def test_pattern_list_unpack():
    integer = pattern(int, allow_coercion=False)
    floating = pattern(float, allow_coercion=False)

    assert match([1, 2, *floating], [1, 2, 3]) is NoMatch
    assert match([1, 2, *floating], [1, 2, 3.0]) == [1, 2, 3.0]
    assert match([1, 2, *floating], [1, 2, 3.0, 4.0]) == [1, 2, 3.0, 4.0]
    assert match([1, *floating, *integer], [1, 2.0, 3.0, 4]) == [1, 2.0, 3.0, 4]
    assert match([1, *floating, *integer], [1, 2.0, 3.0, 4, 5]) == [
        1,
        2.0,
        3.0,
        4,
        5,
    ]
    assert match([1, *floating, *integer], [1, 2.0, 3, 4.0]) is NoMatch


def test_matching():
    assert match("foo", "foo") == "foo"
    assert match("foo", "bar") is NoMatch

    assert match(IsType(int), 1) == 1
    assert match(IsType(int), "foo") is NoMatch

    assert Capture("pi", IsType(float)) == "pi" @ IsType(float)
    assert Capture("pi", IsType(float)) == "pi" @ IsType(float)

    assert match(Capture("pi", IsType(float)), 3.14, ctx := {}) == 3.14
    assert ctx == {"pi": 3.14}
    assert match("pi" @ IsType(float), 3.14, ctx := {}) == 3.14
    assert ctx == {"pi": 3.14}

    assert match("pi" @ IsType(float), 3.14, ctx := {}) == 3.14
    assert ctx == {"pi": 3.14}

    assert match(IsType(int) | IsType(float), 3) == 3
    assert match(IsType(object) & IsType(float), 3.14) == 3.14


# def test_replace_passes_matched_value_as_underscore():
#     class MyInt:
#         def __init__(self, value):
#             self.value = value

#         def __eq__(self, other):
#             return self.value == other.value

#     p = InstanceOf(int) >> Call(MyInt, value=_)
#     assert p.apply(1, context={}) == MyInt(1)


def test_replace_in_nested_object_pattern():
    # simple example using reference to replace a value
    b = Var("b")
    p = Object(Foo, 1, b=Replace(Anything(), b))
    f = p.apply(Foo(1, 2), {"b": 3})
    assert f.a == 1
    assert f.b == 3

    # nested example using reference to replace a value
    d = Var("d")
    p = Object(Foo, 1, b=Object(Bar, 2, d=Replace(Anything(), d)))
    g = p.apply(Foo(1, Bar(2, 3)), {"d": 4})
    assert g.b.c == 2
    assert g.b.d == 4

    # nested example using reference to replace a value with a captured value
    p = Object(
        Foo,
        1,
        b=Replace(Object(Bar, 2, d="d" @ Anything()), lambda _, d: Foo(-1, b=d)),
    )
    h = p.apply(Foo(1, Bar(2, 3)), {})
    assert isinstance(h, Foo)
    assert h.a == 1
    assert isinstance(h.b, Foo)
    assert h.b.b == 3

    d = Var("d")
    p = Object(Foo, 1, b=Object(Bar, 2, d=d @ Anything()) >> Call(Foo, -1, b=d))
    h1 = p.apply(Foo(1, Bar(2, 3)), {})
    assert isinstance(h1, Foo)
    assert h1.a == 1
    assert isinstance(h1.b, Foo)
    assert h1.b.b == 3


# def test_replace_decorator():
#     @replace(int)
#     def sub(_):
#         return _ - 1

#     assert match(sub, 1) == 0
#     assert match(sub, 2) == 1


def test_replace_using_deferred():
    x = Deferred(Var("x"))
    y = Deferred(Var("y"))

    pat = Object(Foo, Capture(x), b=Capture(y)) >> Call(Foo, x, b=y)
    assert pat.apply(Foo(1, 2)) == Foo(1, 2)

    pat = Object(Foo, Capture(x), b=Capture(y)) >> Call(Foo, x, b=(y + 1) * x)
    assert pat.apply(Foo(2, 3)) == Foo(2, 8)

    pat = Object(Foo, "x" @ Anything(), y @ IsType(Bar)) >> Call(Foo, x, b=y.c + y.d)
    assert pat.apply(Foo(1, Bar(2, 3))) == Foo(1, 5)


def test_matching_sequence_pattern():
    assert match([], []) == []
    assert match([], [1]) is NoMatch

    assert match([1, 2, 3, 4, SomeOf(...)], list(range(1, 9))) == list(range(1, 9))
    assert match([1, 2, 3, 4, SomeOf(...)], list(range(1, 3))) is NoMatch
    assert match([1, 2, 3, 4, SomeOf(...)], list(range(1, 5))) == list(range(1, 5))
    assert match([1, 2, 3, 4, SomeOf(...)], list(range(1, 6))) == list(range(1, 6))

    assert match([SomeOf(...), 3, 4], list(range(5))) == list(range(5))
    assert match([SomeOf(...), 3, 4], list(range(3))) is NoMatch

    assert match([0, 1, SomeOf(...), 4], list(range(5))) == list(range(5))
    assert match([0, 1, SomeOf(...), 4], list(range(4))) is NoMatch

    assert match([SomeOf(...)], list(range(5))) == list(range(5))
    assert match([SomeOf(...), 2, 3, 4, SomeOf(...)], list(range(8))) == list(range(8))


def test_matching_sequence_pattern_keeps_original_type():
    assert match([1, 2, 3, 4, SomeOf(...)], tuple(range(1, 9))) == list(range(1, 9))
    assert match((1, 2, 3, SomeOf(...)), [1, 2, 3, 4, 5]) == (1, 2, 3, 4, 5)


def test_matching_sequence_with_captures():
    x = Deferred(Var("x"))

    v = list(range(1, 9))
    assert match([1, 2, 3, 4, SomeOf(...)], v) == v
    assert match([1, 2, 3, 4, "rest" @ SomeOf(...)], v, ctx := {}) == v
    assert ctx == {"rest": [5, 6, 7, 8]}

    v = list(range(5))
    assert match([0, 1, x @ SomeOf(...), 4], v, ctx := {}) == v
    assert ctx == {"x": [2, 3]}
    assert match([0, 1, "var" @ SomeOf(...), 4], v, ctx := {}) == v
    assert ctx == {"var": [2, 3]}

    p = [
        0,
        1,
        "ints" @ SomeOf(int),
        SomeOf("last_float" @ IsType(float)),
        6,
    ]
    v = [0, 1, 2, 3, 4.0, 5.0, 6]
    assert match(p, v, ctx := {}) == v
    assert ctx == {"ints": [2, 3], "last_float": 5.0}


def test_matching_sequence_remaining():
    three = [1, 2, 3]
    four = [1, 2, 3, 4]
    five = [1, 2, 3, 4, 5]

    assert match([1, 2, 3, SomeOf(int, at_least=1)], four) == four
    assert match([1, 2, 3, SomeOf(int, at_least=1)], three) is NoMatch
    assert match([1, 2, 3, SomeOf(int)], three) == three
    assert match([1, 2, 3, SomeOf(int, at_most=1)], three) == three
    # assert match([1, 2, 3, SomeOf(InstanceOf(int) & Between(0, 10))], five) == five
    # assert match([1, 2, 3, SomeOf(InstanceOf(int) & Between(0, 4))], five) is NoMatch
    assert match([1, 2, 3, SomeOf(int, at_least=2)], four) is NoMatch
    assert match([1, 2, 3, "res" @ SomeOf(int, at_least=2)], five, ctx := {}) == five
    assert ctx == {"res": [4, 5]}


def test_matching_sequence_complicated():
    pat = [
        1,
        "a" @ SomeOf(IsType(int) & If(lambda x: x < 10)),
        4,
        "b" @ SomeOf(...),
        8,
        9,
    ]
    expected = {
        "a": [2, 3],
        "b": [5, 6, 7],
    }
    assert match(pat, range(1, 10), ctx := {}) == list(range(1, 10))
    assert ctx == expected

    pat = [1, 2, Capture("remaining", SomeOf(...))]
    expected = {"remaining": [3, 4, 5, 6, 7, 8, 9]}
    assert match(pat, range(1, 10), ctx := {}) == list(range(1, 10))
    assert ctx == expected

    v = [0, [1, 2, "3"], [1, 2, "4"], 3]
    assert match([0, SomeOf([1, 2, str]), 3], v) == v


def test_pattern_sequence_with_nested_some_of():
    assert SomeChunksOf(1, 2) == SomeOf(1, 2)

    ctx = {}
    res = match([0, "subseq" @ SomeChunksOf(1, 2), 3], [0, 1, 2, 1, 2, 3], ctx)
    assert res == [0, 1, 2, 1, 2, 3]
    assert ctx == {"subseq": [1, 2, 1, 2]}

    res = match([0, "subseq" @ SomeOf(1, 2), 3], [0, 1, 2, 3], ctx)
    assert res == [0, 1, 2, 3]
    assert ctx == {"subseq": [1, 2]}

    assert match([0, SomeOf(1), 2, 3], [0, 2, 3]) == [0, 2, 3]
    assert match([0, SomeOf(1, at_least=1), 2, 3], [0, 2, 3]) is NoMatch
    assert match([0, SomeOf(1, at_least=1), 2, 3], [0, 1, 2, 3]) == [0, 1, 2, 3]
    assert match([0, SomeOf(1, at_least=2), 2, 3], [0, 1, 2, 3]) is NoMatch
    assert match([0, SomeOf(1, at_least=2), 2, 3], [0, 1, 1, 2, 3]) == [0, 1, 1, 2, 3]
    assert match([0, SomeOf(1, at_most=2), 2, 3], [0, 1, 1, 2, 3]) == [0, 1, 1, 2, 3]
    assert match([0, SomeOf(1, at_most=1), 2, 3], [0, 1, 1, 2, 3]) is NoMatch
    assert match([0, SomeOf(1, exactly=1), 2, 3], [0, 2, 3]) is NoMatch
    assert match([0, SomeOf(1, exactly=1), 2, 3], [0, 1, 2, 3]) == [0, 1, 2, 3]
    assert match([0, SomeOf(1, exactly=0), 2, 3], [0, 2, 3]) == [0, 2, 3]
    assert match([0, SomeOf(1, exactly=0), 2, 3], [0, 1, 2, 3]) is NoMatch

    assert match([0, SomeOf(1, SomeOf(2)), 3], [0, 3]) == [0, 3]
    assert match([0, SomeOf(1, SomeOf(2)), 3], [0, 1, 3]) == [0, 1, 3]
    assert match([0, SomeOf(1, SomeOf(2)), 3], [0, 1, 2, 3]) == [0, 1, 2, 3]
    assert match([0, SomeOf(1, SomeOf(2)), 3], [0, 1, 2, 2, 3]) == [0, 1, 2, 2, 3]
    assert match([0, SomeOf(1, SomeOf(2)), 3], [0, 1, 2, 2, 2, 3]) == [0, 1, 2, 2, 2, 3]
    assert match([0, SomeOf(1, SomeOf(2)), 3], [0, 1, 2, 1, 2, 2, 3]) == [
        0,
        1,
        2,
        1,
        2,
        2,
        3,
    ]

    assert match([0, SomeOf(1, SomeOf(2), at_least=1), 3], [0, 1, 2, 3]) == [0, 1, 2, 3]
    assert match([0, SomeOf(1, SomeOf(2), at_least=1), 3], [0, 1, 3]) == [0, 1, 3]
    assert (
        match([0, SomeOf(1, SomeOf(2, at_least=2), at_least=1), 3], [0, 1, 3])
        is NoMatch
    )
    assert (
        match([0, SomeOf(1, SomeOf(2, at_least=2), at_least=1), 3], [0, 1, 2, 3])
        is NoMatch
    )
    assert match(
        [0, SomeOf(1, SomeOf(2, at_least=2), at_least=1), 3], [0, 1, 2, 2, 3]
    ) == [
        0,
        1,
        2,
        2,
        3,
    ]


@pytest.mark.parametrize(
    ("pattern", "value", "expected"),
    [
        (IsType(bool), True, True),
        (IsType(str), "foo", "foo"),
        (IsType(int), 8, 8),
        (IsType(int), 1, 1),
        (IsType(float), 1.0, 1.0),
        (IsIn({"a", "b"}), "a", "a"),
        (IsIn({"a": 1, "b": 2}), "a", "a"),
        (IsIn(["a", "b"]), "a", "a"),
        (IsIn(("a", "b")), "b", "b"),
        (IsIn({"a", "b", "c"}), "c", "c"),
        (TupleOf(IsType(int)), (1, 2, 3), (1, 2, 3)),
        (PatternList((IsType(int), IsType(str))), (1, "a"), [1, "a"]),
        (ListOf(IsType(str)), ["a", "b"], ["a", "b"]),
        (AnyOf(IsType(str), IsType(int)), "foo", "foo"),
        (AnyOf(IsType(str), IsType(int)), 7, 7),
        (
            AllOf(IsType(int), If(lambda v: v >= 3), If(lambda v: v >= 8)),
            10,
            10,
        ),
        (
            MappingOf(IsType(str), IsType(int)),
            {"a": 1, "b": 2},
            {"a": 1, "b": 2},
        ),
        (AnyOf(IsType(str)) | IsType(int), 7, 7),
        (AllOf(IsType(int)) & IsType(int), 7, 7),
        (IsType(int) | AnyOf(IsType(str)), 7, 7),
        (IsType(int) & AllOf(IsType(int)), 7, 7),
        (AnyOf(IsType(str)) | AnyOf(IsType(int)), 7, 7),
        (AllOf(IsType(int)) & AllOf(IsType(int)), 7, 7),
    ],
)
def test_various_patterns(pattern, value, expected):
    assert pattern.apply(value, context={}) == expected


@pytest.mark.parametrize(
    ("pattern", "value"),
    [
        (IsType(bool), "foo"),
        (IsType(str), True),
        (IsType(int), 8.1),
        (pattern(Min(3)), 2),
        (IsType(int), None),
        (IsType(float), 1),
        (IsIn(["a", "b"]), "c"),
        (IsIn({"a", "b"}), "c"),
        (IsIn({"a": 1, "b": 2}), "d"),
        (TupleOf(IsType(int)), (1, 2.0, 3)),
        (ListOf(IsType(str)), ["a", "b", None]),
        (AnyOf(IsType(str), Min(4)), 3.14),
        (AnyOf(Is(str), Min(10)), 9),
        (AllOf(Is(int), Min(3), Min(8)), 7),
        (DictOf(IsType(int), IsType(str)), {"a": 1, "b": 2}),
    ],
)
def test_various_not_matching_patterns(pattern, value):
    with pytest.raises(MatchError):
        pattern.apply(value, context={})


@pattern
def endswith_d(s, **ctx):
    if not s.endswith("d"):
        return NoMatch
    return s


def test_pattern_decorator():
    assert endswith_d.apply("abcd", context={}) == "abcd"
    assert endswith_d.apply("abc", context={}) is NoMatch


@pytest.mark.parametrize(
    ("annot", "expected"),
    [
        (int, IsType(int)),
        (str, IsType(str)),
        (bool, IsType(bool)),
        (Optional[int], Option(IsType(int))),
        (Optional[Union[str, int]], Option(AnyOf(IsType(str), IsType(int)))),
        (Union[int, str], AnyOf(IsType(int), IsType(str))),
        (Annotated[str, endswith_d], AllOf(IsType(str), endswith_d)),
        (list[int], SequenceOf(IsType(int), list, allow_coercion=False)),
        (
            tuple[int, float, str],
            PatternList((IsType(int), IsType(float), IsType(str)), type_=tuple),
        ),
        (tuple[int, ...], TupleOf(IsType(int))),
        (
            dict[str, float],
            DictOf(IsType(str), IsType(float), allow_coercion=False),
        ),
        (
            FrozenDict[str, int],
            MappingOf(IsType(str), IsType(int), FrozenDict, allow_coercion=False),
        ),
        (Literal["alpha", "beta", "gamma"], IsIn(("alpha", "beta", "gamma"))),
        (
            Callable[[str, int], str],
            CallableWith((IsType(str), IsType(int)), IsType(str)),
        ),
    ],
)
def test_pattern_from_typehint_no_coercion(annot, expected):
    assert Pattern.from_typehint(annot, allow_coercion=False) == expected


@pytest.mark.parametrize(
    ("annot", "expected"),
    [
        (int, AsInt()),
        (bool, AsBool()),
        (Optional[int], Option(AsInt())),
        (Optional[Union[str, int]], Option(AnyOf(IsType(str), AsInt()))),
        (Union[int, str], AnyOf(AsInt(), IsType(str))),
        (Annotated[int, Min(3)], AllOf(AsInt(), Min(3))),
        (list[int], SequenceOf(AsInt(), list)),
        (
            tuple[int, float, str],
            PatternList((AsInt(), AsBuiltin(float), IsType(str)), type_=tuple),
        ),
        (tuple[int, ...], TupleOf(AsInt())),
        (
            dict[str, float],
            DictOf(IsType(str), AsBuiltin(float)),
        ),
        (FrozenDict[str, int], MappingOf(IsType(str), AsInt(), FrozenDict)),
        (Literal["alpha", "beta", "gamma"], IsIn(("alpha", "beta", "gamma"))),
        (
            Callable[[str, int], str],
            CallableWith((IsType(str), AsInt()), IsType(str)),
        ),
    ],
)
def test_pattern_from_typehint_with_coercion(annot, expected):
    assert Pattern.from_typehint(annot, allow_coercion=True) == expected


def test_pattern_from_annotated():
    p = Pattern.from_typehint(Annotated[Any, ...])
    assert p == AllOf(Anything(), Anything())
    assert pattern(Any) == Anything()


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_pattern_from_typehint_uniontype():
    # uniontype marks `type1 | type2` annotations and it's different from
    # Union[type1, type2]
    validator = Pattern.from_typehint(str | int | float, allow_coercion=False)
    assert validator == AnyOf(IsType(str), IsType(int), IsType(float))

    validator = Pattern.from_typehint(str | int | float, allow_coercion=True)
    assert validator == AnyOf(IsType(str), AsInt(), AsBuiltin(float))


def test_pattern_from_coercible_typehint_disable_coercion():
    class MyFloat(float):
        @classmethod
        def __coerce__(cls, obj):
            return cls(float(obj))

    p = Pattern.from_typehint(MyFloat, allow_coercion=True)
    assert isinstance(p, AsCoercible)

    p = Pattern.from_typehint(MyFloat, allow_coercion=False)
    assert isinstance(p, IsType)


def test_pattern_from_self_typehint():
    p = Pattern.from_typehint(
        List[Self], self_qualname="koerce.tests.test_patterns.MyClass"
    )
    assert p == SequenceOf(IsTypeLazy("koerce.tests.test_patterns.MyClass"), list)
    result = p.apply([MyClass(), MyClass()])
    assert isinstance(result, list)
    assert isinstance(result[0], MyClass)
    assert isinstance(result[1], MyClass)

    with pytest.raises(
        MatchError,
        match="`1` is not an instance of 'koerce.tests.test_patterns.MyClass'",
    ):
        p.apply([MyClass(), 1])


class PlusOne:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    @classmethod
    def __coerce__(cls, obj):
        return cls(obj + 1)

    def __eq__(self, other):
        return type(self) is type(other) and self.value == other.value


class PlusOneRaise(PlusOne):
    @classmethod
    def __coerce__(cls, obj):
        if isinstance(obj, cls):
            return obj
        else:
            raise ValueError("raise on coercion")


class PlusOneChild(PlusOne):
    pass


class PlusTwo(PlusOne):
    @classmethod
    def __coerce__(cls, obj):
        return obj + 2


def test_pattern_from_coercible_protocol():
    s = Pattern.from_typehint(PlusOne, allow_coercion=True)
    assert s == AsCoercible(PlusOne)
    assert s.apply(1) == PlusOne(2)
    assert s.apply(10) == PlusOne(11)

    s = Pattern.from_typehint(PlusOneRaise, allow_coercion=False)
    assert s == IsType(PlusOneRaise)
    assert s.apply(PlusOneRaise(10), context={}) == PlusOneRaise(10)
    msg = re.escape(
        "`1` is not an instance of <class 'koerce.tests.test_patterns.PlusOneRaise'>"
    )
    with pytest.raises(MatchError, match=msg):
        s.apply(1)


def test_pattern_coercible_bypass_coercion():
    s = Pattern.from_typehint(PlusOneRaise, allow_coercion=True)
    # bypass coercion since it's already an instance of SomethingRaise
    assert s.apply(PlusOneRaise(10), context={}) == PlusOneRaise(10)
    # but actually call __coerce__ if it's not an instance
    msg = re.escape(
        "`10` cannot be coerced to <class 'koerce.tests.test_patterns.PlusOneRaise'>"
    )
    with pytest.raises(MatchError, match=msg):
        s.apply(10)


def test_pattern_coercible_checks_type():
    s = Pattern.from_typehint(PlusOneChild, allow_coercion=True)
    v = Pattern.from_typehint(PlusTwo, allow_coercion=True)

    assert s.apply(1, context={}) == PlusOneChild(2)

    assert PlusTwo.__coerce__(1) == 3
    msg = re.escape(
        "`PlusTwo.__coerce__(3)` did not return an instance of <class 'koerce.tests.test_patterns.PlusTwo'>"
    )
    with pytest.raises(MatchError, match=msg):
        v.apply(1)


class DoubledList(list[T]):
    @classmethod
    def __coerce__(cls, obj):
        return cls(list(obj) * 2)


def test_pattern_coercible_sequence_type():
    s = Pattern.from_typehint(Sequence[PlusOne], allow_coercion=True)
    # with pytest.raises(TypeError, match=r"Sequence\(\) takes no arguments"):
    assert s.apply([1, 2, 3], context={}) == [PlusOne(2), PlusOne(3), PlusOne(4)]

    s = Pattern.from_typehint(list[PlusOne], allow_coercion=True)
    assert s == SequenceOf(AsCoercible(PlusOne), list)
    assert s.apply([1, 2, 3], context={}) == [PlusOne(2), PlusOne(3), PlusOne(4)]

    s = Pattern.from_typehint(tuple[PlusOne, ...], allow_coercion=True)
    assert s == TupleOf(AsCoercible(PlusOne))
    assert s.apply([1, 2, 3], context={}) == (PlusOne(2), PlusOne(3), PlusOne(4))

    s = Pattern.from_typehint(DoubledList[PlusOne], allow_coercion=True)
    assert s == SequenceOf(AsCoercible(PlusOne), DoubledList)
    assert s.apply([1, 2, 3], context={}) == DoubledList(
        [PlusOne(2), PlusOne(3), PlusOne(4), PlusOne(2), PlusOne(3), PlusOne(4)]
    )


def test_pattern_function():
    class MyNegativeInt(int):
        @classmethod
        def __coerce__(cls, other):
            return cls(-int(other))

    class Box(Generic[T]):
        value: T

    def f(x):
        return x > 0

    # ... is treated the same as Any()
    assert pattern(...) == Anything()
    assert pattern(Anything()) == Anything()
    assert pattern(True) == EqValue(True)

    # plain types are converted to InstanceOf patterns
    assert pattern(int) == IsType(int)
    assert pattern(int, allow_coercion=False) == IsType(int)
    assert pattern(int, allow_coercion=True) == AsInt()
    # no matter whether the type implements the coercible protocol or not
    assert pattern(MyNegativeInt) == IsType(MyNegativeInt)
    assert pattern(MyNegativeInt, allow_coercion=True) == AsCoercible(MyNegativeInt)
    assert pattern(MyNegativeInt, allow_coercion=False) == IsType(MyNegativeInt)

    # generic types are converted to GenericInstanceOf patterns
    assert pattern(Box[int]) == IsGeneric(Box[int])
    assert pattern(Box[int], allow_coercion=False) == IsGeneric(Box[int])
    # no matter whethwe the origin type implements the coercible protocol or not
    assert pattern(Box[MyNegativeInt]) == IsGeneric(Box[MyNegativeInt])

    # sequence typehints are converted to the appropriate sequence checkers
    assert pattern(List[int], allow_coercion=True) == ListOf(
        AsInt(), allow_coercion=True
    )
    assert pattern(List[int], allow_coercion=False) == ListOf(
        IsType(int), allow_coercion=False
    )

    # spelled out sequences construct a more advanced pattern sequence
    assert pattern([int, str, 1], allow_coercion=True) == PatternList(
        [AsInt(), IsType(str), EqValue(1)]
    )
    assert pattern([int, str, 1], allow_coercion=False) == PatternList(
        [IsType(int), IsType(str), EqValue(1)]
    )

    # matching deferred to user defined functions
    assert pattern(f) == Custom(f)

    # matching mapping values
    assert pattern({"a": 1, "b": 2}) == PatternMap({"a": EqValue(1), "b": EqValue(2)})


def test_callable_with():
    def func(a, b):
        return str(a) + b

    def func_with_args(a, b, *args):
        return sum((a, b) + args)

    def func_with_kwargs(a, b, c=1, **kwargs):
        return str(a) + b + str(c)

    def func_with_optional_keyword_only_kwargs(a, *, c=1):
        return a + c

    def func_with_required_keyword_only_kwargs(*, c):
        return c

    p = CallableWith([IsType(int), IsType(str)])
    with pytest.raises(MatchError, match="is not a callable"):
        assert p.apply(10)

    msg = "Callable has mandatory keyword-only arguments which cannot be specified"
    with pytest.raises(TypeError, match=msg):
        p.apply(func_with_required_keyword_only_kwargs, context={})

    # Callable has more positional arguments than expected
    p = CallableWith([IsType(int)] * 2)
    assert p.apply(func_with_kwargs, context={}) is func_with_kwargs

    # Callable has less positional arguments than expected
    p = CallableWith([IsType(int)] * 4)
    msg = "has less positional arguments than the expected 4"
    with pytest.raises(MatchError, match=msg):
        assert p.apply(func_with_kwargs)

    p = CallableWith([IsType(int)] * 4, IsType(int))
    wrapped = p.apply(func_with_args, context={})
    assert wrapped(1, 2, 3, 4) == 10

    p = CallableWith([IsType(int), IsType(str)], IsType(str))
    wrapped = p.apply(func, context={})
    assert wrapped(1, "st") == "1st"

    p = CallableWith([IsType(int)])
    wrapped = p.apply(func_with_optional_keyword_only_kwargs, context={})
    assert wrapped(1) == 2


def test_callable_with_default_arguments():
    def f(a: int, b: str, c: str):
        return a + int(b) + int(c)

    def g(a: int, b: str, c: str = "0"):
        return a + int(b) + int(c)

    h = functools.partial(f, c="0")

    p = Pattern.from_typehint(Callable[[int, str], int])
    with pytest.raises(
        MatchError, match="has more positional arguments than the required 2"
    ):
        p.apply(f)
    assert p.apply(g) == g
    assert p.apply(h) == h


def test_instance_of_with_metaclass():
    class Meta(type): ...

    class Class(metaclass=Meta): ...

    class OtherMeta(type): ...

    class OtherClass(metaclass=OtherMeta): ...

    my_instance = Class()
    my_other_instance = OtherClass()

    assert IsType(Class).apply(my_instance) == my_instance
    assert IsType(OtherClass).apply(my_other_instance) == my_other_instance

    with pytest.raises(MatchError):
        assert IsType(Class).apply(my_other_instance)
    with pytest.raises(MatchError):
        assert IsType(OtherClass).apply(my_instance)


def test_as_dispatch():
    p = AsType(datetime)
    assert p.apply(datetime(2021, 1, 1)) == datetime(2021, 1, 1)
    assert p.apply("2021-01-01") == datetime(2021, 1, 1)
    with pytest.raises(
        MatchError, match="failed to construct <class 'datetime.datetime'> from `3.14`"
    ):
        p.apply(3.14)

    p = pattern(datetime)
    assert p.apply(datetime(2021, 1, 1)) == datetime(2021, 1, 1)
    with pytest.raises(
        MatchError,
        match="`'2021-01-01'` is not an instance of <class 'datetime.datetime'>",
    ):
        p.apply("2021-01-01")

    p = pattern(datetime, allow_coercion=True)
    assert p.apply(datetime(2021, 1, 1)) == datetime(2021, 1, 1)
    assert p.apply("2021-01-01") == datetime(2021, 1, 1)
    with pytest.raises(
        MatchError, match="failed to construct <class 'datetime.datetime'> from `3.14`"
    ):
        p.apply(3.14)
