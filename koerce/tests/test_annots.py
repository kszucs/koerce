from __future__ import annotations

import copy
import pickle
import weakref
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import pytest
from typing_extensions import Self

from koerce._internal import (
    EMPTY,
    AbstractMeta,
    Annotable,
    AnnotableMeta,
    Anything,
    As,
    FrozenDictOf,
    Hashable,
    Immutable,
    Is,
    IsType,
    MatchError,
    Option,
    Parameter,
    Pattern,
    Signature,
    TupleOf,
    annotated,
    argument,
    attribute,
    optional,
    pattern,
    varargs,
    varkwargs,
)


def test_parameter():
    p = Parameter(Parameter.POSITIONAL_OR_KEYWORD, typehint=int)
    assert p.kind is Parameter.POSITIONAL_OR_KEYWORD
    assert p.format("x") == "x: int"

    p = Parameter(Parameter.POSITIONAL_OR_KEYWORD, default=1)
    assert p.kind is Parameter.POSITIONAL_OR_KEYWORD
    assert p.default_ == 1
    assert p.format("x") == "x=1"
    assert p.pattern == Anything()

    p = Parameter(
        Parameter.POSITIONAL_OR_KEYWORD, typehint=int, default=1, pattern=is_int
    )
    assert p.kind is Parameter.POSITIONAL_OR_KEYWORD
    assert p.default_ == 1
    assert p.typehint is int
    assert p.format("x") == "x: int = 1"
    assert p.pattern == is_int

    p = Parameter(Parameter.VAR_POSITIONAL, typehint=int, pattern=is_int)
    assert p.kind is Parameter.VAR_POSITIONAL
    assert p.typehint is int
    assert p.format("y") == "*y: int"
    assert p.pattern == TupleOf(is_int)

    p = Parameter(Parameter.VAR_KEYWORD, typehint=int, pattern=is_int)
    assert p.kind is Parameter.VAR_KEYWORD
    assert p.typehint is int
    assert p.format("z") == "**z: int"
    assert p.pattern == FrozenDictOf(Anything(), is_int)


def test_signature_contruction():
    a = Parameter(Parameter.POSITIONAL_OR_KEYWORD, typehint=int)
    b = Parameter(Parameter.POSITIONAL_OR_KEYWORD, typehint=str)
    c = Parameter(Parameter.POSITIONAL_OR_KEYWORD, typehint=int, default=1)
    d = Parameter(Parameter.VAR_POSITIONAL, typehint=int, pattern=is_int)

    sig = Signature({"a": a, "b": b, "c": c, "d": d})
    assert sig.parameters == {"a": a, "b": b, "c": c, "d": d}
    assert sig.return_typehint is EMPTY
    assert sig.return_pattern == Anything()


def test_signature_equality_comparison():
    # order of parameters matters
    a = Parameter(Parameter.POSITIONAL_OR_KEYWORD, typehint=int)
    b = Parameter(Parameter.POSITIONAL_OR_KEYWORD, typehint=str)
    c = Parameter(Parameter.POSITIONAL_OR_KEYWORD, typehint=int, default=1)

    sig1 = Signature({"a": a, "b": b, "c": c})
    sig2 = Signature({"a": a, "b": b, "c": c})
    assert sig1 == sig2

    sig3 = Signature({"a": a, "c": c, "b": b})
    assert sig1 != sig3


def test_signature_from_callable():
    def func(a: int, b: str, *args, c=1, **kwargs) -> float: ...

    sig = Signature.from_callable(func)
    assert sig.parameters == {
        "a": Parameter(Parameter.POSITIONAL_OR_KEYWORD, typehint=int),
        "b": Parameter(Parameter.POSITIONAL_OR_KEYWORD, typehint=str),
        "args": Parameter(Parameter.VAR_POSITIONAL),
        "c": Parameter(Parameter.KEYWORD_ONLY, default=1),
        "kwargs": Parameter(Parameter.VAR_KEYWORD),
    }
    assert sig.return_typehint is float


def test_signature_bind_various():
    # with positional or keyword default
    def func(a: int, b: str, c=1) -> float: ...

    sig = Signature.from_callable(func)
    bound = sig(1, "2")
    assert bound == {"a": 1, "b": "2", "c": 1}

    # with variable positional arguments
    def func(a: int, b: str, *args: int, c=1) -> float: ...

    sig = Signature.from_callable(func)
    bound = sig(1, "2", 3, 4)
    assert bound == {"a": 1, "b": "2", "args": (3, 4), "c": 1}

    # with both variadic positional and variadic keyword arguments
    def func(a: int, b: str, *args: int, c=1, **kwargs: int) -> float: ...

    sig = Signature.from_callable(func)
    bound = sig(1, "2", 3, 4, x=5, y=6)
    assert bound == {
        "a": 1,
        "b": "2",
        "args": (3, 4),
        "c": 1,
        "kwargs": {"x": 5, "y": 6},
    }

    # with positional only arguments
    def func(a: int, b: str, /, c=1) -> float: ...

    sig = Signature.from_callable(func)
    bound = sig(1, "2")
    assert bound == {"a": 1, "b": "2", "c": 1}

    with pytest.raises(TypeError, match="passed as keyword argument"):
        sig(a=1, b="2", c=3)

    # with keyword only arguments
    def func(a: int, b: str, *, c=1) -> float: ...

    sig = Signature.from_callable(func)
    bound = sig(1, "2", c=3)
    assert bound == {"a": 1, "b": "2", "c": 3}

    with pytest.raises(TypeError, match="too many positional arguments"):
        sig(1, "2", 3)

    def func(a, *args, b, z=100, **kwargs): ...

    sig = Signature.from_callable(func)
    bound = sig(10, 20, b=30, c=40, args=50, kwargs=60)
    assert bound == {
        "a": 10,
        "args": (20,),
        "b": 30,
        "z": 100,
        "kwargs": {"c": 40, "args": 50, "kwargs": 60},
    }


def call(func, *args, **kwargs):
    # it also tests the unbind method
    sig = Signature.from_callable(func)
    bound = sig(*args, **kwargs)
    ubargs, ubkwargs = sig.unbind(bound)
    return func(*ubargs, **ubkwargs)


def test_signature_bind_no_arguments():
    def func(): ...

    sig = Signature.from_callable(func)
    assert sig() == {}

    with pytest.raises(TypeError, match="too many positional arguments"):
        sig(1)
    with pytest.raises(TypeError, match="too many positional arguments"):
        sig(1, keyword=2)
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'keyword'"):
        sig(keyword=1)


def test_signature_bind_positional_or_keyword_arguments():
    def func(a, b, c):
        return a, b, c

    with pytest.raises(TypeError, match="missing a required argument: 'a'"):
        call(func)
    with pytest.raises(TypeError, match="missing a required argument: 'b'"):
        call(func, 1)
    with pytest.raises(TypeError, match="missing a required argument: 'c'"):
        call(func, 1, 2)
    assert call(func, 1, 2, 3) == (1, 2, 3)

    # one optional argument
    def func(a, b, c=0):
        return a, b, c

    assert call(func, 1, 2, 3) == (1, 2, 3)
    assert call(func, 1, 2) == (1, 2, 0)

    # two optional arguments
    def func(a, b=0, c=0):
        return a, b, c

    assert call(func, 1, 2, 3) == (1, 2, 3)
    assert call(func, 1, 2) == (1, 2, 0)
    assert call(func, 1) == (1, 0, 0)

    # three optional arguments
    def func(a=0, b=0, c=0):
        return a, b, c

    assert call(func, 1, 2, 3) == (1, 2, 3)
    assert call(func, 1, 2) == (1, 2, 0)
    assert call(func, 1) == (1, 0, 0)
    assert call(func) == (0, 0, 0)


def test_signature_bind_varargs():
    def func(*args):
        return args

    assert call(func) == ()
    assert call(func, 1) == (1,)
    assert call(func, 1, 2) == (1, 2)
    assert call(func, 1, 2, 3) == (1, 2, 3)

    def func(a, b, c=3, *args):
        return a, b, c, args

    assert call(func, 1, 2) == (1, 2, 3, ())
    assert call(func, 1, 2, 3) == (1, 2, 3, ())
    assert call(func, 1, 2, 3, 4) == (1, 2, 3, (4,))
    assert call(func, 1, 2, 3, 4, 5) == (1, 2, 3, (4, 5))
    assert call(func, 1, 2, 4) == (1, 2, 4, ())
    assert call(func, a=1, b=2, c=3) == (1, 2, 3, ())
    assert call(func, c=3, a=1, b=2) == (1, 2, 3, ())

    with pytest.raises(TypeError, match="multiple values for argument 'c'"):
        call(func, 1, 2, 3, c=4)

    def func(a, *args):
        return a, args

    with pytest.raises(TypeError, match="got an unexpected keyword argument 'args'"):
        call(func, a=0, args=1)

    def func(*args, **kwargs):
        return args, kwargs

    assert call(func, args=1) == ((), {"args": 1})

    sig = Signature.from_callable(func)
    ba = sig(args=1)
    assert ba == {"args": (), "kwargs": {"args": 1}}


def test_signature_bind_varkwargs():
    def func(**kwargs):
        return kwargs

    assert call(func) == {}
    assert call(func, foo="bar") == {"foo": "bar"}
    assert call(func, foo="bar", spam="ham") == {"foo": "bar", "spam": "ham"}

    def func(a, b, c=3, **kwargs):
        return a, b, c, kwargs

    assert call(func, 1, 2) == (1, 2, 3, {})
    assert call(func, 1, 2, foo="bar") == (1, 2, 3, {"foo": "bar"})
    assert call(func, 1, 2, foo="bar", spam="ham") == (
        1,
        2,
        3,
        {"foo": "bar", "spam": "ham"},
    )
    assert call(func, 1, 2, foo="bar", spam="ham", c=4) == (
        1,
        2,
        4,
        {"foo": "bar", "spam": "ham"},
    )
    assert call(func, 1, 2, c=4, foo="bar", spam="ham") == (
        1,
        2,
        4,
        {"foo": "bar", "spam": "ham"},
    )
    assert call(func, 1, 2, c=4, foo="bar", spam="ham", args=10) == (
        1,
        2,
        4,
        {"foo": "bar", "spam": "ham", "args": 10},
    )
    assert call(func, b=2, a=1, c=4, foo="bar", spam="ham") == (
        1,
        2,
        4,
        {"foo": "bar", "spam": "ham"},
    )


def test_signature_bind_varargs_and_varkwargs():
    def func(*args, **kwargs):
        return args, kwargs

    assert call(func) == ((), {})
    assert call(func, 1) == ((1,), {})
    assert call(func, 1, 2) == ((1, 2), {})
    assert call(func, foo="bar") == ((), {"foo": "bar"})
    assert call(func, 1, foo="bar") == ((1,), {"foo": "bar"})
    assert call(func, args=10), () == {"args": 10}
    assert call(func, 1, 2, foo="bar") == ((1, 2), {"foo": "bar"})
    assert call(func, 1, 2, foo="bar", spam="ham") == (
        (1, 2),
        {"foo": "bar", "spam": "ham"},
    )
    assert call(func, foo="bar", spam="ham", args=10) == (
        (),
        {"foo": "bar", "spam": "ham", "args": 10},
    )


def test_signature_bind_positional_only_arguments():
    def func(a, b, /, c=3):
        return a, b, c

    assert call(func, 1, 2) == (1, 2, 3)
    assert call(func, 1, 2, 4) == (1, 2, 4)
    assert call(func, 1, 2, c=4) == (1, 2, 4)
    with pytest.raises(TypeError, match="multiple values for argument 'c'"):
        call(func, 1, 2, 3, c=4)

    def func(a, b=2, /, c=3, *args):
        return a, b, c, args

    assert call(func, 1, 2) == (1, 2, 3, ())
    assert call(func, 1, 2, 4) == (1, 2, 4, ())
    assert call(func, 1, c=3) == (1, 2, 3, ())

    def func(a, b, c=3, /, foo=42, *, bar=50, **kwargs):
        return a, b, c, foo, bar, kwargs

    assert call(func, 1, 2, 4, 5, bar=6) == (1, 2, 4, 5, 6, {})
    assert call(func, 1, 2) == (1, 2, 3, 42, 50, {})
    assert call(func, 1, 2, foo=4, bar=5) == (1, 2, 3, 4, 5, {})
    assert call(func, 1, 2, foo=4, bar=5, c=10) == (1, 2, 3, 4, 5, {"c": 10})
    assert call(func, 1, 2, 30, c=31, foo=4, bar=5) == (1, 2, 30, 4, 5, {"c": 31})
    assert call(func, 1, 2, 30, foo=4, bar=5, c=31) == (1, 2, 30, 4, 5, {"c": 31})
    assert call(func, 1, 2, c=4) == (1, 2, 3, 42, 50, {"c": 4})
    assert call(func, 1, 2, c=4, foo=5) == (1, 2, 3, 5, 50, {"c": 4})

    with pytest.raises(
        TypeError, match="positional only argument 'a' passed as keyword argument"
    ):
        call(func, a=1, b=2)

    def func(a=1, b=2, /):
        return a, b

    with pytest.raises(TypeError, match="got an unexpected keyword argument 'a'"):
        call(func, a=3, b=4)

    def func(a, /, **kwargs):
        return a, kwargs

    assert call(func, "pos-only", bar="keyword") == ("pos-only", {"bar": "keyword"})


def test_signature_bind_keyword_only_arguments():
    def func(*, a, b, c=3):
        return a, b, c

    with pytest.raises(TypeError, match="too many positional arguments"):
        call(func, 1)

    assert call(func, a=1, b=2) == (1, 2, 3)
    assert call(func, a=1, b=2, c=4) == (1, 2, 4)

    def func(a, *, b, c=3, **kwargs):
        return a, b, c, kwargs

    with pytest.raises(TypeError, match="missing a required argument: 'b'"):
        call(func, 1)

    assert call(func, 1, b=2) == (1, 2, 3, {})
    assert call(func, 1, b=2, c=4) == (1, 2, 4, {})

    def func(*, a, b, c=3, foo=42, **kwargs):
        return a, b, c, foo, kwargs

    assert call(func, a=1, b=2) == (1, 2, 3, 42, {})
    assert call(func, a=1, b=2, foo=4) == (1, 2, 3, 4, {})
    assert call(func, a=1, b=2, foo=4, bar=5) == (1, 2, 3, 4, {"bar": 5})
    assert call(func, a=1, b=2, foo=4, bar=5, c=10) == (1, 2, 10, 4, {"bar": 5})
    assert call(func, a=1, b=2, foo=4, bar=5, c=10, spam=6) == (
        1,
        2,
        10,
        4,
        {"bar": 5, "spam": 6},
    )

    with pytest.raises(TypeError, match="missing a required argument: 'a'"):
        call(func, b=2)
    with pytest.raises(TypeError, match="missing a required argument: 'b'"):
        call(func, a=1)

    def func(a, *, b):
        return a, b

    assert call(func, 1, b=2) == (1, 2)
    with pytest.raises(TypeError, match="missing a required argument: 'b'"):
        call(func, 1)
    with pytest.raises(TypeError, match="missing a required argument: 'a'"):
        call(func, b=2)
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'c'"):
        call(func, a=1, b=2, c=3)
    with pytest.raises(TypeError, match="too many positional arguments"):
        call(func, 1, 2)
    with pytest.raises(TypeError, match="too many positional arguments"):
        call(func, 1, 2, c=3)

    def func(a, *, b, **kwargs):
        return a, b, kwargs

    assert call(func, 1, b=2) == (1, 2, {})
    assert call(func, 1, b=2, c=3) == (1, 2, {"c": 3})
    assert call(func, 1, b=2, c=3, d=4) == (1, 2, {"c": 3, "d": 4})
    assert call(func, a=1, b=2) == (1, 2, {})
    assert call(func, c=3, a=1, b=2) == (1, 2, {"c": 3})
    with pytest.raises(TypeError, match="missing a required argument: 'b'"):
        call(func, a=1)
    with pytest.raises(TypeError, match="missing a required argument: 'a'"):
        call(func, c=3, b=2)


def test_signature_bind_with_arg_named_self():
    def test(a, self, b):
        pass

    sig = Signature.from_callable(test)
    ba = sig(1, 2, 3)
    args, _ = sig.unbind(ba)
    assert args == (1, 2, 3)
    ba = sig(1, self=2, b=3)
    args, _ = sig.unbind(ba)
    assert args == (1, 2, 3)


def test_signature_unbind_from_callable():
    def test(a: int, b: int, c: int = 1): ...

    sig = Signature.from_callable(test)
    bound = sig(2, 3)

    assert bound == {"a": 2, "b": 3, "c": 1}

    args, kwargs = sig.unbind(bound)
    assert args == (2, 3, 1)
    assert kwargs == {}


def test_signature_unbind_from_callable_with_varargs():
    def test(a: int, b: int, *args: int): ...

    sig = Signature.from_callable(test)
    bound = sig(2, 3)

    assert bound == {"a": 2, "b": 3, "args": ()}
    args, kwargs = sig.unbind(bound)
    assert args == (2, 3)
    assert kwargs == {}

    bound = sig(2, 3, 4, 5)
    assert bound == {"a": 2, "b": 3, "args": (4, 5)}
    args, kwargs = sig.unbind(bound)
    assert args == (2, 3, 4, 5)
    assert kwargs == {}


def test_signature_unbind_from_callable_with_positional_only_arguments():
    def test(a: int, b: int, /, c: int = 1): ...

    sig = Signature.from_callable(test)
    bound = sig(2, 3)
    assert bound == {"a": 2, "b": 3, "c": 1}

    args, kwargs = sig.unbind(bound)
    assert args == (2, 3, 1)
    assert kwargs == {}

    bound = sig(2, 3, 4)
    assert bound == {"a": 2, "b": 3, "c": 4}

    args, kwargs = sig.unbind(bound)
    assert args == (2, 3, 4)
    assert kwargs == {}


def test_signature_unbind_from_callable_with_keyword_only_arguments():
    def test(a: int, b: int, *, c: float, d: float = 0.0): ...

    sig = Signature.from_callable(test)
    bound = sig(2, 3, c=4.0)
    assert bound == {"a": 2, "b": 3, "c": 4.0, "d": 0.0}

    args, kwargs = sig.unbind(bound)
    assert args == (2, 3)
    assert kwargs == {"c": 4.0, "d": 0.0}


def test_signature_unbind():
    def func(a, b, c=1): ...

    sig = Signature.from_callable(func)
    bound = sig(1, 2)

    assert bound == {"a": 1, "b": 2, "c": 1}

    args, kwargs = sig.unbind(bound)
    assert args == (1, 2, 1)
    assert kwargs == {}


@pytest.mark.parametrize("d", [(), (5, 6, 7)])
def test_signature_unbind_with_empty_variadic(d):
    def func(a, b, c, *args, e=None):
        return a, b, c, args, e

    sig = Signature.from_callable(func)
    bound = sig(1, 2, 3, *d, e=4)
    assert bound == {"a": 1, "b": 2, "c": 3, "args": d, "e": 4}

    args, kwargs = sig.unbind(bound)
    assert args == (1, 2, 3, *d)
    assert kwargs == {"e": 4}


def test_signature_merge():
    def f1(a, b, c=1): ...

    def f2(d, e, f=2): ...

    def f3(d, a=1, **kwargs): ...

    sig1 = Signature.from_callable(f1)
    sig2 = Signature.from_callable(f2)
    sig3 = Signature.from_callable(f3)

    sig12 = Signature.merge([sig1, sig2])
    assert sig12.parameters == {
        "a": Parameter(Parameter.POSITIONAL_OR_KEYWORD),
        "b": Parameter(Parameter.POSITIONAL_OR_KEYWORD),
        "d": Parameter(Parameter.POSITIONAL_OR_KEYWORD),
        "e": Parameter(Parameter.POSITIONAL_OR_KEYWORD),
        "c": Parameter(Parameter.POSITIONAL_OR_KEYWORD, default=1),
        "f": Parameter(Parameter.POSITIONAL_OR_KEYWORD, default=2),
    }
    assert tuple(sig12.parameters.keys()) == ("a", "b", "d", "e", "c", "f")

    sig13 = Signature.merge([sig1, sig3])
    assert sig13.parameters == {
        "b": Parameter(Parameter.POSITIONAL_OR_KEYWORD),
        "d": Parameter(Parameter.POSITIONAL_OR_KEYWORD),
        "a": Parameter(Parameter.POSITIONAL_OR_KEYWORD, default=1),
        "c": Parameter(Parameter.POSITIONAL_OR_KEYWORD, default=1),
        "kwargs": Parameter(Parameter.VAR_KEYWORD),
    }
    assert tuple(sig13.parameters.keys()) == ("b", "d", "a", "c", "kwargs")


def test_annotated_function():
    @annotated(a=IsType(int), b=IsType(int), c=IsType(int))
    def test(a, b, c=1):
        return a + b + c

    assert test(2, 3) == 6
    assert test(2, 3, 4) == 9
    assert test(2, 3, c=4) == 9
    assert test(a=2, b=3, c=4) == 9

    with pytest.raises(MatchError):
        test(2, 3, c="4")

    @annotated(a=IsType(int))
    def test(a, b, c=1):
        return (a, b, c)

    assert test(2, "3") == (2, "3", 1)


def test_annotated_function_with_type_annotations():
    @annotated()
    def test(a: int, b: int, c: int = 1):
        return a + b + c

    assert test(2, 3) == 6

    @annotated
    def test(a: int, b: int, c: int = 1):
        return a + b + c

    assert test(2, 3) == 6

    @annotated
    def test(a: int, b, c=1):
        return (a, b, c)

    assert test(2, 3, "4") == (2, 3, "4")


def test_annotated_function_with_return_type_annotation():
    @annotated
    def test_ok(a: int, b: int, c: int = 1) -> int:
        return a + b + c

    @annotated
    def test_wrong(a: int, b: int, c: int = 1) -> int:
        return "invalid result"

    assert test_ok(2, 3) == 6
    with pytest.raises(MatchError):
        test_wrong(2, 3)


def test_annotated_function_with_keyword_overrides():
    @annotated(b=IsType(float))
    def test(a: int, b: int, c: int = 1):
        return a + b + c

    with pytest.raises(MatchError):
        test(2, 3)

    assert test(2, 3.0) == 6.0


def test_annotated_function_with_list_overrides():
    with pytest.raises(MatchError):

        @annotated([IsType(int), IsType(int), IsType(str)])
        def test(a: int, b: int, c: int = 1):
            return a + b + c

    @annotated([IsType(int), IsType(int), IsType(float)])
    def test(a: int, b: int, c: int = 1.0):
        return a + b + c

    assert test(2, 3) == 6.0
    assert isinstance(test(2, 3), float)
    with pytest.raises(MatchError):
        test(2, 3, 4)


def test_annotated_function_with_list_overrides_and_return_override():
    with pytest.raises(MatchError):

        @annotated([IsType(int), IsType(int), IsType(float)], IsType(float))
        def test(a: int, b: int, c: int = 1):
            return a + b + c

    @annotated([IsType(int), IsType(int), IsType(float)], IsType(float))
    def test(a: int, b: int, c: int = 1.1):
        return a + b + c

    assert test(2, 3) == 6.1
    with pytest.raises(MatchError):
        test(2, 3, 4)

    assert test(2, 3, 4.0) == 9.0


@pattern
def short_str(x, **context):
    if len(x) > 3:
        return x
    else:
        raise ValueError("string is too short")


@pattern
def endswith_d(x, **context):
    if x.endswith("d"):
        return x
    else:
        raise ValueError("string does not end with 'd'")


def test_annotated_function_with_complex_type_annotations():
    @annotated
    def test(a: Annotated[str, short_str, endswith_d], b: Union[int, float]):
        return a, b

    assert test("abcd", 1) == ("abcd", 1)
    assert test("---d", 1.0) == ("---d", 1.0)

    with pytest.raises(MatchError):
        test("---c", 1)
    with pytest.raises(MatchError):
        test("123", 1)
    with pytest.raises(MatchError):
        test("abcd", "qweqwe")


def test_annotated_function_without_annotations():
    @annotated
    def test(a, b, c):
        return a, b, c

    assert test(1, 2, 3) == (1, 2, 3)
    assert list(test.__signature__.parameters.keys()) == ["a", "b", "c"]


def test_annotated_function_without_decoration():
    def test(a, b, c):
        return a + b + c

    func = annotated(test)
    with pytest.raises(TypeError):
        func(1, 2)

    assert func(1, 2, c=3) == 6


def test_annotated_function_with_varargs():
    @annotated
    def test(a: float, b: float, *args: int):
        return sum((a, b) + args)

    assert test(1.0, 2.0, 3, 4) == 10.0
    assert test(1.0, 2.0, 3, 4, 5) == 15.0
    assert test(1.0, 2.0, 3, 4, 5, 6.0) == 21.0


def test_annotated_function_with_varkwargs():
    @annotated
    def test(a: float, b: float, **kwargs: int):
        return sum((a, b) + tuple(kwargs.values()))

    assert test(1.0, 2.0, c=3, d=4) == 10.0
    assert test(1.0, 2.0, c=3, d=4, e=5) == 15.0
    assert test(1.0, 2.0, c=3, d=4, e=5, f=6.0) == 21.0


# def test_multiple_validation_failures():
#     @annotated
#     def test(a: float, b: float, *args: int, **kwargs: int): ...

#     with pytest.raises(ValidationError) as excinfo:
#         test(1.0, 2.0, 3.0, 4, c=5.0, d=6)

#     assert len(excinfo.value.errors) == 2


def test_signature_patterns():
    def func(a: int, b: str) -> str: ...

    sig = Signature.from_callable(func, allow_coercion=False)
    assert sig.parameters["a"].pattern == IsType(int)
    assert sig.parameters["b"].pattern == IsType(str)
    assert sig.return_pattern == IsType(str)

    def func(a: int, b: str, c: str = "0") -> str: ...

    sig = Signature.from_callable(func, allow_coercion=True)
    assert sig.parameters["a"].pattern == As(int)
    assert sig.parameters["b"].pattern == Is(str)
    assert sig.parameters["c"].pattern == Is(str)
    assert sig.return_pattern == IsType(str)

    def func(a: int, b: str, *args): ...

    sig = Signature.from_callable(func)
    assert sig.parameters["a"].pattern == As(int)
    assert sig.parameters["b"].pattern == Is(str)
    assert sig.parameters["args"].pattern == TupleOf(Anything())
    assert sig.return_pattern == Anything()

    def func(a: int, b: str, c: str = "0", *args, **kwargs: int) -> float: ...

    sig = Signature.from_callable(func)
    assert sig.parameters["a"].pattern == As(int)
    assert sig.parameters["b"].pattern == Is(str)
    assert sig.parameters["c"].pattern == Is(str)
    assert sig.parameters["args"].pattern == TupleOf(Anything())
    assert sig.parameters["kwargs"].pattern == FrozenDictOf(Anything(), As(int))
    assert sig.return_pattern == As(float)


def test_annotated_with_class():
    @annotated
    class A:
        a: int
        b: str
        c: float
        d: Optional[int]

        def __init__(self, a, b, c, d=1):
            self.a = a
            self.b = b
            self.c = c
            self.d = d

    with pytest.raises(MatchError):
        A(1, "2", "d")


def test_annotated_with_dataclass():
    @annotated
    @dataclass
    class InventoryItem:
        name: str
        unit_price: float
        quantity_on_hand: int = 10

    @annotated
    @dataclass
    class InventoryItemStrict:
        name: str
        unit_price: Is[float]
        quantity_on_hand: Is[int] = 0

    @annotated
    @dataclass
    class InventoryItemLoose:
        name: str
        unit_price: As[float]
        quantity_on_hand: As[int] = 10

    items = [
        InventoryItem("widget", 3.0, 10),
        InventoryItem("widget", 3.0),
        InventoryItem("widget", "3.0", 10),
        InventoryItem("widget", 3.0, "10"),
        InventoryItemLoose("widget", 3.0, 10),
        InventoryItemLoose("widget", 3.0),
        InventoryItemLoose("widget", "3.0", 10),
        InventoryItemLoose("widget", 3.0, "10"),
    ]
    for item in items:
        assert item.name == "widget"
        assert item.unit_price == 3.0
        assert item.quantity_on_hand == 10

    with pytest.raises(MatchError):
        InventoryItem("widget", 3.0, "10.1")
    with pytest.raises(MatchError):
        InventoryItem("widget", 3.0, 10.1)

    item = InventoryItemStrict("widget", 3.0, 10)
    assert item.name == "widget"
    assert item.unit_price == 3.0
    assert item.quantity_on_hand == 10

    with pytest.raises(MatchError):
        InventoryItemStrict("widget", "3.0", 10)
    with pytest.raises(MatchError):
        InventoryItemStrict("widget", 3.0, "10")


##################################################


is_any = IsType(object)
is_bool = IsType(bool)
is_float = IsType(float)
is_int = IsType(int)
is_str = IsType(str)
is_list = IsType(list)


class Op(Annotable):
    pass


class Value(Op):
    arg = argument(IsType(object))


class StringOp(Value):
    arg = argument(IsType(str))


class BetweenSimple(Annotable):
    value = argument(is_int)
    lower = optional(is_int, default=0)
    upper = optional(is_int, default=None)


class BetweenWithExtra(Annotable):
    extra = attribute(is_int)
    value = argument(is_int)
    lower = optional(is_int, default=0)
    upper = optional(is_int, default=None)


class BetweenWithCalculated(Annotable, immutable=True, hashable=True):
    value = argument(is_int)
    lower = optional(is_int, default=0)
    upper = optional(is_int, default=None)

    @attribute
    def calculated(self):
        return self.value + self.lower


class VariadicArgs(Annotable, immutable=True, hashable=True):
    args = varargs(is_int)


class VariadicKeywords(Annotable, immutable=True, hashable=True):
    kwargs = varkwargs(is_int)


class VariadicArgsAndKeywords(Annotable, immutable=True, hashable=True):
    args = varargs(is_int)
    kwargs = varkwargs(is_int)


T = TypeVar("T", covariant=True)
K = TypeVar("K", covariant=True)
V = TypeVar("V", covariant=True)


class List(Annotable, Generic[T]):
    @classmethod
    def __coerce__(self, values, T=None):
        values = tuple(values)
        if values:
            head, *rest = values
            return ConsList(head, rest)
        else:
            return EmptyList()

    def __eq__(self, other) -> bool:
        if not isinstance(other, List):
            return NotImplemented
        if len(self) != len(other):
            return False
        for a, b in zip(self, other):
            if a != b:
                return False
        return True


# AnnotableMeta doesn't extend ABCMeta, so we need to register the class
# this is due to performance reasons since ABCMeta overrides
# __instancecheck__ and __subclasscheck__ which makes
# issubclass and isinstance way slower
Sequence.register(List)


class EmptyList(List[T]):
    def __getitem__(self, key):
        raise IndexError(key)

    def __len__(self):
        return 0


class ConsList(List[T]):
    head: T
    rest: List[T]

    def __getitem__(self, key):
        if key == 0:
            return self.head
        else:
            return self.rest[key - 1]

    def __len__(self):
        return len(self.rest) + 1


class Map(Annotable, Generic[K, V]):
    @classmethod
    def __coerce__(self, pairs, K=None, V=None):
        pairs = dict(pairs)
        if pairs:
            head_key = next(iter(pairs))
            head_value = pairs.pop(head_key)
            rest = pairs
            return ConsMap((head_key, head_value), rest)
        else:
            return EmptyMap()

    def __eq__(self, other) -> bool:
        if not isinstance(other, Map):
            return NotImplemented
        if len(self) != len(other):
            return False
        for key in self:
            if self[key] != other[key]:
                return False
        return True

    def items(self):
        for key in self:
            yield key, self[key]


# AnnotableMeta doesn't extend ABCMeta, so we need to register the class
# this is due to performance reasons since ABCMeta overrides
# __instancecheck__ and __subclasscheck__ which makes
# issubclass and isinstance way slower
Mapping.register(Map)


class EmptyMap(Map[K, V]):
    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


class ConsMap(Map[K, V]):
    head: tuple[K, V]
    rest: Map[K, V]

    def __getitem__(self, key):
        if key == self.head[0]:
            return self.head[1]
        else:
            return self.rest[key]

    def __iter__(self):
        yield self.head[0]
        yield from self.rest

    def __len__(self):
        return len(self.rest) + 1


class Integer(int):
    @classmethod
    def __coerce__(cls, value):
        return Integer(value)


class Float(float):
    @classmethod
    def __coerce__(cls, value):
        return Float(value)


class MyExpr(Annotable):
    a: Integer
    b: List[Float]
    c: Map[str, Integer]


class MyInt(int):
    @classmethod
    def __coerce__(cls, value):
        return cls(value)


class MyFloat(float):
    @classmethod
    def __coerce__(cls, value):
        return cls(value)


J = TypeVar("J", bound=MyInt, covariant=True)
F = TypeVar("F", bound=MyFloat, covariant=True)
N = TypeVar("N", bound=Union[MyInt, MyFloat], covariant=True)


class MyValue(Annotable, Generic[J, F]):
    integer: J
    floating: F
    numeric: N


def test_annotable():
    class Between(BetweenSimple):
        pass

    assert not issubclass(type(Between), ABCMeta)
    assert type(Between) is AnnotableMeta

    argnames = ("value", "lower", "upper")
    signature = BetweenSimple.__signature__
    assert isinstance(signature, Signature)
    assert BetweenSimple.__slots__ == argnames

    obj = BetweenSimple(10, lower=2)
    assert obj.value == 10
    assert obj.lower == 2
    assert obj.upper is None
    assert obj.__argnames__ == argnames
    assert obj.__slots__ == ("value", "lower", "upper")
    assert not hasattr(obj, "__dict__")
    assert obj.__module__ == __name__
    assert type(obj).__qualname__ == "BetweenSimple"

    # test that a child without additional arguments doesn't have __dict__
    obj = Between(10, lower=2)
    assert obj.__slots__ == tuple()
    assert not hasattr(obj, "__dict__")
    assert obj.__module__ == __name__
    assert type(obj).__qualname__ == "test_annotable.<locals>.Between"

    copied = copy.copy(obj)
    assert obj == copied
    assert obj is not copied

    # copied = obj.copy()
    # assert obj == copied
    # assert obj is not copied

    # obj2 = Between(10, lower=8)
    # assert obj.copy(lower=8) == obj2


def test_annotable_keeps_annotations_classvar():
    class MyClass(Annotable):
        a: int
        b: str

    assert MyClass.__annotations__ == {"a": "int", "b": "str"}


def test_annotable_with_bound_typevars_properly_coerce_values():
    v = MyValue(1.1, 2.2, 3.3)
    assert isinstance(v.integer, MyInt)
    assert v.integer == 1
    assert isinstance(v.floating, MyFloat)
    assert v.floating == 2.2
    assert isinstance(v.numeric, MyInt)
    assert v.numeric == 3


def test_annotable_picklable_with_additional_attributes():
    a = BetweenWithExtra(10, lower=2)
    b = BetweenWithExtra(10, lower=2)
    assert a == b
    assert a is not b

    a.extra = 1
    assert a.extra == 1
    assert a != b

    assert a == pickle.loads(pickle.dumps(a))


def test_annotable_is_mutable_by_default():
    # TODO(kszucs): more exhaustive testing of mutability, e.g. setting
    # optional value to None doesn't set to the default value
    class Op(Annotable):
        __slots__ = ("custom",)

        a = argument(is_int)
        b = argument(is_int)

    p = Op(1, 2)
    assert p.a == 1
    p.a = 3
    assert p.a == 3
    assert p == Op(a=3, b=2)

    # test that non-annotable attributes can be set as well
    p.custom = 1
    assert p.custom == 1


def test_annotable_with_type_annotations() -> None:
    class Op1(Annotable):
        foo: int
        bar: str = ""

    p = Op1(1)
    assert p.foo == 1
    assert p.bar == ""

    with pytest.raises(MatchError):

        class Op2(Annotable):
            bar: str = None

    class Op2(Annotable):
        bar: str | None = None

    op = Op2()
    assert op.bar is None


class RecursiveNode(Annotable):
    child: Optional[Self] = None


def test_annotable_with_self_typehint():
    node = RecursiveNode(RecursiveNode(RecursiveNode(None)))
    assert isinstance(node, RecursiveNode)
    assert isinstance(node.child, RecursiveNode)
    assert isinstance(node.child.child, RecursiveNode)
    assert node.child.child.child is None

    with pytest.raises(MatchError):
        RecursiveNode(1)


def test_annotable_with_recursive_generic_type_annotations():
    # testing cons list
    pattern = Pattern.from_typehint(List[Integer], allow_coercion=True)
    values = ["1", 2.0, 3]
    result = pattern.apply(values, {})
    expected = ConsList(1, ConsList(2, ConsList(3, EmptyList())))
    assert result == expected
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert len(result) == 3
    with pytest.raises(IndexError):
        result[3]

    # testing cons map
    pattern = Pattern.from_typehint(Map[Integer, Float], allow_coercion=True)
    values = {"1": 2, 3: "4.0", 5: 6.0}
    result = pattern.apply(values, {})
    expected = ConsMap((1, 2.0), ConsMap((3, 4.0), ConsMap((5, 6.0), EmptyMap())))
    assert result == expected
    assert result[1] == 2.0
    assert result[3] == 4.0
    assert result[5] == 6.0
    assert len(result) == 3
    with pytest.raises(KeyError):
        result[7]

    # testing both encapsulated in a class
    expr = MyExpr(a="1", b=["2.0", 3, True], c={"a": "1", "b": 2, "c": 3.0})
    assert expr.a == 1
    assert expr.b == ConsList(2.0, ConsList(3.0, ConsList(1.0, EmptyList())))
    assert expr.c == ConsMap(("a", 1), ConsMap(("b", 2), ConsMap(("c", 3), EmptyMap())))


def test_annotable_as_immutable():
    class AnnImm(Annotable, immutable=True):
        value = argument(is_int)
        lower = optional(is_int, default=0)
        upper = optional(is_int, default=None)

    assert AnnImm.__mro__ == (AnnImm, Immutable, Annotable, object)

    obj = AnnImm(3, lower=0, upper=4)
    with pytest.raises(AttributeError):
        obj.value = 1


def test_annotable_equality_checks():
    class Between(Annotable):
        value = argument(is_int)
        lower = optional(is_int, default=0)
        upper = optional(is_int, default=None)

    a = Between(3, lower=0, upper=4)
    b = Between(3, lower=0, upper=4)
    c = Between(2, lower=0, upper=4)

    assert a == b
    assert b == a
    assert a != c
    assert c != a
    assert a.__eq__(b)
    assert not a.__eq__(c)


def test_maintain_definition_order():
    class Between(Annotable):
        value = argument(is_int)
        lower = optional(is_int, default=0)
        upper = optional(is_int, default=None)

    assert Between.__argnames__ == ("value", "lower", "upper")


def test_signature_inheritance():
    class IntBinop(Annotable):
        left = argument(is_int)
        right = argument(is_int)

    class FloatAddRhs(IntBinop):
        right = argument(is_float)

    class FloatAddClip(FloatAddRhs):
        left = argument(is_float)
        clip_lower = optional(is_int, default=0)
        clip_upper = optional(is_int, default=10)

    class IntAddClip(FloatAddClip, IntBinop):
        pass

    assert IntBinop.__signature__ == Signature(
        {
            "left": Parameter(Parameter.POSITIONAL_OR_KEYWORD, is_int),
            "right": Parameter(Parameter.POSITIONAL_OR_KEYWORD, is_int),
        }
    )

    assert FloatAddRhs.__signature__ == Signature(
        {
            "left": Parameter(Parameter.POSITIONAL_OR_KEYWORD, is_int),
            "right": Parameter(Parameter.POSITIONAL_OR_KEYWORD, is_float),
        }
    )

    assert FloatAddClip.__signature__ == Signature(
        {
            "left": Parameter(Parameter.POSITIONAL_OR_KEYWORD, is_float),
            "right": Parameter(Parameter.POSITIONAL_OR_KEYWORD, is_float),
            "clip_lower": Parameter(Parameter.POSITIONAL_OR_KEYWORD, is_int, default=0),
            "clip_upper": Parameter(
                Parameter.POSITIONAL_OR_KEYWORD, is_int, default=10
            ),
        }
    )

    assert IntAddClip.__signature__ == Signature(
        {
            "left": Parameter(Parameter.POSITIONAL_OR_KEYWORD, is_float),
            "right": Parameter(Parameter.POSITIONAL_OR_KEYWORD, is_float),
            "clip_lower": Parameter(Parameter.POSITIONAL_OR_KEYWORD, is_int, default=0),
            "clip_upper": Parameter(
                Parameter.POSITIONAL_OR_KEYWORD, is_int, default=10
            ),
        }
    )


def test_positional_argument_reordering():
    class Farm(Annotable):
        ducks = argument(is_int)
        donkeys = argument(is_int)
        horses = argument(is_int)
        goats = argument(is_int)
        chickens = argument(is_int)

    class NoHooves(Farm):
        horses = optional(is_int, default=0)
        goats = optional(is_int, default=0)
        donkeys = optional(is_int, default=0)

    f1 = Farm(1, 2, 3, 4, 5)
    f2 = Farm(1, 2, goats=4, chickens=5, horses=3)
    f3 = Farm(1, 0, 0, 0, 100)
    assert f1 == f2
    assert f1 != f3

    g1 = NoHooves(1, 2, donkeys=-1)
    assert g1.ducks == 1
    assert g1.chickens == 2
    assert g1.donkeys == -1
    assert g1.horses == 0
    assert g1.goats == 0


def test_keyword_argument_reordering():
    class Alpha(Annotable):
        a = argument(is_int)
        b = argument(is_int)

    class Beta(Alpha):
        c = argument(is_int)
        d = optional(is_int, default=0)
        e = argument(is_int)

    obj = Beta(1, 2, 3, 4)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 3
    assert obj.e == 4
    assert obj.d == 0

    obj = Beta(1, 2, 3, 4, 5)
    assert obj.d == 5
    assert obj.e == 4


def test_variadic_argument_reordering():
    class Test(Annotable):
        a = argument(is_int)
        b = argument(is_int)
        args = varargs(is_int)

    class Test2(Test):
        c = argument(is_int)
        args = varargs(is_int)

    with pytest.raises(TypeError, match="missing a required argument: 'c'"):
        Test2(1, 2)

    a = Test2(1, 2, 3)
    assert a.a == 1
    assert a.b == 2
    assert a.c == 3
    assert a.args == ()

    b = Test2(*range(5))
    assert b.a == 0
    assert b.b == 1
    assert b.c == 2
    assert b.args == (3, 4)

    msg = "only one variadic \\*args parameter is allowed"
    with pytest.raises(TypeError, match=msg):

        class Test3(Test):
            another_args = varargs(is_int)


def test_variadic_keyword_argument_reordering():
    class Test(Annotable):
        a = argument(is_int)
        b = argument(is_int)
        options = varkwargs(is_int)

    class Test2(Test):
        c = argument(is_int)
        options = varkwargs(is_int)

    with pytest.raises(TypeError, match="missing a required argument: 'c'"):
        Test2(1, 2)

    a = Test2(1, 2, c=3)
    assert a.a == 1
    assert a.b == 2
    assert a.c == 3
    assert a.options == {}

    b = Test2(1, 2, c=3, d=4, e=5)
    assert b.a == 1
    assert b.b == 2
    assert b.c == 3
    assert b.options == {"d": 4, "e": 5}

    msg = "only one variadic \\*\\*kwargs parameter is allowed"
    with pytest.raises(TypeError, match=msg):

        class Test3(Test):
            another_options = varkwargs(is_int)


def test_variadic_argument():
    class Test(Annotable):
        a = argument(is_int)
        b = argument(is_int)
        args = varargs(is_int)

    assert Test(1, 2).args == ()
    assert Test(1, 2, 3).args == (3,)
    assert Test(1, 2, 3, 4, 5).args == (3, 4, 5)


def test_variadic_keyword_argument():
    class Test(Annotable):
        first = argument(is_int)
        second = argument(is_int)
        options = varkwargs(is_int)

    assert Test(1, 2).options == {}
    assert Test(1, 2, a=3).options == {"a": 3}
    assert Test(1, 2, a=3, b=4, c=5).options == {"a": 3, "b": 4, "c": 5}


# def test_copy_with_variadic_argument():
#     class Foo(Annotable):
#         a = is_int
#         b = is_int
#         args = varargs(is_int)

#     class Bar(Concrete):
#         a = is_int
#         b = is_int
#         args = varargs(is_int)

#     for t in [Foo(1, 2, 3, 4, 5), Bar(1, 2, 3, 4, 5)]:
#         assert t.a == 1
#         assert t.b == 2
#         assert t.args == (3, 4, 5)

#         u = t.copy(a=6, args=(8, 9, 10))
#         assert u.a == 6
#         assert u.b == 2
#         assert u.args == (8, 9, 10)


# def test_concrete_copy_with_unknown_argument_raise():
#     class Bar(Concrete):
#         a = is_int
#         b = is_int

#     t = Bar(1, 2)
#     assert t.a == 1
#     assert t.b == 2

#     with pytest.raises(AttributeError, match="Unexpected arguments"):
#         t.copy(c=3, d=4)


def test_immutable_pickling_variadic_arguments():
    v = VariadicArgs(1, 2, 3, 4, 5)
    assert v.args == (1, 2, 3, 4, 5)
    assert v == pickle.loads(pickle.dumps(v))

    v = VariadicKeywords(a=3, b=4, c=5)
    assert v.kwargs == {"a": 3, "b": 4, "c": 5}
    assert v == pickle.loads(pickle.dumps(v))

    v = VariadicArgsAndKeywords(1, 2, 3, 4, 5, a=3, b=4, c=5)
    assert v.args == (1, 2, 3, 4, 5)
    assert v.kwargs == {"a": 3, "b": 4, "c": 5}
    assert v == pickle.loads(pickle.dumps(v))


def test_dont_copy_default_argument():
    default = tuple()

    class Op(Annotable):
        arg = optional(IsType(tuple), default=default)

    op = Op()
    assert op.arg is default


# def test_copy_mutable_with_default_attribute():
#     class Test(Annotable):
#         a = attribute(InstanceOf(dict), default={})
#         b = argument(InstanceOf(str))  # required argument

#         @attribute
#         def c(self):
#             return self.b.upper()

#     t = Test("t")
#     assert t.a == {}
#     assert t.b == "t"
#     assert t.c == "T"

#     with pytest.raises(ValidationError):
#         t.a = 1
#     t.a = {"map": "ping"}
#     assert t.a == {"map": "ping"}

#     assert t.copy() == t

#     u = t.copy(b="u")
#     assert u.b == "u"
#     assert u.c == "T"
#     assert u.a == {"map": "ping"}

#     x = t.copy(a={"emp": "ty"})
#     assert x.a == {"emp": "ty"}
#     assert x.b == "t"


def test_slots_are_inherited_and_overridable():
    class Op(Annotable):
        __slots__ = ("_cache",)  # first definition
        arg = argument(Anything())

    class StringOp(Op):
        arg = argument(IsType(str))  # new overridden slot

    class StringSplit(StringOp):
        sep = argument(IsType(str))  # new slot

    class StringJoin(StringOp):
        __slots__ = ("_memoize",)  # new slot
        sep = argument(IsType(str))  # new overridden slot

    assert Op.__slots__ == ("_cache", "arg")
    assert StringOp.__slots__ == ("arg",)
    assert StringSplit.__slots__ == ("sep",)
    assert StringJoin.__slots__ == ("_memoize", "sep")


def test_multiple_inheritance():
    # multiple inheritance is allowed only if one of the parents has non-empty
    # __slots__ definition, otherwise python will raise lay-out conflict

    class Op(Annotable):
        __slots__ = ("_hash",)

    class Value(Annotable):
        arg = argument(IsType(object))

    class Reduction(Value):
        pass

    class UDF(Value):
        func = argument(IsType(Callable))

    class UDAF(UDF, Reduction):
        arity = argument(is_int)

    class A(Annotable):
        a = argument(is_int)

    class B(Annotable):
        b = argument(is_int)

    msg = "multiple bases have instance lay-out conflict"
    with pytest.raises(TypeError, match=msg):

        class AB(A, B):
            ab = argument(is_int)

    assert UDAF.__slots__ == ("arity",)
    strlen = UDAF(arg=2, func=lambda value: len(str(value)), arity=1)
    assert strlen.arg == 2
    assert strlen.arity == 1


@pytest.mark.parametrize(
    "obj",
    [
        StringOp("something"),
        StringOp(arg="something"),
    ],
)
def test_pickling_support(obj):
    assert obj == pickle.loads(pickle.dumps(obj))


def test_multiple_inheritance_argument_order():
    class Value(Annotable):
        arg = argument(is_any)

    class VersionedOp(Value):
        version = argument(is_int)

    class Reduction(Annotable):
        pass

    class Sum(VersionedOp, Reduction):
        where = optional(is_bool, default=False)

    assert tuple(Sum.__signature__.parameters.keys()) == ("arg", "version", "where")


def test_multiple_inheritance_optional_argument_order():
    class Value(Annotable):
        pass

    class ConditionalOp(Annotable):
        where = optional(is_bool, default=False)

    class Between(Value, ConditionalOp):
        min = argument(is_int)
        max = argument(is_int)
        how = optional(is_str, default="strict")

    assert tuple(Between.__signature__.parameters.keys()) == (
        "min",
        "max",
        "how",
        "where",
    )


def test_immutability():
    class Value(Annotable, immutable=True):
        a = argument(is_int)

    op = Value(1)
    with pytest.raises(AttributeError):
        op.a = 3


class BaseValue(Annotable):
    i = argument(is_int)
    j = attribute(is_int)


class Value2(BaseValue):
    @attribute
    def k(self):
        return 3


class Value3(BaseValue):
    k = attribute(is_int, default=3)


class Value4(BaseValue):
    k = attribute(Option(is_int), default=None)


def test_annotable_with_dict_slot():
    class Flexible(Annotable):
        __slots__ = ("__dict__",)

    v = Flexible()
    v.a = 1
    v.b = 2
    assert v.a == 1
    assert v.b == 2


def test_annotable_attribute():
    with pytest.raises(TypeError, match="too many positional arguments"):
        BaseValue(1, 2)

    v = BaseValue(1)
    assert v.__slots__ == ("i", "j")
    assert v.i == 1
    assert not hasattr(v, "j")
    v.j = 2
    assert v.j == 2

    # TODO(kszucs)
    # with pytest.raises(TypeError):
    #     v.j = "foo"


def test_annotable_attribute_init():
    assert Value2.__slots__ == ("k",)
    v = Value2(1)

    assert v.i == 1
    assert not hasattr(v, "j")
    v.j = 2
    assert v.j == 2
    assert v.k == 3

    v = Value3(1)
    assert v.k == 3

    v = Value4(1)
    assert v.k is None


def test_annotable_mutability_and_serialization():
    v_ = BaseValue(1)
    v_.j = 2
    v = BaseValue(1)
    v.j = 2
    assert v_ == v
    assert v_.j == v.j == 2

    assert repr(v) == "BaseValue(i=1)"
    w = pickle.loads(pickle.dumps(v))
    assert w.i == 1
    assert w.j == 2
    assert v == w

    v.j = 4
    assert v_ != v
    w = pickle.loads(pickle.dumps(v))
    assert w == v
    assert repr(w) == "BaseValue(i=1)"


def test_initialized_attribute_basics():
    class Value(Annotable):
        a = argument(is_int)

        @attribute
        def double_a(self):
            return 2 * self.a

    op = Value(1)
    assert op.a == 1
    assert op.double_a == 2
    assert "double_a" in Value.__slots__


def test_initialized_attribute_with_validation():
    class Value(Annotable):
        a = argument(is_int)

        @attribute(int)
        def double_a(self):
            return 2 * self.a

    op = Value(1)
    assert op.a == 1
    assert op.double_a == 2
    assert "double_a" in Value.__slots__

    op.double_a = 3
    assert op.double_a == 3

    with pytest.raises(MatchError):
        op.double_a = "foo"


def test_initialized_attribute_mixed_with_classvar():
    class Value(Annotable):
        arg = argument(is_int)

        shape = "like-arg"
        dtype = "like-arg"

    class Reduction(Value):
        shape = "scalar"

    class Variadic(Value):
        @attribute
        def shape(self):
            if self.arg > 10:
                return "columnar"
            else:
                return "scalar"

    r = Reduction(1)
    assert r.shape == "scalar"
    assert "shape" not in r.__slots__

    v = Variadic(1)
    assert v.shape == "scalar"
    assert "shape" in v.__slots__

    v = Variadic(100)
    assert v.shape == "columnar"
    assert "shape" in v.__slots__


def test_hashable():
    assert BetweenWithCalculated.__mro__ == (
        BetweenWithCalculated,
        Hashable,
        Immutable,
        Annotable,
        object,
    )

    assert BetweenWithCalculated.__eq__ is Hashable.__eq__
    assert BetweenWithCalculated.__argnames__ == ("value", "lower", "upper")

    # annotable
    obj = BetweenWithCalculated(10, lower=5, upper=15)
    obj2 = BetweenWithCalculated(10, lower=5, upper=15)
    assert obj.value == 10
    assert obj.lower == 5
    assert obj.upper == 15
    assert obj.calculated == 15
    assert obj == obj2
    assert obj is not obj2
    assert obj != (10, 5, 15)
    assert obj.__args__ == (10, 5, 15)
    # assert obj.args == (10, 5, 15)
    # assert obj.argnames == ("value", "lower", "upper")

    # immutable
    with pytest.raises(AttributeError):
        obj.value = 11

    # hashable
    assert {obj: 1}.get(obj) == 1

    # weakrefable
    ref = weakref.ref(obj)
    assert ref() == obj

    # serializable
    assert pickle.loads(pickle.dumps(obj)) == obj


def test_init_subclass_keyword_arguments():
    class Test(Annotable):
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__()
            cls.kwargs = kwargs

    class Test2(Test, something="value", value="something"):
        pass

    assert Test2.kwargs == {"something": "value", "value": "something"}


def test_argument_order_using_optional_annotations():
    class Case1(Annotable):
        results: Optional[tuple[int, ...]] = ()
        default: Optional[int] = None

    class SimpleCase1(Case1):
        base: int
        cases: Optional[tuple[int, ...]] = ()

    class Case2(Annotable):
        results = optional(TupleOf(is_int), default=())
        default = optional(is_int)

    class SimpleCase2(Case2):
        base = argument(is_int)
        cases = optional(TupleOf(is_int), default=())

    assert (
        SimpleCase1.__argnames__
        == SimpleCase2.__argnames__
        == ("base", "cases", "results", "default")
    )


def test_annotable_with_optional_coercible_typehint():
    class Example(Annotable):
        value: Optional[MyInt] = None

    assert Example().value is None
    assert Example(None).value is None
    assert Example(1).value == 1
    assert isinstance(Example(1).value, MyInt)


# def test_error_message(snapshot):
#     class Example(Annotable):
#         a: int
#         b: int = 0
#         c: str = "foo"
#         d: Optional[float] = None
#         e: tuple[int, ...] = (1, 2, 3)
#         f: As[int] = 1

#     with pytest.raises(ValidationError) as exc_info:
#         Example("1", "2", "3", "4", "5", [])

#     # assert "Failed" in str(exc_info.value)

#     if sys.version_info >= (3, 11):
#         target = "error_message_py311.txt"
#     else:
#         target = "error_message.txt"
#     snapshot.assert_match(str(exc_info.value), target)


def test_abstract_meta():
    class Foo(metaclass=AbstractMeta):
        @abstractmethod
        def foo(self): ...

        @property
        @abstractmethod
        def bar(self): ...

    assert not issubclass(type(Foo), ABCMeta)
    assert issubclass(type(Foo), AbstractMeta)
    assert Foo.__abstractmethods__ == frozenset({"foo", "bar"})

    with pytest.raises(TypeError, match="Can't instantiate abstract class .*Foo.*"):
        Foo()

    class Bar(Foo):
        def foo(self):
            return 1

        @property
        def bar(self):
            return 2

    bar = Bar()
    assert bar.foo() == 1
    assert bar.bar == 2
    assert isinstance(bar, Foo)
    assert Bar.__abstractmethods__ == frozenset()


def test_annotable_supports_abstractmethods():
    class Foo(Annotable):
        @abstractmethod
        def foo(self): ...

        @property
        @abstractmethod
        def bar(self): ...

    assert not issubclass(type(Foo), ABCMeta)
    assert issubclass(type(Foo), AnnotableMeta)
    assert Foo.__abstractmethods__ == frozenset({"foo", "bar"})

    with pytest.raises(TypeError, match="Can't instantiate abstract class .*Foo.*"):
        Foo()

    class Bar(Foo):
        def foo(self):
            return 1

        @property
        def bar(self):
            return 2

    bar = Bar()
    assert bar.foo() == 1
    assert bar.bar == 2
    assert isinstance(bar, Foo)
    assert isinstance(bar, Annotable)
    assert Bar.__abstractmethods__ == frozenset()


def test_annotable_recalculates_inherited_abstractmethods():
    class Abstract(Annotable):
        @abstractmethod
        def foo(self): ...

        @property
        @abstractmethod
        def bar(self): ...

    class Mixin:
        def foo(self):
            return 1

        def bar(self):
            return 2

    class Foo(Mixin, Abstract):
        pass

    assert Abstract.__abstractmethods__ == frozenset({"foo", "bar"})
    assert Foo.__abstractmethods__ == frozenset()


# TODO(kszucs): test __new__ as well
def test_annotable_with_custom_init():
    called_with = None

    class MyInit(Annotable):
        a = argument(int)
        b = argument(IsType(str))
        c = optional(float, default=0.0)

        def __init__(self, a, b, c):
            nonlocal called_with
            called_with = (a, b, c)
            super().__init__(a=a, b=b, c=c)

        @attribute
        def called_with(self):
            return (self.a, self.b, self.c)

    assert MyInit.__spec__.initable is True
    with pytest.raises(MatchError):
        MyInit(1, 2, 3)

    mi = MyInit(1, "2", 3.3)
    assert called_with == (1, "2", 3.3)
    assert isinstance(mi, MyInit)
    assert mi.a == 1
    assert mi.b == "2"
    assert mi.c == 3.3
    assert mi.called_with == called_with


def test_any():
    class MyClass(Annotable):
        foo: Any

    assert MyClass(1).foo == 1


class Bar(Annotable):
    x: int


def test_nested():
    class Foo(Annotable):
        bar: Bar

    assert Foo(Bar(1)).bar.x == 1

    class Quux(Annotable):
        bar: Bar = Bar(2)

    assert Quux().bar.x == 2
    assert Quux(Bar(3)).bar.x == 3


def test_annotable_spec_flags():
    assert Annotable.__spec__.initable is False
    assert Annotable.__spec__.immutable is False
    assert Annotable.__spec__.hashable is False


def test_annotable_spec_flag_inheritance():
    class A(Annotable):
        pass

    class B(Annotable, immutable=True):
        pass

    assert A.__spec__.initable is False
    assert A.__spec__.immutable is False
    assert A.__spec__.hashable is False
    assert B.__spec__.initable is False
    assert B.__spec__.immutable is True
    assert B.__spec__.hashable is False

    class C(A, B):
        pass

    assert C.__spec__.initable is False
    assert C.__spec__.immutable is True
    assert C.__spec__.hashable is False

    class D(B, A, hashable=True):
        pass

    assert D.__spec__.initable is False
    assert D.__spec__.immutable is True
    assert D.__spec__.hashable is True

    with pytest.raises(TypeError):

        class E(A, B, immutable=False):
            pass

    with pytest.raises(TypeError):

        class F(D, hashable=False):
            pass

    with pytest.raises(TypeError):

        class G(D, immutable=False):
            pass


def test_user_model():
    class User(Annotable):
        id: int
        name: str = "Jane Doe"
        age: int | None = None
        children: list[str] = []

    assert User.__spec__.initable is False
    assert User.__spec__.immutable is False
    assert User.__spec__.hashable is False
