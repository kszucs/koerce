from __future__ import annotations

from typing import Dict, Generic, List, Optional, TypeVar

import pytest

from koerce.utils import (
    EMPTY,
    Parameter,
    Signature,
    get_type_boundvars,
    get_type_hints,
    get_type_params,
)

T = TypeVar("T", covariant=True)
S = TypeVar("S", covariant=True)
U = TypeVar("U", covariant=True)


class My(Generic[T, S, U]):
    a: T
    b: S
    c: str

    @property
    def d(self) -> Optional[str]: ...

    @property
    def e(self) -> U:  # type: ignore
        ...


class MyChild(My): ...


def example(a: int, b: str) -> str:  # type: ignore
    ...


def test_get_type_hints() -> None:
    hints = get_type_hints(My)
    assert hints == {"a": T, "b": S, "c": str}

    hints = get_type_hints(My, include_properties=True)
    assert hints == {"a": T, "b": S, "c": str, "d": Optional[str], "e": U}

    hints = get_type_hints(MyChild, include_properties=True)
    assert hints == {"a": T, "b": S, "c": str, "d": Optional[str], "e": U}

    # test that we don't actually mutate the My.__annotations__
    hints = get_type_hints(My)
    assert hints == {"a": T, "b": S, "c": str}

    hints = get_type_hints(example)
    assert hints == {"a": int, "b": str, "return": str}

    hints = get_type_hints(example, include_properties=True)
    assert hints == {"a": int, "b": str, "return": str}


class A(Generic[T, S, U]):
    a: int
    b: str

    t: T
    s: S

    @property
    def u(self) -> U:  # type: ignore
        ...


class B(A[T, S, bytes]): ...


class C(B[T, str]): ...


class D(C[bool]): ...


class MyDict(Dict[T, U]): ...


class MyList(List[T]): ...


def test_get_type_params() -> None:
    # collecting all type params is done by GenericCoercedTo
    assert get_type_params(A[int, float, str]) == {"T": int, "S": float, "U": str}
    assert get_type_params(B[int, bool]) == {"T": int, "S": bool}
    assert get_type_params(C[int]) == {"T": int}
    assert get_type_params(D) == {}
    assert get_type_params(MyDict[int, str]) == {"T": int, "U": str}
    assert get_type_params(MyList[int]) == {"T": int}


def test_get_type_boundvars() -> None:
    expected = {
        "t": int,
        "s": float,
        "u": str,
    }
    assert get_type_boundvars(A[int, float, str]) == expected

    expected = {
        "t": int,
        "s": bool,
    }
    assert get_type_boundvars(B[int, bool]) == expected


def test_get_type_boundvars_unable_to_deduce() -> None:
    msg = "Unable to deduce corresponding type attributes..."
    with pytest.raises(ValueError, match=msg):
        get_type_boundvars(MyDict[int, str])


def test_parameter():
    p = Parameter("x", Parameter.POSITIONAL_OR_KEYWORD, annotation=int)
    assert p.name == "x"
    assert p.kind is Parameter.POSITIONAL_OR_KEYWORD
    assert str(p) == "x: int"

    p = Parameter("x", Parameter.POSITIONAL_OR_KEYWORD, default=1)
    assert p.name == "x"
    assert p.kind is Parameter.POSITIONAL_OR_KEYWORD
    assert p.default_ == 1
    assert str(p) == "x=1"

    p = Parameter("x", Parameter.POSITIONAL_OR_KEYWORD, annotation=int, default=1)
    assert p.name == "x"
    assert p.kind is Parameter.POSITIONAL_OR_KEYWORD
    assert p.default_ == 1
    assert p.annotation is int
    assert str(p) == "x: int = 1"

    p = Parameter("y", Parameter.VAR_POSITIONAL, annotation=int)
    assert p.name == "y"
    assert p.kind is Parameter.VAR_POSITIONAL
    assert p.annotation is int
    assert str(p) == "*y: int"

    p = Parameter("z", Parameter.VAR_KEYWORD, annotation=int)
    assert p.name == "z"
    assert p.kind is Parameter.VAR_KEYWORD
    assert p.annotation is int
    assert str(p) == "**z: int"


def test_signature_contruction():
    a = Parameter("a", Parameter.POSITIONAL_OR_KEYWORD, annotation=int)
    b = Parameter("b", Parameter.POSITIONAL_OR_KEYWORD, annotation=str)
    c = Parameter("c", Parameter.POSITIONAL_OR_KEYWORD, annotation=int, default=1)
    d = Parameter("d", Parameter.VAR_POSITIONAL, annotation=int)

    sig = Signature([a, b, c, d])
    assert sig.parameters == [a, b, c, d]
    assert sig.return_annotation is EMPTY


def test_signature_from_callable():
    def func(a: int, b: str, *args, c=1, **kwargs) -> float: ...

    sig = Signature.from_callable(func)
    assert sig.parameters == [
        Parameter("a", Parameter.POSITIONAL_OR_KEYWORD, annotation="int"),
        Parameter("b", Parameter.POSITIONAL_OR_KEYWORD, annotation="str"),
        Parameter("args", Parameter.VAR_POSITIONAL),
        Parameter("c", Parameter.KEYWORD_ONLY, default=1),
        Parameter("kwargs", Parameter.VAR_KEYWORD),
    ]
    assert sig.return_annotation == "float"


def test_signature_bind_various():
    # with positional or keyword default
    def func(a: int, b: str, c=1) -> float: ...

    sig = Signature.from_callable(func)
    bound = sig.bind(1, "2")
    assert bound == {"a": 1, "b": "2", "c": 1}

    # with variable positional arguments
    def func(a: int, b: str, *args: int, c=1) -> float: ...

    sig = Signature.from_callable(func)
    bound = sig.bind(1, "2", 3, 4)
    assert bound == {"a": 1, "b": "2", "args": (3, 4), "c": 1}

    # with both variadic positional and variadic keyword arguments
    def func(a: int, b: str, *args: int, c=1, **kwargs: int) -> float: ...

    sig = Signature.from_callable(func)
    bound = sig.bind(1, "2", 3, 4, x=5, y=6)
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
    bound = sig.bind(1, "2")
    assert bound == {"a": 1, "b": "2", "c": 1}

    with pytest.raises(TypeError, match="passed as keyword argument"):
        sig.bind(a=1, b="2", c=3)

    # with keyword only arguments
    def func(a: int, b: str, *, c=1) -> float: ...

    sig = Signature.from_callable(func)
    bound = sig.bind(1, "2", c=3)
    assert bound == {"a": 1, "b": "2", "c": 3}

    with pytest.raises(TypeError, match="too many positional arguments"):
        sig.bind(1, "2", 3)

    def func(a, *args, b, z=100, **kwargs): ...

    sig = Signature.from_callable(func)
    bound = sig.bind(10, 20, b=30, c=40, args=50, kwargs=60)
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
    bound = sig.bind(*args, **kwargs)
    ubargs, ubkwargs = sig.unbind(bound)
    return func(*ubargs, **ubkwargs)


def test_signature_bind_no_arguments():
    def func(): ...

    sig = Signature.from_callable(func)
    assert sig.bind() == {}

    with pytest.raises(TypeError, match="too many positional arguments"):
        sig.bind(1)
    with pytest.raises(TypeError, match="too many positional arguments"):
        sig.bind(1, keyword=2)
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'keyword'"):
        sig.bind(keyword=1)


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
    ba = sig.bind(args=1)
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
    ba = sig.bind(1, 2, 3)
    args, _ = sig.unbind(ba)
    assert args == (1, 2, 3)
    ba = sig.bind(1, self=2, b=3)
    args, _ = sig.unbind(ba)
    assert args == (1, 2, 3)
