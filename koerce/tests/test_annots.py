from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Annotated,
    Optional,
    Union,
)

import pytest

from koerce.annots import (
    EMPTY,
    Parameter,
    Signature,
    ValidationError,
    annotated,
)
from koerce.patterns import (
    Anything,
    InstanceOf,
    MappingOf,
    NoMatchError,
    Option,
    PatternMap,
    TupleOf,
    pattern,
)


def test_parameter():
    p = Parameter("x", Parameter.POSITIONAL_OR_KEYWORD, typehint=int)
    assert p.name == "x"
    assert p.kind is Parameter.POSITIONAL_OR_KEYWORD
    assert str(p) == "x: int"

    p = Parameter("x", Parameter.POSITIONAL_OR_KEYWORD, default=1)
    assert p.name == "x"
    assert p.kind is Parameter.POSITIONAL_OR_KEYWORD
    assert p.default_ == 1
    assert str(p) == "x=1"

    p = Parameter("x", Parameter.POSITIONAL_OR_KEYWORD, typehint=int, default=1)
    assert p.name == "x"
    assert p.kind is Parameter.POSITIONAL_OR_KEYWORD
    assert p.default_ == 1
    assert p.typehint is int
    assert str(p) == "x: int = 1"

    p = Parameter("y", Parameter.VAR_POSITIONAL, typehint=int)
    assert p.name == "y"
    assert p.kind is Parameter.VAR_POSITIONAL
    assert p.typehint is int
    assert str(p) == "*y: int"

    p = Parameter("z", Parameter.VAR_KEYWORD, typehint=int)
    assert p.name == "z"
    assert p.kind is Parameter.VAR_KEYWORD
    assert p.typehint is int
    assert str(p) == "**z: int"


def test_signature_contruction():
    a = Parameter("a", Parameter.POSITIONAL_OR_KEYWORD, typehint=int)
    b = Parameter("b", Parameter.POSITIONAL_OR_KEYWORD, typehint=str)
    c = Parameter("c", Parameter.POSITIONAL_OR_KEYWORD, typehint=int, default=1)
    d = Parameter("d", Parameter.VAR_POSITIONAL, typehint=int)

    sig = Signature([a, b, c, d])
    assert sig.parameters == [a, b, c, d]
    assert sig.return_typehint is EMPTY


def test_signature_from_callable():
    def func(a: int, b: str, *args, c=1, **kwargs) -> float: ...

    sig = Signature.from_callable(func)
    assert sig.parameters == [
        Parameter("a", Parameter.POSITIONAL_OR_KEYWORD, typehint=int),
        Parameter("b", Parameter.POSITIONAL_OR_KEYWORD, typehint=str),
        Parameter("args", Parameter.VAR_POSITIONAL),
        Parameter("c", Parameter.KEYWORD_ONLY, default=1),
        Parameter("kwargs", Parameter.VAR_KEYWORD),
    ]
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

    sig12 = Signature.merge(sig1, sig2)
    assert sig12.parameters == [
        Parameter("a", Parameter.POSITIONAL_OR_KEYWORD),
        Parameter("b", Parameter.POSITIONAL_OR_KEYWORD),
        Parameter("d", Parameter.POSITIONAL_OR_KEYWORD),
        Parameter("e", Parameter.POSITIONAL_OR_KEYWORD),
        Parameter("c", Parameter.POSITIONAL_OR_KEYWORD, default=1),
        Parameter("f", Parameter.POSITIONAL_OR_KEYWORD, default=2),
    ]

    sig13 = Signature.merge(sig1, sig3)
    assert sig13.parameters == [
        Parameter("b", Parameter.POSITIONAL_OR_KEYWORD),
        Parameter("d", Parameter.POSITIONAL_OR_KEYWORD),
        Parameter("a", Parameter.POSITIONAL_OR_KEYWORD, default=1),
        Parameter("c", Parameter.POSITIONAL_OR_KEYWORD, default=1),
        Parameter("kwargs", Parameter.VAR_KEYWORD),
    ]


def test_annotated_function():
    @annotated(a=InstanceOf(int), b=InstanceOf(int), c=InstanceOf(int))
    def test(a, b, c=1):
        return a + b + c

    assert test(2, 3) == 6
    assert test(2, 3, 4) == 9
    assert test(2, 3, c=4) == 9
    assert test(a=2, b=3, c=4) == 9

    with pytest.raises(ValidationError):
        test(2, 3, c="4")

    @annotated(a=InstanceOf(int))
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
    with pytest.raises(ValidationError):
        test_wrong(2, 3)


def test_annotated_function_with_keyword_overrides():
    @annotated(b=InstanceOf(float))
    def test(a: int, b: int, c: int = 1):
        return a + b + c

    with pytest.raises(ValidationError):
        test(2, 3)

    assert test(2, 3.0) == 6.0


def test_annotated_function_with_list_overrides():
    @annotated([InstanceOf(int), InstanceOf(int), InstanceOf(str)])
    def test(a: int, b: int, c: int = 1):
        return a + b + c

    with pytest.raises(ValidationError):
        test(2, 3, 4)


def test_annotated_function_with_list_overrides_and_return_override():
    @annotated([InstanceOf(int), InstanceOf(int), InstanceOf(float)], InstanceOf(float))
    def test(a: int, b: int, c: int = 1):
        return a + b + c

    with pytest.raises(ValidationError):
        test(2, 3, 4)

    assert test(2, 3, 4.0) == 9.0


@pattern
def short_str(x, **context):
    if len(x) > 3:
        return x
    else:
        raise NoMatchError()


@pattern
def endswith_d(x, **context):
    if x.endswith("d"):
        return x
    else:
        raise NoMatchError()


def test_annotated_function_with_complex_type_annotations():
    @annotated
    def test(a: Annotated[str, short_str, endswith_d], b: Union[int, float]):
        return a, b

    assert test("abcd", 1) == ("abcd", 1)
    assert test("---d", 1.0) == ("---d", 1.0)

    with pytest.raises(ValidationError):
        test("---c", 1)
    with pytest.raises(ValidationError):
        test("123", 1)
    with pytest.raises(ValidationError):
        test("abcd", "qweqwe")


def test_annotated_function_without_annotations():
    @annotated
    def test(a, b, c):
        return a, b, c

    assert test(1, 2, 3) == (1, 2, 3)
    assert [p.name for p in test.__signature__.parameters] == ["a", "b", "c"]


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

    with pytest.raises(ValidationError):
        test(1.0, 2.0, 3, 4, 5, 6.0)


def test_annotated_function_with_varkwargs():
    @annotated
    def test(a: float, b: float, **kwargs: int):
        return sum((a, b) + tuple(kwargs.values()))

    assert test(1.0, 2.0, c=3, d=4) == 10.0
    assert test(1.0, 2.0, c=3, d=4, e=5) == 15.0

    with pytest.raises(ValidationError):
        test(1.0, 2.0, c=3, d=4, e=5, f=6.0)


# def test_multiple_validation_failures():
#     @annotated
#     def test(a: float, b: float, *args: int, **kwargs: int): ...

#     with pytest.raises(ValidationError) as excinfo:
#         test(1.0, 2.0, 3.0, 4, c=5.0, d=6)

#     assert len(excinfo.value.errors) == 2


def test_signature_to_pattern():
    def func(a: int, b: str) -> str: ...

    args, ret = Signature.from_callable(func).to_pattern()
    assert args == PatternMap({"a": InstanceOf(int), "b": InstanceOf(str)})
    assert ret == InstanceOf(str)

    def func(a: int, b: str, c: str = "0") -> str: ...

    args, ret = Signature.from_callable(func).to_pattern()
    assert args == PatternMap(
        {"a": InstanceOf(int), "b": InstanceOf(str), "c": Option(InstanceOf(str), "0")}
    )
    assert ret == InstanceOf(str)

    def func(a: int, b: str, *args): ...

    args, ret = Signature.from_callable(func).to_pattern()
    assert args == PatternMap(
        {"a": InstanceOf(int), "b": InstanceOf(str), "args": TupleOf(Anything())}
    )
    assert ret == Anything()

    def func(a: int, b: str, c: str = "0", *args, **kwargs: int) -> float: ...

    args, ret = Signature.from_callable(func).to_pattern()
    assert args == PatternMap(
        {
            "a": InstanceOf(int),
            "b": InstanceOf(str),
            "c": Option(InstanceOf(str), "0"),
            "args": TupleOf(Anything()),
            "kwargs": MappingOf(Anything(), InstanceOf(int)),
        }
    )
    assert ret == InstanceOf(float)


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

    with pytest.raises(ValidationError):
        A(1, "2", "d")


def test_annotated_with_dataclass():
    @annotated
    @dataclass
    class InventoryItem:
        name: str
        unit_price: float
        quantity_on_hand: int = 0

    item = InventoryItem("widget", 3.0, 10)
    assert item.name == "widget"
    assert item.unit_price == 3.0
    assert item.quantity_on_hand == 10

    item = InventoryItem("widget", 3.0)
    assert item.name == "widget"
    assert item.unit_price == 3.0
    assert item.quantity_on_hand == 0

    with pytest.raises(ValidationError):
        InventoryItem("widget", "3.0", 10)

    with pytest.raises(ValidationError):
        InventoryItem("widget", 3.0, "10")
