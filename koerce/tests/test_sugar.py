from __future__ import annotations

from typing import Annotated, Union

import pytest

from koerce.patterns import InstanceOf, NoMatchError, pattern
from koerce.sugar import NoMatch, ValidationError, annotated, match, var
from koerce.utils import Signature


def test_capture_shorthand():
    a = var("a")
    b = var("b")

    ctx = {}
    assert match((~a, ~b), (1, 2), ctx) == (1, 2)
    assert ctx == {"a": 1, "b": 2}

    ctx = {}
    assert match((~a, a, a), (1, 2, 3), ctx) is NoMatch
    assert ctx == {"a": 1}

    ctx = {}
    assert match((~a, a, a), (1, 1, 1), ctx) == (1, 1, 1)
    assert ctx == {"a": 1}

    ctx = {}
    assert match((~a, a, a), (1, 1, 2), ctx) is NoMatch
    assert ctx == {"a": 1}


def test_namespace():
    pass


def test_signature_unbind_from_callable():
    def test(a: int, b: int, c: int = 1): ...

    sig = Signature.from_callable(test)
    bound = sig.bind(2, 3)
    bound.apply_defaults()

    assert bound.arguments == {"a": 2, "b": 3, "c": 1}

    args, kwargs = sig.unbind(bound.arguments)
    assert args == (2, 3, 1)
    assert kwargs == {}


def test_signature_unbind_from_callable_with_varargs():
    def test(a: int, b: int, *args: int): ...

    sig = Signature.from_callable(test)
    bound = sig.bind(2, 3)
    bound.apply_defaults()

    assert bound.arguments == {"a": 2, "b": 3, "args": ()}
    args, kwargs = sig.unbind(bound.arguments)
    assert args == (2, 3)
    assert kwargs == {}

    bound = sig.bind(2, 3, 4, 5)
    bound.apply_defaults()
    assert bound.arguments == {"a": 2, "b": 3, "args": (4, 5)}
    args, kwargs = sig.unbind(bound.arguments)
    assert args == (2, 3, 4, 5)
    assert kwargs == {}


def test_signature_unbind_from_callable_with_positional_only_arguments():
    def test(a: int, b: int, /, c: int = 1): ...

    sig = Signature.from_callable(test)
    bound = sig.bind(2, 3)
    bound.apply_defaults()
    assert bound.arguments == {"a": 2, "b": 3, "c": 1}

    args, kwargs = sig.unbind(bound.arguments)
    assert args == (2, 3, 1)
    assert kwargs == {}

    bound = sig.bind(2, 3, 4)
    bound.apply_defaults()
    assert bound.arguments == {"a": 2, "b": 3, "c": 4}

    args, kwargs = sig.unbind(bound.arguments)
    assert args == (2, 3, 4)
    assert kwargs == {}


def test_signature_unbind_from_callable_with_keyword_only_arguments():
    def test(a: int, b: int, *, c: float, d: float = 0.0): ...

    sig = Signature.from_callable(test)
    bound = sig.bind(2, 3, c=4.0)
    bound.apply_defaults()
    assert bound.arguments == {"a": 2, "b": 3, "c": 4.0, "d": 0.0}

    args, kwargs = sig.unbind(bound.arguments)
    assert args == (2, 3)
    assert kwargs == {"c": 4.0, "d": 0.0}


def test_signature_unbind():
    def func(a, b, c=1): ...

    sig = Signature.from_callable(func)
    bound = sig.bind(1, 2)
    bound.apply_defaults()

    assert bound.arguments == {"a": 1, "b": 2, "c": 1}

    args, kwargs = sig.unbind(bound.arguments)
    assert args == (1, 2, 1)
    assert kwargs == {}


@pytest.mark.parametrize("d", [(), (5, 6, 7)])
def test_signature_unbind_with_empty_variadic(d):
    def func(a, b, c, *args, e=None):
        return a, b, c, args, e

    sig = Signature.from_callable(func)
    bound = sig.bind(1, 2, 3, *d, e=4)
    bound.apply_defaults()
    assert bound.arguments == {"a": 1, "b": 2, "c": 3, "args": d, "e": 4}

    args, kwargs = sig.unbind(bound.arguments)
    assert args == (1, 2, 3, *d)
    assert kwargs == {"e": 4}


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
    @annotated([InstanceOf(int), InstanceOf(int), InstanceOf(float)])
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
    assert test.__signature__.parameters.keys() == {"a", "b", "c"}


# def test_annotated_function_without_decoration(snapshot):
#     def test(a, b, c):
#         return a + b + c

#     func = annotated(test)
#     with pytest.raises(ValidationError) as excinfo:
#         func(1, 2)
#     snapshot.assert_match(str(excinfo.value), "error.txt")

#     assert func(1, 2, c=3) == 6


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
