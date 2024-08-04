from __future__ import annotations

import functools
import sys
from typing import Any

from .builders import Deferred, Variable
from .patterns import (
    Capture,
    Context,
    Eq,
    If,
    NoMatch,
    Pattern,
    pattern,
)
from .utils import Signature


class ValidationError(Exception):
    pass


class Namespace:
    """Convenience class for creating patterns for various types from a module.

    Useful to reduce boilerplate when creating patterns for various types from
    a module.

    Parameters
    ----------
    factory
        The pattern to construct with the looked up types.
    module
        The module object or name to look up the types.

    """

    __slots__ = ("_factory", "_module")
    # _factory: Callable
    # _module: ModuleType

    def __init__(self, factory, module):
        if isinstance(module, str):
            module = sys.modules[module]
        self._module = module
        self._factory = factory

    def __getattr__(self, name: str):
        obj = getattr(self._module, name)
        return self._factory(obj)


class Var(Deferred):
    def __init__(self, name: str):
        builder = Variable(name)
        super().__init__(builder)

    def __invert__(self):
        return Capture(self)


var = Var


def match(pat: Pattern, value: Any, context: Context = None) -> Any:
    """Match a value against a pattern.

    Parameters
    ----------
    pat
        The pattern to match against.
    value
        The value to match.
    context
        Arbitrary mapping of values to be used while matching.

    Returns
    -------
    The matched value if the pattern matches, otherwise :obj:`NoMatch`.

    Examples
    --------
    >>> assert match(Any(), 1) == 1
    >>> assert match(1, 1) == 1
    >>> assert match(1, 2) is NoMatch
    >>> assert match(1, 1, context={"x": 1}) == 1
    >>> assert match(1, 2, context={"x": 1}) is NoMatch
    >>> assert match([1, int], [1, 2]) == [1, 2]
    >>> assert match([1, int, "a" @ InstanceOf(str)], [1, 2, "three"]) == [
    ...     1,
    ...     2,
    ...     "three",
    ... ]

    """
    pat = pattern(pat)
    return pat.apply(value, context)


def annotated(_1=None, _2=None, _3=None, **kwargs):
    """Create functions with arguments validated at runtime.

    There are various ways to apply this decorator:

    1. With type annotations

    >>> @annotated
    ... def foo(x: int, y: str) -> float:
    ...     return float(x) + float(y)

    2. With argument patterns passed as keyword arguments

    >>> from ibis.common.patterns import InstanceOf as instance_of
    >>> @annotated(x=instance_of(int), y=instance_of(str))
    ... def foo(x, y):
    ...     return float(x) + float(y)

    3. With mixing type annotations and patterns where the latter takes precedence

    >>> @annotated(x=instance_of(float))
    ... def foo(x: int, y: str) -> float:
    ...     return float(x) + float(y)

    4. With argument patterns passed as a list and/or an optional return pattern

    >>> @annotated([instance_of(int), instance_of(str)], instance_of(float))
    ... def foo(x, y):
    ...     return float(x) + float(y)

    Parameters
    ----------
    *args : Union[
                tuple[Callable],
                tuple[list[Pattern], Callable],
                tuple[list[Pattern], Pattern, Callable]
            ]
        Positional arguments.
        - If a single callable is passed, it's wrapped with the signature
        - If two arguments are passed, the first one is a list of patterns for the
          arguments and the second one is the callable to wrap
        - If three arguments are passed, the first one is a list of patterns for the
          arguments, the second one is a pattern for the return value and the third
          one is the callable to wrap
    **kwargs : dict[str, Pattern]
        Patterns for the arguments.

    Returns
    -------
    Callable

    """
    if _1 is None:
        return functools.partial(annotated, **kwargs)
    elif _2 is None:
        if callable(_1):
            func, patterns, return_pattern = _1, None, None
        else:
            return functools.partial(annotated, _1, **kwargs)
    elif _3 is None:
        if not isinstance(_2, Pattern):
            func, patterns, return_pattern = _2, _1, None
        else:
            return functools.partial(annotated, _1, _2, **kwargs)
    else:
        func, patterns, return_pattern = _3, _1, _2

    sig = Signature.from_callable(func)
    argpats, retpat = Pattern.from_callable(
        func, sig=sig, args=patterns or kwargs, return_=return_pattern
    )

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # 0. Bind the arguments to the signature
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # 1. Validate the passed arguments
        values = argpats.apply(bound.arguments)
        if values is NoMatch:
            raise ValidationError()

        # 2. Reconstruction of the original arguments
        args, kwargs = sig.unbind(values)

        # 3. Call the function with the validated arguments
        result = func(*args, **kwargs)

        # 4. Validate the return value
        result = retpat.apply(result)
        if result is NoMatch:
            raise ValidationError()

        return result

    wrapped.__signature__ = sig

    return wrapped


if_ = If
eq = Eq
_ = var("_")
