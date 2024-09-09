from __future__ import annotations

import sys

from ._internal import *


class _Variable(Deferred):
    def __init__(self, name: str):
        builder = Var(name)
        super().__init__(builder)

    def __pos__(self):
        return Capture(self)

    def __neg__(self):
        return self


class _Namespace:
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

    def __init__(self, factory, module):
        if isinstance(module, str):
            module = sys.modules[module]
        self._module = module
        self._factory = factory

    def __getattr__(self, name: str):
        obj = getattr(self._module, name)
        return self._factory(obj)


def var(name):
    return _Variable(name)


def namespace(module):
    p = _Namespace(pattern, module)
    d = _Namespace(deferred, module)
    return p, d


def replace(matcher):
    """More convenient syntax for replacing a value with the output of a function."""

    def decorator(replacer):
        return Replace(matcher, replacer)

    return decorator


class NoMatch:
    __slots__ = ()

    def __init__(self):
        raise ValueError("Cannot instantiate NoMatch")


def koerce(
    pat: Pattern, value: Any, context: Context = None, allow_coercion: bool = False
) -> Any:
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
    >>> assert koerce(Any(), 1) == 1
    >>> assert koerce(1, 1) == 1
    >>> assert koerce(1, 2) is NoMatch
    >>> assert koerce(1, 1, context={"x": 1}) == 1
    >>> assert koerce(1, 2, context={"x": 1}) is NoMatch
    >>> assert koerce([1, int], [1, 2]) == [1, 2]
    >>> assert koerce([1, int, "a" @ InstanceOf(str)], [1, 2, "three"]) == [
    ...     1,
    ...     2,
    ...     "three",
    ... ]

    """
    pat = pattern(pat, allow_coercion=allow_coercion)
    try:
        return pat.apply(value, context)
    except MatchError:
        return NoMatch


_ = var("_")

match = koerce

# define __all__
