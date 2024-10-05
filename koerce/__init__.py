from __future__ import annotations

import sys

from ._internal import *  # noqa: F403


class _Variable(Deferred):  #noqa: F405
    def __init__(self, name: str):
        builder = Var(name)  # noqa: F405
        super().__init__(builder)

    def __pos__(self):
        return Capture(self)  # noqa: F405

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
    p = _Namespace(pattern, module)  # noqa: F405
    d = _Namespace(deferred, module)  # noqa: F405
    return p, d


def replace(matcher):
    """More convenient syntax for replacing a value with the output of a function."""

    def decorator(replacer):
        return Replace(matcher, replacer)  # noqa: F405

    return decorator


class NoMatch:
    __slots__ = ()

    def __init__(self):
        raise ValueError("Cannot instantiate NoMatch")


def koerce(
    pat: Pattern, value: Any, context: Context = None, allow_coercion: bool = False  # noqa: F405
) -> Any:  # noqa: F405
    """Match a value against a pattern.

    Parameters
    ----------
    pat
        The pattern to match against.
    value
        The value to match.
    context
        Arbitrary mapping of values to be used while matching.
    allow_coercion
        Whether to allow coercion of values to match the pattern.

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
    pat = pattern(pat, allow_coercion=allow_coercion) # noqa: F405
    try:
        return pat.apply(value, context)
    except MatchError: # noqa: F405
        return NoMatch


_ = var("_")

match = koerce

# define __all__
