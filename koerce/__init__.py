from __future__ import annotations

import sys

from ._internal import *


class _Variable(Deferred):
    def __init__(self, name: str):
        builder = Var(name)
        super().__init__(builder)

    def __invert__(self):
        return Capture(self)


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


def match(
    pat: Pattern, value: Any, context: Context = None, allow_coercion: bool = True
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
    pat = pattern(pat, allow_coercion=allow_coercion)
    return pat.apply(value, context)


_ = var("_")


# define __all__
