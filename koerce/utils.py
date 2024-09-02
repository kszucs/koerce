from __future__ import annotations

import itertools
import sys
import typing
from collections.abc import Hashable
from typing import Any, Optional, TypeVar

from typing_extensions import Self

_SpecialForm = type(Self)

K = TypeVar("K")
V = TypeVar("V")

get_type_args = typing.get_args
get_type_origin = typing.get_origin


class FakeType:
    def __init__(self, module: str, annotations: dict[str, Any]):
        self.__module__ = module
        self.__annotations__ = annotations


def get_type_hints(
    obj: Any,
    globalns=None,
    localns=None,
    module: Optional[str] = None,
    include_extras: bool = True,
    include_properties: bool = False,
) -> dict[str, Any]:
    """Get type hints for a callable or class.

    Extension of typing.get_type_hints that supports getting type hints for
    class properties.

    Parameters
    ----------
    obj
        Callable or class to get type hints for.
    globalns
        Global namespace to use for resolving type variables.
    localns
        Local namespace to use for resolving type variables.
    module
        Module name to use for resolving globalns if not provided.
    include_extras
        Whether to include extra type hints such as Annotated.
    include_properties
        Whether to include type hints for class properties.

    Returns
    -------
    Mapping of parameter or attribute name to type hint.
    """
    if isinstance(obj, dict):
        assert module is not None
        obj = FakeType(module=module, annotations=obj)
        mod = sys.modules.get(module, None)
        globalns = getattr(mod, "__dict__", None)

    try:
        hints = typing.get_type_hints(
            obj, globalns=globalns, localns=localns, include_extras=include_extras
        )
    except TypeError:
        return {}

    if include_properties:
        for name in dir(obj):
            attr = getattr(obj, name)
            if isinstance(attr, property):
                annots = typing.get_type_hints(
                    attr.fget,
                    globalns=globalns,
                    localns=localns,
                    include_extras=include_extras,
                )
                if return_hint := annots.get("return"):
                    hints[name] = return_hint

    return hints


# TODO(kszucs): memoize this function
def get_type_params(typ: Any) -> dict[TypeVar, type]:
    """Get type parameters for a generic class.

    Parameters
    ----------
    typ
        Generic class to get type parameters for.

    Returns
    -------
    Mapping of type parameter name to type.

    Examples
    --------
    >>> from typing import Dict, List
    >>> class MyList(List[T]): ...
    >>> get_type_params(MyList[int])
    {'T': <class 'int'>}
    >>> class MyDict(Dict[T, U]): ...
    >>> get_type_params(MyDict[int, str])
    {'T': <class 'int'>, 'U': <class 'str'>}
    """
    args: tuple = get_type_args(typ)
    origin: Any = get_type_origin(typ)  # or typ
    params: tuple = getattr(origin, "__parameters__", ())

    result: dict[str, type] = {}
    for param, arg in zip(params, args):
        result[param.__name__] = arg

    return result


def get_type_boundvars(typ: Any) -> dict[TypeVar, tuple[str, type]]:
    """Get the attribute names and concrete types of a generic class.

    Parameters
    ----------
    typ
        Generic class to get type variables for.
    only_covariant
        Whether to only include covariant type variables.

    Returns
    -------
    Mapping of type variable to attribute name and type.

    Examples
    --------
    >>> from typing import Generic
    >>> class MyStruct(Generic[T, U]):
    ...     a: T
    ...     b: U
    >>> get_bound_typevars(MyStruct[int, str])
    {~T: ('a', <class 'int'>), ~U: ('b', <class 'str'>)}
    >>>
    >>> class MyStruct(Generic[T, U]):
    ...     a: T
    ...
    ...     @property
    ...     def myprop(self) -> U: ...
    >>> get_bound_typevars(MyStruct[float, bytes])
    {~T: ('a', <class 'float'>), ~U: ('myprop', <class 'bytes'>)}
    """
    origin = get_type_origin(typ) or typ
    hints = get_type_hints(origin, include_properties=True)
    params = get_type_params(typ)

    result = {}
    for attr, hint in hints.items():
        if isinstance(hint, TypeVar) and hint.__name__ in params:
            if not hint.__covariant__:
                raise TypeError(
                    f"Typevar {hint} is not covariant and currently only "
                    "covariant typevars are supported"
                )
            result[attr] = params.pop(hint.__name__)

    if params:
        raise ValueError(
            f"Unable to deduce corresponding attributes for type parameters of {typ}.\n"
            f"Missing attributes with typehints for the following type variables: {params}.\n"
            f"Available type hints: {hints}"
        )

    return result


_hint_types = (type, TypeVar, _SpecialForm)


def is_typehint(obj: Any) -> bool:
    return isinstance(obj, _hint_types) or get_type_origin(obj)


class FrozenDict(dict[K, V], Hashable):
    __slots__ = ("__precomputed_hash__",)
    __precomputed_hash__: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hashable = frozenset(self.items())
        object.__setattr__(self, "__precomputed_hash__", hash(hashable))

    def __hash__(self) -> int:
        return self.__precomputed_hash__

    def __setitem__(self, key: K, value: V) -> None:
        raise TypeError(
            f"'{self.__class__.__name__}' object does not support item assignment"
        )

    def __setattr__(self, name: str, _: Any) -> None:
        raise TypeError(f"Attribute {name!r} cannot be assigned to frozendict")

    def __reduce__(self) -> tuple:
        return (self.__class__, (dict(self),))


frozendict = FrozenDict


class RewindableIterator:
    """Iterator that can be rewound to a checkpoint.

    Examples
    --------
    >>> it = RewindableIterator(range(5))
    >>> next(it)
    0
    >>> next(it)
    1
    >>> it.checkpoint()
    >>> next(it)
    2
    >>> next(it)
    3
    >>> it.rewind()
    >>> next(it)
    2
    >>> next(it)
    3
    >>> next(it)
    4

    """

    __slots__ = ("_iterator", "_checkpoint")

    def __init__(self, iterable):
        self._iterator = iter(iterable)
        self._checkpoint = None

    def __next__(self):
        return next(self._iterator)

    def rewind(self):
        """Rewind the iterator to the last checkpoint."""
        if self._checkpoint is None:
            raise ValueError("No checkpoint to rewind to.")
        self._iterator, self._checkpoint = itertools.tee(self._checkpoint)

    def checkpoint(self):
        """Create a checkpoint of the current iterator state."""
        self._iterator, self._checkpoint = itertools.tee(self._iterator)
