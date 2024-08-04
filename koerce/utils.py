from __future__ import annotations

import inspect
import itertools
import typing
from typing import Any, TypeVar

get_type_args = typing.get_args
get_type_origin = typing.get_origin


def get_type_hints(
    obj: Any,
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
    include_extras
        Whether to include extra type hints such as Annotated.
    include_properties
        Whether to include type hints for class properties.

    Returns
    -------
    Mapping of parameter or attribute name to type hint.
    """
    try:
        hints = typing.get_type_hints(obj, include_extras=include_extras)
    except TypeError:
        return {}

    if include_properties:
        for name in dir(obj):
            attr = getattr(obj, name)
            if isinstance(attr, property):
                annots = typing.get_type_hints(attr.fget, include_extras=include_extras)
                if return_annot := annots.get("return"):
                    hints[name] = return_annot

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
    origin: type = get_type_origin(typ)  # or typ
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
            f"Unable to deduce corresponding type attributes for the following type variables: {params}"
        )

    return result


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


class Signature(inspect.Signature):
    def unbind(self, this: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Reverse bind of the parameters.

        Attempts to reconstructs the original arguments as keyword only arguments.

        Parameters
        ----------
        this : Any
            Object with attributes matching the signature parameters.

        Returns
        -------
        args : (args, kwargs)
            Tuple of positional and keyword arguments.

        """
        # does the reverse of bind, but doesn't apply defaults
        args: list = []
        kwargs: dict = {}
        for name, param in self.parameters.items():
            value = this[name]
            if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                args.append(value)
            elif param.kind is inspect.Parameter.VAR_POSITIONAL:
                args.extend(value)
            elif param.kind is inspect.Parameter.VAR_KEYWORD:
                kwargs.update(value)
            elif param.kind is inspect.Parameter.KEYWORD_ONLY:
                kwargs[name] = value
            elif param.kind is inspect.Parameter.POSITIONAL_ONLY:
                args.append(value)
            else:
                raise TypeError(f"unsupported parameter kind {param.kind}")

        return tuple(args), kwargs
