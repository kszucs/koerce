from __future__ import annotations

import inspect
import itertools
import typing
from typing import Any, TypeVar

import cython

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


EMPTY = inspect.Parameter.empty

POSITIONAL_ONLY = cython.declare(cython.int, int(inspect.Parameter.POSITIONAL_ONLY))
POSITIONAL_OR_KEYWORD = cython.declare(
    cython.int, int(inspect.Parameter.POSITIONAL_OR_KEYWORD)
)
VAR_POSITIONAL = cython.declare(cython.int, int(inspect.Parameter.VAR_POSITIONAL))
KEYWORD_ONLY = cython.declare(cython.int, int(inspect.Parameter.KEYWORD_ONLY))
VAR_KEYWORD = cython.declare(cython.int, int(inspect.Parameter.VAR_KEYWORD))


@cython.final
@cython.cclass
class Parameter:
    POSITIONAL_ONLY: typing.ClassVar[int] = int(inspect.Parameter.POSITIONAL_ONLY)
    POSITIONAL_OR_KEYWORD: typing.ClassVar[int] = int(
        inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    VAR_POSITIONAL: typing.ClassVar[int] = int(inspect.Parameter.VAR_POSITIONAL)
    KEYWORD_ONLY: typing.ClassVar[int] = int(inspect.Parameter.KEYWORD_ONLY)
    VAR_KEYWORD: typing.ClassVar[int] = int(inspect.Parameter.VAR_KEYWORD)

    name = cython.declare(str, visibility="readonly")
    kind = cython.declare(cython.int, visibility="readonly")
    # Cannot use C reserved keyword 'default' here
    default_ = cython.declare(object, visibility="readonly")
    annotation = cython.declare(object, visibility="readonly")

    def __init__(
        self, name: str, kind: int, default: Any = EMPTY, annotation: Any = EMPTY
    ):
        self.name = name
        self.kind = kind
        self.default_ = default
        self.annotation = annotation

    def __str__(self) -> str:
        result: str = self.name
        if self.annotation is not EMPTY:
            if hasattr(self.annotation, "__qualname__"):
                result += f": {self.annotation.__qualname__}"
            else:
                result += f": {self.annotation}"
        if self.default_ is not EMPTY:
            if self.annotation is EMPTY:
                result = f"{result}={self.default_}"
            else:
                result = f"{result} = {self.default_!r}"
        if self.kind == VAR_POSITIONAL:
            result = f"*{result}"
        elif self.kind == VAR_KEYWORD:
            result = f"**{result}"
        return result

    def __repr__(self):
        return f'<{self.__class__.__name__} "{self}">'

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Parameter):
            return NotImplemented
        right: Parameter = cython.cast(Parameter, other)
        return (
            self.name == right.name
            and self.kind == right.kind
            and self.default_ == right.default_
            and self.annotation == right.annotation
        )


@cython.final
@cython.cclass
class Signature:
    parameters = cython.declare(list[Parameter], visibility="readonly")
    return_annotation = cython.declare(object, visibility="readonly")

    def __init__(self, parameters: list[Parameter], return_annotation: Any = EMPTY):
        self.parameters = parameters
        self.return_annotation = return_annotation

    @staticmethod
    def from_callable(func: Any) -> Signature:
        sig = inspect.signature(func)
        params: list[Parameter] = [
            Parameter(p.name, int(p.kind), p.default, p.annotation)
            for p in sig.parameters.values()
        ]
        return Signature(params, sig.return_annotation)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Signature):
            return NotImplemented
        right: Signature = cython.cast(Signature, other)
        return (
            self.parameters == right.parameters
            and self.return_annotation == right.return_annotation
        )

    def bind(self, /, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Bind the arguments to the signature.

        Parameters
        ----------
        args : Any
            Positional arguments.
        kwargs : Any
            Keyword arguments.

        Returns
        -------
        dict
            Mapping of parameter names to argument values.

        """
        i: cython.int = 0
        kind: cython.int
        param: Parameter
        bound: dict[str, Any] = {}

        # 1. HANDLE ARGS
        for i in range(len(args)):
            if i >= len(self.parameters):
                raise TypeError("too many positional arguments")

            param = self.parameters[i]
            kind = param.kind
            if kind is POSITIONAL_OR_KEYWORD:
                if param.name in kwargs:
                    raise TypeError(f"multiple values for argument '{param.name}'")
                bound[param.name] = args[i]
            elif kind is VAR_KEYWORD or kind is KEYWORD_ONLY:
                raise TypeError("too many positional arguments")
            elif kind is VAR_POSITIONAL:
                bound[param.name] = args[i:]
                break
            elif kind is POSITIONAL_ONLY:
                bound[param.name] = args[i]
            else:
                raise TypeError("unreachable code")

        # 2. INCREMENT PARAMETER INDEX
        if args:
            i += 1

        # 3. HANDLE KWARGS
        for param in self.parameters[i:]:
            if param.kind is POSITIONAL_OR_KEYWORD or param.kind is KEYWORD_ONLY:
                if param.name in kwargs:
                    bound[param.name] = kwargs.pop(param.name)
                elif param.default_ is EMPTY:
                    raise TypeError(f"missing a required argument: '{param.name}'")
                else:
                    bound[param.name] = param.default_
            elif param.kind is VAR_POSITIONAL:
                bound[param.name] = ()
            elif param.kind is VAR_KEYWORD:
                bound[param.name] = kwargs
                break
            elif param.kind is POSITIONAL_ONLY:
                if param.default_ is EMPTY:
                    if param.name in kwargs:
                        raise TypeError(
                            f"positional only argument '{param.name}' passed as keyword argument"
                        )
                    else:
                        raise TypeError(
                            f"missing required positional argument {param.name}"
                        )
                else:
                    bound[param.name] = param.default_
            else:
                raise TypeError("unreachable code")
        else:
            if kwargs:
                raise TypeError(
                    f"got an unexpected keyword argument '{next(iter(kwargs))}'"
                )

        return bound

    def unbind(self, bound: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Reverse bind of the parameters.

        Attempts to reconstructs the original arguments as keyword only arguments.

        Parameters
        ----------
        bound
            Object with attributes matching the signature parameters.

        Returns
        -------
        args : (args, kwargs)
            Tuple of positional and keyword arguments.

        """
        # does the reverse of bind, but doesn't apply defaults
        args: list = []
        kwargs: dict = {}
        param: Parameter
        for param in self.parameters:
            value = bound[param.name]
            if param.kind is POSITIONAL_OR_KEYWORD:
                args.append(value)
            elif param.kind is VAR_POSITIONAL:
                args.extend(value)
            elif param.kind is VAR_KEYWORD:
                kwargs.update(value)
            elif param.kind is KEYWORD_ONLY:
                kwargs[param.name] = value
            elif param.kind is POSITIONAL_ONLY:
                args.append(value)
            else:
                raise TypeError(f"unsupported parameter kind {param.kind}")

        return tuple(args), kwargs
