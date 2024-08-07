from __future__ import annotations

import functools
import inspect
import typing
from typing import Any

import cython

from .patterns import (
    DictOf,
    NoMatch,
    Option,
    Pattern,
    PatternMap,
    TupleOf,
    _any,
    pattern,
)
from .utils import get_type_hints

EMPTY = inspect.Parameter.empty
_POSITIONAL_ONLY = cython.declare(cython.int, int(inspect.Parameter.POSITIONAL_ONLY))
_POSITIONAL_OR_KEYWORD = cython.declare(
    cython.int, int(inspect.Parameter.POSITIONAL_OR_KEYWORD)
)
_VAR_POSITIONAL = cython.declare(cython.int, int(inspect.Parameter.VAR_POSITIONAL))
_KEYWORD_ONLY = cython.declare(cython.int, int(inspect.Parameter.KEYWORD_ONLY))
_VAR_KEYWORD = cython.declare(cython.int, int(inspect.Parameter.VAR_KEYWORD))


@cython.final
@cython.cclass
class Parameter:
    POSITIONAL_ONLY: typing.ClassVar[int] = _POSITIONAL_ONLY
    POSITIONAL_OR_KEYWORD: typing.ClassVar[int] = _POSITIONAL_OR_KEYWORD
    VAR_POSITIONAL: typing.ClassVar[int] = _VAR_POSITIONAL
    KEYWORD_ONLY: typing.ClassVar[int] = _KEYWORD_ONLY
    VAR_KEYWORD: typing.ClassVar[int] = _VAR_KEYWORD

    name = cython.declare(str, visibility="readonly")
    kind = cython.declare(cython.int, visibility="readonly")
    # Cannot use C reserved keyword 'default' here
    default_ = cython.declare(object, visibility="readonly")
    typehint = cython.declare(object, visibility="readonly")

    def __init__(
        self, name: str, kind: int, default: Any = EMPTY, typehint: Any = EMPTY
    ):
        self.name = name
        self.kind = kind
        self.default_ = default
        self.typehint = typehint

    def __str__(self) -> str:
        result: str = self.name
        if self.typehint is not EMPTY:
            if hasattr(self.typehint, "__qualname__"):
                result += f": {self.typehint.__qualname__}"
            else:
                result += f": {self.typehint}"
        if self.default_ is not EMPTY:
            if self.typehint is EMPTY:
                result = f"{result}={self.default_}"
            else:
                result = f"{result} = {self.default_!r}"
        if self.kind == _VAR_POSITIONAL:
            result = f"*{result}"
        elif self.kind == _VAR_KEYWORD:
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
            and self.typehint == right.typehint
        )


@cython.final
@cython.cclass
class Signature:
    parameters = cython.declare(list[Parameter], visibility="readonly")
    return_typehint = cython.declare(object, visibility="readonly")

    def __init__(self, parameters: list[Parameter], return_typehint: Any = EMPTY):
        self.parameters = parameters
        self.return_typehint = return_typehint

    @staticmethod
    def from_callable(func: Any) -> Signature:
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        params: list[Parameter] = [
            Parameter(p.name, int(p.kind), p.default, hints.get(p.name, EMPTY))
            for p in sig.parameters.values()
        ]
        return Signature(params, return_typehint=hints.get("return", EMPTY))

    @staticmethod
    def merge(*signatures: Signature, **annotations):
        """Merge multiple signatures.

        In addition to concatenating the parameters, it also reorders the
        parameters so that optional arguments come after mandatory arguments.

        Parameters
        ----------
        *signatures : Signature
            Signature instances to merge.
        **annotations : dict
            Annotations to add to the merged signature.

        Returns
        -------
        Signature

        """
        name: str
        param: Parameter
        params: dict[str, Parameter] = {}
        for sig in signatures:
            for param in sig.parameters:
                params[param.name] = param

        inherited: set[str] = set(params.keys())
        for name, annot in annotations.items():
            params[name] = Parameter(name, annotation=annot)

        # mandatory fields without default values must precede the optional
        # ones in the function signature, the partial ordering will be kept
        old_args: list = []
        new_args: list = []
        var_args: list = []
        var_kwargs: list = []
        new_kwargs: list = []
        old_kwargs: list = []
        for name, param in params.items():
            if param.kind == _VAR_POSITIONAL:
                if var_args:
                    raise TypeError("only one variadic *args parameter is allowed")
                var_args.append(param)
            elif param.kind == _VAR_KEYWORD:
                if var_kwargs:
                    raise TypeError("only one variadic **kwargs parameter is allowed")
                var_kwargs.append(param)
            elif name in inherited:
                if param.default_ is EMPTY:
                    old_args.append(param)
                else:
                    old_kwargs.append(param)
            elif param.default_ is EMPTY:
                new_args.append(param)
            else:
                new_kwargs.append(param)

        return Signature(
            old_args + new_args + var_args + new_kwargs + old_kwargs + var_kwargs
        )

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
            if kind is _POSITIONAL_OR_KEYWORD:
                if param.name in kwargs:
                    raise TypeError(f"multiple values for argument '{param.name}'")
                bound[param.name] = args[i]
            elif kind is _VAR_KEYWORD or kind is _KEYWORD_ONLY:
                raise TypeError("too many positional arguments")
            elif kind is _VAR_POSITIONAL:
                bound[param.name] = args[i:]
                break
            elif kind is _POSITIONAL_ONLY:
                bound[param.name] = args[i]
            else:
                raise TypeError("unreachable code")

        # 2. INCREMENT PARAMETER INDEX
        if args:
            i += 1

        # 3. HANDLE KWARGS
        for param in self.parameters[i:]:
            if param.kind is _POSITIONAL_OR_KEYWORD or param.kind is _KEYWORD_ONLY:
                if param.name in kwargs:
                    bound[param.name] = kwargs.pop(param.name)
                elif param.default_ is EMPTY:
                    raise TypeError(f"missing a required argument: '{param.name}'")
                else:
                    bound[param.name] = param.default_
            elif param.kind is _VAR_POSITIONAL:
                bound[param.name] = ()
            elif param.kind is _VAR_KEYWORD:
                bound[param.name] = kwargs
                break
            elif param.kind is _POSITIONAL_ONLY:
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
            if param.kind is _POSITIONAL_OR_KEYWORD:
                args.append(value)
            elif param.kind is _VAR_POSITIONAL:
                args.extend(value)
            elif param.kind is _VAR_KEYWORD:
                kwargs.update(value)
            elif param.kind is _KEYWORD_ONLY:
                kwargs[param.name] = value
            elif param.kind is _POSITIONAL_ONLY:
                args.append(value)
            else:
                raise TypeError(f"unsupported parameter kind {param.kind}")

        return tuple(args), kwargs

    def to_pattern(
        self,
        overrides: dict[str, Any] | list[Any] | None = None,
        return_override: Any = None,
    ) -> Pattern:
        """Create patterns from a Signature.

        Two patterns are created, one for the arguments and one for the return value.

        Parameters
        ----------
        overrides : dict, default None
            Pass patterns to add missing or override existing argument type
            annotations.
        return_override : Pattern, default None
            Pattern for the return value of the callable.

        Returns
        -------
        Tuple of patterns for the arguments and the return value.
        """
        arg_overrides: dict[str, Any]
        if overrides is None:
            arg_overrides = {}
        elif isinstance(overrides, (list, tuple)):
            # create a mapping of parameter name to pattern
            arg_overrides = {
                param.name: arg for param, arg in zip(self.parameters, overrides)
            }
        elif isinstance(overrides, dict):
            arg_overrides = overrides
        else:
            raise TypeError(f"patterns must be a list or dict, got {type(overrides)}")

        retpat: Pattern
        argpat: Pattern
        argpats: dict[str, Pattern] = {}
        for param in self.parameters:
            name: str = param.name

            if name in arg_overrides:
                argpat = pattern(arg_overrides[name])
            elif param.typehint is not EMPTY:
                argpat = Pattern.from_typehint(param.typehint)
            else:
                argpat = _any

            if param.kind is _VAR_POSITIONAL:
                argpat = TupleOf(argpat)
            elif param.kind is _VAR_KEYWORD:
                argpat = DictOf(_any, argpat)
            elif param.default_ is not EMPTY:
                argpat = Option(argpat, default=param.default_)

            argpats[name] = argpat

        if return_override is not None:
            retpat = pattern(return_override)
        elif self.return_typehint is not EMPTY:
            retpat = Pattern.from_typehint(self.return_typehint)
        else:
            retpat = _any

        return (PatternMap(argpats), retpat)


class ValidationError(Exception):
    pass


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
    argpats, retpat = sig.to_pattern(
        overrides=patterns or kwargs, return_override=return_pattern
    )

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # 0. Bind the arguments to the signature
        bound = sig.bind(*args, **kwargs)

        # 1. Validate the passed arguments
        values = argpats.apply(bound)
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
