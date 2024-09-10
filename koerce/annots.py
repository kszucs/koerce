from __future__ import annotations

import functools
import inspect
import typing
from collections.abc import Mapping, Sequence
from types import FunctionType, MethodType
from typing import Any, ClassVar, Optional

import cython

from .patterns import (
    FrozenDictOf,
    Option,
    Pattern,
    PatternMap,
    TupleOf,
    _any,
    pattern,
)
from .utils import get_type_hints, get_type_origin

EMPTY = inspect.Parameter.empty
_ensure_pattern = pattern


@cython.final
@cython.cclass
class Attribute:
    pattern = cython.declare(Pattern, visibility="readonly")
    default_ = cython.declare(object, visibility="readonly")

    def __init__(self, pattern: Any = _any, default: Any = EMPTY):
        self.pattern = _ensure_pattern(pattern)
        self.default_ = default

    def __repr__(self):
        return f"<{self.__class__.__name__} pattern={self.pattern!r} default={self.default_!r}>"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Attribute):
            return NotImplemented
        right: Attribute = cython.cast(Attribute, other)
        return self.pattern == right.pattern and self.default_ == right.default_

    def __call__(self, default):
        """Needed to support the decorator syntax."""
        return self.__class__(self.pattern, default)


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

    kind = cython.declare(cython.int, visibility="readonly")
    pattern = cython.declare(Pattern, visibility="readonly")
    default_ = cython.declare(object, visibility="readonly")
    typehint = cython.declare(object, visibility="readonly")

    def __init__(
        self,
        kind: int,
        pattern: Any = _any,
        default: Any = EMPTY,
        typehint: Any = EMPTY,
    ):
        self.kind = kind
        self.typehint = typehint
        if kind is _VAR_POSITIONAL:
            self.pattern = TupleOf(pattern)
        elif kind is _VAR_KEYWORD:
            # TODO(kszucs): remove FrozenDict?
            self.pattern = FrozenDictOf(_any, pattern)
        else:
            self.pattern = _ensure_pattern(pattern)

        # validate that the default value matches the pattern
        if default is not EMPTY:
            # TODO(kszucs): try/except MatchError raise an error indicating that the default value doesn't match the pattern
            self.default_ = self.pattern.match(default, {})
        else:
            self.default_ = default

    def format(self, name) -> str:
        result: str = name
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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Parameter):
            return NotImplemented
        right: Parameter = cython.cast(Parameter, other)
        return (
            self.kind == right.kind
            and self.default_ == right.default_
            and self.typehint == right.typehint
        )


@cython.final
@cython.cclass
class Signature:
    length = cython.declare(cython.int, visibility="readonly")
    parameters = cython.declare(dict[str, Parameter], visibility="readonly")
    return_pattern = cython.declare(Pattern, visibility="readonly")
    return_typehint = cython.declare(object, visibility="readonly")

    def __init__(
        self,
        parameters: dict[str, Parameter],
        return_pattern: Pattern = _any,
        return_typehint: Any = EMPTY,
    ):
        self.length = len(parameters)
        self.parameters = parameters
        self.return_pattern = return_pattern
        self.return_typehint = return_typehint

    @staticmethod
    def from_callable(
        func: Any,
        arg_patterns: Sequence[Any] | Mapping[str, Any] | None = None,
        return_pattern: Any = None,
        allow_coercion: bool = True,
    ) -> Signature:
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        params: dict[str, Parameter] = {}

        if arg_patterns is None:
            arg_patterns = {}
        elif isinstance(arg_patterns, Sequence):
            # create a mapping of parameter name to pattern
            arg_patterns = dict(zip(sig.parameters.keys(), arg_patterns))
        elif not isinstance(arg_patterns, Mapping):
            raise TypeError("arg_patterns must be a sequence or a mapping")

        for name, param in sig.parameters.items():
            typehint = hints.get(name, EMPTY)
            if name in arg_patterns:
                argpat = pattern(arg_patterns[name])
            elif typehint is not EMPTY:
                argpat = Pattern.from_typehint(typehint, allow_coercion=allow_coercion)
            else:
                argpat = _any

            params[name] = Parameter(
                kind=int(param.kind),
                default=param.default,
                pattern=argpat,
                typehint=typehint,
            )

        return_typehint = hints.get("return", EMPTY)
        if return_pattern is not None:
            retpat = pattern(return_pattern)
        elif return_typehint is not EMPTY:
            retpat = Pattern.from_typehint(
                return_typehint, allow_coercion=allow_coercion
            )
        else:
            retpat = _any

        return Signature(params, return_typehint=return_typehint, return_pattern=retpat)

    @staticmethod
    def merge(
        signatures: Sequence[Signature],
        parameters: Optional[dict[str, Parameter]] = None,
    ):
        """Merge multiple signatures.

        In addition to concatenating the parameters, it also reorders the
        parameters so that optional arguments come after mandatory arguments.

        Parameters
        ----------
        signatures :
            Signature instances to merge.
        parameters :
            Parameters to add to the merged signature.

        Returns
        -------
        Signature

        """
        name: str
        param: Parameter
        params: dict[str, Parameter] = {}
        for sig in signatures:
            params.update(sig.parameters)

        inherited: set[str] = set(params.keys())
        if parameters:
            for name, param in parameters.items():
                params[name] = param

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
                var_args.append(name)
            elif param.kind == _VAR_KEYWORD:
                if var_kwargs:
                    raise TypeError("only one variadic **kwargs parameter is allowed")
                var_kwargs.append(name)
            elif name in inherited:
                if param.default_ is EMPTY:
                    old_args.append(name)
                else:
                    old_kwargs.append(name)
            elif param.default_ is EMPTY:
                new_args.append(name)
            else:
                new_kwargs.append(name)

        order: list[str] = (
            old_args + new_args + var_args + new_kwargs + old_kwargs + var_kwargs
        )
        return Signature({name: params[name] for name in order})

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Signature):
            return NotImplemented
        right: Signature = cython.cast(Signature, other)
        return (
            tuple(self.parameters.items()) == tuple(right.parameters.items())
            and self.return_pattern == right.return_pattern
            and self.return_typehint == right.return_typehint
        )

    def __call__(self, /, *args, **kwargs):
        return self.bind(args, kwargs)

    def __len__(self) -> int:
        return self.length

    def __str__(self):
        params_str = ", ".join(
            param.format(name) for name, param in self.parameters.items()
        )
        if self.return_typehint is not EMPTY:
            return_str = f" -> {self.return_typehint}"
        else:
            return_str = ""
        return f"({params_str}){return_str}"

    @cython.ccall
    def bind(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
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
        params = iter(self.parameters.items())

        # 1. HANDLE ARGS
        for i in range(len(args)):
            try:
                name, param = next(params)
            except StopIteration:
                raise TypeError("too many positional arguments")

            kind = param.kind
            if kind is _POSITIONAL_OR_KEYWORD:
                if name in kwargs:
                    raise TypeError(f"multiple values for argument '{name}'")
                bound[name] = args[i]
            elif kind is _VAR_KEYWORD or kind is _KEYWORD_ONLY:
                raise TypeError("too many positional arguments")
            elif kind is _VAR_POSITIONAL:
                bound[name] = args[i:]
                break
            elif kind is _POSITIONAL_ONLY:
                bound[name] = args[i]
            else:
                raise TypeError("unreachable code")

        # 2. HANDLE KWARGS
        while True:
            try:
                name, param = next(params)
            except StopIteration:
                if kwargs:
                    raise TypeError(
                        f"got an unexpected keyword argument '{next(iter(kwargs))}'"
                    )
                break

            kind = param.kind
            if kind is _POSITIONAL_OR_KEYWORD or kind is _KEYWORD_ONLY:
                if name in kwargs:
                    bound[name] = kwargs.pop(name)
                elif param.default_ is EMPTY:
                    raise TypeError(f"missing a required argument: '{name}'")
                else:
                    bound[name] = param.default_
            elif kind is _VAR_POSITIONAL:
                bound[name] = ()
            elif kind is _VAR_KEYWORD:
                bound[name] = kwargs
                break
            elif kind is _POSITIONAL_ONLY:
                if param.default_ is EMPTY:
                    if name in kwargs:
                        raise TypeError(
                            f"positional only argument '{name}' passed as keyword argument"
                        )
                    else:
                        raise TypeError(f"missing required positional argument {name}")
                else:
                    bound[name] = param.default_
            else:
                raise TypeError("unreachable code")

        return bound

    @cython.ccall
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
        kind: cython.int
        kwargs: dict = {}
        param: Parameter
        for name, param in self.parameters.items():
            value = bound[name]
            kind = param.kind
            if kind is _POSITIONAL_OR_KEYWORD:
                args.append(value)
            elif kind is _VAR_POSITIONAL:
                args.extend(value)
            elif kind is _VAR_KEYWORD:
                kwargs.update(value)
            elif kind is _KEYWORD_ONLY:
                kwargs[name] = value
            elif kind is _POSITIONAL_ONLY:
                args.append(value)
            else:
                raise TypeError(f"unsupported parameter kind {kind}")

        return tuple(args), kwargs


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

    sig: Signature = Signature.from_callable(
        func,
        arg_patterns=patterns or kwargs,
        return_pattern=return_pattern,
        allow_coercion=True,
    )
    pat: Pattern = PatternMap(
        {name: param.pattern for name, param in sig.parameters.items()}
    )

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # 0. Bind the arguments to the signature
        bound: dict[str, Any] = sig.bind(args, kwargs)

        # 1. Validate the passed arguments
        values: Any = pat.match(bound, {})

        # 2. Reconstruction of the original arguments
        args, kwargs = sig.unbind(values)

        # 3. Call the function with the validated arguments
        result = func(*args, **kwargs)

        # 4. Validate the return value
        result = sig.return_pattern.match(result, {})

        return result

    wrapped.__signature__ = sig

    return wrapped


def attribute(pattern=_any, default=EMPTY):
    """Annotation to mark a field in a class."""
    if default is EMPTY and isinstance(pattern, (FunctionType, MethodType)):
        return Attribute(pattern=_any, default=pattern)
    else:
        return Attribute(pattern=pattern, default=default)


def argument(pattern=_any, default=EMPTY, typehint=EMPTY):
    """Annotation type for all fields which should be passed as arguments."""
    return Parameter(
        kind=_POSITIONAL_OR_KEYWORD, default=default, pattern=pattern, typehint=typehint
    )


def optional(pattern=_any, default=None, typehint=EMPTY):
    """Annotation to allow and treat `None` values as missing arguments."""
    if default is None:
        pattern = Option(pattern)
    return Parameter(
        kind=_POSITIONAL_OR_KEYWORD, default=default, pattern=pattern, typehint=typehint
    )


def varargs(pattern=_any, typehint=EMPTY):
    """Annotation to mark a variable length positional arguments."""
    return Parameter(kind=_VAR_POSITIONAL, pattern=pattern, typehint=typehint)


def varkwargs(pattern=_any, typehint=EMPTY):
    """Annotation to mark a variable length keyword arguments."""
    return Parameter(kind=_VAR_KEYWORD, pattern=pattern, typehint=typehint)


__create__ = cython.declare(object, type.__call__)
if cython.compiled:
    from cython.cimports.cpython.object import PyObject_GenericSetAttr as __setattr__
else:
    __setattr__ = object.__setattr__


@cython.final
@cython.cclass
class AnnotableSpec:
    # make them readonly
    initable = cython.declare(cython.bint, visibility="readonly")
    hashable = cython.declare(cython.bint, visibility="readonly")
    immutable = cython.declare(cython.bint, visibility="readonly")
    signature = cython.declare(Signature, visibility="readonly")
    attributes = cython.declare(dict[str, Attribute], visibility="readonly")
    hasattribs = cython.declare(cython.bint, visibility="readonly")

    def __init__(
        self,
        initable: bool,
        hashable: bool,
        immutable: bool,
        signature: Signature,
        attributes: dict[str, Attribute],
    ):
        self.initable = initable
        self.hashable = hashable
        self.immutable = immutable
        self.signature = signature
        self.attributes = attributes
        self.hasattribs = bool(attributes)

    @cython.cfunc
    @cython.inline
    def new(self, cls: type, args: tuple[Any, ...], kwargs: dict[str, Any]):
        ctx: dict[str, Any] = {}
        bound: dict[str, Any]
        param: Parameter

        if not args and len(kwargs) == self.signature.length:
            bound = kwargs
        else:
            bound = self.signature.bind(args, kwargs)

        if self.initable:
            # slow initialization calling __init__
            for name, param in self.signature.parameters.items():
                bound[name] = param.pattern.match(bound[name], ctx)
            return __create__(cls, **bound)
        else:
            # fast initialization directly setting the arguments
            this = cls.__new__(cls)
            for name, param in self.signature.parameters.items():
                __setattr__(this, name, param.pattern.match(bound[name], ctx))
            # TODO(kszucs): test order ot precomputes and attributes calculations
            if self.hashable:
                self.init_precomputes(this)
            if self.hasattribs:
                self.init_attributes(this)
            return this

    @cython.cfunc
    @cython.inline
    def init_attributes(self, this) -> cython.void:
        attr: Attribute
        for name, attr in self.attributes.items():
            if attr.default_ is not EMPTY:
                if callable(attr.default_):
                    value = attr.default_(this)
                else:
                    value = attr.default_
                __setattr__(this, name, value)

    @cython.cfunc
    @cython.inline
    def init_precomputes(self, this) -> cython.void:
        arguments = tuple(getattr(this, name) for name in self.signature.parameters)
        hashvalue = hash((this.__class__, arguments))
        __setattr__(this, "__args__", arguments)
        __setattr__(this, "__precomputed_hash__", hashvalue)


class AbstractMeta(type):
    """Base metaclass for many of the ibis core classes.

    Enforce the subclasses to define a `__slots__` attribute and provide a
    `__create__` classmethod to change the instantiation behavior of the class.

    Support abstract methods without extending `abc.ABCMeta`. While it provides
    a reduced feature set compared to `abc.ABCMeta` (no way to register virtual
    subclasses) but avoids expensive instance checks by enforcing explicit
    subclassing.
    """

    __slots__ = ()

    def __new__(metacls, clsname, bases, dct, **kwargs):
        # # enforce slot definitions
        # dct.setdefault("__slots__", ())

        # construct the class object
        cls = super().__new__(metacls, clsname, bases, dct, **kwargs)

        # calculate abstract methods existing in the class
        abstracts = {
            name
            for name, value in dct.items()
            if getattr(value, "__isabstractmethod__", False)
        }
        for parent in bases:
            for name in getattr(parent, "__abstractmethods__", set()):
                value = getattr(cls, name, None)
                if getattr(value, "__isabstractmethod__", False):
                    abstracts.add(name)

        # set the abstract methods for the class
        cls.__abstractmethods__ = frozenset(abstracts)

        return cls


class AnnotableMeta(AbstractMeta):
    def __new__(
        metacls,
        clsname,
        bases,
        dct,
        initable=None,
        hashable=None,
        immutable=None,
        allow_coercion=True,
        **kwargs,
    ):
        # inherit annotable specifications from parent classes
        spec: AnnotableSpec
        signatures: list = []
        attributes: dict[str, Attribute] = {}
        is_initable: cython.bint
        is_hashable: cython.bint = hashable is True
        is_immutable: cython.bint = immutable is True
        if initable is None:
            is_initable = "__init__" in dct or "__new__" in dct
        else:
            is_initable = initable
        for parent in bases:
            try:  # noqa: SIM105
                spec = parent.__spec__
            except AttributeError:
                continue
            is_initable |= spec.initable
            is_hashable |= spec.hashable
            is_immutable |= spec.immutable
            signatures.append(spec.signature)
            attributes.update(spec.attributes)

        # create the base classes for the new class
        traits: list[type] = []
        if is_immutable and immutable is False:
            raise TypeError(
                "One of the base classes is immutable so the child class cannot be mutable"
            )
        if is_hashable and hashable is False:
            raise TypeError(
                "One of the base classes is hashable so this child class must be hashable"
            )
        if is_hashable and not is_immutable:
            raise TypeError("Only immutable classes can be hashable")
        if hashable:
            traits.append(Hashable)
        if immutable:
            traits.append(Immutable)

        # collect type annotations and convert them to patterns
        slots: list[str] = list(dct.pop("__slots__", []))
        module: str | None = dct.pop("__module__", None)
        qualname: str = dct.pop("__qualname__", clsname)
        annotations: dict[str, Any] = dct.get("__annotations__", {})
        if module is None:
            self_qualname = None
        else:
            self_qualname = f"{module}.{qualname}"

        # TODO(kszucs): pass dct as localns to evaluate_annotations
        typehints = get_type_hints(annotations, module=module)
        for name, typehint in typehints.items():
            if get_type_origin(typehint) is ClassVar:
                continue
            dct[name] = Parameter(
                kind=_POSITIONAL_OR_KEYWORD,
                pattern=Pattern.from_typehint(
                    typehint, allow_coercion=allow_coercion, self_qualname=self_qualname
                ),
                default=dct.get(name, EMPTY),
                typehint=typehint,
            )

        namespace: dict[str, Any] = {}
        parameters: dict[str, Parameter] = {}
        for name, value in dct.items():
            if isinstance(value, Parameter):
                parameters[name] = value
                slots.append(name)
            elif isinstance(value, Attribute):
                attributes[name] = value
                slots.append(name)
            else:
                namespace[name] = value

        # merge the annotations with the parent annotations
        signature = Signature.merge(signatures, parameters)
        argnames = tuple(signature.parameters.keys())
        bases = tuple(traits) + bases
        spec = AnnotableSpec(
            initable=is_initable,
            hashable=is_hashable,
            immutable=is_immutable,
            signature=signature,
            attributes=attributes,
        )

        namespace.update(
            __argnames__=argnames,
            __match_args__=argnames,
            __module__=module,
            __qualname__=qualname,
            __signature__=signature,
            __slots__=tuple(slots),
            __spec__=spec,
        )
        return super().__new__(metacls, clsname, bases, namespace, **kwargs)

    def __call__(cls, *args, **kwargs):
        spec: AnnotableSpec = cython.cast(AnnotableSpec, cls.__spec__)
        return spec.new(cython.cast(type, cls), args, kwargs)


class Immutable:
    __slots__ = ()

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __setattr__(self, name: str, _: Any) -> None:
        raise AttributeError(
            f"Attribute {name!r} cannot be assigned to immutable instance of "
            f"type {type(self)}"
        )


class Hashable:
    __slots__ = ("__args__", "__precomputed_hash__")

    def __hash__(self) -> int:
        return self.__precomputed_hash__

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return (
            self.__precomputed_hash__ == other.__precomputed_hash__
            and self.__args__ == other.__args__
        )


class Annotable(metaclass=AnnotableMeta, initable=False):
    __argnames__: ClassVar[tuple[str, ...]]
    __match_args__: ClassVar[tuple[str, ...]]
    __signature__: ClassVar[Signature]

    __slots__ = ("__weakref__",)

    def __init__(self, **kwargs):
        spec: AnnotableSpec = self.__spec__
        for name, value in kwargs.items():
            __setattr__(self, name, value)
        if spec.hashable:
            spec.init_precomputes(self)
        if spec.hasattribs:
            spec.init_attributes(self)

    def __setattr__(self, name, value) -> None:
        spec: AnnotableSpec = self.__spec__
        attr: Attribute
        param: Parameter
        if param := spec.signature.parameters.get(name):
            # try to look up the parameter
            value = param.pattern.match(value, {})
        elif attr := spec.attributes.get(name):
            # try to look up the attribute
            value = attr.pattern.match(value, {})
        __setattr__(self, name, value)

    def __eq__(self, other):
        spec: AnnotableSpec = self.__spec__
        # compare types
        if type(self) is not type(other):
            return NotImplemented
        # compare parameters
        for name in spec.signature.parameters:
            if getattr(self, name) != getattr(other, name):
                return False
        # compare attributes
        for name in spec.attributes:
            if getattr(self, name, EMPTY) != getattr(other, name, EMPTY):
                return False
        return True

    def __getstate__(self):
        spec: AnnotableSpec = self.__spec__
        state: dict[str, Any] = {}
        for name in spec.signature.parameters:
            state[name] = getattr(self, name)
        for name in spec.attributes:
            value = getattr(self, name, EMPTY)
            if value is not EMPTY:
                state[name] = value
        return state

    def __setstate__(self, state):
        spec: AnnotableSpec = self.__spec__
        for name, value in state.items():
            __setattr__(self, name, value)
        if spec.hashable:
            spec.init_precomputes(self)

    def __repr__(self) -> str:
        args = (f"{n}={getattr(self, n)!r}" for n in self.__argnames__)
        argstring = ", ".join(args)
        return f"{self.__class__.__name__}({argstring})"

    @property
    def __args__(self) -> tuple[Any, ...]:
        return tuple(getattr(self, name) for name in self.__argnames__)
