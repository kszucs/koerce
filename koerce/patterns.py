from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from enum import Enum
from types import UnionType
from typing import (
    Annotated,
    Any,
    ClassVar,
    ForwardRef,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
)

import cython
from typing_extensions import GenericMeta, Self, get_original_bases

# TODO(kszucs): would be nice to cimport Signature and Builder
from .builders import Builder, Deferred, Var, builder
from .utils import (
    RewindableIterator,
    frozendict,
    get_type_args,
    get_type_boundvars,
    get_type_origin,
    get_type_params,
    is_typehint,
)

T = TypeVar("T")

Context = dict[str, Any]


@cython.final
@cython.cclass
class MatchError(Exception):
    value: Any
    reason: str
    pattern: Optional[Pattern]

    def __init__(self, pattern: Pattern, value: Any, reason: str = ""):
        self.pattern = pattern
        self.value = value
        self.reason = reason

    def __str__(self):
        return self.pattern.describe(self.value, self.reason)


@cython.cclass
class Pattern:
    @staticmethod
    def from_typehint(
        annot: Any, allow_coercion: bool = False, self_qualname: Optional[str] = None
    ) -> Pattern:
        """Construct a validator from a python type annotation.

        Parameters
        ----------
        annot
            The typehint annotation to construct the pattern from. This must be
            an already evaluated type annotation.
        allow_coercion
            Whether to use coercion if the typehint is a Coercible type.
        self_qualname
            The qualname of the class to use for Self typehints.

        Returns
        -------
        A pattern that matches the given type annotation.
        """
        args: tuple = get_type_args(annot)
        origin: Any = get_type_origin(annot)
        options: dict[str, Any] = {
            "allow_coercion": allow_coercion,
            "self_qualname": self_qualname,
        }

        if origin is None:
            # the typehint is not generic
            if annot is Ellipsis or annot is Any:
                # treat both `Any` and `...` as wildcard
                return _any
            elif annot is Self:
                if self_qualname is None:
                    raise ValueError("self_qualname must be provided for Self typehint")
                return IsTypeLazy(self_qualname)
            elif isinstance(annot, type):
                # the typehint is a concrete type (e.g. int, str, etc.)
                if allow_coercion:
                    if hasattr(annot, "__coerce__"):
                        return AsCoercible(annot)
                    elif issubclass(annot, bool):
                        return AsBool()
                    elif issubclass(annot, int):
                        return AsInt()
                    elif issubclass(annot, (float, list, tuple, dict, set)):
                        return AsBuiltin(annot)
                    else:
                        with suppress(TypeError):
                            return AsType(annot)
                return IsType(annot)
            elif isinstance(annot, TypeVar):
                # if the typehint is a type variable we try to construct a
                # validator from it only if it is covariant and has a bound
                if not annot.__covariant__:
                    raise NotImplementedError(
                        "Only covariant typevars are supported for now"
                    )
                if annot.__bound__:
                    return Pattern.from_typehint(annot.__bound__, **options)
                else:
                    return _any
            elif isinstance(annot, Enum):
                # for enums we check the value against the enum values
                return EqValue(annot)
            elif isinstance(annot, str):
                # for strings and forward references we check in a lazy way
                return IsTypeLazy(annot)
            elif isinstance(annot, ForwardRef):
                return IsTypeLazy(annot.__forward_arg__)
            else:
                raise TypeError(f"Cannot create validator from annotation {annot!r}")
        elif origin is Is:
            return Pattern.from_typehint(args[0], allow_coercion=False)
        elif origin is As:
            return Pattern.from_typehint(args[0], allow_coercion=True)
        elif origin is Literal:
            # for literal types we check the value against the literal values
            return IsIn(args)
        elif origin is UnionType or origin is Union:
            # this is slightly more complicated because we need to handle
            # Optional[T] which is Union[T, None] and Union[T1, T2, ...]
            *rest, last = args
            if last is type(None):
                # the typehint is Optional[*rest] which is equivalent to
                # Union[*rest, None], so we construct an Option pattern
                if len(rest) == 1:
                    inner = rest[0]
                else:
                    inner = AnyOf(*rest, **options)
                return Option(inner, **options)
            else:
                # the typehint is Union[*args] so we construct an AnyOf pattern
                return AnyOf(*args, **options)
        elif origin is Annotated:
            # the Annotated typehint can be used to add extra validation logic
            # to the typehint, e.g. Annotated[int, Positive], the first argument
            # is used for isinstance checks, the rest are applied in conjunction
            return AllOf(*args, **options)
        elif origin is Callable:
            # the Callable typehint is used to annotate functions, e.g. the
            # following typehint annotates a function that takes two integers
            # and returns a string: Callable[[int, int], str]
            if args:
                # callable with args and return typehints construct a special
                # CallableWith validator
                return CallableWith(*args, **options)
            else:
                # in case of Callable without args we check for the Callable
                # protocol only
                return IsType(Callable)
        elif issubclass(origin, tuple):
            # construct validators for the tuple elements, but need to treat
            # variadic tuples differently, e.g. tuple[int, ...] is a variadic
            # tuple of integers, while tuple[int] is a tuple with a single int
            first, *rest = args
            if rest == [Ellipsis]:
                return TupleOf(first, **options)
            else:
                return PatternList(args, origin, **options)
        elif issubclass(origin, Sequence):
            # construct a validator for the sequence elements where all elements
            # must be of the same type, e.g. Sequence[int] is a sequence of ints
            return SequenceOf(args[0], type_=origin, **options)
        elif issubclass(origin, Mapping):
            # construct a validator for the mapping keys and values, e.g.
            # Mapping[str, int] is a mapping with string keys and int values
            return MappingOf(args[0], args[1], type_=origin, **options)
        elif isinstance(origin, GenericMeta):
            # construct a validator for the generic type, see the specific
            # Generic* validators for more details
            if allow_coercion and hasattr(origin, "__coerce__") and args:
                return AsCoercibleGeneric(annot)
            return IsGeneric(annot)
        else:
            raise TypeError(
                f"Cannot create validator from annotation {annot!r} {origin!r}"
            )

    def apply(self, value, context=None):
        if context is None:
            context = {}
        return self.match(value, context)

    @cython.cfunc
    def match(self, value, ctx: Context): ...

    def describe(self, value, reason) -> str: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self.equals(other)

    def __invert__(self) -> Not:
        """Syntax sugar for matching the inverse of the pattern."""
        return Not(self)

    def __or__(self, other: Pattern) -> AnyOf:
        """Syntax sugar for matching either of the patterns.

        Parameters
        ----------
        other
            The other pattern to match against.

        Returns
        -------
        New pattern that matches if either of the patterns match.
        """
        if isinstance(other, AnyOf):
            return AnyOf(self, *cython.cast(AnyOf, other).inners)
        else:
            return AnyOf(self, other)

    def __and__(self, other: Pattern) -> AllOf:
        """Syntax sugar for matching both of the patterns.

        Parameters
        ----------
        other
            The other pattern to match against.

        Returns
        -------
        New pattern that matches if both of the patterns match.
        """
        if isinstance(other, AllOf):
            return AllOf(self, *cython.cast(AllOf, other).inners)
        else:
            return AllOf(self, other)

    def __rmatmul__(self, name) -> Capture:
        """Syntax sugar for capturing a value.

        Parameters
        ----------
        name
            The name of the capture.

        Returns
        -------
        New capture pattern.

        """
        return Capture(name, self)

    def __rshift__(self, other) -> Replace:
        """Syntax sugar for replacing a value.

        Parameters
        ----------
        other
            The deferred to use for constructing the replacement value.

        Returns
        -------
        New replace pattern.

        """
        return Replace(self, other)

    def __iter__(self) -> SomeOf:
        yield SomeOf(self)


@cython.final
@cython.cclass
class Anything(Pattern):
    def equals(self, other: Anything) -> bool:
        return True

    def describe(self, value, reason):
        return "Anything() always matches"

    @cython.cfunc
    @cython.inline
    def match(self, value, ctx: Context):
        return value


_any = Anything()


@cython.final
@cython.cclass
class Nothing(Pattern):
    def equals(self, other: Nothing) -> bool:
        return True

    def describe(self, value, reason):
        return "Nothing() never matches"

    @cython.cfunc
    @cython.inline
    def match(self, value, ctx: Context):
        raise MatchError(self, value)


@cython.final
@cython.cclass
class IdenticalTo(Pattern):
    value: Any

    def __init__(self, value):
        self.value = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value!r})"

    def equals(self, other: IdenticalTo) -> bool:
        return self.value == other.value

    def describe(self, value, reason):
        return f"{value!r} is not identical to {self.value!r}"

    @cython.cfunc
    @cython.inline
    def match(self, value, ctx: Context):
        if value is self.value:
            return value
        else:
            raise MatchError(self, value)


@cython.ccall
def Eq(value) -> Pattern:
    if isinstance(value, (Deferred, Builder)):
        return EqDeferred(value)
    else:
        return EqValue(value)


@cython.final
@cython.cclass
class EqValue(Pattern):
    value: Any

    def __init__(self, value: Any):
        self.value = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value!r})"

    def equals(self, other: EqValue) -> bool:
        return self.value == other.value

    def describe(self, value, reason) -> str:
        return f"`{value!r}` is not equal to the expected `{self.value!r}`"

    @cython.cfunc
    @cython.inline
    def match(self, value, ctx: Context):
        if value == self.value:
            return value
        else:
            raise MatchError(self, value)


@cython.final
@cython.cclass
class EqDeferred(Pattern):
    """Pattern that checks a value equals to the given value.

    Parameters
    ----------
    value
        The value to check against.

    """

    value: Builder

    def __init__(self, value):
        self.value = builder(value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value!r})"

    def equals(self, other: EqDeferred) -> bool:
        return self.value == other.value

    def describe(self, value, reason) -> str:
        return f"`{value!r}` is not equal to deferred {self.value!r}"

    @cython.cfunc
    def match(self, value, ctx: Context):
        # ctx["_"] = value
        # TODO(kszucs): Builder is not cimported so self.value.build() cannot be
        # used, hence using .apply() instead
        if value == self.value.apply(ctx):
            return value
        else:
            raise MatchError(self, value)


@cython.final
@cython.cclass
class TypeOf(Pattern):
    type_: Any

    def __init__(self, type_):
        assert isinstance(type_, type)
        self.type_ = type_

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.type_!r})"

    def equals(self, other: TypeOf) -> bool:
        return self.type_ == other.type_

    def describe(self, value, reason):
        return f"`{value!r}` doesn't have the exact type of {self.type_}"

    @cython.cfunc
    @cython.inline
    def match(self, value, ctx: Context):
        if type(value) is self.type_:
            return value
        else:
            raise MatchError(self, value)


class Is(Generic[T]):
    def __new__(cls, type_) -> Pattern:
        if isinstance(type_, tuple):
            return IsType(type_)
        else:
            return Pattern.from_typehint(type_, allow_coercion=False)


@cython.final
@cython.cclass
class IsType(Pattern):
    # performance doesn't seem to be affected:
    # https://github.com/kszucs/koerce/pull/6#discussion_r1705833034
    type_: Any

    def __init__(self, type_: Any):
        self.type_ = type_

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.type_!r})"

    def __call__(self, *args, **kwargs):
        return ObjectOf(self.type_, args, kwargs)

    def equals(self, other: IsType) -> bool:
        return self.type_ == other.type_

    def describe(self, value, reason) -> str:
        return f"`{value!r}` is not an instance of {self.type_}"

    @cython.cfunc
    @cython.inline
    def match(self, value, ctx: Context):
        if isinstance(value, self.type_):
            return value
        else:
            raise MatchError(self, value)


@cython.final
@cython.cclass
class IsTypeLazy(Pattern):
    qualname: str
    package: str
    type_: Any

    def __init__(self, qualname: str):
        if not isinstance(qualname, str):
            raise TypeError("qualname must be a string")
        _common_package_aliases: dict[str, str] = {
            "pa": "pyarrow",
            "pd": "pandas",
            "np": "numpy",
            "tf": "tensorflow",
        }
        package: str = qualname.split(".")[0]
        self.qualname = qualname
        self.package = _common_package_aliases.get(package, package)
        self.type_ = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.qualname!r})"

    def __call__(self, *args, **kwargs):
        return ObjectOf(self, args, kwargs)

    def equals(self, other: IsTypeLazy) -> bool:
        return self.qualname == other.qualname

    def describe(self, value, reason) -> str:
        return f"`{value!r}` is not an instance of {self.qualname!r}"

    @cython.cfunc
    def _import_type(self):
        module_name, type_name = self.qualname.rsplit(".", 1)
        module = importlib.import_module(module_name)
        try:
            self.type_ = getattr(module, type_name)
        except AttributeError:
            raise ImportError(f"Could not import {type_name} from {module_name}")

    @cython.cfunc
    def match(self, value, ctx: Context):
        if self.type_ is not None:
            if isinstance(value, self.type_):
                return value
            else:
                raise MatchError(self, value)

        klass: Any
        package: str
        for klass in type(value).__mro__:
            package = klass.__module__.split(".", 1)[0]
            if package == self.package:
                self._import_type()
                if isinstance(value, self.type_):
                    return value
                else:
                    raise MatchError(self, value)

        raise MatchError(self, value)


@cython.ccall
def IsGeneric(typ) -> Pattern:
    # TODO(kszucs): could be expressed using ObjectOfN..
    nparams: int = len(get_type_params(typ))
    if nparams == 1:
        return IsGeneric1(typ)
    elif nparams == 2:
        return IsGeneric2(typ)
    else:
        return IsGenericN(typ)


@cython.final
@cython.cclass
class IsGeneric1(Pattern):
    origin: Any
    name1: str
    pattern1: Pattern

    def __init__(self, typ):
        self.origin = get_type_origin(typ)

        ((self.name1, type1),) = get_type_boundvars(typ).items()
        self.pattern1 = Pattern.from_typehint(type1, allow_coercion=False)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.origin!r}, "
            f"name1={self.name1!r}, pattern1={self.pattern1!r})"
        )

    def __call__(self, *args, **kwargs):
        return ObjectOf(self, args, kwargs)

    def equals(self, other: IsGeneric1) -> bool:
        return (
            self.origin == other.origin
            and self.name1 == other.name1
            and self.pattern1 == other.pattern1
        )

    def describe(self, value, reason) -> str:
        return f"{value!r} is not an instance of {self.origin!r}"

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not isinstance(value, self.origin):
            raise MatchError(self, value)

        attr1 = getattr(value, self.name1)
        self.pattern1.match(attr1, ctx)

        return value


@cython.final
@cython.cclass
class IsGeneric2(Pattern):
    origin: Any
    name1: str
    name2: str
    pattern1: Pattern
    pattern2: Pattern

    def __init__(self, typ):
        self.origin = get_type_origin(typ)

        (self.name1, type1), (self.name2, type2) = get_type_boundvars(typ).items()
        self.pattern1 = Pattern.from_typehint(type1, allow_coercion=False)
        self.pattern2 = Pattern.from_typehint(type2, allow_coercion=False)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.origin!r}, "
            f"name1={self.name1!r}, pattern1={self.pattern1!r}, "
            f"name2={self.name2!r}, pattern2={self.pattern2!r})"
        )

    def __call__(self, *args, **kwargs):
        return ObjectOf(self, *args, **kwargs)

    def describe(self, value, reason) -> str:
        return f"{value!r} is not an instance of {self.origin!r}"

    def equals(self, other: IsGeneric2) -> bool:
        return (
            self.origin == other.origin
            and self.name1 == other.name1
            and self.name2 == other.name2
            and self.pattern1 == other.pattern1
            and self.pattern2 == other.pattern2
        )

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not isinstance(value, self.origin):
            raise MatchError(self, value)

        attr1 = getattr(value, self.name1)
        self.pattern1.match(attr1, ctx)

        attr2 = getattr(value, self.name2)
        self.pattern2.match(attr2, ctx)

        return value


@cython.final
@cython.cclass
class IsGenericN(Pattern):
    origin: Any
    fields: dict[str, Pattern]

    def __init__(self, typ):
        self.origin = get_type_origin(typ)

        name: str
        self.fields = {}
        for name, type_ in get_type_boundvars(typ).items():
            self.fields[name] = Pattern.from_typehint(type_, allow_coercion=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.origin!r}, fields={self.fields!r})"

    def __call__(self, *args, **kwargs):
        return ObjectOf(self, args, kwargs)

    def equals(self, other: IsGenericN) -> bool:
        return self.origin == other.origin and self.fields == other.fields

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not isinstance(value, self.origin):
            raise MatchError(self, value)

        name: str
        pattern: Pattern
        for name, pattern in self.fields.items():
            attr = getattr(value, name)
            pattern.match(attr, ctx)

        return value


@cython.final
@cython.cclass
class SubclassOf(Pattern):
    type_: Any

    def __init__(self, type_: Any):
        self.type_ = type_

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.type_!r})"

    def equals(self, other: SubclassOf) -> bool:
        return self.type_ == other.type_

    def describe(self, value, reason) -> str:
        return f"{value!r} is not a subclass of {self.type_!r}"

    @cython.cfunc
    @cython.inline
    def match(self, value, ctx: Context):
        if issubclass(value, self.type_):
            return value
        else:
            raise MatchError(self, value)


class As(Generic[T]):
    def __new__(cls, type_) -> Self:
        return Pattern.from_typehint(type_, allow_coercion=True)


@cython.final
@cython.cclass
class AsBool(Pattern):
    def equals(self, other: AsBool) -> bool:
        return True

    def describe(self, value, reason) -> str:
        return f"Cannot losslessly convert {value!r} to a boolean."

    @cython.cfunc
    def match(self, value, ctx: Context):
        if isinstance(value, bool):
            # Check if the value is already a boolean
            return value
        if value is None:
            raise MatchError(self, value)
        if isinstance(value, int):
            # Allow conversion only for values clearly boolean-like (0, 1, "true", "false", etc.)
            if value == 0:
                return False
            elif value == 1:
                return True
        if isinstance(value, str):
            lowered = value.lower()
            if lowered == "true" or lowered == "1":
                return True
            elif lowered == "false" or lowered == "0":
                return False
        raise MatchError(self, value)


@cython.final
@cython.cclass
class AsInt(Pattern):
    def equals(self, other: AsInt) -> bool:
        return True

    def describe(self, value, reason) -> str:
        return f"Cannot losslessly convert {value!r} to an integer."

    @cython.cfunc
    def match(self, value, ctx: Context):
        if isinstance(value, int):
            # Check if the value is already an integer
            return value
        if value is None:
            raise MatchError(self, value)
        if isinstance(value, float) and value.is_integer():
            # Check if it's a float but an integer in essence (e.g., 5.0 -> 5)
            return int(value)
        if isinstance(value, str):
            # Check if it's a string representation of an integer
            try:
                # Check if converting to int and back doesn't change the value
                if float(value).is_integer():
                    return int(value)
            except ValueError:
                pass
        raise MatchError(self, value)


@cython.final
@cython.cclass
class AsType(Pattern):
    _registry: ClassVar[dict[type, Any]] = {}
    type_: Any
    func: Any

    def __init__(self, type_: Any):
        self.type_ = type_
        self.func = self.lookup(type_)

    @classmethod
    def register(cls, type_: Any):
        def decorator(func):
            cls._registry[type_] = func
            return func

        return decorator

    @classmethod
    def lookup(cls, type_: Any):
        if not isinstance(type_, type):
            raise TypeError(f"{type_} is not a type")

        for klass in type_.__mro__:
            try:
                impl = cls._registry[klass]
            except KeyError:
                pass
            else:
                if type_ is not klass:
                    # Cache implementation
                    cls._registry[type_] = impl
                return impl

        raise TypeError(f"Could not find a coerce implementation for {type_}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.type_!r})"

    def equals(self, other: AsType) -> bool:
        return self.type_ == other.type_ and self.func == other.func

    def describe(self, value, reason) -> str:
        return f"failed to construct {self.type_!r} from `{value!r}`"

    @cython.cfunc
    def match(self, value, ctx: Context):
        if isinstance(value, self.type_):
            return value
        try:
            return self.func(self.type_, value)
        except Exception as exc:
            raise MatchError(self, value) from exc


@cython.final
@cython.cclass
class AsBuiltin(Pattern):
    type_: Any

    def __init__(self, type_: Any):
        self.type_ = type_

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.type_!r})"

    def equals(self, other: AsBuiltin) -> bool:
        return self.type_ == other.type_

    def describe(self, value, reason) -> str:
        return f"`{value!r}` cannot be coerced to builtin type {self.type_!r}"

    @cython.cfunc
    def match(self, value, ctx: Context):
        if isinstance(value, self.type_):
            return value
        if value is None:
            raise MatchError(self, value)
        try:
            return self.type_(value)
        except Exception as exc:
            raise MatchError(self, value) from exc


@cython.final
@cython.cclass
class AsCoercible(Pattern):
    type_: Any

    def __init__(self, type_: Any):
        if not hasattr(type_, "__coerce__"):
            raise TypeError(f"{type_} does not implement the Coercible protocol")
        self.type_ = type_

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.type_!r})"

    def __call__(self, *args, **kwargs):
        return ObjectOf(self, args, kwargs)

    def equals(self, other: AsCoercible) -> bool:
        return self.type_ == other.type_

    def describe(self, value, reason) -> str:
        if reason == "failed-to-coerce":
            return f"`{value!r}` cannot be coerced to {self.type_!r}"
        elif reason == "not-an-instance":
            return (
                f"`{self.type_.__name__}.__coerce__({value!r})` did not "
                f"return an instance of {self.type_!r}"
            )
        else:
            raise ValueError(f"Unknown reason: {reason}")

    @cython.cfunc
    def match(self, value, ctx: Context):
        try:
            value = self.type_.__coerce__(value)
        except Exception as exc:
            raise MatchError(self, value, "failed-to-coerce") from exc

        if isinstance(value, self.type_):
            return value
        else:
            raise MatchError(self, value, "not-an-instance")


@cython.final
@cython.cclass
class AsCoercibleGeneric(Pattern):
    origin: Any
    params: dict[str, type]
    checker: Pattern

    def __init__(self, typ):
        self.origin = get_type_origin(typ)
        if not hasattr(self.origin, "__coerce__"):
            raise TypeError(f"{self.origin} does not implement the Coercible protocol")
        self.checker = IsGeneric(typ)

        # get all type parameters for the generic class in its type hierarchy
        self.params = {}
        for base in get_original_bases(self.origin):
            self.params.update(get_type_params(base))
        self.params.update(get_type_params(typ))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.origin!r}, params={self.params!r})"

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return ObjectOf(self, args, kwds)

    def equals(self, other: AsCoercibleGeneric) -> bool:
        return self.origin == other.origin and self.params == other.params

    def describe(self, value, reason) -> str:
        if reason == "failed-to-coerce":
            return f"{value!r} cannot be coerced to {self.origin!r}"
        else:
            raise ValueError(f"Unknown reason: {reason}")

    @cython.cfunc
    def match(self, value, ctx: Context):
        try:
            value = self.origin.__coerce__(value, **self.params)
        except Exception as exc:
            raise MatchError(self, value, "failed-to-coerce") from exc

        self.checker.match(value, ctx)

        return value


@cython.final
@cython.cclass
class Not(Pattern):
    inner: Pattern

    def __init__(self, inner, **options):
        self.inner = pattern(inner, **options)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.inner!r})"

    def equals(self, other: Not) -> bool:
        return self.inner == other.inner

    def describe(self, value, reason) -> str:
        return f"`{value!r}` is matching {self.inner!r} whereas it should not"

    @cython.cfunc
    def match(self, value, ctx: Context):
        try:
            self.inner.match(value, ctx)
        except MatchError:
            return value
        else:
            raise MatchError(self, value)


@cython.final
@cython.cclass
class AnyOf(Pattern):
    inners: list[Pattern]

    def __init__(self, *inners: Pattern, **options):
        self.inners = [pattern(inner, **options) for inner in inners]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.inners!r})"

    def equals(self, other: AnyOf) -> bool:
        return self.inners == other.inners

    def describe(self, value, reason) -> str:
        return f"`{value!r}` does not match any of {self.inners!r}"

    @cython.cfunc
    def match(self, value, ctx: Context):
        inner: Pattern
        for inner in self.inners:
            try:
                return inner.match(value, ctx)
            except MatchError:
                pass
        raise MatchError(self, value)

    def __or__(self, other: Pattern) -> AnyOf:
        """Syntax sugar for matching either of the patterns.

        Parameters
        ----------
        other
            The other pattern to match against.

        Returns
        -------
        New pattern that matches if either of the patterns match.
        """
        if isinstance(other, AnyOf):
            return AnyOf(*self.inners, *cython.cast(AnyOf, other).inners)
        else:
            return AnyOf(*self.inners, other)


@cython.final
@cython.cclass
class AllOf(Pattern):
    inners: list[Pattern]

    def __init__(self, *inners: Pattern, **options):
        self.inners = [pattern(inner, **options) for inner in inners]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.inners!r})"

    def equals(self, other: AllOf) -> bool:
        return self.inners == other.inners

    @cython.cfunc
    def match(self, value, ctx: Context):
        inner: Pattern
        for inner in self.inners:
            value = inner.match(value, ctx)
        return value

    def __and__(self, other: Pattern) -> AllOf:
        """Syntax sugar for matching both of the patterns.

        Parameters
        ----------
        other
            The other pattern to match against.

        Returns
        -------
        New pattern that matches if both of the patterns match.
        """
        if isinstance(other, AllOf):
            return AllOf(*self.inners, *cython.cast(AllOf, other).inners)
        else:
            return AllOf(*self.inners, other)


def NoneOf(*args) -> Pattern:
    """Match none of the passed patterns."""
    return Not(AnyOf(*args))


@cython.final
@cython.cclass
class Option(Pattern):
    """Pattern that matches `None` or a value that passes the inner validator.

    Parameters
    ----------
    pattern
        The inner pattern to use.

    """

    pattern: Pattern
    default: Any

    def __init__(self, pat, default=None, **options):
        self.default = default
        self.pattern = pattern(pat, **options)
        if isinstance(self.pattern, Option):
            self.pattern = cython.cast(Option, self.pattern).pattern

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.pattern!r}, default={self.default!r})"

    def equals(self, other: Option) -> bool:
        return self.pattern == other.pattern and self.default == other.default

    @cython.cfunc
    def match(self, value, ctx: Context):
        if value is None:
            return self.default
        else:
            return self.pattern.match(value, ctx)


@cython.ccall
def If(predicate) -> Pattern:
    if isinstance(predicate, (Deferred, Builder)):
        return IfDeferred(predicate)
    elif callable(predicate):
        return IfFunction(predicate)
    else:
        raise TypeError("Predicate must be a callable or a deferred value")


@cython.final
@cython.cclass
class IfFunction(Pattern):
    """Pattern that checks a value against a predicate.

    Parameters
    ----------
    predicate
        The predicate to use.

    """

    predicate: Callable

    def __init__(self, predicate):
        self.predicate = predicate

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.predicate!r})"

    def equals(self, other: IfFunction) -> bool:
        return self.predicate == other.predicate

    def describe(self, value, reason) -> str:
        return f"`{value!r}` does not satisfy the condition {self.predicate!r}"

    @cython.cfunc
    def match(self, value, ctx: Context):
        if self.predicate(value):
            return value
        else:
            raise MatchError(self, value)


@cython.final
@cython.cclass
class IfDeferred(Pattern):
    """Pattern that checks a value against a predicate.

    Parameters
    ----------
    predicate
        The predicate to use.

    """

    builder: Builder

    def __init__(self, obj):
        self.builder = builder(obj)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.builder!r})"

    def equals(self, other: IfDeferred) -> bool:
        return self.builder == other.builder

    def describe(self, value, reason) -> str:
        return f"{value!r} does not satisfy the deferred predicate {self.builder!r}"

    @cython.cfunc
    def match(self, value, ctx: Context):
        # TODO(kszucs): Builder is not cimported so self.builder.build()
        # is not available, hence using .apply() instead
        ctx["_"] = value
        if self.builder.apply(ctx):
            return value
        else:
            raise MatchError(self, value)


@cython.final
@cython.cclass
class IsIn(Pattern):
    """Pattern that matches if a value is in a given set.

    Parameters
    ----------
    haystack
        The set of values that the passed value should be in.

    """

    haystack: frozenset

    def __init__(self, haystack):
        self.haystack = frozenset(haystack)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.haystack})"

    def equals(self, other: IsIn) -> bool:
        return self.haystack == other.haystack

    def describe(self, value, reason) -> str:
        return f"`{value!r}` is not in {self.haystack!r}"

    @cython.cfunc
    def match(self, value, ctx: Context):
        if value in self.haystack:
            return value
        else:
            raise MatchError(self, value)


@cython.final
@cython.cclass
class SequenceOf(Pattern):
    """Pattern that matches if all of the items in a sequence match a given pattern.

    Specialization of the more flexible GenericSequenceOf pattern which uses two
    additional patterns to possibly coerce the sequence type and to match on
    the length of the sequence.

    Parameters
    ----------
    item
        The pattern to match against each item in the sequence.
    type
        The type to coerce the sequence to. Defaults to tuple.

    """

    item: Pattern
    type_: Pattern

    def __init__(self, item: Any, type_: Any = list, **options):
        self.item = pattern(item, **options)
        self.type_ = As(type_)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.item!r}, type_={self.type_!r})"

    def equals(self, other: SequenceOf) -> bool:
        return self.item == other.item and self.type_ == other.type_

    def describe(self, value, reason) -> str:
        if reason == "is-a-string":
            return f"`{value!r}` is a string or bytes, not a sequence"
        elif reason == "not-iterable":
            return f"`{value!r}` is not iterable"
        else:
            raise ValueError(f"Unknown reason: {reason}")

    @cython.cfunc
    def match(self, values, ctx: Context):
        if isinstance(values, (str, bytes)):
            raise MatchError(self, values, "is-a-string")

        # could initialize the result list with the length of values
        result: list = []
        try:
            for item in values:
                result.append(self.item.match(item, ctx))
        except TypeError as exc:
            raise MatchError(self, values, "not-iterable") from exc

        return self.type_.match(result, ctx)


def ListOf(item, **options) -> Pattern:
    return SequenceOf(item, type_=list, **options)


def TupleOf(item, **options) -> Pattern:
    return SequenceOf(item, type_=tuple, **options)


@cython.final
@cython.cclass
class MappingOf(Pattern):
    """Pattern that matches if all of the keys and values match the given patterns.

    Parameters
    ----------
    key
        The pattern to match the keys against.
    value
        The pattern to match the values against.
    type
        The type to coerce the mapping to. Defaults to dict.

    """

    __slots__ = ("key", "value", "type")
    key: Pattern
    value: Pattern
    type_: Pattern

    def __init__(self, key: Any, value: Any, type_: Any = dict, **options):
        self.key = pattern(key, **options)
        self.value = pattern(value, **options)
        self.type_ = As(type_)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.key!r}, {self.value!r}, {self.type_!r})"
        )

    def equals(self, other: MappingOf) -> bool:
        return (
            self.key == other.key
            and self.value == other.value
            and self.type_ == other.type_
        )

    def describe(self, value, reason) -> str:
        return f"`{value!r}` is not a mapping"

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not isinstance(value, Mapping):
            raise MatchError(self, value)

        result = {}
        for k, v in value.items():
            k = self.key.match(k, ctx)
            v = self.value.match(v, ctx)
            result[k] = v

        return self.type_.match(result, ctx)


def DictOf(key, value, **options) -> Pattern:
    return MappingOf(key, value, type_=dict, **options)


def FrozenDictOf(key, value, **options) -> Pattern:
    return MappingOf(key, value, type_=frozendict, **options)


@cython.final
@cython.cclass
class Custom(Pattern):
    """Pattern that matches if a custom function returns True.

    Parameters
    ----------
    func
        The function to use for matching.

    """

    func = cython.declare(object, visibility="readonly")

    def __init__(self, func):
        self.func = func

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.func!r})"

    def equals(self, other: Custom) -> bool:
        return self.func == other.func

    @cython.cfunc
    def match(self, value, ctx: Context):
        try:
            return self.func(value, **ctx)
        except ValueError as exc:
            raise MatchError(self, value) from exc


@cython.final
@cython.cclass
class Capture(Pattern):
    """Pattern that captures a value in the context.

    Parameters
    ----------
    pattern
        The pattern to match against.
    key
        The key to use in the context if the pattern matches.

    """

    key: str
    what: Pattern

    def __init__(self, key: Any, what=_any, **options):
        if isinstance(key, (Deferred, Builder)):
            key = builder(key)
            if isinstance(key, Var):
                key = key.name
            else:
                raise TypeError("Only variables can be used as capture keys")
        self.key = key
        self.what = pattern(what, **options)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.key!r}, {self.what!r})"

    def equals(self, other: Capture) -> bool:
        return self.key == other.key and self.what == other.what

    @cython.cfunc
    def match(self, value, ctx: Context):
        value = self.what.match(value, ctx)
        ctx[self.key] = value
        return value


@cython.final
@cython.cclass
class Replace(Pattern):
    """Pattern that replaces a value with the output of another pattern.

    Parameters
    ----------
    matcher
        The pattern to match against.
    replacer
        The deferred to use as a replacement.

    """

    searcher: Pattern
    replacer: Builder

    def __init__(self, searcher, replacer, **options):
        self.searcher = pattern(searcher, **options)
        self.replacer = builder(replacer, allow_custom=True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.searcher!r}, {self.replacer!r})"

    @cython.cfunc
    def match(self, value, ctx: Context):
        value = self.searcher.match(value, ctx)
        # use the `_` reserved variable to record the value being replaced
        # in the context, so that it can be used in the replacer pattern
        ctx["_"] = value
        # TODO(kszucs): Builder is not cimported so self.replacer.build() cannot be
        # used, hence using .apply() instead
        return self.replacer.apply(ctx)


def Object(type_, *args, **kwargs) -> Pattern:
    return ObjectOf(type_, args, kwargs)


def ObjectOf(type_, args, kwargs, **options) -> Pattern:
    if isinstance(type_, type):
        if len(type_.__match_args__) < len(args):
            raise ValueError(
                "The type to match has fewer `__match_args__` than the number "
                "of positional arguments in the pattern"
            )
        fields = dict(zip(type_.__match_args__, args))
        fields.update(kwargs)
        if len(fields) == 1:
            return ObjectOf1(type_, fields, **options)
        elif len(fields) == 2:
            return ObjectOf2(type_, fields, **options)
        elif len(fields) == 3:
            return ObjectOf3(type_, fields, **options)
        else:
            return ObjectOfN(type_, fields, **options)
    else:
        return ObjectOfX(type_, args, kwargs, **options)


@cython.cfunc
@cython.inline
def _reconstruct(value: Any, changed: dict[str, Any]):
    # use it with caution because it mutates the changed dict
    name: str
    args: tuple[str] = value.__match_args__
    for name in args:
        if name not in changed:
            changed[name] = getattr(value, name)
    return type(value)(**changed)


# TODO(kszucs): pass **options to pattern everywhere
@cython.final
@cython.cclass
class ObjectOf1(Pattern):
    type_: Any
    field1: str
    pattern1: Pattern

    def __init__(self, type_: Any, fields, **options):
        assert len(fields) == 1
        self.type_ = type_
        ((self.field1, pattern1),) = fields.items()
        self.pattern1 = pattern(pattern1, **options)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.type_!r}, {self.field1!r}={self.pattern1!r})"

    def equals(self, other: ObjectOf1) -> bool:
        return (
            self.type_ == other.type_
            and self.field1 == other.field1
            and self.pattern1 == other.pattern1
        )

    def describe(self, value, reason) -> str:
        return f"{value!r} is not an instance of {self.type_!r}"

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not isinstance(value, self.type_):
            raise MatchError(self, value)

        attr1 = getattr(value, self.field1)
        result1 = self.pattern1.match(attr1, ctx)

        if result1 is not attr1:
            changed: dict = {self.field1: result1}
            return _reconstruct(value, changed)
        else:
            return value


@cython.final
@cython.cclass
class ObjectOf2(Pattern):
    type_: Any
    field1: str
    field2: str
    pattern1: Pattern
    pattern2: Pattern

    def __init__(self, type_: Any, fields, **options):
        assert len(fields) == 2
        self.type_ = type_
        (self.field1, pattern1), (self.field2, pattern2) = fields.items()
        self.pattern1 = pattern(pattern1, **options)
        self.pattern2 = pattern(pattern2, **options)

    def __repr__(self) -> str:
        return f"ObjectOf2({self.type_!r}, {self.field1!r}={self.pattern1!r}, {self.field2!r}={self.pattern2!r})"

    def equals(self, other: ObjectOf2) -> bool:
        return (
            self.type_ == other.type_
            and self.field1 == other.field1
            and self.field2 == other.field2
            and self.pattern1 == other.pattern1
            and self.pattern2 == other.pattern2
        )

    def describe(self, value, reason) -> str:
        return f"{value!r} is not an instance of {self.type_!r}"

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not isinstance(value, self.type_):
            raise MatchError(self, value)

        attr1 = getattr(value, self.field1)
        result1 = self.pattern1.match(attr1, ctx)

        attr2 = getattr(value, self.field2)
        result2 = self.pattern2.match(attr2, ctx)

        if result1 is not attr1 or result2 is not attr2:
            changed: dict = {self.field1: result1, self.field2: result2}
            return _reconstruct(value, changed)
        else:
            return value


@cython.final
@cython.cclass
class ObjectOf3(Pattern):
    type_: Any
    field1: str
    field2: str
    field3: str
    pattern1: Pattern
    pattern2: Pattern
    pattern3: Pattern

    def __init__(self, type_: Any, fields, **options):
        assert len(fields) == 3
        self.type_ = type_
        (self.field1, pattern1), (self.field2, pattern2), (self.field3, pattern3) = (
            fields.items()
        )
        self.pattern1 = pattern(pattern1, **options)
        self.pattern2 = pattern(pattern2, **options)
        self.pattern3 = pattern(pattern3, **options)

    def __repr__(self) -> str:
        return (
            f"ObjectOf3({self.type_!r}, "
            f"{self.field1!r}={self.pattern1!r}, "
            f"{self.field2!r}={self.pattern2!r}, "
            f"{self.field3!r}={self.pattern3!r})"
        )

    def equals(self, other: ObjectOf3) -> bool:
        return (
            self.type_ == other.type_
            and self.field1 == other.field1
            and self.field2 == other.field2
            and self.field3 == other.field3
            and self.pattern1 == other.pattern1
            and self.pattern2 == other.pattern2
            and self.pattern3 == other.pattern3
        )

    def describe(self, value, reason) -> str:
        return f"{value!r} is not an instance of {self.type_!r}"

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not isinstance(value, self.type_):
            raise MatchError(self, value)

        attr1 = getattr(value, self.field1)
        result1 = self.pattern1.match(attr1, ctx)

        attr2 = getattr(value, self.field2)
        result2 = self.pattern2.match(attr2, ctx)

        attr3 = getattr(value, self.field3)
        result3 = self.pattern3.match(attr3, ctx)

        if result1 is not attr1 or result2 is not attr2 or result3 is not attr3:
            changed: dict = {
                self.field1: result1,
                self.field2: result2,
                self.field3: result3,
            }
            return _reconstruct(value, changed)
        else:
            return value


@cython.final
@cython.cclass
class ObjectOfN(Pattern):
    """Pattern that matches if the object has the given attributes and they match the given patterns.

    The type must conform the structural pattern matching protocol, e.g. it must have a
    __match_args__ attribute that is a tuple of the names of the attributes to match.

    Parameters
    ----------
    type
        The type of the object.
    *args
        The positional arguments to match against the attributes of the object.
    **kwargs
        The keyword arguments to match against the attributes of the object.

    """

    type_: Any
    fields: dict[str, Pattern]

    def __init__(self, type_: Any, fields, **options):
        self.type_ = type_
        self.fields = {k: pattern(v, **options) for k, v in fields.items()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.type_!r}, {self.fields!r})"

    def equals(self, other: ObjectOfN) -> bool:
        return self.type_ == other.type_ and self.fields == other.fields

    def describe(self, value, reason) -> str:
        return f"{value!r} is not an instance of {self.type_!r}"

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not isinstance(value, self.type_):
            raise MatchError(self, value)

        pattern: Pattern
        changed: dict[str, Any] = {}
        for name, pattern in self.fields.items():
            attr = getattr(value, name)
            result = pattern.match(attr, ctx)
            if result is not attr:
                changed[name] = result

        if changed:
            return _reconstruct(value, changed)
        else:
            return value


@cython.final
@cython.cclass
class ObjectOfX(Pattern):
    type_: Pattern
    args: list[Pattern]
    kwargs: dict[str, Pattern]

    def __init__(self, type_, args, kwargs, **options):
        self.type_ = pattern(type_, **options)
        self.args = [pattern(arg, **options) for arg in args]
        self.kwargs = {k: pattern(v, **options) for k, v in kwargs.items()}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.type_!r}, {self.args!r}, {self.kwargs!r})"
        )

    def equals(self, other: ObjectOfX) -> bool:
        return (
            self.type_ == self.type_
            and self.args == other.args
            and self.kwargs == other.kwargs
        )

    def describe(self, value, reason) -> str:
        if reason == "more-args":
            return (
                f"{value!r} has fewer {len(value.__match_args__)} "
                f"positional arguments than required {len(self.args)}"
            )
        else:
            return f"{value!r} does not have the attribute `{reason}`"

    @cython.cfunc
    def match(self, value, ctx: Context):
        self.type_.match(value, ctx)

        # the pattern requires more positional arguments than the object has
        if len(value.__match_args__) < len(self.args):
            raise MatchError(self, value, "more-args")

        patterns: dict[str, Pattern] = dict(zip(value.__match_args__, self.args))
        patterns.update(self.kwargs)

        name: str
        pattern: Pattern
        changed: dict[str, Any] = {}
        for name, pattern in patterns.items():
            try:
                attr = getattr(value, name)
            except AttributeError:
                raise MatchError(self, value, name)

            result = pattern.match(attr, ctx)
            if result is not attr:
                changed[name] = result

        if changed:
            return _reconstruct(value, changed)
        else:
            return value


@cython.final
@cython.cclass
class CallableWith(Pattern):
    args: list[Pattern]
    return_: Pattern

    def __init__(self, args, return_=_any, **options):
        self.args = [pattern(arg, **options) for arg in args]
        self.return_ = pattern(return_, **options)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.args!r}, return_={self.return_!r})"

    def equals(self, other: CallableWith) -> bool:
        return self.args == other.args and self.return_ == other.return_

    def describe(self, value, reason) -> str:
        if reason == "is-not-callable":
            return f"`{value!r}` is not a callable"
        elif reason == "more-args":
            return (
                f"`{value!r}` has more positional arguments than "
                f"the required {len(self.args)}"
            )
        elif reason == "less-args":
            return (
                f"`{value!r}` has less positional arguments than "
                f"the expected {len(self.args)}"
            )
        else:
            raise ValueError(f"Unknown reason: {reason}")

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not callable(value):
            raise MatchError(self, value, "is-not-callable")

        sig = inspect.signature(value)

        has_varargs: bool = False
        positional: list = []
        required_positional: list = []
        for p in sig.parameters.values():
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                positional.append(p)
                if p.default is inspect.Parameter.empty:
                    required_positional.append(p)
            elif (
                p.kind is inspect.Parameter.KEYWORD_ONLY
                and p.default is inspect.Parameter.empty
            ):
                raise TypeError(
                    "Callable has mandatory keyword-only arguments which cannot be specified"
                )
            elif p.kind is inspect.Parameter.VAR_POSITIONAL:
                has_varargs = True

        if len(required_positional) > len(self.args):
            # Callable has more positional arguments than expected")
            raise MatchError(self, value, "more-args")
        elif len(positional) < len(self.args) and not has_varargs:
            # Callable has less positional arguments than expected")
            raise MatchError(self, value, "less-args")
        else:
            return value


@cython.final
@cython.cclass
class Length(Pattern):
    """Pattern that matches if the length of a value is within a given range.

    Parameters
    ----------
    exactly
        The exact length of the value. If specified, `at_least` and `at_most`
        must be None.
    at_least
        The minimum length of the value.
    at_most
        The maximum length of the value.

    """

    at_least: int
    at_most: int

    def __init__(
        self,
        exactly: Optional[int] = None,
        at_least: Optional[int] = None,
        at_most: Optional[int] = None,
    ):
        if exactly is not None:
            if at_least is not None or at_most is not None:
                raise ValueError("Can't specify both exactly and at_least/at_most")
            at_least = exactly
            at_most = exactly
        self.at_least = at_least
        self.at_most = at_most

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(at_least={self.at_least}, at_most={self.at_most})"

    def equals(self, other: Length) -> bool:
        return self.at_least == other.at_least and self.at_most == other.at_most

    def describe(self, value, reason) -> str:
        if reason == "too-short":
            return (
                f"`{value!r}` is too short, expected at least {self.at_least} elements"
            )
        elif reason == "too-long":
            return f"`{value!r}` is too long, expected at most {self.at_most} elements"
        else:
            raise ValueError(f"Unknown reason: {reason}")

    @cython.cfunc
    def match(self, value, ctx: Context):
        length = len(value)
        if self.at_least is not None and length < self.at_least:
            raise MatchError(self, value, "too-short")
        if self.at_most is not None and length > self.at_most:
            raise MatchError(self, value, "too-long")
        return value


def SomeOf(*args, type_=list, **kwargs) -> Pattern:
    if len(args) == 1:
        return SomeItemsOf(*args, type_=type_, **kwargs)
    else:
        return SomeChunksOf(*args, type_=type_, **kwargs)


@cython.final
@cython.cclass
class SomeItemsOf(Pattern):
    pattern: SequenceOf
    delimiter: Pattern
    length: Length

    def __init__(self, item, type_=list, **kwargs):
        self.pattern = SequenceOf(item, type_=type_)
        self.delimiter = self.pattern.item
        self.length = Length(**kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.pattern!r})"

    def equals(self, other: SomeItemsOf) -> bool:
        return self.pattern == other.pattern

    @cython.cfunc
    def match(self, values, ctx: Context):
        result = self.pattern.match(values, ctx)
        return self.length.match(result, ctx)


@cython.final
@cython.cclass
class SomeChunksOf(Pattern):
    """Pattern that unpacks a value into its elements.

    Designed to be used inside a `PatternList` pattern with the `*` syntax.
    """

    pattern: SequenceOf
    delimiter: Pattern
    length: Length

    def __init__(self, *args, type_=list, **kwargs):
        pl = PatternList(args)
        self.pattern = SequenceOf(pl, type_=type_)
        self.delimiter = cython.cast(Pattern, pl.delimiter)
        self.length = Length(**kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.pattern!r}, {self.delimiter!r})"

    def equals(self, other: SomeChunksOf) -> bool:
        return self.pattern == other.pattern and self.delimiter == other.delimiter

    def chunk(self, values, context):
        chunk: list = []
        for item in values:
            try:
                self.delimiter.match(item, context)
            except MatchError:
                chunk.append(item)
            else:
                if chunk:  # only yield if there are items in the chunk
                    yield chunk
                chunk = [item]  # start a new chunk with the delimiter
        if chunk:
            yield chunk

    @cython.cfunc
    def match(self, values, ctx: Context):
        chunks = self.chunk(values, ctx)
        result = self.pattern.match(chunks, ctx)
        result = self.length.match(result, ctx)
        return [el for lst in result for el in lst]


def _maybe_unwrap_capture(obj):
    if isinstance(obj, Capture):
        return cython.cast(Capture, obj).what
    else:
        return obj


def PatternList(patterns, type_=list, **options) -> Pattern:
    if patterns == ():
        return EqValue(patterns)

    patterns = [pattern(p, **options) for p in patterns]
    for pat in patterns:
        pat = _maybe_unwrap_capture(pat)
        if isinstance(pat, (SomeItemsOf, SomeChunksOf)):
            return VariadicPatternList(patterns, type_, **options)

    return FixedPatternList(patterns, type_, **options)


@cython.final
@cython.cclass
class FixedPatternList(Pattern):
    """Pattern that matches if the respective items in a tuple match the given patterns.

    Parameters
    ----------
    fields
        The patterns to match the respective items in the tuple.

    """

    type_: type
    patterns: list[Pattern]

    def __init__(self, patterns, type_=list, **options):
        self.type_ = type_
        self.patterns = [pattern(p, **options) for p in patterns]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.patterns!r}, type_={self.type_!r})"

    def equals(self, other: FixedPatternList) -> bool:
        return self.patterns == other.patterns and self.type_ == other.type_

    def describe(self, value, reason) -> str:
        if reason == "is-string":
            return f"`{value!r}` is a string or bytes object"
        elif reason == "not-iterable":
            return f"`{value!r}` is not iterable"
        elif reason == "length-mismatch":
            return f"`{value!r}` does not have the same length as the pattern"
        else:
            raise ValueError(f"Unknown reason: {reason}")

    @property
    def delimiter(self) -> Pattern:
        return self.patterns[0]

    @cython.cfunc
    def match(self, values, ctx: Context):
        if isinstance(values, (str, bytes)):
            raise MatchError(self, values, "is-string")

        try:
            values = list(values)
        except TypeError as exc:
            raise MatchError(self, values, "not-iterable") from exc

        if len(values) != len(self.patterns):
            raise MatchError(self, values, "length-mismatch")

        result = []
        pattern: Pattern
        for pattern, value in zip(self.patterns, values):
            value = pattern.match(value, ctx)
            result.append(value)

        return self.type_(result)


@cython.final
@cython.cclass
class VariadicPatternList(Pattern):
    type_: type
    patterns: list[Pattern]

    def __init__(self, patterns, type_=list, **options):
        self.type_ = type_
        self.patterns = [pattern(p, **options) for p in patterns]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.patterns!r}, {self.type_!r})"

    def equals(self, other: VariadicPatternList) -> bool:
        return self.patterns == other.patterns and self.type_ == other.type_

    def describe(self, value, reason) -> str:
        return f"{value!r} does not match the ({self.patterns!r})"

    @property
    def delimiter(self) -> Pattern:
        return self.patterns[0]

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not self.patterns:
            if value:
                raise MatchError(self, value)
            else:
                return self.type_(value)

        it = RewindableIterator(value)

        result: list = []
        current: Pattern
        original: Pattern
        following: Pattern
        following_patterns = self.patterns[1:] + [Nothing()]
        for current, following in zip(self.patterns, following_patterns):
            original = current
            current = _maybe_unwrap_capture(current)
            following = _maybe_unwrap_capture(following)

            if isinstance(current, (SomeItemsOf, SomeChunksOf)):
                if isinstance(following, SomeItemsOf):
                    following = cython.cast(SomeItemsOf, following).delimiter
                elif isinstance(following, SomeChunksOf):
                    following = cython.cast(SomeChunksOf, following).delimiter

                matches = []
                while True:
                    it.checkpoint()
                    try:
                        item = next(it)
                    except StopIteration:
                        break

                    try:
                        res = following.match(item, ctx)
                    except MatchError:
                        matches.append(item)
                    else:
                        it.rewind()
                        break

                res = original.match(matches, ctx)
                result.extend(res)
            else:
                try:
                    item = next(it)
                except StopIteration:
                    raise MatchError(self, value)

                res = original.match(item, ctx)
                result.append(res)

        return self.type_(result)


def PatternMap(fields, **options) -> Pattern:
    if len(fields) == 1:
        return PatternMap1(fields, **options)
    elif len(fields) == 2:
        return PatternMap2(fields, **options)
    elif len(fields) == 3:
        return PatternMap3(fields, **options)
    else:
        return PatternMapN(fields, **options)


@cython.final
@cython.cclass
class PatternMap1(Pattern):
    field1: str
    pattern1: Pattern

    def __init__(self, fields, **options):
        ((self.field1, pattern1),) = fields.items()
        self.pattern1 = pattern(pattern1, **options)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.field1!r}={self.pattern1!r})"

    def equals(self, other: PatternMap1) -> bool:
        return self.field1 == other.field1 and self.pattern1 == other.pattern1

    def describe(self, value, reason) -> str:
        return f"{value!r} is not matching {self!r}"

    @cython.cfunc
    def match(self, value, ctx: Context):
        # TODO(kszucs): checking for Mapping is slow, speed it up e.g. by putting
        # both len and getitem in a try block catching AttributeError indicating
        # that value is not implementing the mapping ABC
        if not isinstance(value, Mapping):
            raise MatchError(self, value)

        if len(value) != 1:
            raise MatchError(self, value)

        try:
            item1 = value[self.field1]
        except KeyError:
            raise MatchError(self, value)

        result1 = self.pattern1.match(item1, ctx)

        if result1 is not item1:
            return type(value)({**value, self.field1: result1})
        else:
            return value


@cython.final
@cython.cclass
class PatternMap2(Pattern):
    field1: str
    field2: str
    pattern1: Pattern
    pattern2: Pattern

    def __init__(self, fields, **options):
        (self.field1, pattern1), (self.field2, pattern2) = fields.items()
        self.pattern1 = pattern(pattern1, **options)
        self.pattern2 = pattern(pattern2, **options)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.field1!r}={self.pattern1!r}, "
            f"{self.field2!r}={self.pattern2!r})"
        )

    def equals(self, other: PatternMap2) -> bool:
        return (
            self.field1 == other.field1
            and self.field2 == other.field2
            and self.pattern1 == other.pattern1
            and self.pattern2 == other.pattern2
        )

    def describe(self, value, reason) -> str:
        return f"{value!r} is not matching {self!r}"

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not isinstance(value, Mapping):
            raise MatchError(self, value)

        if len(value) != 2:
            raise MatchError(self, value)

        try:
            item1 = value[self.field1]
            item2 = value[self.field2]
        except KeyError:
            raise MatchError(self, value)

        result1 = self.pattern1.match(item1, ctx)
        result2 = self.pattern2.match(item2, ctx)

        if result1 is not item1 or result2 is not item2:
            return type(value)({**value, self.field1: result1, self.field2: result2})
        else:
            return value


@cython.final
@cython.cclass
class PatternMap3(Pattern):
    field1: str
    field2: str
    field3: str
    pattern1: Pattern
    pattern2: Pattern
    pattern3: Pattern

    def __init__(self, fields, **options):
        (self.field1, pattern1), (self.field2, pattern2), (self.field3, pattern3) = (
            fields.items()
        )
        self.pattern1 = pattern(pattern1, **options)
        self.pattern2 = pattern(pattern2, **options)
        self.pattern3 = pattern(pattern3, **options)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.field1!r}={self.pattern1!r}, "
            f"{self.field2!r}={self.pattern2!r}, "
            f"{self.field3!r}={self.pattern3!r})"
        )

    def equals(self, other: PatternMap3) -> bool:
        return (
            self.field1 == other.field1
            and self.field2 == other.field2
            and self.field3 == other.field3
            and self.pattern1 == other.pattern1
            and self.pattern2 == other.pattern2
            and self.pattern3 == other.pattern3
        )

    def describe(self, value, reason) -> str:
        return f"{value!r} is not matching {self!r}"

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not isinstance(value, Mapping):
            raise MatchError(self, value)

        if len(value) != 3:
            raise MatchError(self, value)

        try:
            item1 = value[self.field1]
            item2 = value[self.field2]
            item3 = value[self.field3]
        except KeyError:
            raise MatchError(self, value)

        result1 = self.pattern1.match(item1, ctx)
        result2 = self.pattern2.match(item2, ctx)
        result3 = self.pattern3.match(item3, ctx)

        if result1 is not item1 or result2 is not item2 or result3 is not item3:
            return type(value)(
                {
                    **value,
                    self.field1: result1,
                    self.field2: result2,
                    self.field3: result3,
                }
            )
        else:
            return value


@cython.final
@cython.cclass
class PatternMapN(Pattern):
    fields: dict[str, Pattern]

    def __init__(self, fields, **options):
        self.fields = {k: pattern(v, **options) for k, v in fields.items()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.fields!r})"

    def equals(self, other: PatternMapN) -> bool:
        return self.fields == other.fields

    def describe(self, value, reason) -> str:
        return f"{value!r} is not matching {self!r}"

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not isinstance(value, Mapping):
            raise MatchError(self, value)

        if len(value) != len(self.fields):
            raise MatchError(self, value)

        name: str
        pattern: Pattern
        changed: dict[str, Any] = {}
        for name, pattern in self.fields.items():
            try:
                item = value[name]
            except KeyError:
                raise MatchError(self, value, "")
            result = pattern.match(item, ctx)
            if result is not item:
                changed[name] = result

        if changed:
            return type(value)({**value, **changed})
        else:
            return value


@cython.ccall
def pattern(
    obj: Any, allow_coercion: bool = False, self_qualname: Optional[str] = None
) -> Pattern:
    """Create a pattern from various types.

    Not that if a Coercible type is passed as argument, the constructed pattern
    won't attempt to coerce the value during matching. In order to allow type
    coercions use `Pattern.from_typehint()` factory method.

    Parameters
    ----------
    obj
        The object to create a pattern from. Can be a pattern, a type, a callable,
        a mapping, an iterable or a value.

    Examples
    --------
    >>> assert pattern(Any()) == Any()
    >>> assert pattern(int) == InstanceOf(int)
    >>>
    >>> @pattern
    ... def as_int(x, context):
    ...     return int(x)
    >>>
    >>> assert as_int.match(1, {}) == 1

    Returns
    -------
    The constructed pattern.

    """
    if obj is Ellipsis or obj is Any:
        return _any
    elif isinstance(obj, Pattern):
        return obj
    elif is_typehint(obj):
        return Pattern.from_typehint(
            obj, allow_coercion=allow_coercion, self_qualname=self_qualname
        )
    elif isinstance(obj, (Deferred, Builder)):
        return EqDeferred(obj)
    elif isinstance(obj, Mapping):
        return PatternMap(
            obj, allow_coercion=allow_coercion, self_qualname=self_qualname
        )
    elif isinstance(obj, Sequence):
        if isinstance(obj, (str, bytes)):
            return EqValue(obj)
        else:
            return PatternList(
                obj,
                type_=type(obj),
                allow_coercion=allow_coercion,
                self_qualname=self_qualname,
            )
    elif callable(obj):
        return Custom(obj)
    else:
        return EqValue(obj)
