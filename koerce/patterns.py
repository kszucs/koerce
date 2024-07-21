from __future__ import annotations
import cython
import importlib
from enum import Enum
from types import UnionType
from typing import (
    Any,
    TypeVar,
    ForwardRef,
    Mapping,
    Literal,
    Sequence,
    Union,
    Annotated,
    Callable,
    Optional,
)
from typing_extensions import GenericMeta
from .utils import (
    get_type_origin,
    get_type_boundvars,
    get_type_args,
    get_type_params,
    get_original_bases,
    RewindableIterator,
)
from abc import abstractmethod

# from cython.cimports.builders import Builder, Deferred, builder
from .builders import Builder, Deferred, builder, Variable


class CoercionError(Exception):
    pass


Context = dict[str, Any]


@cython.cclass
class NoMatch:
    def __init__(self):
        raise ValueError("Cannot instantiate NoMatch")


@cython.cclass
class Pattern:
    @staticmethod
    def from_typehint(annot: Any, allow_coercion: bool = True) -> Pattern:
        """Construct a validator from a python type annotation.

        Parameters
        ----------
        annot
            The typehint annotation to construct the pattern from. This must be
            an already evaluated type annotation.
        allow_coercion
            Whether to use coercion if the typehint is a Coercible type.

        Returns
        -------
        A pattern that matches the given type annotation.
        """
        origin = get_type_origin(annot)
        args: tuple = get_type_args(annot)

        if origin is None:
            # the typehint is not generic
            if annot is Ellipsis or annot is Any:
                # treat both `Any` and `...` as wildcard
                return _any
            elif isinstance(annot, type):
                # the typehint is a concrete type (e.g. int, str, etc.)
                if allow_coercion and hasattr(annot, "__coerce__"):
                    # the type implements the Coercible protocol so we try to
                    # coerce the value to the given type rather than checking
                    return CoercedTo(annot)
                else:
                    return InstanceOf(annot)
            elif isinstance(annot, TypeVar):
                # if the typehint is a type variable we try to construct a
                # validator from it only if it is covariant and has a bound
                if not annot.__covariant__:
                    raise NotImplementedError(
                        "Only covariant typevars are supported for now"
                    )
                if annot.__bound__:
                    return Pattern.from_typehint(annot.__bound__)
                else:
                    return _any
            elif isinstance(annot, Enum):
                # for enums we check the value against the enum values
                return EqualTo(annot)
            elif isinstance(annot, str):
                # for strings and forward references we check in a lazy way
                return LazyInstanceOf(annot)
            elif isinstance(annot, ForwardRef):
                return LazyInstanceOf(annot.__forward_arg__)
            else:
                raise TypeError(f"Cannot create validator from annotation {annot!r}")
        # elif origin is CoercedTo:
        #     return CoercedTo(args[0])
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
                    inner = Pattern.from_typehint(rest[0])
                else:
                    inner = AnyOf(*map(Pattern.from_typehint, rest))
                return Option(inner)
            else:
                # the typehint is Union[*args] so we construct an AnyOf pattern
                return AnyOf(*map(Pattern.from_typehint, args))
        elif origin is Annotated:
            # the Annotated typehint can be used to add extra validation logic
            # to the typehint, e.g. Annotated[int, Positive], the first argument
            # is used for isinstance checks, the rest are applied in conjunction
            annot, *extras = args
            return AllOf(Pattern.from_typehint(annot), *extras)
        # elif origin is Callable:
        #     # the Callable typehint is used to annotate functions, e.g. the
        #     # following typehint annotates a function that takes two integers
        #     # and returns a string: Callable[[int, int], str]
        #     if args:
        #         # callable with args and return typehints construct a special
        #         # CallableWith validator
        #         arg_hints, return_hint = args
        #         arg_patterns = tuple(map(cls.from_typehint, arg_hints))
        #         return_pattern = cls.from_typehint(return_hint)
        #         return CallableWith(arg_patterns, return_pattern)
        #     else:
        #         # in case of Callable without args we check for the Callable
        #         # protocol only
        #         return InstanceOf(Callable)
        elif issubclass(origin, tuple):
            # construct validators for the tuple elements, but need to treat
            # variadic tuples differently, e.g. tuple[int, ...] is a variadic
            # tuple of integers, while tuple[int] is a tuple with a single int
            first, *rest = args
            if rest == [Ellipsis]:
                return TupleOf(Pattern.from_typehint(first))
            else:
                patterns = map(Pattern.from_typehint, args)
                return PatternList(patterns, origin)
        elif issubclass(origin, Sequence):
            # construct a validator for the sequence elements where all elements
            # must be of the same type, e.g. Sequence[int] is a sequence of ints
            (value_inner,) = map(Pattern.from_typehint, args)
            return SequenceOf(value_inner, type_=origin)
        elif issubclass(origin, Mapping):
            # construct a validator for the mapping keys and values, e.g.
            # Mapping[str, int] is a mapping with string keys and int values
            key_inner, value_inner = map(Pattern.from_typehint, args)
            return MappingOf(key_inner, value_inner, origin)
        elif isinstance(origin, GenericMeta):
            # construct a validator for the generic type, see the specific
            # Generic* validators for more details
            if allow_coercion and hasattr(origin, "__coerce__") and args:
                return GenericCoercedTo(annot)
            else:
                return GenericInstanceOf(annot)
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

    def __repr__(self) -> str: ...

    # def __str__(self) -> str:
    #     return repr(self)

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
        if isinstance(self, AnyOf) and isinstance(other, AnyOf):
            return AnyOf(*self.patterns, *other.patterns)
        elif isinstance(self, AnyOf):
            return AnyOf(*self.patterns, other)
        elif isinstance(other, AnyOf):
            return AnyOf(self, *other.patterns)
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
        if isinstance(self, AllOf) and isinstance(other, AllOf):
            return AllOf(*self.patterns, *other.patterns)
        elif isinstance(self, AllOf):
            return AllOf(*self.patterns, other)
        elif isinstance(other, AllOf):
            return AllOf(self, *other.patterns)
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

    def __rshift__(self, other: Deferred) -> Replace:
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
    def __repr__(self) -> str:
        return "Anything()"

    def equals(self, other: Anything) -> bool:
        return True

    @cython.cfunc
    @cython.inline
    def match(self, value, ctx: Context):
        return value


_any = Anything()


@cython.final
@cython.cclass
class Nothing(Pattern):
    def __repr__(self) -> str:
        return "Nothing()"

    def equals(self, other: Nothing) -> bool:
        return True

    @cython.cfunc
    @cython.inline
    def match(self, value, ctx: Context):
        return NoMatch


@cython.final
@cython.cclass
class IdenticalTo(Pattern):
    value: Any

    def __init__(self, value):
        self.value = value

    def __repr__(self) -> str:
        return f"IdenticalTo({self.value!r})"

    def equals(self, other: IdenticalTo) -> bool:
        return self.value == other.value

    @cython.cfunc
    @cython.inline
    def match(self, value, ctx: Context):
        if value is self.value:
            return value
        else:
            return NoMatch


@cython.final
@cython.cclass
class EqualTo(Pattern):
    value: Any

    def __init__(self, value: Any):
        self.value = value

    def __repr__(self) -> str:
        return f"EqualTo({self.value!r})"

    def equals(self, other: EqualTo) -> bool:
        return self.value == other.value

    @cython.cfunc
    @cython.inline
    def match(self, value, ctx: Context):
        if value == self.value:
            return value
        else:
            return NoMatch


@cython.final
@cython.cclass
class DeferredEqualTo(Pattern):
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
        return f"DeferredEqualTo({self.value!r})"

    @cython.cfunc
    def match(self, value, ctx):
        ctx["_"] = value
        # TODO(kszucs): Builder is not cimported so self.value.build() cannot be
        # used, hence using .apply() instead
        if value == self.value.apply(ctx):
            return value
        else:
            return NoMatch


@cython.final
@cython.cclass
class TypeOf(Pattern):
    type_: type

    def __init__(self, type_: type):
        self.type_ = type_

    def __repr__(self) -> str:
        return f"TypeOf({self.type_!r})"

    def equals(self, other: TypeOf) -> bool:
        return self.type_ == other.type_

    @cython.cfunc
    @cython.inline
    def match(self, value, ctx: Context):
        if type(value) is self.type_:
            return value
        else:
            return NoMatch


@cython.final
@cython.cclass
class InstanceOf(Pattern):
    type_: type

    def __init__(self, type_: type):
        self.type_ = type_

    def __repr__(self) -> str:
        return f"InstanceOf({self.type_!r})"

    def __call__(self, *args, **kwargs):
        return ObjectOf(self.type_, *args, **kwargs)

    def equals(self, other: InstanceOf) -> bool:
        return self.type_ == other.type_

    @cython.cfunc
    @cython.inline
    def match(self, value, ctx: Context):
        if isinstance(value, self.type_):
            return value
        else:
            return NoMatch


@cython.final
@cython.cclass
class LazyInstanceOf(Pattern):
    qualname: str
    package: str
    type_: type

    def __init__(self, qualname: str):
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
        return f"LazyInstanceOf({self.qualname!r})"

    def __call__(self, *args, **kwargs):
        return ObjectOf(self, *args, **kwargs)

    def equals(self, other: LazyInstanceOf) -> bool:
        return self.qualname == other.qualname

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
            return value if isinstance(value, self.type_) else NoMatch

        klass: type
        package: str
        for klass in type(value).__mro__:
            package = klass.__module__.split(".", 1)[0]
            if package == self.package:
                self._import_type()
                return value if isinstance(value, self.type_) else NoMatch

        return NoMatch


@cython.ccall
def GenericInstanceOf(typ) -> Pattern:
    nparams: int = len(get_type_params(typ))
    if nparams == 1:
        return GenericInstanceOf1(typ)
    elif nparams == 2:
        return GenericInstanceOf2(typ)
    else:
        return GenericInstanceOfN(typ)


@cython.final
@cython.cclass
class GenericInstanceOf1(Pattern):
    origin: type
    name1: str
    pattern1: Pattern

    def __init__(self, typ):
        self.origin = get_type_origin(typ)

        ((self.name1, type1),) = get_type_boundvars(typ).items()
        self.pattern1 = Pattern.from_typehint(type1, allow_coercion=False)

    def __repr__(self) -> str:
        return f"GenericInstanceOf1({self.origin!r}, name1={self.name1!r}, pattern1={self.pattern1!r})"

    def __call__(self, *args, **kwargs):
        return ObjectOf(self, *args, **kwargs)

    def equals(self, other: GenericInstanceOf1) -> bool:
        return (
            self.origin == other.origin
            and self.name1 == other.name1
            and self.pattern1 == other.pattern1
        )

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not isinstance(value, self.origin):
            return NoMatch

        attr1 = getattr(value, self.name1)
        if self.pattern1.match(attr1, ctx) is NoMatch:
            return NoMatch

        return value


@cython.final
@cython.cclass
class GenericInstanceOf2(Pattern):
    origin: type
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
        return f"GenericInstanceOf2({self.origin!r}, name1={self.name1!r}, pattern1={self.pattern1!r}, name2={self.name2!r}, pattern2={self.pattern2!r})"

    def __call__(self, *args, **kwargs):
        return ObjectOf(self, *args, **kwargs)

    def equals(self, other: GenericInstanceOf2) -> bool:
        return (
            self.origin == other.origin
            and self.name1 == other.name1
            and self.pattern1 == other.pattern1
            and self.name2 == other.name2
            and self.pattern2 == other.pattern2
        )

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not isinstance(value, self.origin):
            return NoMatch

        attr1 = getattr(value, self.name1)
        if self.pattern1.match(attr1, ctx) is NoMatch:
            return NoMatch

        attr2 = getattr(value, self.name2)
        if self.pattern2.match(attr2, ctx) is NoMatch:
            return NoMatch

        return value


@cython.final
@cython.cclass
class GenericInstanceOfN(Pattern):
    origin: type
    fields: dict[str, Pattern]

    def __init__(self, typ):
        self.origin = get_type_origin(typ)

        name: str
        self.fields = {}
        for name, type_ in get_type_boundvars(typ).items():
            self.fields[name] = Pattern.from_typehint(type_, allow_coercion=False)

    def __repr__(self) -> str:
        return f"GenericInstanceOfN({self.origin!r}, fields={self.fields!r})"

    def __call__(self, *args, **kwargs):
        return ObjectOf(self, *args, **kwargs)

    def equals(self, other: GenericInstanceOfN) -> bool:
        return self.origin == other.origin and self.fields == other.fields

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not isinstance(value, self.origin):
            return NoMatch

        name: str
        pattern: Pattern
        for name, pattern in self.fields.items():
            attr = getattr(value, name)
            if pattern.match(attr, ctx) is NoMatch:
                return NoMatch

        return value


@cython.final
@cython.cclass
class SubclassOf(Pattern):
    type_: type

    def __init__(self, type_: type):
        self.type_ = type_

    def __repr__(self) -> str:
        return f"SubclassOf({self.type_!r})"

    def equals(self, other: SubclassOf) -> bool:
        return self.type_ == other.type_

    @cython.cfunc
    @cython.inline
    def match(self, value, ctx: Context):
        if issubclass(value, self.type_):
            return value
        else:
            return NoMatch


# @cython.ccall
# def As(type_: type) -> Pattern:
#     origin = get_type_origin(type_)
#     if origin is None:
#         if hasattr(type_, "__coerce__"):
#             return CoercedTo(type_)
#         else:
#             return AsType(type_)
#     else:
#         if hasattr(origin, "__coerce__"):
#             return GenericCoercedTo(type_)
#         else:
#             return AsType(type_)


@cython.final
@cython.cclass
class AsType(Pattern):
    type_: type

    def __init__(self, type_: type):
        self.type_ = type_

    def __repr__(self) -> str:
        return f"AsType({self.type_!r})"

    def equals(self, other: AsType) -> bool:
        return self.type_ == other.type_

    @cython.cfunc
    def match(self, value, ctx: Context):
        try:
            return self.type_(value)
        except ValueError:
            return NoMatch


@cython.final
@cython.cclass
class CoercedTo(Pattern):
    type_: type

    def __init__(self, type_: type):
        if not hasattr(type_, "__coerce__"):
            raise TypeError(f"{type_} does not implement the Coercible protocol")
        self.type_ = type_

    def __repr__(self) -> str:
        return f"CoercedTo({self.type_!r})"

    def __call__(self, *args, **kwargs):
        return ObjectOf(self, *args, **kwargs)

    def equals(self, other: CoercedTo) -> bool:
        return self.type_ == other.type_

    @cython.cfunc
    def match(self, value, ctx: Context):
        try:
            value = self.type_.__coerce__(value)
        except CoercionError:
            return NoMatch

        if isinstance(value, self.type_):
            return value
        else:
            return NoMatch


@cython.final
@cython.cclass
class GenericCoercedTo(Pattern):
    origin: type
    params: dict[str, type]
    checker: Pattern

    def __init__(self, typ):
        self.origin = get_type_origin(typ)
        if not hasattr(self.origin, "__coerce__"):
            raise TypeError(f"{self.origin} does not implement the Coercible protocol")
        self.checker = GenericInstanceOf(typ)

        # get all type parameters for the generic class in its type hierarchy
        self.params = {}
        for base in get_original_bases(self.origin):
            self.params.update(get_type_params(base))
        self.params.update(get_type_params(typ))

    def __repr__(self) -> str:
        return f"GenericCoercedTo({self.origin!r}, params={self.params!r})"

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return ObjectOf(self, *args, **kwds)

    def equals(self, other: GenericCoercedTo) -> bool:
        return self.origin == other.origin and self.params == other.params

    @cython.cfunc
    def match(self, value, context):
        try:
            value = self.origin.__coerce__(value, **self.params)
        except CoercionError:
            return NoMatch

        if self.checker.match(value, context) is NoMatch:
            return NoMatch

        return value


@cython.final
@cython.cclass
class Not(Pattern):
    inner: Pattern

    def __init__(self, inner):
        self.inner = pattern(inner)

    def __repr__(self) -> str:
        return f"Not({self.inner!r})"

    def equals(self, other: Not) -> bool:
        return self.inner == other.inner

    @cython.cfunc
    def match(self, value, ctx: Context):
        if self.inner.match(value, ctx) is NoMatch:
            return value
        else:
            return NoMatch


@cython.final
@cython.cclass
class AnyOf(Pattern):
    inners: list[Pattern]

    def __init__(self, *inners: Pattern):
        self.inners = [pattern(inner) for inner in inners]

    def __repr__(self) -> str:
        return f"AnyOf({self.inners!r})"

    def equals(self, other: AnyOf) -> bool:
        return self.inners == other.inners

    @cython.cfunc
    def match(self, value, ctx: Context):
        inner: Pattern
        for inner in self.inners:
            if inner.match(value, ctx) is not NoMatch:
                return value
        return NoMatch


@cython.final
@cython.cclass
class AllOf(Pattern):
    inners: list[Pattern]

    def __init__(self, *inners: Pattern):
        self.inners = [pattern(inner) for inner in inners]

    def __repr__(self) -> str:
        return f"AllOf({self.inners!r})"

    def equals(self, other: AllOf) -> bool:
        return self.inners == other.inners

    @cython.cfunc
    def match(self, value, ctx: Context):
        inner: Pattern
        for inner in self.inners:
            value = inner.match(value, ctx)
            if value is NoMatch:
                return NoMatch
        return value


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

    def __init__(self, pat, default=None):
        self.pattern = pattern(pat)
        self.default = default

    def __repr__(self) -> str:
        return f"Option({self.pattern!r}, default={self.default!r})"

    def equals(self, other: Option) -> bool:
        return self.pattern == other.pattern and self.default == other.default

    @cython.cfunc
    def match(self, value, ctx: Context):
        if value is None:
            if self.default is None:
                return None
            else:
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
        return f"IfFunction({self.predicate!r})"

    def equals(self, other: IfFunction) -> bool:
        return self.predicate == other.predicate

    @cython.cfunc
    def match(self, value, ctx: Context):
        if self.predicate(value):
            return value
        else:
            return NoMatch


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
        return f"If({self.builder!r})"

    def equals(self, other: IfDeferred) -> bool:
        return self.builder == other.builder

    @cython.cfunc
    def match(self, value, ctx: Context):
        # TODO(kszucs): Builder is not cimported so self.builder.build()
        # is not available, hence using .apply() instead
        ctx["_"] = value
        if self.builder.apply(ctx):
            return value
        else:
            return NoMatch


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
        return f"IsIn({self.haystack})"

    def equals(self, other: IsIn) -> bool:
        return self.haystack == other.haystack

    @cython.cfunc
    def match(self, value, context):
        if value in self.haystack:
            return value
        else:
            return NoMatch


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

    def __init__(self, item, type_=list):
        self.item = pattern(item)
        if hasattr(type_, "__coerce__"):
            self.type_ = CoercedTo(type_)
        else:
            try:
                type_([])
            except TypeError:
                self.type_ = AsType(list)
            else:
                self.type_ = AsType(type_)

    def __repr__(self) -> str:
        return f"SequenceOf({self.item!r}, type_={self.type_!r})"

    def equals(self, other: SequenceOf) -> bool:
        return self.item == other.item and self.type_ == other.type_

    @cython.cfunc
    def match(self, values, ctx: Context):
        if isinstance(values, (str, bytes)):
            return NoMatch

        # optimization to avoid unnecessary iteration
        if isinstance(self.item, Anything):
            return values

        try:
            it = iter(values)
        except TypeError:
            return NoMatch

        result: list = []
        for item in it:
            res = self.item.match(item, ctx)
            if res is NoMatch:
                return NoMatch
            result.append(res)

        return self.type_.match(result, ctx)


def ListOf(item) -> Pattern:
    return SequenceOf(item, list)


def TupleOf(item) -> Pattern:
    return SequenceOf(item, tuple)


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

    def __init__(self, key: Pattern, value: Pattern, type_=dict):
        self.key = pattern(key)
        self.value = pattern(value)
        if isinstance(type_, type):
            self.type_ = AsType(type_)
        elif hasattr(type_, "__coerce__"):
            self.type_ = CoercedTo(type_)
        else:
            raise TypeError(f"Cannot coerce to container type {type_}")

    def __repr__(self) -> str:
        return f"MappingOf({self.key!r}, {self.value!r}, {self.type_!r})"

    def equals(self, other: MappingOf) -> bool:
        return (
            self.key == other.key
            and self.value == other.value
            and self.type_ == other.type_
        )

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not isinstance(value, Mapping):
            return NoMatch

        result = {}
        for k, v in value.items():
            if (k := self.key.match(k, ctx)) is NoMatch:
                return NoMatch
            if (v := self.value.match(v, ctx)) is NoMatch:
                return NoMatch
            result[k] = v

        return self.type_.match(result, ctx)


def DictOf(key, value) -> Pattern:
    return MappingOf(key, value, dict)


@cython.final
@cython.cclass
class Custom(Pattern):
    """Pattern that matches if a custom function returns True.

    Parameters
    ----------
    func
        The function to use for matching.

    """

    func: Callable

    def __init__(self, func):
        self.func = func

    def __repr__(self) -> str:
        return f"Custom({self.func!r})"

    def equals(self, other: Custom) -> bool:
        return self.func == other.func

    @cython.cfunc
    def match(self, value, ctx: Context):
        result = self.func(value, **ctx)
        if result is NoMatch:
            return NoMatch
        else:
            return result


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

    def __init__(self, key, what=_any):
        if isinstance(key, (Deferred, Builder)):
            key = builder(key)
            if isinstance(key, Variable):
                key = key.name
            else:
                raise TypeError("Only variables can be used as capture keys")
        self.key = key
        self.what = pattern(what)

    def __repr__(self) -> str:
        return f"Capture({self.key!r}, {self.what!r})"

    def equals(self, other: Capture) -> bool:
        return self.key == other.key and self.what == other.what

    @cython.cfunc
    def match(self, value, ctx: Context):
        value = self.what.match(value, ctx)
        if value is NoMatch:
            return NoMatch
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

    def __init__(self, searcher, replacer):
        self.searcher = pattern(searcher)
        self.replacer = builder(replacer, allow_custom=True)

    def __repr__(self) -> str:
        return f"Replace({self.searcher!r}, {self.replacer!r})"

    @cython.cfunc
    def match(self, value, ctx: Context):
        value = self.searcher.match(value, ctx)
        if value is NoMatch:
            return NoMatch
        # use the `_` reserved variable to record the value being replaced
        # in the context, so that it can be used in the replacer pattern
        ctx["_"] = value
        # TODO(kszucs): Builder is not cimported so self.replacer.build() cannot be
        # used, hence using .apply() instead
        return self.replacer.apply(ctx)


def ObjectOf(type_, *args, **kwargs) -> Pattern:
    if isinstance(type_, type):
        return ObjectOfN(type_, *args, **kwargs)
    else:
        return ObjectOfX(type_, *args, **kwargs)
    # else:
    #     raise TypeError(f"Expected a type, got {type_!r}")


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

    type_: type
    fields: dict[str, Pattern]

    def __init__(self, type_: type, *args, **kwargs):
        if len(type_.__match_args__) < len(args):
            raise ValueError(
                "The type to match has fewer `__match_args__` than the number "
                "of positional arguments in the pattern"
            )
        self.type_ = type_
        self.fields = {}
        k: str
        for k, v in zip(type_.__match_args__, args):
            self.fields[k] = pattern(v)
        for k, v in kwargs.items():
            self.fields[k] = pattern(v)

    def __repr__(self) -> str:
        return f"ObjectOfN({self.type_!r}, {self.fields!r})"

    def equals(self, other: ObjectOfN) -> bool:
        return self.type_ == other.type_ and self.fields == other.fields

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not isinstance(value, self.type_):
            return NoMatch

        name: str
        fields: dict[str, Any] = {}
        changed: cython.bint = False
        pattern: Pattern
        for name, pattern in self.fields.items():
            try:
                attr = getattr(value, name)
            except AttributeError:
                return NoMatch

            result = pattern.match(attr, ctx)
            if result is NoMatch:
                return NoMatch
            # elif result != attr:
            elif result is not attr:
                changed = True
                fields[name] = result
            else:
                fields[name] = attr

        if changed:
            return type(value)(**fields)
        else:
            return value


@cython.final
@cython.cclass
class ObjectOfX(Pattern):
    type_: Pattern
    args: list[Pattern]
    kwargs: dict[str, Pattern]

    def __init__(self, type_, *args, **kwargs):
        self.type_ = pattern(type_)
        self.args = [pattern(arg) for arg in args]
        self.kwargs = {k: pattern(v) for k, v in kwargs.items()}

    def __repr__(self) -> str:
        return f"ObjectOfX({self.type_!r}, {self.args!r}, {self.kwargs!r})"

    def equals(self, other: ObjectOfX) -> bool:
        return (
            self.type_ == self.type_
            and self.args == other.args
            and self.kwargs == other.kwargs
        )

    @cython.cfunc
    def match(self, value, context):
        if self.type_.match(value, context) is NoMatch:
            return NoMatch

        # the pattern requirest more positional arguments than the object has
        if len(value.__match_args__) < len(self.args):
            return NoMatch

        patterns: dict[str, Pattern] = dict(zip(value.__match_args__, self.args))
        patterns.update(self.kwargs)

        name: str
        pattern: Pattern
        fields: dict[str, Any] = {}
        changed: bool = False
        for name, pattern in patterns.items():
            try:
                attr = getattr(value, name)
            except AttributeError:
                return NoMatch

            result = pattern.match(attr, context)
            if result is NoMatch:
                return NoMatch
            # elif result != attr:
            elif result is not attr:
                changed = True
                fields[name] = result
            else:
                fields[name] = attr

        if changed:
            return type(value)(**fields)
        else:
            return value


@cython.final
@cython.cclass
class WithLength(Pattern):
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
        return f"WithLength(at_least={self.at_least}, at_most={self.at_most})"

    def equals(self, other: WithLength) -> bool:
        return self.at_least == other.at_least and self.at_most == other.at_most

    @cython.cfunc
    def match(self, value, ctx: Context):
        length = len(value)
        if self.at_least is not None and length < self.at_least:
            return NoMatch
        if self.at_most is not None and length > self.at_most:
            return NoMatch
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
    length: WithLength

    def __init__(self, item, type_=list, **kwargs):
        self.pattern = SequenceOf(item, type_=type_)
        self.delimiter = self.pattern.item
        self.length = WithLength(**kwargs)

    def __repr__(self) -> str:
        return f"SomeItemsOf({self.pattern!r})"

    def equals(self, other: SomeItemsOf) -> bool:
        return self.pattern == other.pattern

    @cython.cfunc
    def match(self, values, ctx: Context):
        result = self.pattern.match(values, ctx)
        if result is NoMatch:
            return NoMatch

        return self.length.match(result, ctx)


@cython.final
@cython.cclass
class SomeChunksOf(Pattern):
    """Pattern that unpacks a value into its elements.

    Designed to be used inside a `PatternList` pattern with the `*` syntax.
    """

    pattern: SequenceOf
    delimiter: Pattern
    length: WithLength

    def __init__(self, *args, type_=list, **kwargs):
        pl = PatternList(args)
        self.pattern = SequenceOf(pl, type_=type_)
        self.delimiter = cython.cast(Pattern, pl.delimiter)
        self.length = WithLength(**kwargs)

    def __repr__(self) -> str:
        return f"SomeChunksOf({self.pattern!r}, {self.delimiter!r})"

    def equals(self, other: SomeChunksOf) -> bool:
        return self.pattern == other.pattern and self.delimiter == other.delimiter

    def chunk(self, values, context):
        chunk: list = []
        for item in values:
            if self.delimiter.match(item, context) is NoMatch:
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
        if result is NoMatch:
            return NoMatch

        result = self.length.match(result, ctx)
        if result is NoMatch:
            return NoMatch

        return [el for lst in result for el in lst]


def _maybe_unwrap_capture(obj):
    if isinstance(obj, Capture):
        return cython.cast(Capture, obj).what
    else:
        return obj


def PatternList(patterns, type=list):
    if patterns == ():
        return EqualTo(patterns)

    patterns = tuple(map(pattern, patterns))
    for pat in patterns:
        pat = _maybe_unwrap_capture(pat)
        if isinstance(pat, (SomeItemsOf, SomeChunksOf)):
            return VariadicPatternList(patterns, type)

    return FixedPatternList(patterns, type)


@cython.final
@cython.cclass
class FixedPatternList(Pattern):
    """Pattern that matches if the respective items in a tuple match the given patterns.

    Parameters
    ----------
    fields
        The patterns to match the respective items in the tuple.

    """

    patterns: list[Pattern]
    type_: type

    def __init__(self, patterns, type):
        self.patterns = list(map(pattern, patterns))
        self.type_ = type

    def __repr__(self) -> str:
        return f"FixedPatternList({self.patterns!r}, type_={self.type_!r})"

    def equals(self, other: FixedPatternList) -> bool:
        return self.patterns == other.patterns and self.type_ == other.type_

    @property
    def delimiter(self) -> Pattern:
        return self.patterns[0]

    @cython.cfunc
    def match(self, values, ctx: Context):
        if isinstance(values, (str, bytes)):
            return NoMatch

        try:
            values = list(values)
        except TypeError:
            return NoMatch

        if len(values) != len(self.patterns):
            return NoMatch

        result = []
        pattern: Pattern
        for pattern, value in zip(self.patterns, values):
            value = pattern.match(value, ctx)
            if value is NoMatch:
                return NoMatch
            result.append(value)

        return self.type_(result)


@cython.final
@cython.cclass
class VariadicPatternList(Pattern):
    patterns: list[Pattern]
    type_: type

    def __init__(self, patterns, type=list):
        self.patterns = list(map(pattern, patterns))
        self.type_ = type

    def __repr__(self) -> str:
        return f"VariadicPatternList({self.patterns!r}, {self.type_!r})"

    def equals(self, other: VariadicPatternList) -> bool:
        return self.patterns == other.patterns and self.type_ == other.type_

    @property
    def delimiter(self) -> Pattern:
        return self.patterns[0]

    @cython.cfunc
    def match(self, value, ctx: Context):
        if not self.patterns:
            return NoMatch if value else []

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
                # if isinstance(following, (SomeItemsOf, SomeChunksOf)):
                #     following = following.delimiter

                matches = []
                while True:
                    it.checkpoint()
                    try:
                        item = next(it)
                    except StopIteration:
                        break

                    res = following.match(item, ctx)
                    if res is NoMatch:
                        matches.append(item)
                    else:
                        it.rewind()
                        break

                res = original.match(matches, ctx)
                if res is NoMatch:
                    return NoMatch
                else:
                    result.extend(res)
            else:
                try:
                    item = next(it)
                except StopIteration:
                    return NoMatch

                res = original.match(item, ctx)
                if res is NoMatch:
                    return NoMatch
                else:
                    result.append(res)

        return self.type_(result)


@cython.ccall
def pattern(obj: Any, allow_custom: bool = True) -> Pattern:
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
    if obj is Ellipsis:
        return _any
    elif isinstance(obj, Pattern):
        return obj
    elif isinstance(obj, (Deferred, Builder)):
        # return Capture(obj)
        return DeferredEqualTo(obj)
    elif isinstance(obj, Mapping):
        return EqualTo(obj)
    elif isinstance(obj, Sequence):
        if isinstance(obj, (str, bytes)):
            return EqualTo(obj)
        else:
            return PatternList(obj, type(obj))
    elif isinstance(obj, type):
        return InstanceOf(obj)
    elif get_type_origin(obj):
        return Pattern.from_typehint(obj, allow_coercion=False)
    elif callable(obj) and allow_custom:
        return Custom(obj)
    else:
        return EqualTo(obj)


# barhol ahol callback-et lehet hasznalni oda kell egy deferred verzio is
