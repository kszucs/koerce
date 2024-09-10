from __future__ import annotations

import collections.abc
import operator
from typing import Any

import cython

Context = dict[str, Any]


@cython.cclass
class Deferred:
    """The user facing wrapper object providing syntactic sugar for deferreds.

    Provides a natural-like syntax for constructing deferred expressions by
    overloading all of the available dunder methods including the equality
    operator.

    Its sole purpose is to provide a nicer syntax for constructing deferred
    expressions, thus it gets unwrapped to the underlying deferred expression
    when used by the rest of the library.
    """

    _builder: Builder

    def __init__(self, builder: Builder):
        self._builder = builder

    def __repr__(self):
        return repr(self._builder)

    def __getattr__(self, name):
        return Deferred(Attr(self, name))

    def __iter__(self):
        raise TypeError(f"{self.__class__.__name__!r} object is not iterable")

    def __bool__(self):
        raise TypeError(
            f"The truth value of {self.__class__.__name__} objects is not defined"
        )

    def __getitem__(self, name):
        return Deferred(Item(self, name))

    def __call__(self, *args, **kwargs):
        return Deferred(Call(self, *args, **kwargs))

    # def __contains__(self, item):
    #     return Deferred(Binop(operator.contains, self, item))

    def __invert__(self) -> Deferred:
        return Deferred(Unop(operator.invert, self))

    def __neg__(self) -> Deferred:
        return Deferred(Unop(operator.neg, self))

    def __pos__(self) -> Deferred:
        return Deferred(Unop(operator.pos, self))

    def __add__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.add, self, other))

    def __radd__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.add, other, self))

    def __sub__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.sub, self, other))

    def __rsub__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.sub, other, self))

    def __mul__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.mul, self, other))

    def __rmul__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.mul, other, self))

    def __truediv__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.truediv, self, other))

    def __rtruediv__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.truediv, other, self))

    def __floordiv__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.floordiv, self, other))

    def __rfloordiv__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.floordiv, other, self))

    def __pow__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.pow, self, other))

    def __rpow__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.pow, other, self))

    def __mod__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.mod, self, other))

    def __rmod__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.mod, other, self))

    def __rshift__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.rshift, self, other))

    def __rrshift__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.rshift, other, self))

    def __lshift__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.lshift, self, other))

    def __rlshift__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.lshift, other, self))

    def __eq__(self, other: Any) -> Deferred:  # type: ignore
        return Deferred(Binop(operator.eq, self, other))

    def __ne__(self, other: Any) -> Deferred:  # type: ignore
        return Deferred(Binop(operator.ne, self, other))

    def __lt__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.lt, self, other))

    def __le__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.le, self, other))

    def __gt__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.gt, self, other))

    def __ge__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.ge, self, other))

    def __and__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.and_, self, other))

    def __rand__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.and_, other, self))

    def __or__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.or_, self, other))

    def __ror__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.or_, other, self))

    def __xor__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.xor, self, other))

    def __rxor__(self, other: Any) -> Deferred:
        return Deferred(Binop(operator.xor, other, self))


@cython.cclass
class Builder:
    @staticmethod
    def __coerce__(value) -> Builder:
        if isinstance(value, Builder):
            return value
        elif isinstance(value, Deferred):
            return cython.cast(Deferred, value)._builder
        else:
            raise ValueError(f"Cannot coerce {type(value).__name__!r} to Builder")

    def apply(self, ctx: Context):
        return self.build(ctx)

    @cython.cfunc
    def build(self, ctx: Context): ...

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other) and self.equals(other)


@cython.final
@cython.cclass
class Func(Builder):
    """Construct a value by calling a function.

    The function is called with two positional arguments:
    1. the value being matched
    2. the context dictionary

    The function must return the constructed value.

    Parameters
    ----------
    func
        The function to apply.
    """

    func: Any

    def __init__(self, func: Any):
        self.func = func

    def __repr__(self):
        return f"{self.func.__name__}(...)"

    def equals(self, other: Func) -> bool:
        return self.func == other.func

    @cython.cfunc
    def build(self, ctx: Context):
        return self.func(**ctx)


@cython.final
@cython.cclass
class Just(Builder):
    """Construct exactly the given value.

    Parameters
    ----------
    value
        The value to return when the deferred is called.
    """

    value: Any

    def __init__(self, value: Any):
        if isinstance(value, Just):
            self.value = cython.cast(Just, value).value
        else:
            self.value = value

    def __repr__(self):
        if hasattr(self.value, "__deferred_repr__"):
            return self.value.__deferred_repr__()
        elif callable(self.value):
            return getattr(self.value, "__name__", repr(self.value))
        else:
            return repr(self.value)

    def equals(self, other: Just) -> bool:
        return self.value == other.value

    @cython.cfunc
    def build(self, ctx: Context):
        return self.value


@cython.cclass
class Var(Builder):
    """Retrieve a value from the context.

    Parameters
    ----------
    name
        The key to retrieve from the state.
    """

    name = cython.declare(str, visibility="readonly")

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"${self.name}"

    def equals(self, other: Var) -> bool:
        return self.name == other.name

    @cython.cfunc
    def build(self, ctx: Context):
        return ctx[self.name]


def Call(func, *args, **kwargs) -> Builder:
    """Call a function with the given arguments.

    Parameters
    ----------
    func
        The function to call.
    args
        The positional arguments to pass to the function.
    kwargs
        The keyword arguments to pass to the function.
    """
    if kwargs:
        return CallN(func, *args, **kwargs)
    elif len(args) == 0:
        return Call0(func)
    elif len(args) == 1:
        return Call1(func, *args)
    elif len(args) == 2:
        return Call2(func, *args)
    elif len(args) == 3:
        return Call3(func, *args)
    else:
        return CallN(func, *args)


@cython.final
@cython.cclass
class Call0(Builder):
    """Pattern that calls a function with no arguments.

    Parameters
    ----------
    func
        The function to call.
    """

    func: Builder

    def __init__(self, func):
        self.func = builder(func)

    def __repr__(self):
        return f"{self.func!r}()"

    def equals(self, other: Call0) -> bool:
        return self.func == other.func

    @cython.cfunc
    def build(self, ctx: Context):
        func = self.func.build(ctx)
        return func()


@cython.final
@cython.cclass
class Call1(Builder):
    """Pattern that calls a function with one argument.

    Parameters
    ----------
    func
        The function to call.
    arg
        The argument to pass to the function.
    """

    func: Builder
    arg: Builder

    def __init__(self, func, arg):
        self.func = builder(func)
        self.arg = builder(arg)

    def __repr__(self):
        return f"{self.func!r}({self.arg!r})"

    def equals(self, other: Call1) -> bool:
        return self.func == other.func and self.arg == other.arg

    @cython.cfunc
    def build(self, ctx: Context):
        func = self.func.build(ctx)
        arg = self.arg.build(ctx)
        return func(arg)


@cython.final
@cython.cclass
class Call2(Builder):
    """Pattern that calls a function with two arguments.

    Parameters
    ----------
    func
        The function to call.
    arg1
        The first argument to pass to the function.
    arg2
        The second argument to pass to the function.
    """

    func: Builder
    arg1: Builder
    arg2: Builder

    def __init__(self, func, arg1, arg2):
        self.func = builder(func)
        self.arg1 = builder(arg1)
        self.arg2 = builder(arg2)

    def __repr__(self):
        return f"{self.func!r}({self.arg1!r}, {self.arg2!r})"

    def equals(self, other: Call2) -> bool:
        return (
            self.func == other.func
            and self.arg1 == other.arg1
            and self.arg2 == other.arg2
        )

    @cython.cfunc
    def build(self, ctx: Context):
        func = self.func.build(ctx)
        arg1 = self.arg1.build(ctx)
        arg2 = self.arg2.build(ctx)
        return func(arg1, arg2)


@cython.final
@cython.cclass
class Call3(Builder):
    """Pattern that calls a function with three arguments.

    Parameters
    ----------
    func
        The function to call.
    arg1
        The first argument to pass to the function.
    arg2
        The second argument to pass to the function.
    arg3
        The third argument to pass to the function.
    """

    func: Builder
    arg1: Builder
    arg2: Builder
    arg3: Builder

    def __init__(self, func, arg1, arg2, arg3):
        self.func = builder(func)
        self.arg1 = builder(arg1)
        self.arg2 = builder(arg2)
        self.arg3 = builder(arg3)

    def __repr__(self):
        return f"{self.func!r}({self.arg1!r}, {self.arg2!r}, {self.arg3!r})"

    def equals(self, other: Call3) -> bool:
        return (
            self.func == other.func
            and self.arg1 == other.arg1
            and self.arg2 == other.arg2
            and self.arg3 == other.arg3
        )

    @cython.cfunc
    def build(self, ctx: Context):
        func = self.func.build(ctx)
        arg1 = self.arg1.build(ctx)
        arg2 = self.arg2.build(ctx)
        arg3 = self.arg3.build(ctx)
        return func(arg1, arg2, arg3)


@cython.final
@cython.cclass
class CallN(Builder):
    """Pattern that calls a function with the given arguments.

    Both positional and keyword arguments are coerced into patterns.

    Parameters
    ----------
    func
        The function to call.
    args
        The positional argument patterns.
    kwargs
        The keyword argument patterns.
    """

    func: Builder
    args: list[Builder]
    kwargs: dict[str, Builder]

    def __init__(self, func, *args, **kwargs):
        self.func = builder(func)
        self.args = [builder(arg) for arg in args]
        self.kwargs = {k: builder(v) for k, v in kwargs.items()}

    def __repr__(self):
        args = ", ".join(map(repr, self.args))
        kwargs = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        if self.args and self.kwargs:
            return f"{self.func!r}({args}, {kwargs})"
        elif self.args:
            return f"{self.func!r}({args})"
        elif self.kwargs:
            return f"{self.func!r}({kwargs})"
        else:
            return f"{self.func!r}()"

    def equals(self, other: CallN) -> bool:
        return (
            self.func == other.func
            and self.args == other.args
            and self.kwargs == other.kwargs
        )

    @cython.cfunc
    def build(self, ctx: Context):
        arg: Builder
        args: list[Any] = []
        kwargs: dict[str, Any] = {}

        func = self.func.build(ctx)
        for arg in self.args:
            args.append(arg.build(ctx))
        for key, arg in self.kwargs.items():
            kwargs[key] = arg.build(ctx)

        return func(*args, **kwargs)


_operator_symbols = {
    operator.add: "+",
    operator.sub: "-",
    operator.mul: "*",
    operator.truediv: "/",
    operator.floordiv: "//",
    operator.pow: "**",
    operator.mod: "%",
    operator.eq: "==",
    operator.ne: "!=",
    operator.lt: "<",
    operator.le: "<=",
    operator.gt: ">",
    operator.ge: ">=",
    operator.and_: "&",
    operator.or_: "|",
    operator.xor: "^",
    operator.rshift: ">>",
    operator.lshift: "<<",
    operator.inv: "~",
    operator.neg: "-",
    operator.invert: "~",
}


@cython.final
@cython.cclass
class Unop(Builder):
    """Pattern that applies a unary operator to a value.

    Parameters
    ----------
    op
        The unary operator to apply.
    arg
        The argument to apply the operator to.
    """

    op: Any
    arg: Builder

    def __init__(self, op: Any, arg: Any):
        self.op = op
        self.arg = builder(arg)

    def __repr__(self):
        symbol = _operator_symbols[self.op]
        return f"{symbol}{self.arg!r}"

    def equals(self, other: Unop) -> bool:
        return self.op == other.op and self.arg == other.arg

    @cython.cfunc
    def build(self, ctx: Context):
        arg = self.arg.build(ctx)
        return self.op(arg)


@cython.final
@cython.cclass
class Binop(Builder):
    """Pattern that applies a binary operator to two values.

    Parameters
    ----------
    op
        The binary operator to apply.
    arg1
        The left-hand side argument.
    arg2
        The right-hand side argument.
    """

    op: Any
    arg1: Builder
    arg2: Builder

    def __init__(self, op: Any, arg1: Any, arg2: Any):
        self.op = op
        self.arg1 = builder(arg1)
        self.arg2 = builder(arg2)

    def __repr__(self):
        symbol = _operator_symbols[self.op]
        return f"({self.arg1!r} {symbol} {self.arg2!r})"

    def equals(self, other: Binop) -> bool:
        return (
            self.op == other.op and self.arg1 == other.arg1 and self.arg2 == other.arg2
        )

    @cython.cfunc
    def build(self, ctx: Context):
        arg1 = self.arg1.build(ctx)
        arg2 = self.arg2.build(ctx)
        return self.op(arg1, arg2)


@cython.final
@cython.cclass
class Item(Builder):
    """Pattern that retrieves an item from a container.

    Parameters
    ----------
    container
        The container to retrieve the item from.
    key
        The key to retrieve from the container.
    """

    obj: Builder
    key: Builder

    def __init__(self, obj, key):
        self.obj = builder(obj)
        self.key = builder(key)

    def __repr__(self):
        return f"{self.obj!r}[{self.key!r}]"

    def equals(self, other: Item) -> bool:
        return self.obj == other.obj and self.key == other.key

    @cython.cfunc
    def build(self, ctx: Context):
        obj = self.obj.build(ctx)
        key = self.key.build(ctx)
        return obj[key]


@cython.final
@cython.cclass
class Attr(Builder):
    """Pattern that retrieves an attribute from an object.

    Parameters
    ----------
    obj
        The object to retrieve the attribute from.
    attr
        The attribute to retrieve from the object.
    """

    obj: Builder
    attr: str

    def __init__(self, obj: Any, attr: str):
        self.obj = builder(obj)
        self.attr = attr

    def __repr__(self):
        return f"{self.obj!r}.{self.attr}"

    def equals(self, other: Attr) -> bool:
        return self.obj == other.obj and self.attr == other.attr

    @cython.cfunc
    def build(self, ctx: Context):
        obj = self.obj.build(ctx)
        return getattr(obj, self.attr)


@cython.final
@cython.cclass
class Seq(Builder):
    """Pattern that constructs a sequence from the given items.

    Parameters
    ----------
    items
        The items to construct the sequence from.
    """

    type_: Any
    items: list[Builder]

    def __init__(self, items):
        self.type_ = type(items)
        self.items = [builder(item) for item in items]

    def __repr__(self):
        elems = ", ".join(map(repr, self.items))
        if self.type_ is tuple:
            return f"({elems})"
        elif self.type_ is list:
            return f"[{elems}]"
        else:
            return f"{self.type_.__name__}({elems})"

    def equals(self, other: Seq) -> bool:
        return self.type_ == other.type_ and self.items == other.items

    @cython.cfunc
    def build(self, ctx: Context):
        item: Builder
        result: list[Any] = []
        for item in self.items:
            result.append(item.build(ctx))
        return self.type_(result)


@cython.final
@cython.cclass
class Map(Builder):
    """Pattern that constructs a mapping from the given items.

    Parameters
    ----------
    items
        The items to construct the mapping from.
    """

    type_: Any
    items: dict[Any, Builder]

    def __init__(self, items):
        self.type_ = type(items)
        self.items = {k: builder(v) for k, v in items.items()}

    def __repr__(self):
        items = ", ".join(f"{k!r}: {v!r}" for k, v in self.items.items())
        if self.type_ is dict:
            return f"{{{items}}}"
        else:
            return f"{self.type_.__name__}({{{items}}})"

    def equals(self, other: Map) -> bool:
        return self.type_ == other.type_ and self.items == other.items

    @cython.cfunc
    def build(self, ctx: Context):
        k: Any
        v: Builder
        result: dict = {}
        for k, v in self.items.items():
            result[k] = v.build(ctx)
        return self.type_(result)


@cython.ccall
def builder(obj, allow_custom=False) -> Builder:
    if isinstance(obj, Deferred):
        return cython.cast(Deferred, obj)._builder
    elif isinstance(obj, Builder):
        return obj
    elif isinstance(obj, collections.abc.Mapping):
        # allow nesting deferred patterns in dicts
        return Map(obj)
    elif isinstance(obj, collections.abc.Sequence):
        # allow nesting deferred patterns in tuples/lists
        if isinstance(obj, (str, bytes)):
            return Just(obj)
        else:
            return Seq(obj)
    elif callable(obj) and allow_custom:
        # the object is used as a custom builder function
        return Func(obj)
    else:
        # the object is used as a constant value
        return Just(obj)


def deferred(obj, allow_custom=False) -> Deferred:
    return Deferred(builder(obj, allow_custom))


def resolve(obj, **context):
    bldr: Builder = builder(obj)
    return bldr.build(context)
