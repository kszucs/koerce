from __future__ import annotations

import operator

import pytest

from koerce.builders import (
    Attr,
    Binop,
    Call,
    Call0,
    Call1,
    Call2,
    Call3,
    CallN,
    Custom,
    Deferred,
    Item,
    Just,
    Mapping,
    Sequence,
    Unop,
    Variable,
    builder,
)

_ = Deferred(Variable("_"))


def test_builder():
    class MyClass:
        pass

    def fn(x):
        return x + 1

    assert builder(1) == Just(1)
    assert builder(Just(1)) == Just(1)
    assert builder(Just(Just(1))) == Just(1)
    assert builder(MyClass) == Just(MyClass)
    assert builder(fn) == Just(fn)
    assert builder(()) == Sequence(())
    assert builder((1, 2, _)) == Sequence((Just(1), Just(2), _))
    assert builder({}) == Mapping({})
    assert builder({"a": 1, "b": _}) == Mapping({"a": Just(1), "b": _})
    assert builder("string") == Just("string")

    # assert builder(var("x")) == Variable("x")
    # assert builder(Variable("x")) == Variable("x")


def test_builder_just():
    p = Just(1)
    assert p.apply({}) == 1
    assert p.apply({"a": 1}) == 1

    # unwrap subsequently nested Just instances
    assert Just(p) == p

    # disallow creating a Just builder from other builders or deferreds
    # with pytest.raises(TypeError, match="cannot be used as a Just value"):
    #     Just(_)
    # with pytest.raises(TypeError, match="cannot be used as a Just value"):
    #     Just(Factory(lambda _: _))


def test_builder_variable():
    p = Variable("other")
    context = {"other": 10}
    assert p.apply(context) == 10


def test_builder_custom():
    f = Custom(lambda _: _ + 1)
    assert f.apply({"_": 1}) == 2
    assert f.apply({"_": 2}) == 3

    def fn(**kwargs):
        assert kwargs == {"_": 10, "a": 5}
        return -1

    f = Custom(fn)
    assert f.apply({"_": 10, "a": 5}) == -1


def test_builder_call():
    def fn(*args):
        return args

    def func(a, b, c=1):
        return a + b + c

    c = Call0(Just(fn))
    assert c.apply({}) == ()

    c = Call1(Just(fn), Just(1))
    assert c.apply({}) == (1,)

    c = Call2(Just(fn), Just(1), Just(2))
    assert c.apply({}) == (1, 2)

    c = Call3(Just(fn), Just(1), Just(2), Just(3))
    assert c.apply({}) == (1, 2, 3)

    c = Call(Just(func), Just(1), Just(2), c=Just(3))
    assert isinstance(c, CallN)
    assert c.apply({}) == 6

    c = Call(Just(func), Just(-1), Just(-2))
    assert isinstance(c, Call2)
    assert c.apply({}) == -2

    c = Call(Just(dict), a=Just(1), b=Just(2))
    assert isinstance(c, CallN)
    assert c.apply({}) == {"a": 1, "b": 2}

    c = Call(Just(float), Just("1.1"))
    assert isinstance(c, Call1)
    assert c.apply({}) == 1.1

    c = Call(Just(list))
    assert isinstance(c, Call0)
    assert c.apply({}) == []


def test_builder_unop():
    b = Unop(operator.neg, Just(1))
    assert b.apply({}) == -1

    b = Unop(operator.abs, Just(-1))
    assert b.apply({}) == 1


def test_builder_binop():
    b = Binop(operator.add, Just(1), Just(2))
    assert b.apply({}) == 3

    b = Binop(operator.mul, Just(2), Just(3))
    assert b.apply({}) == 6


def test_builder_attr():
    class MyType:
        def __init__(self, a, b):
            self.a = a
            self.b = b

        def __hash__(self):
            return hash((type(self), self.a, self.b))

    v = Variable("v")
    b = Attr(v, "b")
    assert b.apply({"v": MyType(1, 2)}) == 2

    b = Attr(Just(MyType(1, 2)), "a")
    assert b.apply({}) == 1

    # name is not allowed to be a deferred
    b = Attr(v, "a")
    assert b.apply({"v": MyType(1, 2)}) == 1


def test_builder_item():
    v = Variable("v")
    b = Item(v, Just(1))
    assert b.apply({"v": [1, 2, 3]}) == 2

    b = Item(Just(dict(a=1, b=2)), Just("a"))
    assert b.apply({}) == 1

    name = Variable("name")
    # test that name can be a deferred as well
    b = Item(v, name)
    assert b.apply({"v": {"a": 1, "b": 2}, "name": "b"}) == 2


def test_builder_sequence():
    b = Sequence([Just(1), Just(2), Just(3)])
    assert b.apply({}) == [1, 2, 3]

    b = Sequence((Just(1), Just(2), Just(3)))
    assert b.apply({}) == (1, 2, 3)


def test_builder_mapping():
    b = Mapping({"a": Just(1), "b": Just(2)})
    assert b.apply({}) == {"a": 1, "b": 2}

    b = Mapping({"a": Just(1), "b": Just(2)})
    assert b.apply({}) == {"a": 1, "b": 2}


## Deferred tests


def resolve(deferred, _):
    return builder(deferred).apply({"_": _})


def test_deferred_object_are_not_hashable():
    # since __eq__ is overloaded, Deferred objects are not hashable
    with pytest.raises(TypeError, match="unhashable type"):
        hash(_.a)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ((), ()),
        ([], []),
        ({}, {}),
        ((1, 2, 3), (1, 2, 3)),
        ([1, 2, 3], [1, 2, 3]),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
    ],
)
def test_deferred_builds(value, expected):
    assert builder(value).apply({}) == expected


def test_deferred_supports_string_arguments():
    # deferred() is applied on all arguments of Call() except the first one and
    # sequences are transparently handled, the check far sequences was incorrect
    # for strings causing infinite recursion
    b = builder("3.14")
    assert b.apply({}) == "3.14"


def test_deferred_variable_getattr():
    v = Deferred(Variable("v"))
    p = v.copy
    assert builder(p) == Attr(v, "copy")
    assert builder(p).apply({"v": [1, 2, 3]})() == [1, 2, 3]

    p = v.copy()
    assert builder(p) == Call(Attr(v, "copy"))
    assert builder(p).apply({"v": [1, 2, 3]}) == [1, 2, 3]


class TableMock(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __eq__(self, other):
        return isinstance(other, TableMock) and super().__eq__(other)


def _binop(name, switch=False):
    def method(self, other):
        if switch:
            return BinaryMock(name=name, left=other, right=self)
        else:
            return BinaryMock(name=name, left=self, right=other)

    return method


class ValueMock:
    def log(self, base=None):
        return UnaryMock(name="log", arg=base)

    def sum(self):
        return UnaryMock(name="sum", arg=self)

    def __neg__(self):
        return UnaryMock(name="neg", arg=self)

    def __invert__(self):
        return UnaryMock(name="invert", arg=self)

    __lt__ = _binop("lt")
    __gt__ = _binop("gt")
    __le__ = _binop("le")
    __ge__ = _binop("ge")
    __add__ = _binop("add")
    __radd__ = _binop("add", switch=True)
    __sub__ = _binop("sub")
    __rsub__ = _binop("sub", switch=True)
    __mul__ = _binop("mul")
    __rmul__ = _binop("mul", switch=True)
    __mod__ = _binop("mod")
    __rmod__ = _binop("mod", switch=True)
    __truediv__ = _binop("div")
    __rtruediv__ = _binop("div", switch=True)
    __floordiv__ = _binop("floordiv")
    __rfloordiv__ = _binop("floordiv", switch=True)
    __rshift__ = _binop("shift")
    __rrshift__ = _binop("shift", switch=True)
    __lshift__ = _binop("shift")
    __rlshift__ = _binop("shift", switch=True)
    __pow__ = _binop("pow")
    __rpow__ = _binop("pow", switch=True)
    __xor__ = _binop("xor")
    __rxor__ = _binop("xor", switch=True)
    __and__ = _binop("and")
    __rand__ = _binop("and", switch=True)
    __or__ = _binop("or")
    __ror__ = _binop("or", switch=True)


class ColumnMock(ValueMock):
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype

    def __eq__(self, other):
        return (
            isinstance(other, ColumnMock)
            and self.name == other.name
            and self.dtype == other.dtype
        )

    def __deferred_repr__(self):
        return f"<column[{self.dtype}]>"


class UnaryMock(ValueMock):
    def __init__(self, name, arg):
        self.name = name
        self.arg = arg

    def __eq__(self, other):
        return (
            isinstance(other, UnaryMock)
            and self.name == other.name
            and self.arg == other.arg
        )


class BinaryMock(ValueMock):
    def __init__(self, name, left, right):
        self.name = name
        self.left = left
        self.right = right

    def __eq__(self, other):
        return (
            isinstance(other, BinaryMock)
            and self.name == other.name
            and self.left == other.left
            and self.right == other.right
        )


@pytest.fixture
def table():
    return TableMock(
        a=ColumnMock(name="a", dtype="int"),
        b=ColumnMock(name="b", dtype="int"),
        c=ColumnMock(name="c", dtype="string"),
    )


def test_deferred_getitem(table):
    expr = _["a"]
    assert resolve(expr, table) == table["a"]
    assert repr(expr) == "$_['a']"


def test_deferred_getattr(table):
    expr = _.a
    assert resolve(expr, table) == table.a
    assert repr(expr) == "$_.a"


def test_deferred_call(table):
    expr = Deferred(Call(operator.add, _.a, 2))
    res = resolve(expr, table)
    assert res == table.a + 2
    assert repr(expr) == "add($_.a, 2)"

    func = lambda a, b: a + b
    expr = Deferred(Call(func, a=_.a, b=2))
    res = resolve(expr, table)
    assert res == table.a + 2
    assert func.__name__ in repr(expr)
    assert "a=$_.a, b=2" in repr(expr)

    expr = Deferred(Call(operator.add, (_.a, 2)))
    assert repr(expr) == "add(($_.a, 2))"


def test_deferred_method(table):
    expr = _.a.log()
    res = resolve(expr, table)
    assert res == table.a.log()
    assert repr(expr) == "$_.a.log()"


def test_deferred_method_with_args(table):
    expr = _.a.log(1)
    res = resolve(expr, table)
    assert res == table.a.log(1)
    assert repr(expr) == "$_.a.log(1)"

    expr = _.a.log(_.b)
    res = resolve(expr, table)
    assert res == table.a.log(table.b)
    assert repr(expr) == "$_.a.log($_.b)"


def test_deferred_method_with_kwargs(table):
    expr = _.a.log(base=1)
    res = resolve(expr, table)
    assert res == table.a.log(base=1)
    assert repr(expr) == "$_.a.log(base=1)"

    expr = _.a.log(base=_.b)
    res = resolve(expr, table)
    assert res == table.a.log(base=table.b)
    assert repr(expr) == "$_.a.log(base=$_.b)"


def test_deferred_apply(table):
    expr = Deferred(Call(operator.add, _.a, 2))
    res = resolve(expr, table)
    assert res == table.a + 2
    assert repr(expr) == "add($_.a, 2)"

    func = lambda a, b: a + b
    expr = Deferred(Call(func, _.a, 2))
    res = resolve(expr, table)
    assert res == table.a + 2
    assert func.__name__ in repr(expr)


@pytest.mark.parametrize(
    "symbol, op",
    [
        ("+", operator.add),
        ("-", operator.sub),
        ("*", operator.mul),
        ("/", operator.truediv),
        ("//", operator.floordiv),
        ("**", operator.pow),
        ("%", operator.mod),
        ("&", operator.and_),
        ("|", operator.or_),
        ("^", operator.xor),
        (">>", operator.rshift),
        ("<<", operator.lshift),
    ],
)
def test_deferred_binary_operations(symbol, op, table):
    expr = op(_.a, _.b)
    sol = op(table.a, table.b)
    res = resolve(expr, table)
    assert res == sol
    assert repr(expr) == f"($_.a {symbol} $_.b)"

    expr = op(1, _.a)
    sol = op(1, table.a)
    res = resolve(expr, table)
    assert res == sol
    assert repr(expr) == f"(1 {symbol} $_.a)"


@pytest.mark.parametrize(
    "sym, rsym, op",
    [
        ("==", "==", operator.eq),
        ("!=", "!=", operator.ne),
        ("<", ">", operator.lt),
        ("<=", ">=", operator.le),
        (">", "<", operator.gt),
        (">=", "<=", operator.ge),
    ],
)
def test_deferred_compare_operations(sym, rsym, op, table):
    expr = op(_.a, _.b)
    sol = op(table.a, table.b)
    res = resolve(expr, table)
    assert res == sol
    assert repr(expr) == f"($_.a {sym} $_.b)"

    expr = op(1, _.a)
    sol = op(1, table.a)
    res = resolve(expr, table)
    assert res == sol
    assert repr(expr) == f"($_.a {rsym} 1)"


@pytest.mark.parametrize(
    "symbol, op",
    [
        ("-", operator.neg),
        ("~", operator.invert),
    ],
)
def test_deferred_unary_operations(symbol, op, table):
    expr = op(_.a)
    sol = op(table.a)
    res = resolve(expr, table)
    assert res == sol
    assert repr(expr) == f"{symbol}$_.a"


@pytest.mark.parametrize("obj", [_, _.a, _.a.b[0]])
def test_deferred_is_not_iterable(obj):
    with pytest.raises(TypeError, match="object is not iterable"):
        sorted(obj)

    with pytest.raises(TypeError, match="object is not iterable"):
        iter(obj)

    with pytest.raises(TypeError, match="is not an iterator"):
        next(obj)


@pytest.mark.parametrize("obj", [_, _.a, _.a.b[0]])
def test_deferred_is_not_truthy(obj):
    with pytest.raises(
        TypeError, match="The truth value of Deferred objects is not defined"
    ):
        bool(obj)
