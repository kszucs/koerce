from __future__ import annotations

import inspect
import pickle
from typing import Dict, Generic, List, Mapping, Optional, TypeVar, Union

import pytest
from typing_extensions import Self

from koerce.utils import (
    FrozenDict,
    PseudoHashable,
    RewindableIterator,
    get_type_boundvars,
    get_type_hints,
    get_type_params,
)

T = TypeVar("T", covariant=True)
S = TypeVar("S", covariant=True)
U = TypeVar("U", covariant=True)


class My(Generic[T, S, U]):
    a: T
    b: S
    c: str

    @property
    def d(self) -> Optional[str]: ...

    @property
    def e(self) -> U:  # type: ignore
        ...


class MyChild(My): ...


def example(a: int, b: str) -> str:  # type: ignore
    ...


def test_get_type_hints() -> None:
    hints = get_type_hints(My)
    assert hints == {"a": T, "b": S, "c": str}

    hints = get_type_hints(My, include_properties=True)
    assert hints == {"a": T, "b": S, "c": str, "d": Optional[str], "e": U}

    hints = get_type_hints(MyChild, include_properties=True)
    assert hints == {"a": T, "b": S, "c": str, "d": Optional[str], "e": U}

    # test that we don't actually mutate the My.__annotations__
    hints = get_type_hints(My)
    assert hints == {"a": T, "b": S, "c": str}

    hints = get_type_hints(example)
    assert hints == {"a": int, "b": str, "return": str}

    hints = get_type_hints(example, include_properties=True)
    assert hints == {"a": int, "b": str, "return": str}


def test_type_hints_with_fake_type():
    annots = inspect.get_annotations(My)
    hints = get_type_hints(annots, module=__name__)
    assert hints == {"a": T, "b": S, "c": str}

    annots = {"a": "Union[int, str]", "b": "Optional[str]"}
    hints = get_type_hints(annots, module=__name__)
    assert hints == {"a": Union[int, str], "b": Optional[str]}

    annots = {"a": "Union[int, Self]", "b": "Optional[Self]"}
    hints = get_type_hints(annots, module=__name__)
    assert hints == {"a": Union[int, Self], "b": Optional[Self]}


class A(Generic[T, S, U]):
    a: int
    b: str

    t: T
    s: S

    @property
    def u(self) -> U:  # type: ignore
        ...


class B(A[T, S, bytes]): ...


class C(B[T, str]): ...


class D(C[bool]): ...


class MyDict(Dict[T, U]): ...


class MyList(List[T]): ...


def test_get_type_params() -> None:
    # collecting all type params is done by GenericCoercedTo
    assert get_type_params(A[int, float, str]) == {"T": int, "S": float, "U": str}
    assert get_type_params(B[int, bool]) == {"T": int, "S": bool}
    assert get_type_params(C[int]) == {"T": int}
    assert get_type_params(D) == {}
    assert get_type_params(MyDict[int, str]) == {"T": int, "U": str}
    assert get_type_params(MyList[int]) == {"T": int}


def test_get_type_boundvars() -> None:
    expected = {
        "t": int,
        "s": float,
        "u": str,
    }
    assert get_type_boundvars(A[int, float, str]) == expected

    expected = {
        "t": int,
        "s": bool,
    }
    assert get_type_boundvars(B[int, bool]) == expected


def test_get_type_boundvars_unable_to_deduce() -> None:
    msg = "Unable to deduce corresponding attributes..."
    with pytest.raises(ValueError, match=msg):
        get_type_boundvars(MyDict[int, str])


def test_rewindable_iterator():
    it = RewindableIterator(range(10))
    assert next(it) == 0
    assert next(it) == 1
    with pytest.raises(ValueError, match="No checkpoint to rewind to"):
        it.rewind()

    it.checkpoint()
    assert next(it) == 2
    assert next(it) == 3
    it.rewind()
    assert next(it) == 2
    assert next(it) == 3
    assert next(it) == 4
    it.checkpoint()
    assert next(it) == 5
    assert next(it) == 6
    it.rewind()
    assert next(it) == 5
    assert next(it) == 6
    assert next(it) == 7
    it.rewind()
    assert next(it) == 5
    assert next(it) == 6
    assert next(it) == 7
    assert next(it) == 8
    assert next(it) == 9
    with pytest.raises(StopIteration):
        next(it)


def test_frozendict():
    d = FrozenDict({"a": 1, "b": 2, "c": 3})
    e = FrozenDict(a=1, b=2, c=3)
    f = FrozenDict(a=1, b=2, c=3, d=4)

    assert isinstance(d, Mapping)

    assert d == e
    assert d != f

    assert d["a"] == 1
    assert d["b"] == 2

    msg = "'FrozenDict' object does not support item assignment"
    with pytest.raises(TypeError, match=msg):
        d["a"] = 2
    with pytest.raises(TypeError, match=msg):
        d["d"] = 4

    assert hash(FrozenDict(a=1, b=2)) == hash(FrozenDict(b=2, a=1))
    assert hash(FrozenDict(a=1, b=2)) != hash(d)

    assert d == pickle.loads(pickle.dumps(d))


def test_pseudo_hashable():
    class Unhashable:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, Unhashable) and self.value == other.value

    class MyList(list):
        pass

    class MyMap(dict):
        pass

    for obj in [1, "a", b"a", 2.0, object(), (), frozenset()]:
        assert PseudoHashable(obj) is obj

    # test unhashable sequences
    lst1 = [1, 2, 3]
    lst2 = [1, 2, 4]
    lst3 = MyList([1, 2, 3])
    ph1 = PseudoHashable(lst1)
    ph2 = PseudoHashable(lst2)
    ph3 = PseudoHashable(lst3)
    ph4 = PseudoHashable(lst1.copy())

    assert hash(ph1) == hash(ph1)
    assert hash(ph1) != hash(ph2)
    assert hash(ph1) != hash(ph3)
    assert hash(ph2) != hash(ph3)
    assert hash(ph3) == hash(ph3)
    assert hash(ph1) == hash(ph4)
    assert ph1 == ph1
    assert ph1 != ph2
    assert ph1 == ph3
    assert ph2 != ph3
    assert ph3 == ph3
    assert ph1 == ph4

    # test unhashable mappings
    dct1 = {"a": 1, "b": 2}
    dct2 = {"a": 1, "b": 3}
    dct3 = MyMap({"a": 1, "b": 2})
    ph1 = PseudoHashable(dct1)
    ph2 = PseudoHashable(dct2)
    ph3 = PseudoHashable(dct3)
    ph4 = PseudoHashable(dct1.copy())

    assert hash(ph1) == hash(ph1)
    assert hash(ph1) != hash(ph2)
    assert hash(ph1) != hash(ph3)
    assert hash(ph2) != hash(ph3)
    assert hash(ph3) == hash(ph3)
    assert hash(ph1) == hash(ph4)
    assert ph1 == ph1
    assert ph1 != ph2
    assert ph1 == ph3
    assert ph2 != ph3
    assert ph3 == ph3
    assert ph1 == ph4

    # test unhashable objects
    obj1 = Unhashable(1)
    obj2 = Unhashable(1)
    obj3 = Unhashable(2)
    obj4 = Unhashable(1)
    ph1 = PseudoHashable(obj1)
    ph2 = PseudoHashable(obj2)
    ph3 = PseudoHashable(obj3)
    ph4 = PseudoHashable(obj4)

    assert hash(ph1) == hash(ph1)
    assert hash(ph1) != hash(ph2)
    assert hash(ph1) != hash(ph3)
    assert hash(ph2) != hash(ph3)
    assert hash(ph3) == hash(ph3)
    assert hash(ph1) != hash(ph4)
    assert ph1 == ph1
    assert ph1 == ph2
    assert ph1 != ph3
    assert ph2 != ph3
    assert ph3 == ph3
    assert ph1 == ph4
