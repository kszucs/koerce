from __future__ import annotations
import pytest

from typing import ForwardRef, Generic, Optional, Union, List, Dict

from typing_extensions import TypeVar
from koerce.utils import get_type_params, get_type_boundvars, get_type_hints


T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")


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
    # assert get_type_params(B[int, bool]) == {"T": int, "S": bool, "U": bytes}
    # assert get_type_params(C[int]) == {"T": int, "S": str, "U": bytes}
    # assert get_type_params(D) == {"T": bool, "S": str, "U": bytes}
    assert get_type_params(A[int, float, str]) == {"T": int, "S": float, "U": str}
    assert get_type_params(B[int, bool]) == {"T": int, "S": bool}
    assert get_type_params(C[int]) == {"T": int}
    assert get_type_params(D) == {}
    assert get_type_params(MyDict[int, str]) == {"T": int, "U": str}
    assert get_type_params(MyList[int]) == {"T": int}


# def test_get_type_boundvars() -> None:
#     expected = {
#         T: ("t", int),
#         S: ("s", float),
#         U: ("u", str),
#     }
#     assert get_type_boundvars(A[int, float, str]) == expected

#     expected = {
#         T: ("t", int),
#         S: ("s", bool),
#         U: ("u", bytes),
#     }
#     assert get_type_boundvars(B[int, bool]) == expected


def test_get_type_boundvars_unable_to_deduce() -> None:
    msg = "Unable to deduce corresponding type attributes..."
    with pytest.raises(ValueError, match=msg):
        get_type_boundvars(MyDict[int, str])
