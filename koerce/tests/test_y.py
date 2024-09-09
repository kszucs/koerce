from __future__ import annotations

from dataclasses import dataclass
from inspect import Signature as InspectSignature
from typing import Generic

import pytest

pydantic = pytest.importorskip("pydantic")
msgspec = pytest.importorskip("msgspec")

# from ibis.common.grounds import Annotable as IAnnotable
from pydantic import BaseModel, validate_call
from pydantic_core import SchemaValidator
from typing_extensions import TypeVar

from koerce import (
    Annotable,
    PatternMap,
    Signature,
    annotated,
)

T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")


class A(Generic[T, S, U]):
    a: int
    b: str

    t: T
    s: S

    @property
    def u(self) -> U:  # type: ignore
        ...


@dataclass
class Person:
    name: str
    age: int
    is_developer: bool = True
    has_children: bool = False


v = SchemaValidator(
    {
        "type": "typed-dict",
        "fields": {
            "name": {
                "type": "typed-dict-field",
                "schema": {
                    "type": "str",
                },
            },
            "age": {
                "type": "typed-dict-field",
                "schema": {
                    "type": "int",
                },
            },
            "is_developer": {
                "type": "typed-dict-field",
                "schema": {
                    "type": "bool",
                },
            },
            "has_children": {
                "type": "typed-dict-field",
                "schema": {
                    "type": "bool",
                },
            },
        },
    }
)

p = Person(name="Samuel", age=35, is_developer=True, has_children=False)

pmap_data = {"name": "Samuel", "age": 35, "is_developer": True, "has_children": False}


ITS = 50


def test_patternmap_pydantic(benchmark):
    r1 = benchmark.pedantic(
        v.validate_python, args=(pmap_data,), iterations=ITS, rounds=20000
    )
    assert r1 == pmap_data


def test_patternmap_koerce(benchmark):
    pat = PatternMap(
        {"name": str, "age": int, "is_developer": bool, "has_children": bool}
    )
    r2 = benchmark.pedantic(
        pat.apply, args=(pmap_data, {}), iterations=ITS, rounds=20000
    )
    assert r2 == pmap_data


def func(x: int, y: str, *args: int, z: float = 3.14, **kwargs) -> float: ...


args = (1, "a", 2, 3, 4)
kwargs = dict(z=3.14, w=5, q=6)
expected = {"x": 1, "y": "a", "args": (2, 3, 4), "z": 3.14, "kwargs": {"w": 5, "q": 6}}


def test_signature_stdlib(benchmark):
    sig = InspectSignature.from_callable(func)
    r = benchmark.pedantic(
        sig.bind, args=args, kwargs=kwargs, iterations=ITS, rounds=20000
    )
    assert r.arguments == expected


def test_signature_koerce(benchmark):
    sig = Signature.from_callable(func)
    r = benchmark.pedantic(sig.bind, args=(args, kwargs), iterations=ITS, rounds=20000)
    assert r == expected


@validate_call
def prepeat(s: str, count: int, *, separator: bytes = b"") -> bytes:
    return b""


@annotated
def krepeat(s: str, count: int, *, separator: bytes = b"") -> bytes:
    return b""


def test_validated_call_pydantic(benchmark):
    r1 = benchmark.pedantic(
        prepeat,
        args=("hello", 3),
        kwargs={"separator": b" "},
        iterations=ITS,
        rounds=20000,
    )
    assert r1 == b""


def test_validated_call_annotated(benchmark):
    r2 = benchmark.pedantic(
        krepeat,
        args=("hello", 3),
        kwargs={"separator": b" "},
        iterations=ITS,
        rounds=20000,
    )
    assert r2 == b""


class PUser(BaseModel):
    id: int
    name: str = "Jane Doe"
    age: int | None = None
    children: list[str] = []


class KUser(Annotable):
    id: int
    name: str = "Jane Doe"
    age: int | None = None
    children: list[str] = []


class MUser(msgspec.Struct):
    id: int
    name: str = "Jane Doe"
    age: int | None = None
    children: list[str] = []


data = {"id": 1, "name": "Jane Doe", "age": None, "children": []}


def test_pydantic(benchmark):
    r1 = benchmark.pedantic(
        PUser,
        args=(),
        kwargs=data,
        iterations=ITS,
        rounds=20000,
    )
    assert r1 == PUser(id=1, name="Jane Doe", age=None, children=[])


def test_msgspec(benchmark):
    r1 = benchmark.pedantic(
        msgspec.convert,
        args=(data, MUser),
        kwargs={},
        iterations=ITS,
        rounds=20000,
    )
    assert r1 == MUser(id=1, name="Jane Doe", age=None, children=[])


def test_annotated(benchmark):
    r2 = benchmark.pedantic(
        KUser,
        args=(),
        kwargs=data,
        iterations=ITS,
        rounds=20000,
    )
    assert r2 == KUser(id=1, name="Jane Doe", age=None, children=[])


# def test_ibis(benchmark):
#     r2 = benchmark.pedantic(
#         IUser,
#         args=(),
#         kwargs={"id": 1, "name": "Jane Doe", "age": None, "children": ()},
#         iterations=ITS,
#         rounds=20000,
#     )
#     assert r2 == IUser(id=1, name="Jane Doe", age=None, children=[])
