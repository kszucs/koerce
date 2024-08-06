from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

import ibis
from coercer import NoMatch, dict_of, instance_of
from pydantic_core import SchemaValidator
from koerce.sugar import annotated
from typing_extensions import TypeVar
import msgspec
from koerce.patterns import InstanceOf, ObjectOf, ObjectOfN, PatternMap

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

# V = dict_of({"name": str, "age": int, "is_developer": bool})

data = {"name": "Samuel", "age": 35, "is_developer": True, "has_children": False}


ITS = 50


def test_pydantic(benchmark):
    r1 = benchmark.pedantic(
        v.validate_python, args=(data,), iterations=ITS, rounds=20000
    )
    assert r1 == data


def test_koerce(benchmark):
    pat = PatternMap(
        {"name": str, "age": int, "is_developer": bool, "has_children": bool}
    )
    r2 = benchmark.pedantic(pat.apply, args=(data, {}), iterations=ITS, rounds=20000)
    assert r2 == data


def test_coercer(benchmark):
    pat = dict_of({"name": str, "age": int, "is_developer": bool, "has_children": bool})
    r2 = benchmark.pedantic(pat.apply, args=(data, {}), iterations=ITS, rounds=20000)
    assert r2 == data


class MPerson(msgspec.Struct):
    name: str
    age: int
    is_developer: bool = True
    has_children: bool = False


# def test_object_msgspec(benchmark):
#     m = MPerson(name="Samuel", age=35, is_developer=True, has_children=False)
#     r2 = benchmark.pedantic(msgspec.convert, args=(m, MPerson), iterations=ITS, rounds=20000)
#     #assert r2 == MPerson(name="Samuel", age=35, is_developer=True, has_children=False)

# def test_object_koerce(benchmark):
#     pat = ObjectOf(Person, name=str, age=int, is_developer=bool, has_children=bool)
#     r2 = benchmark.pedantic(pat.apply, args=(p, {}), iterations=ITS, rounds=20000)
#     #assert r2 == Person(name="Samuel", age=35, is_developer=True, has_children=False)

########################################


def test_instance_of():
    pattern = instance_of(int)
    assert pattern.apply(1) == 1
    assert pattern.apply(4.0) is NoMatch


# test pydantic validate_call

# from pydantic import validate_call


# @validate_call(validate_return=True)
# def foo(x: int, *, y: str) -> float:
#     return float(x) + float(y)


# @annotated
# def bar(x: int, y: str) -> float:
#     return float(x) + float(y)


# def test_e():
#     foo(1, y="3.14")


# def test_call_pydantic(benchmark):
#     assert foo(1, "2") == 3.0
#     benchmark.pedantic(foo, args=(1, "2"), iterations=ITS, rounds=20000)


# def test_call_annotated(benchmark):
#     assert bar(1, "2") == 3.0
#     benchmark.pedantic(bar, args=(1, "2"), iterations=ITS, rounds=20000)


# def test_ibis(benchmark):
#     pattern = ibis.common.patterns.Object(Person, str, int, bool)
#     r = benchmark.pedantic(pattern.match, args=(p, {}), iterations=ITS, rounds=20000)
#     assert r == p


# def test_koerce(benchmark):
#     pattern = ObjectOf(Person, str, int, bool)
#     r = benchmark.pedantic(pattern.apply, args=(p, {}), iterations=ITS, rounds=20000)
#     assert r == p
