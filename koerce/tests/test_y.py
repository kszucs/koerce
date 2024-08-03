from __future__ import annotations

from dataclasses import dataclass

# from koerce.matchers import InstanceOf as InstanceOf2
from ibis.common.patterns import Object
from pydantic_core import SchemaValidator, core_schema

from koerce.patterns import InstanceOf, ObjectOf


@dataclass
class Person:
    name: str
    age: int
    is_developer: bool = True
    has_children: bool = False

schema = core_schema.dataclass_schema(
    Person,
    core_schema.dataclass_args_schema(
        'Person',
        [
            core_schema.dataclass_field(name='name', schema=core_schema.str_schema()),
            core_schema.dataclass_field(name='age', schema=core_schema.int_schema()),
            core_schema.dataclass_field(name='is_developer', schema=core_schema.bool_schema()),
            core_schema.dataclass_field(name='has_children', schema=core_schema.bool_schema())
        ],
    ),
    ['name', 'age', 'is_developer', 'has_children'],
)
#s = SchemaSerializer(schema)

v = SchemaValidator(schema)

p = Person(name='Samuel', age=35, is_developer=True, has_children=False)


def test_pydantic(benchmark):
    r1 = benchmark(v.validate_python, p)
    assert r1 == p


def test_koerce(benchmark):
    pat = ObjectOf(Person, str, int, bool, bool)
    r2 = benchmark(pat.apply, p, {})
    assert r2 == p

def test_ibis(benchmark):
    pat = Object(Person, str, int, bool, bool)
    r3 = benchmark(pat.match, p, {})
    assert r3 == p


# def test_patterns_instanceof(benchmark):
#     pat = InstanceOf(str)
#     assert pat.apply('hello') == 'hello'
#     benchmark(pat.apply, "hello")


# def test_patterns_instanceof2(benchmark):
#     pat = InstanceOf2(str)
#     assert pat.apply('hello') == 'hello'
#     benchmark(pat.apply, "hello")

