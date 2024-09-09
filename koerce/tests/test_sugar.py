from __future__ import annotations

from dataclasses import dataclass

from koerce import (
    As,
    Call,
    Deferred,
    Is,
    NoMatch,
    Object,
    builder,
    match,
    namespace,
    pattern,
    replace,
    var,
)


def test_match_strictness():
    assert pattern(int, allow_coercion=False) == Is(int)
    assert pattern(int, allow_coercion=True) == As(int)

    assert match(int, 1, allow_coercion=False) == 1
    assert match(int, 1.1, allow_coercion=False) is NoMatch
    # not lossless
    assert match(int, 1.1, allow_coercion=True) is NoMatch

    # default is allow_coercion=False
    assert match(int, 1.1) is NoMatch


def test_capture_shorthand():
    a = var("a")
    b = var("b")

    ctx = {}
    assert match((+a, +b), (1, 2), ctx) == (1, 2)
    assert ctx == {"a": 1, "b": 2}

    ctx = {}
    assert match((+a, a, a), (1, 2, 3), ctx) is NoMatch
    assert ctx == {"a": 1}

    ctx = {}
    assert match((+a, a, a), (1, 1, 1), ctx) == (1, 1, 1)
    assert ctx == {"a": 1}

    ctx = {}
    assert match((+a, a, a), (1, 1, 2), ctx) is NoMatch
    assert ctx == {"a": 1}


@dataclass
class Point:
    x: int
    y: int


def test_namespace():
    p, d = namespace(__name__)
    assert p.Point == Is(Point)
    assert p.Point(1, 2) == Object(Point, 1, 2)

    point_deferred = d.Point(1, 2)
    point_builder = builder(point_deferred)
    assert isinstance(point_deferred, Deferred)
    assert point_builder == Call(Point, 1, 2)


def test_replace_decorator():
    @replace(int)
    def sub(_):
        return _ - 1

    assert match(sub, 1) == 0
    assert match(sub, 2) == 1
