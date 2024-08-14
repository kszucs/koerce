from __future__ import annotations

from koerce import NoMatch, match, var


def test_capture_shorthand():
    a = var("a")
    b = var("b")

    ctx = {}
    assert match((~a, ~b), (1, 2), ctx) == (1, 2)
    assert ctx == {"a": 1, "b": 2}

    ctx = {}
    assert match((~a, a, a), (1, 2, 3), ctx) is NoMatch
    assert ctx == {"a": 1}

    ctx = {}
    assert match((~a, a, a), (1, 1, 1), ctx) == (1, 1, 1)
    assert ctx == {"a": 1}

    ctx = {}
    assert match((~a, a, a), (1, 1, 2), ctx) is NoMatch
    assert ctx == {"a": 1}


def test_namespace():
    pass
