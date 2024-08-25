from __future__ import annotations

from koerce import NoMatch, match, var, pattern, As, Is


def test_match_strictness():
    assert pattern(int, allow_coercion=False) == Is(int)
    assert pattern(int, allow_coercion=True) == As(int)

    assert match(int, 1, allow_coercion=False) == 1
    assert match(int, 1.1, allow_coercion=False) is NoMatch
    assert match(int, 1.1, allow_coercion=True) == 1

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


def test_namespace():
    pass
