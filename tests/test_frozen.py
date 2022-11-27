from smaug import frozen


def test_frozen_list():
    a = frozen.frozenlist()
    assert len(a) == 0
    assert 1 not in a

    a = a.append(1)
    assert len(a) == 1
    assert 1 in a

    b = frozen.frozenlist((2, 3))
    assert len(b) == 2
    assert 2 in b
    assert 3 in b

    c = a + b
    assert len(c) == 3
    assert c[0] == 1
    assert c[1] == 2
    assert c[2] == 3

    d, val = c.pop()
    assert val == 3
    assert len(d) == 2
    assert d[0] == 1
    assert d[1] == 2

    e, val = c.pop(1)
    assert val == 2
    assert len(e) == 2
    assert e[0] == 1
    assert e[1] == 3
