from smaug import more_functools


def test_pipe():
    def f1(x):
        return x * 2

    def f2(x):
        return x + 4

    def f3(x):
        return x * 3

    pipe_func = more_functools.pipe(f1, f2, f3)

    x = 1
    assert f3(f2(f1(x))) == pipe_func(x)
    assert f1(f2(f3(x))) == more_functools.pipe(f3, f2, f1)(x)
