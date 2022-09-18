import functools


def make(f):
    """Decorates function to return transform processor."""

    def make_processor(*args, **kwargs):
        def processor(stream):
            return f(stream, *args, **kwargs)

        if hasattr(f, "__post_processors__"):
            processor.__post_processors__ = f.__post_processors__

        return processor

    return functools.update_wrapper(make_processor, f)


def post_run(make_proc_cmd, **kwargs):
    """Registers a processor to after the function.

    Post processors should be executed after this function.
    """

    def decorator(f):
        if not hasattr(f, "__post_processors__"):
            f.__post_processors__ = []
        # Insert in position 0 since decorators are evaluated
        # in reverse order (from bottom to top) thus ensuring
        # the post processors appear in the right order.
        f.__post_processors__.insert(0, (make_proc_cmd, kwargs))
        return f

    return decorator


def call(ctx, processor, stream, post_run=True):
    stream = processor(stream)

    if post_run and hasattr(processor, "__post_processors__"):
        for make_proc_cmd, kwargs in processor.__post_processors__:
            post_processor = ctx.invoke(make_proc_cmd, **kwargs)
            stream = post_processor(stream)

    return stream
