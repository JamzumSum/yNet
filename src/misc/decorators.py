from functools import wraps


def autoProperty(func, strict: dict = None):
    assert func.__name__ == "__init__"
    parg_name = func.__code__.co_varnames[1:]
    arg_name: list = parg_name[:func.__code__.co_argcount - 1]
    if (i := func.__defaults__) is None:
        defaults = {}
    else:
        defaults = {k: v for k, v in zip(arg_name[::-1], i)}
    if (i := func.__kwdefaults__) is not None:
        defaults.update(i)

    @wraps(func)
    def initWrapper(self, *args, **kwargs):
        annotation = None if strict is None else strict.copy()
        d = kwargs.copy()
        d.update(zip(parg_name, args))

        for k, v in d.items():
            if annotation is None or k in annotation:
                setattr(self, k, v)
                if annotation:
                    annotation.pop(k)
        if annotation:
            for i in annotation:
                setattr(self, i, defaults[i])
        func(self, *args, **kwargs)

    return initWrapper


def autoPropertyClass(cls, strict=True):
    cls.__init__ = autoProperty(cls.__init__, cls.__annotations__ if strict else None)
    return cls


def noexcept(func):
    @wraps(func)
    def noexceptwrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            from traceback import print_exc
            print_exc()
            raise e

    return noexceptwrapper