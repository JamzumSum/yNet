class switch:
    def __init__(self, key):
        self._d = {}
        self._key = key

    def case(self, key=None, func=None):
        if func: 
            self._register(key, func)
            return func
        def wrapper(func):
            self._register(key, func)
            return func
        return wrapper

    def _register(self, key, func):
        if key is None: key = func.__name__
        assert key not in self._d
        self._d[key] = func

    def default(self, func):
        assert not hasattr(self, '_default')
        self._default = func
        return func

    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        if self._key in self._d:
            return self._d[self._key]()
        else:
            return self._default()