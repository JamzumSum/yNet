from functools import wraps

class KeyboardInterruptWrapper:
    def __init__(self, solution):
        self.s = solution

    def __call__(this, func):
        @wraps(func)
        def wrapped(self, *args, **kwargs):
            try: return func(self, *args, **kwargs)
            except KeyboardInterrupt:
                this.s(self)
        return wrapped
