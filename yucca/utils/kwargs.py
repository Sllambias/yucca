import inspect


def filter_kwargs(function, kwargs):
    sig = inspect.signature(function).parameters.keys()
    return {key: value for key, value in kwargs.items() if key in sig}


def getattr_for_kwargs(obj, kwargs):
    return {key: getattr(obj, key) for key in kwargs.keys()}
