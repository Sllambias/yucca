def without_keys(dict: {}, keys: []) -> {}:
    if keys == [] or keys is None:
        return dict
    return {k: v for k, v in dict.items() if k not in keys}
