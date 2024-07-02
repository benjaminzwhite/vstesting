from types import SimpleNamespace

# TODO: REWRITE YOUR OWN, THIS IS FROM S.O.
def parse_config(data):
    if type(data) is list:
        return list(map(parse_config, data))
    elif type(data) is dict:
        sns = SimpleNamespace()
        for key, value in data.items():
            setattr(sns, key, parse_config(value))
        return sns
    else:
        return data