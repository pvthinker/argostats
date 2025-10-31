import json


def jsonencoder(dico):
    c = [f'"{k}":\n'+indent(jsonencoder(v), 4)
         if isinstance(v, dict)
         else
         json.dumps({k: v})[1:-1]
         for k, v in dico.items()
         ]
    return "{\n"+",\n".join(c)+"\n}"


def indent(string, width):
    return "\n".join([" "*width+s for s in string.split("\n")])


def tuplify(thing):
    if isinstance(thing, list) or isinstance(thing, tuple):
        return tuple([tuplify(x) for x in thing])

    elif isinstance(thing, dict):
        return {k: tuplify(v) for k, v in thing.items()}

    else:
        return thing


def test_indent():
    assert indent("hello\nworld", 2) == '  hello\n  world'
