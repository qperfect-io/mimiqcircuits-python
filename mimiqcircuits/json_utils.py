
from mimiqcircuits.gates import Gate
from mimiqcircuits.barrier import Barrier


def operation_from_json(d):
    name = d['name']

    if name == "Barrier":
        return Barrier.from_json(d)

    return Gate.from_json(d)
