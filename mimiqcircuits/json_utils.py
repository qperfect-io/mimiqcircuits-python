
from mimiqcircuits.gates import Gate
from mimiqcircuits.barrier import Barrier
from mimiqcircuits.measure import Measure
from mimiqcircuits.reset import Reset

def operation_from_json(d):
    name = d['name']

    if name == "Barrier":
        return Barrier.from_json(d)

    elif name == "Measure":
        return Measure.from_json(d)

    elif name == "Reset":
        return Reset.from_json(d)

    return Gate.from_json(d)

