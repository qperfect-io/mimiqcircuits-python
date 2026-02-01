#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
# Copyright © 2023-2025 QPerfect. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from collections import OrderedDict
from mimiqcircuits import QCSResults
from symengine import Symbol
from mimiqcircuits.proto import optim_pb2, circuit_pb2


def toproto_ParametersDictItem(key, value):
    return optim_pb2.ParametersDictItem(
        key=circuit_pb2.Symbol(value=str(key)), value=value
    )


def fromproto_ParametersDictItem(item):
    return Symbol(item.key.value), item.value


def toproto_OptimizationExperiment(experiment):
    from mimiqcircuits.proto.circuitproto import toproto_circuit

    circuit_pb = toproto_circuit(experiment.circuit)
    initparams_pb = [
        toproto_ParametersDictItem(k, v) for k, v in experiment.initparams.items()
    ]

    return optim_pb2.OptimizationExperiment(
        circuit=circuit_pb,
        optimizer=experiment.optimizer,
        initparams=initparams_pb,
        maxiters=experiment.maxiters or 0,
        zregister=experiment.zregister + 1 or 1,
        label=experiment.label,
    )


def fromproto_OptimizationExperiment(msg):
    from mimiqcircuits.proto.circuitproto import fromproto_circuit
    from mimiqcircuits.optimization import OptimizationExperiment

    circuit = fromproto_circuit(msg.circuit)
    optimizer = msg.optimizer
    label = msg.label
    zregister = msg.zregister - 1
    maxiters = msg.maxiters

    initparams = OrderedDict()
    for p in msg.initparams:
        k, v = fromproto_ParametersDictItem(p)
        initparams[k] = v

    return OptimizationExperiment(
        circuit=circuit,
        initparams=initparams,
        optimizer=optimizer,
        label=label,
        maxiters=maxiters,
        zregister=zregister,
    )


def toproto_OptimizationRun(run):
    from mimiqcircuits.proto.qcsrproto import toproto_qcsr

    param_items = [toproto_ParametersDictItem(k, v) for k, v in run.parameters.items()]

    return optim_pb2.OptimizationRun(
        cost=run.cost,
        parameters=param_items,
        results=toproto_qcsr(run.results),
    )


def fromproto_OptimizationRun(msg):
    from mimiqcircuits.optimization import OptimizationRun

    parameters = dict(fromproto_ParametersDictItem(p) for p in msg.parameters)

    if hasattr(QCSResults, "fromproto"):
        results = QCSResults.fromproto(msg.results)
    else:
        from mimiqcircuits.proto.qcsrproto import fromproto_qcsr

        results = fromproto_qcsr(msg.results)

    return OptimizationRun(
        cost=msg.cost,
        parameters=parameters,
        results=results,
    )


def toproto_OptimizationResults(results):
    return optim_pb2.OptimizationResults(
        best=toproto_OptimizationRun(results.best),
        history=[toproto_OptimizationRun(run) for run in results.history],
    )


def fromproto_OptimizationResults(msg):
    from mimiqcircuits.optimization import OptimizationResults

    best = fromproto_OptimizationRun(msg.best)
    history = [fromproto_OptimizationRun(run) for run in msg.history]

    return OptimizationResults(best=best, history=history)
