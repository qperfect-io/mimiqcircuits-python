#
# Copyright © 2026 QPerfect. All Rights Reserved.
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

# Writes tests/data/simplify_julia_reference.json: random circuits and what the
# MimiqCircuitsBase.jl simplification passes make of them, for
# tests/test_simplify.py to check the Python implementation against.
#
# Run with an environment where MimiqCircuitsBase provides `simplify`:
#
#     julia --project=path/to/env scripts/simplify_julia_reference.jl

using MimiqCircuitsBase
using JSON3
using Random

# Gates by the names both sides use for them.
const GATES = Dict(
    "H" => GateH, "X" => GateX, "Y" => GateY, "Z" => GateZ,
    "S" => GateS, "SDG" => GateSDG, "T" => GateT, "TDG" => GateTDG,
    "SX" => GateSX, "SXDG" => GateSXDG, "SY" => GateSY, "SYDG" => GateSYDG,
    "RX" => GateRX, "RY" => GateRY, "RZ" => GateRZ, "P" => GateP, "U" => GateU,
    "CX" => GateCX, "CZ" => GateCZ, "SWAP" => GateSWAP, "CCX" => GateCCX,
    "C3X" => GateC3X, "CRZ" => GateCRZ, "RZZ" => GateRZZ,
)
const NAMES = Dict(T => name for (name, T) in GATES)

function encode(inst)
    op = getoperation(inst)
    q = collect(getqubits(inst)) .- 1
    op isa Measure && return Dict("name" => "M", "qubits" => q, "bits" => collect(getbits(inst)) .- 1)
    params = Float64[MimiqCircuitsBase.unwrapvalue(p) for p in getparams(op)]
    return Dict("name" => NAMES[typeof(op)], "qubits" => q, "params" => params)
end

const ONEQ = ["H", "X", "Y", "Z", "S", "SDG", "T", "TDG", "SX", "SXDG", "SY", "SYDG"]
const ANGLES = [0.0, 0.25, -0.25, 0.5, π / 4, -π / 2, π]

function randomcircuit(rng, nq, n; wide=false)
    c = Circuit()
    for _ in 1:n
        r = rand(rng)
        if r < 0.3
            a, b = randperm(rng, nq)[1:2]
            push!(c, rand(rng, [GateCX(), GateCZ(), GateSWAP(), GateCRZ(0.3), GateRZZ(-0.4)]), a, b)
        elseif wide && nq >= 3 && r < 0.4
            qs = randperm(rng, nq)
            nq >= 4 && rand(rng) < 0.5 ? push!(c, GateC3X(), qs[1:4]...) : push!(c, GateCCX(), qs[1:3]...)
        elseif r < 0.7
            push!(c, GATES[rand(rng, ONEQ)](), rand(rng, 1:nq))
        elseif r < 0.97
            push!(c, rand(rng, [GateRX, GateRY, GateRZ, GateP])(rand(rng, ANGLES)), rand(rng, 1:nq))
        else
            push!(c, Measure(), rand(rng, 1:nq), 1)
        end
    end
    return c
end

const PASSES = [
    "remove_identities" => remove_identities,
    "cancel_inverses" => cancel_inverses,
    "merge_rotations" => merge_rotations,
    "resynthesize_1q" => resynthesize_1q,
    "resynthesize_2q" => resynthesize_2q,
    "resynthesize_linear" => resynthesize_linear,
    "simplify" => simplify,
]

rng = MersenneTwister(2026)
cases = []
for k in 1:40
    nq = rand(rng, 2:6)
    c = randomcircuit(rng, nq, rand(rng, 10:60); wide=k > 20)
    push!(cases, Dict(
        "circuit" => map(encode, c),
        "results" => Dict(name => map(encode, pass(c)) for (name, pass) in PASSES),
    ))
end

path = joinpath(@__DIR__, "..", "tests", "data", "simplify_julia_reference.json")
mkpath(dirname(path))
open(io -> JSON3.write(io, cases), path, "w")
println("wrote ", length(cases), " cases to ", path)
