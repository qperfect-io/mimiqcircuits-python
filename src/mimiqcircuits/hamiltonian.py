#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
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

import mimiqcircuits as mc
import symengine as se
from mimiqcircuits.matrices import kronecker
from typing import Union
import numpy as np

class HamiltonianTerm:
    r"""Single term in a quantum Hamiltonian.

    A `HamiltonianTerm` represents a single tensor product of Pauli operators
    acting on a subset of qubits, scaled by a real-valued coefficient.

    .. math::
        H_j = c_j \cdot P_j = c_j \cdot (P^{(1)} \otimes P^{(2)} \otimes \dots)

    where :math:`c_j` is a real coefficient and :math:`P_j` is a Pauli string such as
    ``X``, ``YZ``, etc.

    Args:
        coefficient (float): The scalar multiplier for this term.
        pauli (PauliString): The Pauli operator string (e.g., `"XY"`).
        qubits (tuple): The tuple of qubit indices this term acts on.

    Raises:
        ValueError: If any qubit index is negative.

    Examples:
        >>> from mimiqcircuits import *
        >>> HamiltonianTerm(0.5, PauliString("XZ"), (0, 1))
        0.5 * XZ @ q[0,1]
    """

    def __init__(
        self, coefficient: Union[float, int], pauli: mc.PauliString, qubits: tuple
    ):
        invalid_qubits = [q for q in qubits if q < 0]
        if invalid_qubits:
            raise ValueError(
                f"Qubit indices must be ≥ 0. Invalid qubits: {invalid_qubits}"
            )
            
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Duplicate qubits are not allowed: {qubits}")

        self.coefficient = coefficient
        self.pauli = pauli
        self.qubits = qubits

    def get_operation(self):
        return self.pauli

    def get_coefficient(self):
        return self.coefficient

    def get_qubits(self):
        return self.qubits

    def __repr__(self):
        qubit_str = ",".join(str(q) for q in self.qubits)
        return f"{self.coefficient} * {self.pauli} @ q[{qubit_str}]"

    def __str__(self):
        return self.__repr__()


class Hamiltonian:
    r"""A quantum Hamiltonian composed of multiple Pauli terms.

    This class models a Hamiltonian of the form:

    .. math::
        H = \sum_j c_j \cdot P_j

    where each term is a `HamiltonianTerm` consisting of a coefficient and a Pauli string.

    Methods:
        - `push(...)`: Add a term by specifying its components.
        - `add_terms(...)`: Add a `HamiltonianTerm` object.
        - `num_qubits()`: Total number of qubits this Hamiltonian acts on.
        - `saveproto(...)`: Save to protobuf format.
        - `loadproto(...)`: Load from protobuf.

    Examples:
        >>> from mimiqcircuits import *
        >>> h = Hamiltonian()
        >>> h.push(1.0, PauliString("XX"), 0, 1)
        2-qubit Hamiltonian with 1 terms:
        └── 1.0 * XX @ q[0,1]
        >>> h.push(1.0, PauliString("YY"), 0, 1)
        2-qubit Hamiltonian with 2 terms:
        ├── 1.0 * XX @ q[0,1]
        └── 1.0 * YY @ q[0,1]
        >>> print(h)
        2-qubit Hamiltonian with 2 terms:
        ├── 1.0 * XX @ q[0,1]
        └── 1.0 * YY @ q[0,1]
    """

    def __init__(self, terms=None):
        if terms is None:
            terms = []
        self.terms = list(terms)

    def add_terms(self, term: HamiltonianTerm):
        self.terms.append(term)
        return self

    def push(self, coefficient: float, pauli: mc.PauliString, *qubits: int):
        self.terms.append(HamiltonianTerm(coefficient, pauli, qubits))
        return self

    def num_terms(self):
        return len(self.terms)

    def num_qubits(self):
        n = -1
        for term in self.terms:
            if not term.qubits:
                continue
            m = max(term.qubits)
            if m > n:
                n = m
        return n + 1

    def __iter__(self):
        return iter(self.terms)

    def __getitem__(self, idx):
        return self.terms[idx]

    def __len__(self):
        return len(self.terms)

    def __repr__(self):
        if not self.terms:
            return "empty Hamiltonian"
        out = f"{self.num_qubits()}-qubit Hamiltonian with {len(self.terms)} terms:\n"
        for i, term in enumerate(self.terms):
            prefix = "└── " if i == len(self.terms) - 1 else "├── "
            out += prefix + str(term) + "\n"
        return out.strip()

    def __str__(self):
        return self.__repr__()

    def saveproto(self, file):
        from mimiqcircuits.proto.hamiltonianproto import toproto_hamiltonian

        data = toproto_hamiltonian(self).SerializeToString()

        if hasattr(file, "write"):
            return file.write(data)
        else:
            try:
                with open(file, "wb") as f:
                    return f.write(data)
            except TypeError:
                raise ValueError(
                    "Invalid file object. Should be a filename or a file-like object."
                )
            except Exception as e:
                raise e

    @staticmethod
    def loadproto(file):
        from mimiqcircuits.proto import hamiltonian_pb2
        from mimiqcircuits.proto.hamiltonianproto import fromproto_hamiltonian

        if isinstance(file, str):
            with open(file, "rb") as f:
                proto = hamiltonian_pb2.Hamiltonian()
                proto.ParseFromString(f.read())
                return fromproto_hamiltonian(proto)
        elif hasattr(file, "read"):
            proto = hamiltonian_pb2.Hamiltonian()
            proto.ParseFromString(file.read())
            return fromproto_hamiltonian(proto)
        else:
            raise ValueError(
                "Invalid file object. Should be a filename or a file-like object."
            )
    @staticmethod     
    def _pauli_matrix(p: str):
        if p == "I":
            return mc.GateID().matrix()
        elif p == "X":
            return mc.GateX().matrix()
        elif p == "Y":
            return mc.GateY().matrix()
        elif p == "Z":
            return mc.GateZ().matrix()
        else:
            raise ValueError(f"Invalid Pauli character: {p}")

    def matrix(self):
        """Return the symbolic matrix representation of the Hamiltonian using symengine."""

        nq = self.num_qubits()
        dim = 2**nq
        H = se.zeros(dim, dim)

        for term in self.terms:
            coeff = term.get_coefficient()
            qubits = term.get_qubits()
            pauli_str = term.get_operation().pauli

            ps = np.argsort(qubits)
            qubits = [qubits[i] for i in ps]
            pstr = ''.join(pauli_str[i] for i in ps)

            i = 0
            qidx = 0
            if qubits[0] == 0:
                term_matrix = self._pauli_matrix(pstr[0])
                qidx = 1
            else:
                term_matrix = self._pauli_matrix("I")
            
            i += 1
            while qidx < len(qubits):
                while i < qubits[qidx]:
                    term_matrix = kronecker(term_matrix, self._pauli_matrix("I"))
                    i += 1
                term_matrix = kronecker(term_matrix, self._pauli_matrix(pstr[qidx]))
                i += 1
                qidx += 1

            while i < nq:
                term_matrix = kronecker(term_matrix, self._pauli_matrix("I"))
                i += 1

            H += coeff * term_matrix

        return H

    def evaluate(self, d: dict):
        """
        Evaluate the symbolic coefficients of each term using the substitution dictionary `d`.

        Parameters:
            d (dict): Dictionary mapping symbols (e.g., {'theta': 0.5}) to values.

        Returns:
            Hamiltonian: A new Hamiltonian with evaluated (numerical or partially symbolic) coefficients.
        """
        evaluated_terms = []
        for term in self.terms:
            coeff = term.get_coefficient()
            if hasattr(coeff, "subs"):
                coeff = coeff.subs(d)

            evaluated_terms.append(
                HamiltonianTerm(coeff, term.get_operation(), term.get_qubits())
            )

        return Hamiltonian(evaluated_terms)


def push_expval(self, hamiltonian: Hamiltonian, *qubits: int, firstzvar=None):
    r"""Push an expectation value estimation circuit for a given Hamiltonian.

    This operation measures the expectation value of a Hamiltonian and stores
    the result in a Z-register, combining individual Pauli term evaluations.

    For each term :math:`c_j P_j`, the circuit performs:

    .. math::
        \langle \psi | c_j P_j | \psi \rangle

    and sums the contributions in a new z-register.

    Args:
        hamiltonian (Hamiltonian): The Hamiltonian to evaluate.
        qubits (int): The qubit mapping to use.
        firstzvar (int, optional): Index of the first Z-register to use.

    Returns:
        Circuit: The modified circuit.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> h = Hamiltonian()
        >>> h.push(1.0, PauliString("ZZ"), 0, 1)
        2-qubit Hamiltonian with 1 terms:
        └── 1.0 * ZZ @ q[0,1]
        >>> c.push_expval(h, 1, 2)
        3-qubit, 1-zvar circuit with 3 instructions:
        ├── ⟨ZZ⟩ @ q[1,2], z[0]
        ├── z[0] *= 1.0
        └── z[0] += 0.0
        <BLANKLINE>

    See Also:
        :class:`ExpectationValue`, :class:`Multiply`, :class:`Add`


    """
    if len(qubits) != hamiltonian.num_qubits():
        raise ValueError("Number of qubits does not match Hamiltonian.")

    if firstzvar is None:
        firstzvar = self.num_zvars()

    zvar = firstzvar
    for term in hamiltonian:
        self.push(
            mc.ExpectationValue(term.get_operation()),
            *[qubits[i] for i in term.get_qubits()],
            zvar,
        )
        self.push(mc.Multiply(1, c=term.get_coefficient()), zvar)
        zvar += 1

    self.push(mc.Add(zvar - firstzvar), *range(firstzvar, zvar))
    return self


def _pauliexp(term: HamiltonianTerm, t, qubits: tuple):
    """Internal helper for Pauli exponentiation."""
    p = term.get_operation()
    s = p.pauli

    param = term.get_coefficient() * t

    if s == "X":
        return mc.Instruction(mc.GateRX(param), qubits)
    elif s == "Y":
        return mc.Instruction(mc.GateRY(param), qubits)
    elif s == "Z":
        return mc.Instruction(mc.GateRZ(param), qubits)
    elif s == "XX":
        return mc.Instruction(mc.GateRXX(param), qubits)
    elif s == "YY":
        return mc.Instruction(mc.GateRYY(param), qubits)
    elif s == "ZZ":
        return mc.Instruction(mc.GateRZZ(param), qubits)
    elif s == "ZX":
        return mc.Instruction(mc.GateRZX(param), qubits)
    elif s == "XZ":
        return mc.Instruction(mc.GateRZX(param), (qubits[::-1]))

    else:
        return mc.Instruction(mc.RPauli(p, param), qubits)


def push_lietrotter(self, h: Hamiltonian, qubits: tuple, t: float, steps: int):
    r"""
    Apply a Lie-Trotter expansion of the Hamiltonian ``h`` to the circuit ``self``
    for the qubits ``qubits`` over total time ``t`` with ``steps`` steps.

    The Lie-Trotter expansion is a first-order approximation of the time evolution
    operator for a Hamiltonian composed of non-commuting terms. It decomposes the
    exponential of a sum of operators into a product of exponentials:

    .. math::

        e^{-i H t} \approx \left[ \prod_{j=1}^m e^{-i c_j P_j \Delta t} \right]^n

    where:
        - :math:`H = \sum_{j=1}^m c_j P_j` is the Hamiltonian
        - :math:`P_j` are Pauli strings
        - :math:`\Delta t = t / n` is the time step size
        - :math:`n` is the number of steps

    This method is particularly useful for simulating quantum systems and time-evolving
    quantum states in quantum algorithms such as VQE or QAOA.

    See Also:
        :func:`push_suzukitrotter`, :class:`GateDecl`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> h = Hamiltonian()
        >>> h.push(1.0, PauliString("ZZ"), 0, 1)
        2-qubit Hamiltonian with 1 terms:
        └── 1.0 * ZZ @ q[0,1]
        >>> c.push_lietrotter(h, (0, 1), t=1.0, steps=3)
        2-qubit circuit with 3 instructions:
        ├── trotter(0.3333333333333333) @ q[0,1]
        ├── trotter(0.3333333333333333) @ q[0,1]
        └── trotter(0.3333333333333333) @ q[0,1]
        <BLANKLINE>
    """
    if len(qubits) != h.num_qubits():
        raise ValueError("Number of qubits does not match Hamiltonian.")

    tstep = t / steps
    Δt = se.Symbol("Δt")

    ch = mc.Circuit()

    for term in h:
        ch.push(_pauliexp(term, 2 * Δt, term.get_qubits()))

    decl = mc.GateDecl("trotter", (Δt,), ch)

    for _ in range(steps):
        self.push(decl(tstep), *qubits)

    return self


def push_suzukitrotter(
    self, h: Hamiltonian, qubits: tuple, t: float, steps: int, order: int = 2
):
    # see e.g. [https://arxiv.org/pdf/quant-ph/0508139]
    # and [https://arxiv.org/abs/2211.02691]
    # and [https://arxiv.org/pdf/math-ph/0506007]

    r"""
    Apply Suzuki-Trotter expansion of the Hamiltonian ``h`` to the circuit ``self``
    for the qubits ``qubits`` over time ``t`` with ``steps`` steps.

    The Suzuki-Trotter expansion approximates the time evolution operator of a
    quantum Hamiltonian using a sequence of exponentiated subterms. This is
    particularly useful for simulating quantum systems where the Hamiltonian is
    composed of non-commuting parts.

    The expansion performed is a ``2k``-th order expansion according to the Suzuki
    construction.

    The second-order expansion is given by:

    .. math::

        e^{-i H t} \approx
        \left[
            \prod_{j=1}^{m} e^{-i \frac{\Delta t}{2} H_j}
            \prod_{j=m}^{1} e^{-i \frac{\Delta t}{2} H_j}
        \right]^n

    where the Hamiltonian :math:`H` is a sum of terms:

    .. math::

        H = \sum_{j=1}^{m} H_j

    and the Trotter step size is:

    .. math::

        \Delta t = \frac{t}{n}

    Higher-order expansions follow the Suzuki recursion relation:

    .. math::

        S_{2k}(\lambda) =
        [S_{2k-2}(p_k \lambda)]^2 \cdot
        S_{2k-2}((1 - 4p_k)\lambda) \cdot
        [S_{2k-2}(p_k \lambda)]^2

    with:

    .. math::

        p_k = \left(4 - 4^{1/(2k - 1)}\right)^{-1}

    See Also:
        :func:`push_lietrotter`, :class:`GateDecl`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> h = Hamiltonian()
        >>> h.push(1.0, PauliString("XX"), 0, 1)
        2-qubit Hamiltonian with 1 terms:
        └── 1.0 * XX @ q[0,1]
        >>> c.push_suzukitrotter(h, (0, 1), t=1.0, steps=5, order=2)
        2-qubit circuit with 5 instructions:
        ├── suzukitrotter_2(0.2) @ q[0,1]
        ├── suzukitrotter_2(0.2) @ q[0,1]
        ├── suzukitrotter_2(0.2) @ q[0,1]
        ├── suzukitrotter_2(0.2) @ q[0,1]
        └── suzukitrotter_2(0.2) @ q[0,1]
        <BLANKLINE>
    """
    if len(qubits) != h.num_qubits():
        raise ValueError("Number of qubits does not match Hamiltonian.")

    if order < 2 or order % 2 != 0:
        raise ValueError(
            f"Suzuki-Trotter order must be an even integer ≥ 2. Got {order}"
        )

    tstep = t / steps
    λ = se.Symbol("λ")

    # Build base expansion S2(λ)
    ch = mc.Circuit()
    for term in h:
        ch.push(_pauliexp(term, λ, term.get_qubits()))
    for term in reversed(h.terms):
        ch.push(_pauliexp(term, λ, term.get_qubits()))

    decls = [mc.GateDecl(f"suzukitrotter_2", (λ,), ch)]

    for k in range(2, order // 2 + 1):
        pk = 1 / (4 - 4 ** (1 / (2 * k - 1)))
        ck = mc.Circuit()

        ck.push(decls[-1](pk * λ), *qubits)
        ck.push(decls[-1](pk * λ), *qubits)
        ck.push(decls[-1]((1 - 4 * pk) * λ), *qubits)
        ck.push(decls[-1](pk * λ), *qubits)
        ck.push(decls[-1](pk * λ), *qubits)

        decls.append(mc.GateDecl(f"suzukitrotter_{2 * k}", (λ,), ck))

    for _ in range(steps):
        self.push(decls[-1](tstep), *qubits)

    return self


def push_yoshidatrotter(
    self, h: Hamiltonian, qubits: tuple, t: float, steps: int, order: int = 4
):
    # see e.g. [https://doi.org/10.1016/0375-9601(90)90092-3] for Yoshida's composition method
    # see e.g. [https://aiichironakano.github.io/phys516/Yoshida-symplectic-PLA00.pdf]

    r"""
    Apply Yoshida-Trotter expansion of the Hamiltonian ``h`` to the circuit ``self``
    for the qubits ``qubits`` over time ``t`` with ``steps`` steps.

    The Yoshida-Trotter expansion approximates the time evolution operator of a
    quantum Hamiltonian using a symmetric composition of second-order Trotter
    formulas. This technique improves accuracy by canceling higher-order error terms
    in the Baker–Campbell–Hausdorff expansion.

    The Yoshida expansion performed is a ``2k``-th order expansion using the symmetric
    structure:

    .. math::

        S_{2(k+1)}(t) =
        S_{2k}(w_1 \cdot t) \cdot
        S_{2k}(w_2 \cdot t) \cdot
        S_{2k}(w_1 \cdot t)

    where the weights are:

    .. math::

        w_1 = \frac{1}{2 - 2^{1/(2k+1)}}, \quad
        w_2 = -\frac{2^{1/(2k+1)}}{2 - 2^{1/(2k+1)}}


    and the base case is the standard second-order Strang splitting:

    .. math::

        S_2(\Delta t) =
        \prod_{j=1}^{m} e^{-i \Delta t H_j / 2}
        \prod_{j=m}^{1} e^{-i \Delta t H_j / 2}

    This method is particularly useful for simulating quantum systems where the
    Hamiltonian is composed of non-commuting parts, and is a computationally efficient
    alternative to full recursive Suzuki methods.

    See Also:
        :func:`push_suzukitrotter`, :class:`GateDecl`

    Args:
        h (Hamiltonian): The Hamiltonian object.
        qubits (tuple): Tuple of qubit indices.
        t (float): Total simulation time.
        steps (int): Number of Trotter steps to apply.
        order (int): Desired even expansion order (must be ≥ 2 and even).

    Returns:
        Circuit: The modified circuit.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> h = Hamiltonian()
        >>> h.push(1.0, PauliString("XY"), 0, 1)
        2-qubit Hamiltonian with 1 terms:
        └── 1.0 * XY @ q[0,1]
        >>> h.push(0.5, PauliString("Z"), 0)
        2-qubit Hamiltonian with 2 terms:
        ├── 1.0 * XY @ q[0,1]
        └── 0.5 * Z @ q[0]
        >>> c.push_yoshidatrotter(h, (0, 1), t=1.0, steps=3, order=4)
        2-qubit circuit with 3 instructions:
        ├── yoshida_4(0.3333333333333333) @ q[0,1]
        ├── yoshida_4(0.3333333333333333) @ q[0,1]
        └── yoshida_4(0.3333333333333333) @ q[0,1]
        <BLANKLINE>
    """

    if len(qubits) != h.num_qubits():
        raise ValueError("Number of qubits does not match Hamiltonian.")
    if order < 2 or order % 2 != 0:
        raise ValueError("Yoshida order must be an even integer ≥ 2.")

    λ = se.Symbol("λ")
    base = mc.Circuit()
    for term in h:
        base.push(_pauliexp(term, λ, term.get_qubits()))
    for term in reversed(h.terms):
        base.push(_pauliexp(term, λ, term.get_qubits()))

    decls = [mc.GateDecl(f"yoshida_2", (λ,), base)]

    for k in range(2, order // 2 + 1):
        alpha = 1 / (2 - 2 ** (1 / (2 * k - 1)))
        beta = -(2 ** (1 / (2 * k - 1))) / (2 - 2 ** (1 / (2 * k - 1)))

        ck = mc.Circuit()
        ck.push(decls[-1](alpha * λ), *qubits)
        ck.push(decls[-1](beta * λ), *qubits)
        ck.push(decls[-1](alpha * λ), *qubits)

        decls.append(mc.GateDecl(f"yoshida_{2 * k}", (λ,), ck))

    tstep = t / steps
    for _ in range(steps):
        self.push(decls[-1](tstep), *qubits)

    return self


# Auto-register on Circuit
mc.Circuit.push_suzukitrotter = push_suzukitrotter
mc.Circuit.push_lietrotter = push_lietrotter
mc.Circuit.push_expval = push_expval
mc.Circuit.push_yoshidatrotter = push_yoshidatrotter


__all__ = [
    "HamiltonianTerm",
    "Hamiltonian",
    "push_expval",
    "push_lietrotter",
    "push_suzukitrotter",
    "push_yoshidatrotter",
]
