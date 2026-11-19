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
"""Pauli Hamiltonians, their expectation values and Trotterized time evolution."""

from mimiqcircuits.circuit import Circuit
from mimiqcircuits.complex import Add, Multiply
from mimiqcircuits.gatedecl import GateDecl
from mimiqcircuits.instruction import Instruction
from mimiqcircuits.operations.expectationvalue import ExpectationValue
from mimiqcircuits.operations.gates.generalized.paulistring import PauliString
from mimiqcircuits.operations.gates.generalized.rpauli import RPauli
from mimiqcircuits.operations.gates.standard.id import GateID
from mimiqcircuits.operations.gates.standard.interactions import (
    GateRXX,
    GateRYY,
    GateRZX,
    GateRZZ,
)
from mimiqcircuits.operations.gates.standard.pauli import GateX, GateY, GateZ
from mimiqcircuits.operations.gates.standard.rotations import GateRX, GateRY, GateRZ
import symengine as se
from mimiqcircuits.matrices import kronecker
from typing import Union
import numpy as np


class HamiltonianTerm:
    r"""One term of a Hamiltonian: a coefficient times a Pauli string.

    The term is

    .. math::
        H_j = c_j P_j = c_j \left(P^{(1)} \otimes P^{(2)} \otimes \dots\right),

    where :math:`c_j` is a real coefficient and :math:`P_j` a Pauli string
    such as ``X`` or ``YZ``, whose :math:`k`-th letter acts on the
    :math:`k`-th qubit of ``qubits``. Terms are usually created with
    :meth:`Hamiltonian.push`.

    Args:
        coefficient: The real coefficient :math:`c_j`. A symbolic
            expression is accepted and can be substituted later with
            :meth:`Hamiltonian.evaluate`.
        pauli: The Pauli string :math:`P_j`. Its length should equal the
            number of qubits; this is not checked.
        qubits: The indices of the qubits the term acts on, counted
            from 0.

    Raises:
        ValueError: If a qubit index is negative or appears twice.

    Examples:
        >>> from mimiqcircuits import *
        >>> # 0.5 X_0 Z_1
        >>> t = HamiltonianTerm(0.5, PauliString("XZ"), (0, 1))
        >>> t
        0.5 * XZ @ q[0,1]
        >>> t.get_coefficient(), t.get_operation(), t.get_qubits()
        (0.5, XZ, (0, 1))
    """

    def __init__(
        self, coefficient: Union[float, int], pauli: PauliString, qubits: tuple
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
        """Return the Pauli string of the term.

        Returns:
            PauliString: The Pauli string :math:`P_j`.
        """
        return self.pauli

    def get_coefficient(self):
        """Return the coefficient of the term.

        Returns:
            The coefficient :math:`c_j`, a number or a symbolic expression.
        """
        return self.coefficient

    def get_qubits(self):
        """Return the qubits the term acts on.

        Returns:
            tuple: The qubit indices, in the order of the letters of the
            Pauli string.
        """
        return self.qubits

    def __repr__(self):
        qubit_str = ",".join(str(q) for q in self.qubits)
        return f"{self.coefficient} * {self.pauli} @ q[{qubit_str}]"

    def __str__(self):
        return self.__repr__()


class Hamiltonian:
    r"""A Hamiltonian written as a sum of Pauli strings.

    The Hamiltonian is

    .. math::
        H = \sum_j c_j P_j,

    where each term :math:`c_j P_j` is a :class:`HamiltonianTerm`. Add terms
    with :meth:`push` or :meth:`add_terms`. Iteration, indexing and ``len``
    act on the list of terms. Use it with :func:`push_expval` to measure its expectation value, and with
    :func:`push_lietrotter`, :func:`push_suzukitrotter` or
    :func:`push_yoshidatrotter` to simulate the time evolution
    :math:`e^{-iHt}`. These are available as methods of
    :class:`~mimiqcircuits.Circuit`.

    Args:
        terms (iterable, optional): The :class:`HamiltonianTerm` objects to
            start with.
            Defaults to no terms.

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
        """Append a term to the Hamiltonian.

        Despite its name, it takes a single term.

        Args:
            term: The term to append.

        Returns:
            Hamiltonian: This Hamiltonian, modified in place.
        """
        self.terms.append(term)
        return self

    def push(self, coefficient: float, pauli: PauliString, *qubits: int):
        """Append the term ``coefficient * pauli`` acting on ``qubits``.

        Args:
            coefficient: The coefficient of the term, a real number or a
                symbolic expression.
            pauli: The Pauli string, one letter per qubit.
            *qubits: The qubit indices, counted from 0.

        Returns:
            Hamiltonian: This Hamiltonian, modified in place.

        Raises:
            ValueError: If a qubit index is negative or appears twice.

        Examples:
            >>> from mimiqcircuits import *
            >>> h = Hamiltonian()
            >>> # -J Z_0 Z_1 with J = 1, then a field on qubit 1
            >>> h.push(-1.0, PauliString("ZZ"), 0, 1)
            2-qubit Hamiltonian with 1 terms:
            └── -1.0 * ZZ @ q[0,1]
            >>> h.push(0.5, PauliString("X"), 1)
            2-qubit Hamiltonian with 2 terms:
            ├── -1.0 * ZZ @ q[0,1]
            └── 0.5 * X @ q[1]
        """
        self.terms.append(HamiltonianTerm(coefficient, pauli, qubits))
        return self

    def num_terms(self):
        """Return the number of terms, the same as ``len(h)``.

        Returns:
            int: The number of terms.
        """
        return len(self.terms)

    def num_qubits(self):
        """Return the number of qubits the Hamiltonian acts on.

        This is one more than the largest qubit index of any term, so it
        counts unused qubits below that index.

        Returns:
            int: The number of qubits, ``0`` for an empty Hamiltonian.
        """
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
        """Save the Hamiltonian to a file in protobuf format.

        Args:
            file: A path, or a binary file object with a ``write`` method.

        Returns:
            int: The number of bytes written.

        Raises:
            ValueError: If writing to a path fails.

        Examples:
            >>> from mimiqcircuits import *
            >>> import io
            >>> h = Hamiltonian()
            >>> h.push(0.5, PauliString("ZZ"), 0, 1)
            2-qubit Hamiltonian with 1 terms:
            └── 0.5 * ZZ @ q[0,1]
            >>> # write to an in-memory file, then read it back
            >>> buf = io.BytesIO()
            >>> _ = h.saveproto(buf)
            >>> _ = buf.seek(0)
            >>> Hamiltonian.loadproto(buf)
            2-qubit Hamiltonian with 1 terms:
            └── 0.5 * ZZ @ q[0,1]
        """
        from mimiqcircuits.proto.protoio import saveproto

        return saveproto(self, file)

    @staticmethod
    def loadproto(file):
        """Load a Hamiltonian from a file in protobuf format.

        Args:
            file: A path, or a binary file object with a ``read`` method,
                holding a Hamiltonian written by :meth:`saveproto`.

        Returns:
            Hamiltonian: The loaded Hamiltonian.

        Raises:
            ValueError: If the file cannot be read or parsed.
        """
        from mimiqcircuits.proto.protoio import loadproto

        return loadproto(file, Hamiltonian)

    @staticmethod
    def _pauli_matrix(p: str):
        if p == "I":
            return GateID().matrix()
        elif p == "X":
            return GateX().matrix()
        elif p == "Y":
            return GateY().matrix()
        elif p == "Z":
            return GateZ().matrix()
        else:
            raise ValueError(f"Invalid Pauli character: {p}")

    def matrix(self):
        """Return the matrix of the Hamiltonian.

        The matrix is :math:`2^n \\times 2^n` for :meth:`num_qubits` qubits
        :math:`n`, with qubit 0 as the leftmost factor of the Kronecker
        product, that is the most significant bit of the row index.

        Returns:
            symengine.DenseMatrix: The matrix, symbolic if any coefficient
            is.

        Examples:
            >>> from mimiqcircuits import *
            >>> h = Hamiltonian()
            >>> h.push(1.0, PauliString("Z"), 0)
            1-qubit Hamiltonian with 1 terms:
            └── 1.0 * Z @ q[0]
            >>> h.matrix()
            [1.0, 0]
            [0, -1.0]
            <BLANKLINE>
        """

        nq = self.num_qubits()
        dim = 2**nq
        H = se.zeros(dim, dim)

        for term in self.terms:
            coeff = term.get_coefficient()
            qubits = term.get_qubits()
            pauli_str = term.get_operation().pauli

            ps = np.argsort(qubits)
            qubits = [qubits[i] for i in ps]
            pstr = "".join(pauli_str[i] for i in ps)

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
        """Substitute values into the symbolic coefficients.

        Coefficients that are plain numbers are kept as they are. The
        Hamiltonian itself is not modified.

        Args:
            d: A dictionary from symbols, or their names, to values.

        Returns:
            Hamiltonian: A new Hamiltonian with the substituted
            coefficients, numeric or still partly symbolic.

        Examples:
            >>> from mimiqcircuits import *
            >>> from symengine import Symbol
            >>> J = Symbol("J")
            >>> h = Hamiltonian()
            >>> h.push(J, PauliString("ZZ"), 0, 1)
            2-qubit Hamiltonian with 1 terms:
            └── J * ZZ @ q[0,1]
            >>> # set the coupling J to 0.25
            >>> h.evaluate({J: 0.25})
            2-qubit Hamiltonian with 1 terms:
            └── 0.25 * ZZ @ q[0,1]
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
    r"""Append the computation of the expectation value of a Hamiltonian.

    Called as a method, ``circuit.push_expval(hamiltonian, *qubits)``. For
    each term :math:`c_j P_j` it appends an :class:`ExpectationValue` of
    :math:`P_j` into z-variable ``firstzvar + j`` and scales it by
    :math:`c_j` with :class:`Multiply`, then sums them all with
    :class:`Add`, so that z-variable ``firstzvar`` ends up holding

    .. math::
        \langle \psi | H | \psi \rangle = \sum_j c_j \langle \psi | P_j | \psi \rangle.

    The other z-variables keep the value of each scaled term. The state is
    not changed.

    Args:
        hamiltonian: The Hamiltonian to evaluate.
        *qubits (int): The circuit qubits that the Hamiltonian qubits
            ``0, 1, ...`` map to. There must be exactly
            ``hamiltonian.num_qubits()`` of them.
        firstzvar (int, optional): First z-variable to use. Defaults to the
            first unused one, ``circuit.num_zvars()``.

    Returns:
        Circuit: The circuit, modified in place.

    Raises:
        ValueError: If the number of qubits does not match the Hamiltonian.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> h = Hamiltonian()
        >>> h.push(1.0, PauliString("ZZ"), 0, 1)
        2-qubit Hamiltonian with 1 terms:
        └── 1.0 * ZZ @ q[0,1]
        >>> # Hamiltonian qubits 0 and 1 are circuit qubits 1 and 2
        >>> c.push_expval(h, 1, 2)
        3-qubit, 1-zvar circuit with 3 instructions:
        ├── ⟨ZZ⟩ @ q[1:2], z[0]
        ├── z[0] = 1.0 * z[0]
        └── z[0] = 0.0 + z[0]
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
            ExpectationValue(term.get_operation()),
            *[qubits[i] for i in term.get_qubits()],
            zvar,
        )
        # In-place scale: z[zvar] = coeff * z[zvar], with the destination
        # also as the only input.
        self.push(Multiply(2, c=term.get_coefficient()), zvar, zvar)
        zvar += 1

    # In-place sum: z[firstzvar] = z[firstzvar] + z[firstzvar+1] + ... +
    # z[zvar-1]. The destination is also the first input, so its own value
    # stays in the sum.
    self.push(Add(zvar - firstzvar + 1), firstzvar, *range(firstzvar, zvar))
    return self


def _pauliexp(term: HamiltonianTerm, t, qubits: tuple):
    """Internal helper for Pauli exponentiation."""
    p = term.get_operation()
    s = p.pauli

    param = term.get_coefficient() * t

    if s == "X":
        return Instruction(GateRX(param), qubits)
    elif s == "Y":
        return Instruction(GateRY(param), qubits)
    elif s == "Z":
        return Instruction(GateRZ(param), qubits)
    elif s == "XX":
        return Instruction(GateRXX(param), qubits)
    elif s == "YY":
        return Instruction(GateRYY(param), qubits)
    elif s == "ZZ":
        return Instruction(GateRZZ(param), qubits)
    elif s == "ZX":
        return Instruction(GateRZX(param), qubits)
    elif s == "XZ":
        return Instruction(GateRZX(param), (qubits[::-1]))

    else:
        return Instruction(RPauli(p, param), qubits)


def push_lietrotter(self, h: Hamiltonian, qubits: tuple, t: float, steps: int):
    r"""Append the first-order (Lie-Trotter) time evolution under a Hamiltonian.

    Called as a method, ``circuit.push_lietrotter(h, qubits, t, steps)``. It
    approximates :math:`e^{-iHt}`, for :math:`H = \sum_{j=1}^m c_j P_j`, by

    .. math::

        e^{-i H t} \approx \left[ \prod_{j=1}^m e^{-i c_j P_j \Delta t} \right]^n,

    with :math:`n` = ``steps`` and :math:`\Delta t = t / n`. The error per
    step is of order :math:`\Delta t^2` and vanishes when the terms commute.

    One step is declared once as a gate named ``trotter`` with parameter
    :math:`\Delta t`, holding one Pauli rotation per term (``GateRX``,
    ``GateRZZ``, ``RPauli``, ...). The circuit gets ``steps`` calls to it.

    Args:
        h: The Hamiltonian.
        qubits: The circuit qubits that the Hamiltonian qubits ``0, 1, ...``
            map to. There must be exactly ``h.num_qubits()`` of them.
        t: The total evolution time.
        steps: The number of Trotter steps :math:`n`.

    Returns:
        Circuit: The circuit, modified in place.

    Raises:
        ValueError: If the number of qubits does not match the Hamiltonian.

    See Also:
        :func:`push_suzukitrotter`, :func:`push_yoshidatrotter`,
        :class:`GateDecl`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> h = Hamiltonian()
        >>> h.push(1.0, PauliString("ZZ"), 0, 1)
        2-qubit Hamiltonian with 1 terms:
        └── 1.0 * ZZ @ q[0,1]
        >>> # evolve for t = 1 in 3 steps of 1/3
        >>> c.push_lietrotter(h, (0, 1), t=1.0, steps=3)
        2-qubit circuit with 3 instructions:
        ├── trotter(0.3333333333333333) @ q[0:1]
        ├── trotter(0.3333333333333333) @ q[0:1]
        └── trotter(0.3333333333333333) @ q[0:1]
        <BLANKLINE>
    """
    if len(qubits) != h.num_qubits():
        raise ValueError("Number of qubits does not match Hamiltonian.")

    tstep = t / steps
    Δt = se.Symbol("Δt")

    ch = Circuit()

    for term in h:
        ch.push(_pauliexp(term, 2 * Δt, term.get_qubits()))

    decl = GateDecl("trotter", (Δt,), ch)

    for _ in range(steps):
        self.push(decl(tstep), *qubits)

    return self


def push_suzukitrotter(
    self, h: Hamiltonian, qubits: tuple, t: float, steps: int, order: int = 2
):
    # see e.g. [https://arxiv.org/pdf/quant-ph/0508139]
    # and [https://arxiv.org/abs/2211.02691]
    # and [https://arxiv.org/pdf/math-ph/0506007]

    r"""Append the time evolution under a Hamiltonian with a Suzuki-Trotter formula.

    Called as a method, ``circuit.push_suzukitrotter(h, qubits, t, steps,
    order)``. It approximates :math:`e^{-iHt}` with the Suzuki product
    formula of the given even ``order``, whose error per step is of order
    :math:`\Delta t^{\text{order}+1}`.

    The second-order formula is

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

    and each higher order is built from the one below by the Suzuki
    recursion

    .. math::

        S_{2k}(\lambda) =
        [S_{2k-2}(p_k \lambda)]^2 \cdot
        S_{2k-2}((1 - 4p_k)\lambda) \cdot
        [S_{2k-2}(p_k \lambda)]^2

    with:

    .. math::

        p_k = \left(4 - 4^{1/(2k - 1)}\right)^{-1}.

    Each order is declared once as a gate named ``suzukitrotter_<order>``,
    the higher ones calling the lower ones five times. The circuit gets
    ``steps`` calls to the gate of the requested order.

    Args:
        h: The Hamiltonian.
        qubits: The circuit qubits that the Hamiltonian qubits ``0, 1, ...``
            map to. There must be exactly ``h.num_qubits()`` of them.
        t: The total evolution time.
        steps: The number of Trotter steps :math:`n`.
        order: The order of the formula, an even integer of at least 2
            (default ``2``).

    Returns:
        Circuit: The circuit, modified in place.

    Raises:
        ValueError: If the number of qubits does not match the Hamiltonian,
            or ``order`` is not an even integer of at least 2.

    See Also:
        :func:`push_lietrotter`, :func:`push_yoshidatrotter`,
        :class:`GateDecl`

    References:
        - M. Suzuki, "General theory of fractal path integrals with
          applications to many-body theories and statistical physics",
          J. Math. Phys. 32, 400 (1991).

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> h = Hamiltonian()
        >>> h.push(1.0, PauliString("XX"), 0, 1)
        2-qubit Hamiltonian with 1 terms:
        └── 1.0 * XX @ q[0,1]
        >>> c.push_suzukitrotter(h, (0, 1), t=1.0, steps=5, order=2)
        2-qubit circuit with 5 instructions:
        ├── suzukitrotter_2(0.2) @ q[0:1]
        ├── suzukitrotter_2(0.2) @ q[0:1]
        ├── suzukitrotter_2(0.2) @ q[0:1]
        ├── suzukitrotter_2(0.2) @ q[0:1]
        └── suzukitrotter_2(0.2) @ q[0:1]
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
    ch = Circuit()
    for term in h:
        ch.push(_pauliexp(term, λ, term.get_qubits()))
    for term in reversed(h.terms):
        ch.push(_pauliexp(term, λ, term.get_qubits()))

    decls = [GateDecl(f"suzukitrotter_2", (λ,), ch)]

    for k in range(2, order // 2 + 1):
        pk = 1 / (4 - 4 ** (1 / (2 * k - 1)))
        ck = Circuit()

        ck.push(decls[-1](pk * λ), *range(len(qubits)))
        ck.push(decls[-1](pk * λ), *range(len(qubits)))
        ck.push(decls[-1]((1 - 4 * pk) * λ), *range(len(qubits)))
        ck.push(decls[-1](pk * λ), *range(len(qubits)))
        ck.push(decls[-1](pk * λ), *range(len(qubits)))

        decls.append(GateDecl(f"suzukitrotter_{2 * k}", (λ,), ck))

    for _ in range(steps):
        self.push(decls[-1](tstep), *qubits)

    return self


def push_yoshidatrotter(
    self, h: Hamiltonian, qubits: tuple, t: float, steps: int, order: int = 4
):
    # see e.g. [https://doi.org/10.1016/0375-9601(90)90092-3] for Yoshida's composition method
    # see e.g. [https://aiichironakano.github.io/phys516/Yoshida-symplectic-PLA00.pdf]

    r"""Append the time evolution under a Hamiltonian with a Yoshida product formula.

    Called as a method, ``circuit.push_yoshidatrotter(h, qubits, t, steps,
    order)``. It approximates :math:`e^{-iHt}` by composing second-order
    Trotter formulas symmetrically, which cancels the error terms up to the
    given even ``order``. Each order makes three calls to the order below,
    against five for :func:`push_suzukitrotter`.

    The order :math:`2(k+1)` formula is built from the order :math:`2k` one as

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

    Each order is declared once as a gate named ``yoshida_<order>``. The
    circuit gets ``steps`` calls to the gate of the requested order.

    Args:
        h: The Hamiltonian.
        qubits: The circuit qubits that the Hamiltonian qubits ``0, 1, ...``
            map to. There must be exactly ``h.num_qubits()`` of them.
        t: The total evolution time.
        steps: The number of Trotter steps :math:`n`.
        order: The order of the formula, an even integer of at least 2
            (default ``4``).

    Returns:
        Circuit: The circuit, modified in place.

    Raises:
        ValueError: If the number of qubits does not match the Hamiltonian,
            or ``order`` is not an even integer of at least 2.

    See Also:
        :func:`push_suzukitrotter`, :func:`push_lietrotter`,
        :class:`GateDecl`

    References:
        - H. Yoshida, "Construction of higher order symplectic integrators",
          Phys. Lett. A 150, 262 (1990).

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
        ├── yoshida_4(0.3333333333333333) @ q[0:1]
        ├── yoshida_4(0.3333333333333333) @ q[0:1]
        └── yoshida_4(0.3333333333333333) @ q[0:1]
        <BLANKLINE>
    """

    if len(qubits) != h.num_qubits():
        raise ValueError("Number of qubits does not match Hamiltonian.")
    if order < 2 or order % 2 != 0:
        raise ValueError("Yoshida order must be an even integer ≥ 2.")

    λ = se.Symbol("λ")
    base = Circuit()
    for term in h:
        base.push(_pauliexp(term, λ, term.get_qubits()))
    for term in reversed(h.terms):
        base.push(_pauliexp(term, λ, term.get_qubits()))

    decls = [GateDecl(f"yoshida_2", (λ,), base)]

    for k in range(2, order // 2 + 1):
        alpha = 1 / (2 - 2 ** (1 / (2 * k - 1)))
        beta = -(2 ** (1 / (2 * k - 1))) / (2 - 2 ** (1 / (2 * k - 1)))

        ck = Circuit()
        ck.push(decls[-1](alpha * λ), *range(len(qubits)))
        ck.push(decls[-1](beta * λ), *range(len(qubits)))
        ck.push(decls[-1](alpha * λ), *range(len(qubits)))

        decls.append(GateDecl(f"yoshida_{2 * k}", (λ,), ck))

    tstep = t / steps
    for _ in range(steps):
        self.push(decls[-1](tstep), *qubits)

    return self


# Auto-register on Circuit
Circuit.push_suzukitrotter = push_suzukitrotter
Circuit.push_lietrotter = push_lietrotter
Circuit.push_expval = push_expval
Circuit.push_yoshidatrotter = push_yoshidatrotter


__all__ = [
    "HamiltonianTerm",
    "Hamiltonian",
    "push_expval",
    "push_lietrotter",
    "push_suzukitrotter",
    "push_yoshidatrotter",
]
