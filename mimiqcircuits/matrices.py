#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
# Copyright © 2032-2024 QPerfect. All Rights Reserved.
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

from symengine import I, cos, sin, pi, eye, Matrix


def cis(x):
    return cos(x) + I * sin(x)


def cispi(x):
    return cos(pi * x) + I * sin(pi * x)


def cospi(x):
    return cos(pi * x)


def sinpi(x):
    return sin(pi * x)


def pmatrixpi(lmbda):
    return Matrix([[1, 0], [0, cispi(lmbda)]])


def pmatrix(lmbda):
    return Matrix([[1, 0], [0, cis(lmbda)]])


def ctrl(mat):
    """
    Returns the controlled version of a given 2x2 matrix.

    Args:
        mat (numpy.ndarray): A nxn matrix to be controlled.

    Returns:
        numpy.ndarray: A 2n x 2n controlled matrix.
    """
    dim = mat.shape[0]
    m = eye(2 * dim)

    # TODO: can it be done via slicing?
    for i in range(dim):
        for j in range(dim):
            m[i + dim, j + dim] = mat[i, j]
    return m


def gphase(lmbda):
    return cis(lmbda)


def gphasepi(lmbda):
    return cispi(lmbda)


def umatrix(theta, phi, lmbda, gamma=0.0):
    costheta2 = cos(theta / 2)
    sintheta2 = sin(theta / 2)
    return Matrix(
        [
            [cis(gamma) * costheta2, -cis(lmbda + gamma) * sintheta2],
            [cis(phi + gamma) * sintheta2, cis(phi + lmbda + gamma) * costheta2],
        ]
    )


def umatrixpi(theta, phi, lmbda, gamma=0):
    costheta2 = cospi(theta / 2)
    sintheta2 = sinpi(theta / 2)
    return Matrix(
        [
            [cispi(gamma) * costheta2, -cispi(lmbda + gamma) * sintheta2],
            [cispi(phi + gamma) * sintheta2, cispi(phi + lmbda + gamma) * costheta2],
        ]
    )


def rxmatrixpi(theta):
    return umatrixpi(theta, -1 / 2, 1 / 2)


def rxmatrix(theta):
    return rxmatrixpi(theta / pi)


def rymatrixpi(theta):
    return umatrixpi(theta, 0, 0)


def rymatrix(theta):
    return rymatrixpi(theta / pi)


def rzmatrixpi(lmbda):
    return gphasepi(-lmbda / 2) * umatrixpi(0, 0, lmbda)


def rzmatrix(lmbda):
    return rzmatrixpi(lmbda / pi)
