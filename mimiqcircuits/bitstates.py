#
# Copyright © 2023 University of Strasbourg. All Rights Reserved.
# See AUTHORS.md for the list of authors.
#

from bitarray import bitarray


class BitState(bitarray):
    def __str__(self):
        return f"BitState('{self.to01()}')"

    def __repr__(self):
        return self.__str__()

    def num_qubits(self):
        return len(self)

    def __hash__(self):
        return hash(self.to01())


__all__ = ["BitState"]
