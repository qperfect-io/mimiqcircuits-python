#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
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
import shutil
import math


def _string_with_square(arr, sep):
    return f"[{sep.join(map(str, arr))}]"


def _find_unit_range(arr):
    if len(arr) < 2:
        return arr

    narr = []
    rangestart = arr[0]
    rangestop = arr[0]

    for v in arr[0:]:
        if v == rangestop + 1:
            rangestop = v
        elif rangestart == rangestop:
            narr.append(rangestart)
            rangestart = v
            rangestop = v
        else:
            narr.append(range(rangestart, rangestop + 1))
            rangestart = v
            rangestop = v

    if rangestart == rangestop:
        narr.append(rangestart)
    else:
        narr.append(range(rangestart, rangestop + 1))

    return narr


def _gate_name_padding(qubits, bits):
    nq = len(qubits)
    qubits_padding = 0 if nq == 1 else math.floor(math.log10(nq)) + 2

    if not bits:
        return qubits_padding

    bits_padding = len(str(bits)) + 1

    return max(qubits_padding, bits_padding)


class AsciiCanvas:
    def __init__(self, width=None):
        if width is None:
            _, self.width = 0, shutil.get_terminal_size().columns
        self.width = width
        self.data = []

    def get_rows(self):
        return len(self.data)

    def get_cols(self):
        return self.width

    def push_line(self):
        self.data.append([" "] * self.width)

    def __getitem__(self, position):
        row, col = position
        if row >= len(self.data):
            return " "
        return self.data[row][col]

    def __setitem__(self, position, value):
        row, col = position
        while row >= self.get_rows():
            self.push_line()
        while col >= len(self.data[row]):
            self.data[row].append(" ")
        self.data[row][col] = value

    def __str__(self):
        return "\n".join("".join(row) for row in self.data if row)

    def draw_hline(self, row, col, width):
        start_col, stop_col = min(col, col + width - 1), max(col, col + width - 1)

        for i in range(start_col, stop_col + 1):
            current_char = self[row, i]
            # Check and replace characters based on conditions
            if current_char == "│":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "├", "┼", "┤"
                )
            elif current_char == "┤":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "┼", "┼", "┤"
                )
            elif current_char == "├":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "├", "┼", "┼"
                )
            elif current_char == "╵":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "└", "┴", "┘"
                )
            elif current_char == "└":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "└", "┴", "┴"
                )
            elif current_char == "┘":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "┴", "┴", "┘"
                )
            elif current_char == "╷":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "┌", "┬", "┐"
                )
            elif current_char == "║":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╟", "╫", "╢"
                )
            elif current_char == "╟":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╟", "╫", "╫"
                )
            elif current_char == "╢":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╫", "╫", "╢"
                )
            elif current_char == "╶":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╶", "─", "─"
                )
            elif current_char == "╴":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "─", "─", "╴"
                )
            else:
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╶", "─", "╴"
                )

        return self

    def _start_mid_stop(self, current, start, stop, start_char, mid_char, stop_char):
        if current == start:
            return start_char
        elif current == stop:
            return stop_char
        return mid_char

    def draw_vline(self, row, col, height):
        start_row, stop_row = min(row, row + height - 1), max(row, row + height - 1)
        for i in range(start_row, stop_row + 1):
            current_char = self[i, col]
            if current_char == "─":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "┬", "┼", "┴"
                )
            elif current_char == "┴":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "┼", "┼", "┴"
                )
            elif current_char == "┬":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "┬", "┼", "┼"
                )
            elif current_char == "╴":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "┐", "┤", "┘"
                )
            elif current_char == "┐":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "┐", "┤", "┤"
                )
            elif current_char == "┘":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "┤", "┤", "┘"
                )
            elif current_char == "╶":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "┌", "├", "└"
                )
            elif current_char == "┌":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "┌", "├", "├"
                )
            elif current_char == "└":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "├", "├", "└"
                )
            elif current_char == "╵":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "│", "│", "╵"
                )
            elif current_char == "╷":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╷", "│", "│"
                )
            elif current_char == "═":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╤", "╪", "╧"
                )
            elif current_char == "╤":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╤", "╪", "╪"
                )
            elif current_char == "╧":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╪", "╪", "╧"
                )
            else:
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╷", "│", "╵"
                )

        return self

    def draw_double_hline(self, row, col, width):
        start_col, stop_col = min(col, col + width - 1), max(col, col + width - 1)
        for i in range(start_col, stop_col + 1):
            if self[row, i] == "│":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╞", "╪", "╡"
                )
            elif self[row, i] == "╞":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╞", "╪", "╪"
                )
            elif self[row, i] == "╡":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╪", "╪", "╡"
                )
            elif self[row, i] == "╵":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╘", "╧", "╛"
                )
            elif self[row, i] == "╘":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╘", "╧", "╧"
                )
            elif self[row, i] == "╛":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╧", "╧", "╛"
                )
            elif self[row, i] == "╷":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╒", "╤", "╕"
                )
            elif self[row, i] == "╒":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╒", "╤", "╤"
                )
            elif self[row, i] == "╕":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╤", "╤", "╕"
                )
            elif self[row, i] == "║":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╠", "╬", "╣"
                )
            elif self[row, i] == "╠":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╠", "╬", "╣"
                )
            elif self[row, i] == "╣":
                self[row, i] = self._start_mid_stop(
                    i, start_col, stop_col, "╠", "╬", "╣"
                )
            else:
                self[row, i] = "═"

    def draw_double_vline(self, row, col, height):
        start_row, stop_row = min(row, row + height - 1), max(row, row + height - 1)
        for i in range(start_row, stop_row + 1):
            if self[i, col] == "─":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╥", "╫", "╨"
                )
            elif self[i, col] == "╥":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╥", "╫", "╫"
                )
            elif self[i, col] == "╨":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╫", "╫", "╨"
                )
            elif self[i, col] == "╶":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╓", "╟", "╙"
                )
            elif self[i, col] == "╓":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╓", "╟", "╟"
                )
            elif self[i, col] == "╙":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╟", "╟", "╙"
                )
            elif self[i, col] == "╴":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╖", "╢", "╜"
                )
            elif self[i, col] == "╖":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╖", "╢", "╢"
                )
            elif self[i, col] == "╜":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╢", "╢", "╜"
                )
            elif self[i, col] == "═":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╦", "╬", "╩"
                )
            elif self[i, col] == "╩":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╬", "╬", "╩"
                )
            elif self[i, col] == "╦":
                self[i, col] = self._start_mid_stop(
                    i, start_row, stop_row, "╦", "╬", "╬"
                )
            else:
                self[i, col] = "║"

    def draw_fill(self, char, row, col, width, height):
        for i in range(row, row + height):
            for j in range(col, col + width):
                self[i, j] = char

    def draw_empty(self, row, col, width, height):
        for i in range(row, row + height):
            while i >= len(self.data):
                self.push_line()
            for j in range(col, col + width):
                while j >= len(self.data[i]):
                    self.data[i].append(" ")
                if j == col and self.data[i][j] == "─":
                    self.data[i][j] = "╴"
                elif j == col + width - 1 and self.data[i][j] == "─":
                    self.data[i][j] = "╶"
                elif i == row and self.data[i][j] == "│":
                    self.data[i][j] = "╵"
                elif i == row + height - 1 and self.data[i][j] == "│":
                    self.data[i][j] = "╷"
                else:
                    self.data[i][j] = " "

    def draw_box(self, row, col, width, height, clean=False):
        if clean:
            self.draw_empty(row, col, width, height)

        self.draw_hline(row, col, width)
        self.draw_hline(row + height - 1, col, width)
        self.draw_vline(row, col, height)
        self.draw_vline(row, col + width - 1, height)
        return self

    def draw_text(self, text, row, col):
        for i, char in enumerate(text):
            self[row, col + i] = char
        return self

    def draw_vtext(self, text, row, col):
        for i, char in enumerate(text):
            self[row + i, col] = char
        return self

    def reset(self):
        self.data = []
        return self


class AsciiCircuit:
    def __init__(self, width=None):
        if width is None:
            _, width = 0, shutil.get_terminal_size().columns
        self.canvas = AsciiCanvas(width)
        self.qubitrow = {}
        self.bitrow = None
        self.currentcol = 0

    def set_current_col(self, col):
        self.currentcol = max(self.currentcol, col)

    def get_current_col(self):
        return self.currentcol

    def get_qubit_row(self, qubit):
        return self.qubitrow.get(qubit)

    def get_bit_row(self):
        return self.bitrow

    def reset(self):
        self.canvas.reset()
        self.qubitrow = {}
        self.bitrow = None
        self.currentcol = 1
        return self

    def draw_wires(self, qubits, bits):
        # Draw qubit wires and labels
        for i, q in enumerate(qubits):
            row = i * 2 + 1
            qubitstr = f"q[{q}]: "
            self.canvas.draw_text(qubitstr, row, 1)
            self.qubitrow[q] = row
            self.set_current_col(len(qubitstr) + 1)

        if len(bits) > 0:
            row = len(qubits) * 2 + 3
            bitstr = "c: "
            self.canvas.draw_text(bitstr, row, 1)
            self.bitrow = row
            self.set_current_col(len(bitstr) + 1)

        ccol = self.get_current_col()

        # Fill space and draw horizontal line for qubits
        for i in range(len(qubits)):
            row = i * 2
            self.canvas.draw_fill(" ", row, ccol, self.canvas.get_cols() - ccol, 1)
            self.canvas.draw_hline(row + 1, ccol, self.canvas.get_cols() - ccol)

            self.canvas.draw_fill(
                " ", (len(qubits) - 1) * 2 + 3, ccol, self.canvas.get_cols() - ccol, 1
            )

        # Prepare space and draw double horizontal line for bits if more than one bit
        if len(bits) > 0:
            row = len(qubits) * 2 + 2
            self.canvas.draw_fill(" ", row, ccol, self.canvas.get_cols() - ccol, 1)
            self.canvas.draw_double_hline(row + 1, ccol, self.canvas.get_cols() - ccol)

        self.set_current_col(ccol + 1)
        return self

    def draw_operation(self, operation, qubits, bits):
        if not isinstance(operation, mc.Operation):
            raise TypeError("operation must be an instance of Operation")

        namepadding = _gate_name_padding(qubits, bits)
        ccol = self.get_current_col()
        qubitrow = [self.get_qubit_row(q) for q in qubits]
        bitrow = [self.get_bit_row()]

        startrow = min(qubitrow) if not bits else min(qubitrow, bitrow) - 1
        stoprow = max(qubitrow) if not bits else max(qubitrow, bitrow) + 1

        gateheight = stoprow - startrow + 1
        midrow = startrow + gateheight // 2

        operation_str = str(operation)
        gw = operation.asciiwidth(qubits, bits)

        # Draw the box dynamically based on content size
        self.canvas.draw_box(startrow - 1, ccol, gw, gateheight + 2, clean=True)

        # Center the text within the box
        text_start_col = ccol + (gw - len(operation_str)) // 2
        self.canvas.draw_text(operation_str, midrow, ccol + namepadding + 1)

        # Draw indices for qubits if there are more than one
        if len(qubits) > 1:
            for i, qr in enumerate(qubitrow):
                self.canvas.draw_text(str(i), qr, ccol + 1)

        # Draw indices for bits if there are more than one
        if len(bits) > 1:
            bitsstr = len(bits)
            self.canvas.draw_text(bitsstr, bitrow, ccol + 2)

        self.set_current_col(ccol + gw)

    def draw_control(self, operation, qubits, _):
        if not isinstance(operation, mc.Control):
            raise TypeError("operation must be an instance of Operation")

        if operation.op.num_qubits == 1:
            target_row = self.get_qubit_row(qubits[-1])
            control_rows = [self.get_qubit_row(q) for q in qubits[:-1]]
            max_row = max(control_rows + [target_row])
            min_row = min(control_rows + [target_row])
            current_col = self.get_current_col() - 1
            gate_width = operation.asciiwidth([qubits[-1]], [])
            middle_column = current_col + gate_width // 2

            self.canvas.draw_vline(min_row, middle_column, max_row - min_row + 1)

            for row in control_rows:
                self.canvas[row, middle_column] = "●"

            self.draw_operation(operation.get_operation(), [qubits[-1]], [])
        else:
            self.draw_operation(operation, qubits, [])

    def draw_measure(self, qubits, bits):
        if not qubits or not bits:
            raise ValueError("Qubits and bits must be provided for measurement.")

        qubit = qubits[0]
        bit = bits[0]
        qubit_row = self.get_qubit_row(qubit)
        bit_row = self.get_bit_row()
        middle_column = self.get_current_col() + 1

        self.canvas.draw_box(qubit_row - 1, middle_column - 1, 3, 3, clean=True)
        self.canvas.draw_text("M", qubit_row, middle_column)
        self.set_current_col(middle_column + 2)

        if bit_row > qubit_row:
            self.canvas.draw_double_vline(
                qubit_row + 1, middle_column, bit_row - qubit_row
            )

        bit_str = str(bit)
        self.canvas.draw_text(bit_str, bit_row + 1, middle_column)
        self.set_current_col(middle_column + len(bit_str))

        return self

    def draw_barrier(self, barrier, qubits):
        for qubit in qubits:
            qubit_row = self.get_qubit_row(qubit)
            current_col = self.get_current_col()

            self.canvas.draw_text("░", qubit_row, current_col)
            self.canvas.draw_text("░", qubit_row, current_col)
            self.canvas.draw_text("░", qubit_row + 1, current_col)

        self.set_current_col(current_col + 1)
        return self

    def draw_ifstatement(self, if_statement, qubits, bits):
        if not isinstance(if_statement, mc.IfStatement):
            raise TypeError("must be an Ifstatement")

        brow = self.get_bit_row()
        val = if_statement.get_unwrapped_value()
        bstr = _string_with_square(_find_unit_range(bits), ",")
        btext = f"c{bstr} == 0x{val}"

        ccol = self.get_current_col()

        self.canvas.draw_box(brow - 1, ccol, len(btext) + 2, 3, clean=True)
        self.canvas.draw_text(btext, brow, ccol + 1)

        self.draw_operation(if_statement.get_operation(), qubits, [])

        qrow = max(self.get_qubit_row(q) for q in qubits)

        self.canvas.draw_double_vline(qrow + 1, ccol + 1, brow - qrow - 1)

        self.set_current_col(ccol + len(btext) + 2)

    def draw_reset(self, reset, qubits, _):
        if not isinstance(reset, mc.Reset):
            raise TypeError("Must be a Reset Operation")

        qubit_row = self.get_qubit_row(qubits[0])
        current_col = self.get_current_col()
        self.canvas.draw_box(qubit_row - 1, current_col, 5, 3, clean=True)

        # Draw the reset symbol (|0⟩) in the center of the box
        self.canvas.draw_text("|0⟩", qubit_row, current_col + 1)

        # Move the current column index forward to prevent overlapping with future elements
        self.set_current_col(current_col + 5)
        return self

    def draw_parallel(self, parallel, qubits, _):
        if not isinstance(parallel, mc.Parallel):
            raise TypeError("Must be a Parallel")

        op = parallel.get_operation()
        nq = op.num_qubits
        nb = op.num_bits
        ccol = self.get_current_col()

        for i in range(parallel.num_repeats):
            self.currentcol = ccol
            self.draw_operation(op, qubits[nq * i : nq * (i + 1)], [])

        return self

    def draw_instruction(self, instruction):
        if not isinstance(instruction, mc.Instruction):
            raise TypeError("Must be an Instruction")

        return self.draw_operation(
            instruction.get_operation(), instruction._qubits, instruction._bits
        )

    def draw_measurereset(self, qubits, bits):
        if not qubits or not bits:
            raise ValueError("Qubits and bits must be provided for measurement.")

        qubit = qubits[0]
        bit = bits[0]
        qubit_row = self.get_qubit_row(qubit)
        bit_row = self.get_bit_row()
        middle_column = self.get_current_col() + 1

        self.canvas.draw_box(qubit_row - 1, middle_column - 1, 4, 3, clean=True)
        self.canvas.draw_text("MR", qubit_row, middle_column)
        self.set_current_col(middle_column + 3)

        if bit_row > qubit_row:
            self.canvas.draw_double_vline(
                qubit_row + 1, middle_column, bit_row - qubit_row
            )

        bit_str = str(bit)
        self.canvas.draw_text(bit_str, bit_row + 1, middle_column)
        self.set_current_col(middle_column + len(bit_str))

        return self


__all__ = ["AsciiCircuit", "AsciiCanvas"]
