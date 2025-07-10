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

    for v in arr[1:]:
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


def _gate_name_padding(qubits, bits, zvars):
    nq = len(qubits)
    qubits_padding = 0 if nq in [1, 0] else math.floor(math.log10(nq)) + 2

    bitspadding = 0 if not bits else len(str(bits)) + 1
    zvarspadding = 0 if not zvars else len(str(zvars)) + 1

    return max(qubits_padding, bitspadding, zvarspadding)


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
                existing = self.data[i][j]
                if existing in ("─", "╶", "╴", "│", "╷", "╵", " "):
                    # Only clear common wire/line characters
                    if j == col and existing == "─":
                        self.data[i][j] = "╴"
                    elif j == col + width - 1 and existing == "─":
                        self.data[i][j] = "╶"
                    elif i == row and existing == "│":
                        self.data[i][j] = "╵"
                    elif i == row + height - 1 and existing == "│":
                        self.data[i][j] = "╷"
                    else:
                        self.data[i][j] = " "

    def draw_box(self, row, col, width, height, clean=False):
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
        self.zvarrow = None
        self.nonerow = None
        self.currentcol = 0

    def set_current_col(self, col):
        self.currentcol = max(self.currentcol, col)

    def get_current_col(self):
        return self.currentcol

    def get_qubit_row(self, qubit):
        return self.qubitrow.get(qubit)

    def get_bit_row(self):
        return self.bitrow

    def get_zvars_row(self):
        return self.zvarrow

    def reset(self):
        self.canvas.reset()
        self.qubitrow = {}
        self.bitrow = None
        self.zvars = None
        self.nonerow = None
        self.currentcol = 1
        return self

    def draw_wires(self, qubits, bits, zvars):
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

        if len(zvars) > 0:
            if len(bits) < 1:
                row = (len(qubits)) * 2 + 3
            else:
                row = (len(qubits) + 1) * 2 + 3

            zstr = "z: "
            self.canvas.draw_text(zstr, row, 1)
            self.zvarrow = row
            self.set_current_col(len(zstr) + 1)

        ccol = self.get_current_col() - 3

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

        if len(zvars) > 0:
            if len(bits) < 1:
                row = (len(qubits)) * 2 + 2
            else:
                row = (len(qubits) + 1) * 2 + 2

            self.canvas.draw_fill(" ", row, ccol, self.canvas.get_cols() - ccol, 1)
            self.canvas.draw_double_hline(row + 1, ccol, self.canvas.get_cols() - ccol)

        self.set_current_col(ccol + 1)
        return self

    def draw_operation(self, operation, qubits, bits=None, zvars=None):
        # Initialize default values for bits and zvars if they are None
        bits = bits or []
        zvars = zvars or []

        # Calculate padding for qubits, bits, and zvars
        namepadding = _gate_name_padding(qubits, bits, zvars)
        ccol = self.get_current_col()

        qubitrow = [self.get_qubit_row(q) for q in qubits]
        bitrow = self.get_bit_row()
        zvarrow = self.get_zvars_row()

        if not qubits and not bits and not zvars:
            # Draw the None row only if it's not already drawn
            if self.nonerow is None:
                none_row_start = (
                    len(self.qubitrow)
                    + (1 if self.bitrow else 0)
                    + (1 if self.zvarrow else 0)
                ) * 2 + 3
                self.nonerow = none_row_start
                ccol = 0
                self.canvas.draw_fill(
                    " ", none_row_start, ccol, self.canvas.get_cols(), 1
                )
                self.canvas.draw_double_hline(
                    none_row_start, ccol, self.canvas.get_cols()
                )

            # Now draw the operation on the None line
            none_row = self.nonerow
            ccol = self.get_current_col()
            gw = operation.asciiwidth([], [], [])
            self.canvas.draw_box(none_row - 1, ccol, gw, 3, clean=True)  # 3 rows height
            self.canvas.draw_text(str(operation), none_row, ccol + 1)
            self.set_current_col(ccol + gw)
            return self

        # Determine start and stop rows for qubits, bits, and zvars
        if bits or zvars:
            (
                qubit_startrow,
                qubit_stoprow,
                bits_startrow,
                bits_stoprow,
                zvar_startrow,
                zvar_stoprow,
            ) = _used_row_bounds(qubitrow, bitrow, zvarrow, bits, zvars)

            namepadding = _gate_name_padding(qubits, bits, zvars)

            if qubitrow and bits and zvars:
                qubit_gateheight = qubit_stoprow - qubit_startrow + 1
                qubit_midrow = qubit_startrow + qubit_gateheight // 2

                gw_qubits = operation.asciiwidth(qubits, [], [])
                gw_bits = operation.asciiwidth([], bits, [])
                gw_zvars = operation.asciiwidth([], [], zvars)
                gw = max(gw_qubits, gw_bits, gw_zvars)
                midcol = ccol + gw // 2

                _draw_box_with_centered_text(
                    self.canvas,
                    qubit_startrow - 1,
                    ccol,
                    gw,
                    qubit_gateheight + 2,
                    str(operation),
                    qubit_midrow,
                )

                if bitrow is not None:
                    bit_string = ",".join(map(str, bits))
                    bit_col = midcol - len(bit_string) // 2
                    self.canvas.draw_text(bit_string, bitrow + 1, bit_col)

                if zvarrow is not None:
                    zvar_string = ",".join(map(str, zvars))
                    zvar_col = midcol - len(zvar_string) // 2
                    self.canvas.draw_text(zvar_string, zvarrow + 1, zvar_col)

                if bitrow is not None:
                    self.canvas.draw_double_vline(
                        qubit_stoprow + 1, midcol, bitrow - qubit_stoprow
                    )

                if zvarrow is not None:
                    None

                endcol = ccol + gw

            elif qubitrow and zvars:
                qubit_gateheight = qubit_stoprow - qubit_startrow + 1
                qubit_midrow = qubit_startrow + qubit_gateheight // 2
                gw = max(
                    operation.asciiwidth(qubits, [], []),
                    operation.asciiwidth([], [], zvars),
                )
                midcol = ccol + gw // 2

                _draw_box_with_centered_text(
                    self.canvas,
                    qubit_startrow - 1,
                    ccol,
                    gw,
                    qubit_gateheight + 2,
                    str(operation),
                    qubit_midrow,
                )

                self.canvas.draw_text(",".join(map(str, zvars)), zvarrow + 1, midcol)
                self.canvas.draw_double_vline(
                    qubit_stoprow + 1, midcol, zvar_startrow - qubit_stoprow
                )

                endcol = ccol + gw

            elif qubitrow and bits:
                qubit_gateheight = qubit_stoprow - qubit_startrow + 1
                qubit_midrow = qubit_startrow + qubit_gateheight // 2
                gw = max(
                    operation.asciiwidth(qubits, [], []),
                    operation.asciiwidth([], bits, []),
                )
                midcol = ccol + gw // 2

                _draw_box_with_centered_text(
                    self.canvas,
                    qubit_startrow - 1,
                    ccol,
                    gw,
                    qubit_gateheight + 2,
                    str(operation),
                    qubit_midrow,
                )

                bit_string = ",".join(map(str, bits))
                bit_col = midcol - len(bit_string) // 2
                self.canvas.draw_text(bit_string, bits_startrow + 1, bit_col)

                self.canvas.draw_double_vline(
                    qubit_stoprow + 1, midcol, bits_stoprow - qubit_stoprow
                )

                endcol = ccol + gw

            elif zvars and not qubitrow:
                startrow = min(zvar_startrow, qubit_startrow) - 1
                stoprow = max(zvar_stoprow, qubit_stoprow) + 1
                gateheight = stoprow - startrow + 1
                midrow = startrow + gateheight // 2

                gw = operation.asciiwidth(qubits, bits, zvars)
                zstr = f"{','.join(map(str, zvars))} ═> "
                self.canvas.draw_box(startrow, ccol, gw, gateheight, clean=True)
                self.canvas.draw_text(zstr, midrow, ccol + 1)
                self.canvas.draw_text(str(operation), midrow, ccol + len(zstr) + 1)

                midcol = ccol + gw // 2
                endcol = ccol + gw

            elif bits and not qubitrow:
                startrow = min(bits_startrow, qubit_startrow) - 1
                stoprow = max(bits_stoprow, qubit_stoprow) + 1
                gateheight = stoprow - startrow + 1
                midrow = startrow + gateheight // 2

                gw = operation.asciiwidth(qubits, bits, zvars)
                bstr = f"{','.join(map(str, bits))} ═> "
                self.canvas.draw_box(startrow, ccol, gw, gateheight, clean=True)
                self.canvas.draw_text(bstr, midrow, ccol + 1)
                self.canvas.draw_text(str(operation), midrow, ccol + len(bstr) + 1)

                midcol = ccol + gw // 2
                endcol = ccol + gw

        else:
            startrow = min(qubitrow) - 1
            stoprow = max(qubitrow) + 1
            gateheight = stoprow - startrow + 1
            midrow = startrow + gateheight // 2

            # Draw the box around the operation
            gw = operation.asciiwidth(qubits, bits, zvars)
            self.canvas.draw_box(startrow, ccol, gw, gateheight, clean=True)
            # Properly centered
            self.canvas.draw_text(str(operation), midrow, ccol + namepadding + 1)
            midcol = ccol + gw // 2
            endcol = ccol + gw + namepadding

        # If more than one qubit, label each qubit within the box
        if len(qubits) > 1:
            for i, qr in enumerate(qubitrow):
                self.canvas.draw_text(str(i), qr, ccol + 1)
        self.set_current_col(endcol)

        return self

    def draw_control(self, operation, qubits, _):
        if not isinstance(operation, mc.Control):
            raise TypeError("operation must be an instance of Operation")

        if operation.op.num_qubits == 1:
            target_row = self.get_qubit_row(qubits[-1])
            control_rows = [self.get_qubit_row(q) for q in qubits[:-1]]
            max_row = max(control_rows + [target_row])
            min_row = min(control_rows + [target_row])
            current_col = self.get_current_col()
            gate_width = operation.get_operation().asciiwidth([qubits[-1]], [], [])
            middle_column = current_col + gate_width // 2

            self.canvas.draw_vline(min_row, middle_column, max_row - min_row + 1)

            for row in control_rows:
                self.canvas[row, middle_column] = "●"

            self.draw_operation(operation.get_operation(), [qubits[-1]], [], [])
        else:
            self.draw_operation(operation, qubits, [])

    def draw_barrier(self, barrier, qubits, bits, zvars):
        for qubit in qubits:
            qubit_row = self.get_qubit_row(qubit)
            current_col = self.get_current_col()

            self.canvas.draw_text("░", qubit_row, current_col)
            self.canvas.draw_text("░", qubit_row, current_col)
            self.canvas.draw_text("░", qubit_row + 1, current_col)

        self.set_current_col(current_col + 1)
        return self

    def draw_ifstatement(self, g, qubits, bits, zvars):
        qubitrow = [self.get_qubit_row(q) for q in qubits]
        bitrow = self.get_bit_row()

        nb = len(bits)
        val = g.get_bitstring()

        ccol = self.get_current_col()

        bstr = _string_with_square(_find_unit_range(bits), ",")
        btext = f"c{bstr}==" + val.to01()

        self.canvas.draw_box(bitrow - 1, ccol, len(btext) + 2, 3, clean=True)
        self.canvas.draw_text(btext, bitrow, ccol + 1)

        self.set_current_col(ccol + len(btext) + 2)
        ifcol = self.get_current_col()

        qstartrow = min(qubitrow) - 1
        qstoprow = max(qubitrow) + 1
        qw = g.op.asciiwidth(qubits, [], [])
        qh = qstoprow - qstartrow + 1
        qmh = qstartrow + qh // 2
        namepadding = _gate_name_padding(qubits, [], [])

        self.canvas.draw_box(qstartrow, ifcol, qw, qh, clean=True)

        for i, qr in enumerate(qubitrow):
            self.canvas.draw_text(str(i), qr, ifcol + 1)

        self.canvas.draw_text(repr(g.op), qmh, ifcol + namepadding + 1)

        midcol = ifcol + qw // 2

        self.canvas.draw_double_vline(qstoprow, midcol, bitrow - qstoprow)
        self.canvas.draw_double_hline(bitrow - 1, ifcol, midcol - ifcol)
        self.canvas.draw_text("╝", bitrow - 1, midcol)
        self.canvas.draw_text("○", bitrow - 1, ifcol)

        self.set_current_col(ifcol + qw)

    def draw_reset(self, reset, qubits, _, zvars):
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

    def draw_parallel(self, parallel, qubits, _, zvars):
        if not isinstance(parallel, mc.Parallel):
            raise TypeError("Must be a Parallel")

        op = parallel.get_operation()
        nq = op.num_qubits
        nb = op.num_bits
        ccol = self.get_current_col()

        for i in range(parallel.num_repeats):
            self.currentcol = ccol
            self.draw_operation(op, qubits[nq * i : nq * (i + 1)], [], [])

        return self

    def draw_paulistring(self, g, qubits, bits, zvars):
        ccol = self.get_current_col()
        qubitrow = [self.get_qubit_row(q) for q in qubits]

        startrow = min(qubitrow) - 1
        stoprow = max(qubitrow) + 1
        gateheight = stoprow - startrow + 1

        gw = g.asciiwidth(qubits, bits, zvars)

        self.canvas.draw_box(startrow, ccol, gw, gateheight, clean=True)

        for i, qr in enumerate(qubitrow):
            pauli_char = g.pauli[i]
            label = f"{i}: {pauli_char}" if len(qubits) > 1 else f"{pauli_char}"
            self.canvas.draw_text(label, qr, ccol + 1)

        self.set_current_col(ccol + gw)

        return self

    def draw_instruction(self, instruction):
        if not isinstance(instruction, mc.Instruction):
            raise TypeError("Must be an Instruction")

        return self.draw_operation(
            instruction.get_operation(),
            instruction._qubits,
            instruction._bits,
            instruction._zvars,
        )


def _used_row_bounds(qubitrow, bitrow, zvarrow, bits, zvars):
    used_qubit_rows = [r for r in qubitrow if r is not None]
    qubit_start = min(used_qubit_rows, default=float("inf"))
    qubit_stop = max(used_qubit_rows, default=0)

    bit_start = bitrow if bits else float("inf")
    bit_stop = bitrow if bits else 0

    zvar_start = zvarrow if zvars else float("inf")
    zvar_stop = zvarrow if zvars else 0

    return qubit_start, qubit_stop, bit_start, bit_stop, zvar_start, zvar_stop


def _draw_box_with_centered_text(
    canvas, row, col, width, height, text, center_row=None
):
    canvas.draw_box(row, col, width, height, clean=True)
    if center_row is None:
        center_row = row + height // 2
    canvas.draw_text(text, center_row, col + (width - len(text)) // 2)


__all__ = ["AsciiCircuit", "AsciiCanvas"]
