#
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


def _allunique(lst):
    seen = set()
    for element in lst:
        if element in seen:
            return False
        seen.add(element)
    return True


def _partition(arr, indices):
    vec = list(arr)
    partitions = [vec[: indices[0]]]

    for i in range(1, len(indices)):
        partitions.append(vec[indices[i - 1] : indices[i]])

    return partitions


def _rangestring(obj):
    if isinstance(obj, range):
        return f"{obj.start}:{obj.stop - 1}"
    return str(obj)


def _string_with_square(arr, sep):
    if isinstance(arr, (range, list)):
        arr = [arr]
    return (
        "["
        + sep.join(
            map(
                lambda e: sep.join(map(_rangestring, e))
                if isinstance(e, list)
                else str(e),
                arr,
            )
        )
        + "]"
    )


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
            narr.append((range(rangestart, rangestop + 1)))
            rangestart = v
            rangestop = v

    if rangestart == rangestop:
        narr.append(rangestart)
    else:
        narr.append((range(rangestart, rangestop + 1)))

    return narr


__all__ = ["_allunique", "_partition", "_string_with_square", "_find_unit_range"]
