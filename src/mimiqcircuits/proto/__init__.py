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

WIRE_FORMAT_VERSION = "1.0.0"
"""
Version of the MIMIQ wire format (protobuf schemas + JSON
request/response envelope) spoken by this build. Versioned independently
of the package release so API-only changes do not force client/executor
incompatibility. Must match ``MimiqCircuitsBase.WIRE_FORMAT_VERSION`` on
the Julia side. See ``WIRE_FORMAT.md`` in ``MimiqCircuitsBase.jl/`` for
the surface and bump rules.
"""
