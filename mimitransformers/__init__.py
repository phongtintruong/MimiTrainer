# Copyright 2024 The HuggingFace Team. All rights reserved.
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

# You can still keep the type checking imports if you use a type checker
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_mimi import MimiConfig

# Direct import of MimiModel
from .modeling_mimi import MimiModel  # Assuming modeling_mimi.py is in the same directory

# You might need other classes from modeling_mimi.py, add them here:
# from .modeling_mimi import MimiPreTrainedModel

# If you're using configuration_mimi.py
# from .configuration_mimi import MimiConfig