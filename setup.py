# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Install script."""

import setuptools

setuptools.setup(
    name="NeuralComprehension",
    version="1.0.0",
    url="https://github.com/WENGSYX/Neural-Comprehension",
    author="Yixuan Weng",
    author_email="wengsyx@qq.com",
    description="Incorporating Neural Network Compilation into a Language Model Framework",
    packages=setuptools.find_packages(),
    install_requires=[
        "pytorch",
        'tracr',
        'transformers'
    ],
)
