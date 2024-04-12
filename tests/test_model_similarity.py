# Project Name: simple-ai-benchmarking
# File Name: test_model_similarity.py
# Author: Timo Leitritz
# Copyright (C) 2024 Timo Leitritz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from simple_ai_benchmarking.model_comparison import calculate_model_similarity
from simple_ai_benchmarking.config_structures import ModelIdentifier

MAXIMAL_ALLOWED_RELATIVE_PARAM_DIFFERENCE = 0.01


def test_model_similarity_resnet50() -> None:

    MODEL = ModelIdentifier.RESNET50

    rel_diff, abs_diff = calculate_model_similarity(MODEL)

    assert rel_diff < MAXIMAL_ALLOWED_RELATIVE_PARAM_DIFFERENCE, "Models are not similar enough"


def test_model_similarity_vitb16() -> None:

    MODEL = ModelIdentifier.VIT_B_16

    rel_diff, abs_diff = calculate_model_similarity(MODEL)

    assert rel_diff < MAXIMAL_ALLOWED_RELATIVE_PARAM_DIFFERENCE, "Models are not similar enough"


def test_model_similarity_simpleclassificationcnn() -> None:

    MODEL = ModelIdentifier.SIMPLE_CLASSIFICATION_CNN

    rel_diff, abs_diff = calculate_model_similarity(MODEL)

    assert rel_diff < MAXIMAL_ALLOWED_RELATIVE_PARAM_DIFFERENCE, "Models are not similar enough"
