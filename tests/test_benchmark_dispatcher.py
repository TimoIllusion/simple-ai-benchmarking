# Project Name: simple-ai-benchmarking
# File Name: test_default_execution.py
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


from simple_ai_benchmarking.entrypoints import BenchmarkDispatcher
from simple_ai_benchmarking.config_structures import AIFramework


def test_pt_benchmark() -> None:

    dispatcher = BenchmarkDispatcher(AIFramework.PYTORCH)
    dispatcher.NUM_BATCHES_INFERENCE = 2
    dispatcher.NUM_BATCHES_TRAINING = 2
    dispatcher.BATCH_SIZE = 2
    dispatcher.REPETITIONS = 1
    dispatcher.NUM_BATCHES_INFERENCE = 5
    dispatcher.NUM_BATCHES_TRAINING = 3
    dispatcher.run()


def test_tf_benchmark() -> None:

    dispatcher = BenchmarkDispatcher(AIFramework.TENSORFLOW)
    dispatcher.NUM_BATCHES_INFERENCE = 2
    dispatcher.NUM_BATCHES_TRAINING = 2
    dispatcher.BATCH_SIZE = 2
    dispatcher.REPETITIONS = 1
    dispatcher.NUM_BATCHES_INFERENCE = 5
    dispatcher.NUM_BATCHES_TRAINING = 3
    dispatcher.run()


if __name__ == "__main__":
    test_pt_benchmark()
    test_tf_benchmark()
