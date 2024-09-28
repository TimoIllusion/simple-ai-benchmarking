# Project Name: simple-ai-benchmarking
# File Name: test_publish.py
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

import unittest

from simple_ai_benchmarking.database import get_git_commit_hash_from_package_version


class TestPublishDatabase(unittest.TestCase):

    def test_get_git_commit_hash_from_package_version(self):

        git_commit = get_git_commit_hash_from_package_version()
        self.assertTrue(git_commit != "N/A" and git_commit != None)


# Entry point for running the tests
if __name__ == "__main__":
    unittest.main()
