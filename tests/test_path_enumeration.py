import unittest

from pyxadd import build
from pyxadd import matrix_vector


class TestPathEnumeration(unittest.TestCase):
    def test_no_numeric_bounds(self):
        b = build.Builder()
        b.ints("x", "y")

        diagram = b.test("x <= y + 10") * b.test("x >= y - 10") * b.exp("10 * x * y")
        matrix_vector.sum_out(diagram.pool, diagram.root_id, ["x"])
