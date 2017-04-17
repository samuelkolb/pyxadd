import sympy
import unittest
from pyxadd.test import Operator

class ResolutionTest(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_resolution(self):
        x = sympy.sympify("x")
        a = sympy.sympify("a")
        b = sympy.sympify("b")
        operator_1 = Operator.compile(x, "<=", a)
        operator_2 = Operator.compile(x, ">=", b)
        resolved = operator_1.resolve("x", operator_2)
        print(resolved)


if __name__ == '__main__':
    unittest.main()

