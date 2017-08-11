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
        self.assertEquals({"a", "b"}, set(resolved.variables))
        self.assertEquals(-1, resolved.coefficient("a"))
        self.assertEquals(1, resolved.coefficient("b"))
        self.assertEquals(0, resolved.rhs)

    def test_resolution2(self):
        x = sympy.sympify("x")
        y = sympy.sympify("y")

        # 1.0 <= y - 1
        # x >= 1 resolve
        operator_1 = Operator.compile(x, "<=", 1.0)
        operator_2 = Operator.compile(x - y, "<=", -1)
        resolved = operator_1.switch_direction().resolve("x", operator_2)
        self.assertEquals(["y"], resolved.variables)
        self.assertEquals(-1, resolved.coefficient("y"))
        self.assertEquals(-2, resolved.rhs)


if __name__ == '__main__':
    unittest.main()

