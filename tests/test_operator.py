import unittest
import sympy as sym

from pyxadd import test


class OperatorTest(unittest.TestCase):
    def test_substitution_rename(self):
        operator_1 = test.Operator.compile(sym.S("x"), "<=", sym.S("a"))
        operator_substituted = operator_1.substitute_expressions({"a": "b"})
        self.assertEquals({"x", "b"}, set(operator_substituted.variables))
        self.assertEquals(1, operator_substituted.coefficient("x"))
        self.assertEquals(-1, operator_substituted.coefficient("b"))
        self.assertEquals(0, operator_substituted.rhs)

    def test_substitution_expression(self):
        operator_1 = test.Operator.compile(sym.S("x"), "<=", sym.S("a"))
        operator_substituted = operator_1.substitute_expressions({"a": "2*b + 1"})
        self.assertEquals({"x", "b"}, set(operator_substituted.variables))
        self.assertEquals(1, operator_substituted.coefficient("x"))
        self.assertEquals(-2, operator_substituted.coefficient("b"))
        self.assertEquals(-1, operator_substituted.rhs)
