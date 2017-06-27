import unittest

from pyxadd.parse import XADDParser


class TestParse(unittest.TestCase):
    test_bool_var = "a"
    test_int_var = "x"

    @staticmethod
    def compile(string):
        return XADDParser().parse_xadd(string)

    def test_bool_values(self, string, val1, val2):
        xadd = self.compile(string)
        self.assertEquals(val1, xadd.evaluate({self.test_bool_var: True}))
        self.assertEquals(val2, xadd.evaluate({self.test_bool_var: False}))

    def test_int_values(self, string, x_values, results):
        xadd = self.compile(string)
        for x_value, result in zip(x_values, results):
            self.equals = self.assertEquals(result, xadd.evaluate({self.test_int_var: x_value}))

    def test_constant(self, string, val):
        self.assertEquals(val, self.compile(string).evaluate({}))

    def testSimpleIte(self):
        val1 = 13
        val2 = 19
        wmi = "(ite (var bool {}) (const real {}) (const real {}))".format(self.test_bool_var, val1, val2)
        self.test_bool_values(wmi, val1, val2)

    def testSimplePower(self):
        power = 2
        values = [1, 5]
        wmi = "(^ (var real {}) (const real {})".format(self.test_int_var, power)
        self.test_int_values(wmi, values, map(lambda it: it * it, values))

    def testSimpleNot(self):
        self.test_constant("(~ (const bool true))", 0)

    def testSimpleAnd(self):
        self.test_bool_values("(& (const bool true) (var bool {})".format(self.test_bool_var), 1, 0)

    def testSimpleOr(self):
        self.test_bool_values("(| (const bool false) (var bool {})".format(self.test_bool_var), 1, 0)

    def testSimpleTimes(self):
        constValue = 17
        values = [1, 5]
        wmi = "(* (var real {}) (const real {})".format(self.test_int_var, constValue)
        self.test_int_values(wmi, values, map(lambda it: it * constValue, values))

    def testSimplePlus(self):
        constValue = 17
        values = [1, 5]
        wmi = "(+ (var real {}) (const real {})".format(self.test_int_var, constValue)
        self.test_int_values(wmi, values, map(lambda it: it + constValue, values))

    def testSimpleLE(self):
        values = [1, 5]
        wmi = "(<= (var real {}) (const real 2))".format(self.test_int_var)
        self.test_int_values(wmi, values, [1, 0])

    def testSimpleLT(self):
        values = [1, 5]
        wmi = "(<= (var real {}) (const real 2))".format(self.test_int_var)
        self.test_int_values(wmi, values, [1, 0])

    def testSimpleConstReal(self):
        number = 2.
        self.test_constant("(const real {})".format(number), number)

    def testSimpleConstBool(self):
        self.test_constant("(const bool true)", 1)

    def testSimpleVar(self):
        values = [1, 5]
        wmi = "(var real {})".format(self.test_int_var)
        self.test_int_values(wmi, values, values)

    def testTautology(self):
        wmi = "(| (const bool true) (var bool {}))".format(self.test_bool_var)
        self.test_constant(wmi, 1)
