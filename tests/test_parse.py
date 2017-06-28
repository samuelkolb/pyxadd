import unittest

from pyxadd.parse import XADDParser


class TestParse(unittest.TestCase):
    def setUp(self):
        self.test_bool_var = "a"
        self.test_int_var = "x"

    @staticmethod
    def compile(string):
        return XADDParser().parse_xadd(string)

    def _test_bool_values(self, string, val1, val2):
        xadd = self.compile(string)
        self.assertEquals(val1, xadd.evaluate({self.test_bool_var: True}))
        self.assertEquals(val2, xadd.evaluate({self.test_bool_var: False}))

    def _test_int_values(self, string, x_values, results):
        xadd = self.compile(string)
        for x_value, result in zip(x_values, results):
            self.equals = self.assertEquals(result, xadd.evaluate({self.test_int_var: x_value}))

    def _test_constant(self, string, val):
        self.assertEquals(val, self.compile(string).evaluate({}))

    def test_simple_ite(self):
        val1 = 13
        val2 = 19
        wmi = "(ite (var bool {}) (const real {}) (const real {}))".format(self.test_bool_var, val1, val2)
        self._test_bool_values(wmi, val1, val2)

    def test_simple_power(self):
        power = 2
        values = [1, 5]
        wmi = "(^ (var real {}) (const real {})".format(self.test_int_var, power)
        self._test_int_values(wmi, values, map(lambda it: it * it, values))

    def test_simple_not(self):
        self._test_constant("(~ (const bool true))", 0)

    def test_simple_and(self):
        self._test_bool_values("(& (const bool true) (var bool {})".format(self.test_bool_var), 1, 0)

    def test_simple_or(self):
        self._test_bool_values("(| (const bool false) (var bool {})".format(self.test_bool_var), 1, 0)

    def test_simple_times(self):
        const_value = 17
        values = [1, 5]
        wmi = "(* (var real {}) (const real {})".format(self.test_int_var, const_value)
        self._test_int_values(wmi, values, map(lambda it: it * const_value, values))

    def test_simple_plus(self):
        const_value = 17
        values = [1, 5]
        wmi = "(+ (var real {}) (const real {})".format(self.test_int_var, const_value)
        self._test_int_values(wmi, values, map(lambda it: it + const_value, values))

    def test_simple_le(self):
        values = [1, 5]
        wmi = "(<= (var real {}) (const real 2))".format(self.test_int_var)
        self._test_int_values(wmi, values, [1, 0])

    def test_simple_lt(self):
        values = [1, 5]
        wmi = "(<= (var real {}) (const real 2))".format(self.test_int_var)
        self._test_int_values(wmi, values, [1, 0])

    def test_simple_const_real(self):
        number = 2.
        self._test_constant("(const real {})".format(number), number)

    def test_simple_const_bool_true(self):
        self._test_constant("(const bool true)", 1)

    def test_simple_const_bool_false(self):
        self._test_constant("(const bool false)", 0)

    def test_simple_real_var(self):
        values = [1, 5]
        wmi = "(var real {})".format(self.test_int_var)
        self._test_int_values(wmi, values, values)

    def test_simple_bool_var(self):
        wmi = "(var bool {})".format(self.test_bool_var)
        self._test_bool_values(wmi, 1, 0)

    def test_tautology(self):
        wmi = "(| (const bool true) (var bool {}))".format(self.test_bool_var)
        self._test_constant(wmi, 1)
