import re

from pyxadd.diagram import Diagram, Pool
from pyxadd.test import LinearTest, BinaryTest


class Builder(object):
    inequality_pattern = r"(.*)(<=?|>=?)(.*)"

    def __init__(self, pool=None):
        if pool is None:
            pool = Pool()
        assert isinstance(pool, Pool)
        self._pool = pool

    @property
    def pool(self):
        return self._pool

    def vars(self, v_type, *args):
        for arg in args:
            self.pool.add_var(arg, v_type)

    def ints(self, *args):
        self.vars("int", *args)

    def terminal(self, exp):
        if exp == "1" or exp == 1:
            node_id = self.pool.one_id
        elif exp == "0" or exp == 0:
            node_id = self.pool.zero_id
        else:
            node_id = self.pool.terminal(exp)
        return Diagram(self.pool, node_id)

    def exp(self, exp):
        return self.terminal(exp)

    def test(self, lhs, symbol=None, rhs=None):
        if symbol is None and rhs is None:
            if isinstance(lhs, str) and re.match(self.inequality_pattern, lhs):
                match = re.match(self.inequality_pattern, lhs)
                return self.test(match.group(1), match.group(2), match.group(3))
            if self.pool.get_var_type(lhs) != "bool":
                raise RuntimeError("'{}' is not a variable of type 'bool'".format(lhs))
            test = BinaryTest(lhs)
        else:
            test = LinearTest(lhs, symbol, rhs)
        return Diagram(self.pool, self.pool.internal(test, self.pool.one_id, self.pool.zero_id))

    def limit(self, var, lb, ub):
        diagram = self.terminal(1)
        if lb is not None:
            diagram &= self.test(var, ">=", lb)
        if ub is not None:
            diagram &= self.test(var, "<=", ub)
        return diagram

    def ite(self, if_diagram, then_diagram, else_diagram=None):
        if not isinstance(if_diagram, Diagram):
            if_diagram = self.test(if_diagram)
        if not isinstance(then_diagram, Diagram):
            then_diagram = self.terminal(then_diagram)
        if not isinstance(else_diagram, Diagram):
            else_diagram = self.terminal(else_diagram)
        assert if_diagram.pool == then_diagram.pool == else_diagram.pool == self.pool

        return if_diagram * then_diagram + (~if_diagram) * else_diagram
