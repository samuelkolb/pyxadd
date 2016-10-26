from pyxadd.diagram import Diagram, Pool
from pyxadd.test import Test


class Builder(object):
    def __init__(self, pool):
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

    def test(self, lhs, symbol, rhs):
        return Diagram(self.pool, self.pool.internal(Test(lhs, symbol, rhs), self.pool.one_id, self.pool.zero_id))

    def limit(self, var, lb, ub):
        diagram = self.terminal(1)
        if lb is not None:
            diagram &= self.test(var, ">=", lb)
        if ub is not None:
            diagram &= self.test(var, "<=", ub)
        return diagram

    def ite(self, if_diagram, then_diagram, else_diagram):
        assert isinstance(if_diagram, Diagram)
        assert isinstance(then_diagram, Diagram)
        assert isinstance(else_diagram, Diagram)
        assert if_diagram.pool == then_diagram.pool == else_diagram.pool == self.pool

        return if_diagram * then_diagram + (~if_diagram) * else_diagram
