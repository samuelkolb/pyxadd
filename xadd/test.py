import sympy


class Operator:
    def __init__(self, op):
        self._op = op

    def strict(self):
        if self._op == "<=":
            return Operators.get("<")
        if self._op == ">=":
            return Operators.get(">")
        return self

    def weak_int(self, lhs):
        if self._op == "<":
            return lhs + 1, Operators.get("<=")
        if self._op == ">":
            return lhs - 1, Operators.get(">=")
        return lhs, self

    def weak(self):
        if self._op == "<":
            return Operators.get("<=")
        if self._op == ">":
            return Operators.get(">=")
        return self

    def flip(self):
        if self._op == "<=":
            return Operators.get(">=")
        if self._op == ">=":
            return Operators.get("<=")
        if self._op == "<":
            return Operators.get(">")
        if self._op == ">":
            return Operators.get("<")

    def test(self, lhs, rhs):
        if self._op == "<=":
            return lhs <= rhs
        if self._op == ">=":
            return lhs >= rhs
        if self._op == "<":
            return lhs < rhs
        if self._op == ">":
            return lhs > rhs
        if self._op == "=":
            return lhs == rhs

    def update_bounds_int(self, lb, ub, rhs):
        if self._op == "<=":
            return lb, min(ub, rhs)
        if self._op == ">=":
            return max(lb, rhs), ub
        if self._op == "<":
            return lb, min(ub, rhs - 1)
        if self._op == ">":
            return max(lb, rhs + 1), ub
        if self._op == "=":
            return max(lb, rhs), min(ub, rhs)

    def flip_int(self):
        if self._op == "<=":
            return Operators.get(">")
        if self._op == ">=":
            return Operators.get("<")
        if self._op == "<":
            return Operators.get(">=")
        if self._op == ">":
            return Operators.get("<=")

    def __repr__(self):
        return self._op

    def __hash__(self):
        return hash(self._op)

    def __eq__(self, other):
        # noinspection PyProtectedMember
        return isinstance(other, Operator) and self._op == other._op


class Operators:
    _objects = {s: Operator(s) for s in ["<=", "<", ">=", ">", "="]}

    @staticmethod
    def get(symbol):
        return Operators._objects[symbol]


class Test:
    def __init__(self, expression, operator):
        if not isinstance(expression, sympy.Basic):
            expression = sympy.sympify(expression)
        self._expression = expression
        if isinstance(operator, str):
            operator = Operators.get(operator)
        self._operator = operator

    @property
    def expression(self):
        return self._expression

    @property
    def operator(self):
        return self._operator

    def update_bounds(self, var, lb, ub, test=True):
        expression, operator = self.rewrite(var, test=test)
        return operator.update_bounds_int(lb, ub, expression)

    def rewrite(self, var, test=True):
        if var not in self.expression.free_symbols:
            raise RuntimeError("Variable {} not in expression {} {} 0".format(var, self.expression, self.operator))
        coefficient = self.expression.coeff(var, 1)
        expression = self.expression.subs(var, 0) * -coefficient
        if coefficient > 0:
            operator = self.operator
        elif coefficient < 0:
            operator = self.operator.flip()
        else:
            raise RuntimeError("Variable {} not in expression {} {} 0".format(var, self.expression, self.operator))
        return (expression, operator) if test else (expression, operator.flip_int())

    def __repr__(self):
        return " ".join([str(self.expression), str(self.operator), "0"])

    def __hash__(self):
        return hash((self.expression, self.operator))

    def __eq__(self, other):
        return isinstance(other, Test) and self.expression == other.expression and self.operator == other.operator
