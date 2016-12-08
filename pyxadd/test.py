import sympy


class Operator:

    constructors = {
        ">": lambda lhs, rhs: GreaterThan(lhs, rhs),
        "<": lambda lhs, rhs: LessThan(lhs, rhs),
        ">=": lambda lhs, rhs: GreaterThanEqual(lhs, rhs),
        "<=": lambda lhs, rhs: LessThanEqual(lhs, rhs),
    }

    def __init__(self, symbol, lhs, rhs):
        self._symbol = symbol
        self._lhs = lhs
        self._rhs = rhs

    @property
    def symbol(self):
        return self._symbol

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    @property
    def children(self):
        return [self.lhs, self.rhs]

    @property
    def variables(self):
        return list(self.lhs.keys())

    def coefficient(self, var, force=False):
        if var in self.lhs:
            return self.lhs[var]
        if not force:
            return 0
        raise RuntimeError("Missing variable {}, only found {}".format(var, self.variables))

    def strict(self):
        raise NotImplementedError()

    def weak(self):
        raise NotImplementedError()

    def __invert__(self):
        raise NotImplementedError()

    def flip(self):
        """
        Flips this operator
        :return Operator:
        """
        raise NotImplementedError()

    def _update(self, lhs, rhs):
        raise NotImplementedError()

    def update_bounds(self, var, lb, ub):
        if len(self.variables) != 1:
            raise RuntimeError("Test does not have exactly one variable (it has {})".format(self.variables))
        rewritten = self.times(1 / self.coefficient(var, force=True))
        assert isinstance(rewritten, Operator)
        lb = -sympy.oo if lb is None else lb
        ub = sympy.oo if ub is None else ub
        return rewritten._update_bounds(lb, ub)

    def _update_bounds(self, lb, ub):
        raise NotImplementedError()

    def to_canonical(self):
        raise NotImplementedError()

    def invert_lhs(self):
        return {k: -v for k, v in self.lhs.items()}

    def times(self, constant):
        lhs = {k: v * constant for k, v in self.lhs.items()}
        rhs = self.rhs * constant
        operator = self if constant >= 0 else self.flip()
        return operator._update(lhs, rhs)

    def is_singular(self):
        return len(self.lhs) == 1

    def evaluate(self, assignment):
        lhs_value = 0
        for v, c in self.lhs.items():
            if v in assignment:
                lhs_value += assignment[v] * c
            else:
                raise RuntimeError("Assignment {} did not include all variables required for test {}"
                                   .format(assignment, self))
        return self.evaluate_values(lhs_value, self.rhs)

    def evaluate_values(self, lhs_value, rhs_value):
        raise NotImplementedError()

    def __repr__(self):
        return "{} {} {}".format(" + ".join("{}{}".format(v, k) for k, v in self.lhs.items()), self.symbol, self.rhs)

    def __hash__(self):
        lhs_tuples = tuple((k, self.lhs[k]) for k in sorted(self.lhs.keys()))
        return hash((lhs_tuples, self.symbol, self.rhs))

    def __eq__(self, other):
        return isinstance(other, Operator) and self.symbol == other.symbol and self.children == other.children

    @staticmethod
    def compile(lhs, symbol, rhs):
        """
        Compiles an operator datastructure from the given parameters
        :param lhs: The left hand side expression (sympy expression)
        :param symbol: The comparison symbol
        :param rhs: The right hand side expression (sympy expression or number)
        :return Operator:
        """
        expression = lhs - rhs
        lhs = {str(var): float(expression.coeff(var, 1)) for var in expression.free_symbols}
        rhs = float(-sympy.lambdify(expression.free_symbols, expression)(*([0] * len(expression.free_symbols))))
        operator = Operator.constructors[symbol](lhs, rhs)
        assert isinstance(operator, Operator)
        return operator

    def is_tautology(self):
        return len(self.variables) == 0

    def partial(self, assignment):
        lhs = dict()
        rhs = self.rhs
        for k, v in self.lhs.items():
            if k in assignment:
                rhs -= assignment[k] * v
            else:
                lhs[k] = v

        updated = self._update(lhs, rhs)
        return updated

    def rename(self, translation):
        return self._update({translation[k] if k in translation else k: v for k, v in self._lhs.items()}, self.rhs)


class LessThan(Operator):
    def __init__(self, lhs, rhs):
        Operator.__init__(self, "<", lhs, rhs)

    def strict(self):
        return self

    def weak(self):
        return LessThanEqual(self.lhs, self.rhs - 1)

    def __invert__(self):
        return GreaterThanEqual(self.lhs, self.rhs)

    def flip(self):
        return GreaterThan(self.invert_lhs(), -self.rhs)

    def _update_bounds(self, lb, ub):
        return lb, min(ub, self.rhs - 1)

    def to_canonical(self):
        return self.weak()

    def evaluate_values(self, lhs_value, rhs_value):
        return lhs_value < rhs_value

    def _update(self, lhs, rhs):
        return LessThan(lhs, rhs)


class GreaterThan(Operator):
    def __init__(self, lhs, rhs):
        Operator.__init__(self, ">", lhs, rhs)

    def strict(self):
        return self

    def weak(self):
        return GreaterThanEqual(self.lhs, self.rhs + 1)

    def __invert__(self):
        return LessThanEqual(self.lhs, self.rhs)

    def flip(self):
        return LessThan(self.invert_lhs(), -self.rhs)

    def _update_bounds(self, lb, ub):
        return max(lb, self.rhs + 1), ub

    def to_canonical(self):
        return self.flip().weak()

    def evaluate_values(self, lhs_value, rhs_value):
        return lhs_value > rhs_value

    def _update(self, lhs, rhs):
        return GreaterThan(lhs, rhs)


class LessThanEqual(Operator):
    def __init__(self, lhs, rhs):
        Operator.__init__(self, "<=", lhs, rhs)

    def strict(self):
        return LessThan(self.lhs, self.rhs + 1)

    def weak(self):
        return self

    def __invert__(self):
        return GreaterThan(self.lhs, self.rhs)

    def flip(self):
        return GreaterThanEqual(self.invert_lhs(), -self.rhs)

    def _update_bounds(self, lb, ub):
        return lb, min(ub, self.rhs)

    def to_canonical(self):
        return self

    def evaluate_values(self, lhs_value, rhs_value):
        return lhs_value <= rhs_value

    def _update(self, lhs, rhs):
        return LessThanEqual(lhs, rhs)


class GreaterThanEqual(Operator):
    def __init__(self, lhs, rhs):
        Operator.__init__(self, ">=", lhs, rhs)

    def strict(self):
        return GreaterThan(self.lhs, self.rhs - 1)

    def weak(self):
        return self

    def __invert__(self):
        return LessThan(self.lhs, self.rhs)

    def flip(self):
        return LessThanEqual(self.invert_lhs(), -self.rhs)

    def _update_bounds(self, lb, ub):
        return max(lb, self.rhs), ub

    def to_canonical(self):
        return self.flip()

    def evaluate_values(self, lhs_value, rhs_value):
        return lhs_value >= rhs_value

    def _update(self, lhs, rhs):
        return GreaterThanEqual(lhs, rhs)


class Test(object):
    @property
    def variables(self):
        raise NotImplementedError()

    def evaluate(self, assignment):
        raise NotImplementedError()

    def rename(self, translation):
        raise NotImplementedError()

    def to_canonical(self, child_true, child_false):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()


class LinearTest(Test):
    def __init__(self, lhs, symbol=None, rhs=0):
        if symbol is None:
            operator = lhs
        else:
            if not isinstance(lhs, sympy.Basic):
                lhs = sympy.sympify(lhs)
            if not isinstance(rhs, sympy.Basic):
                rhs = sympy.sympify(rhs)
            operator = Operator.compile(lhs, symbol, rhs)

        self._operator = operator
        self._negated = None

    @property
    def operator(self):
        return self._operator

    @property
    def _negated_operator(self):
        if self._negated is None:
            self._negated = ~self.operator
        return self._negated

    @property
    def variables(self):
        return self.operator.variables

    def update_bounds(self, var, lb, ub, test=True):
        if len(self.operator.variables) != 1:
            raise RuntimeError("Test does not have exactly one variable (it has {})".format(self.operator.variables))
        if test:
            return self.operator.update_bounds(var, lb, ub)
        else:
            return self._negated_operator.update_bounds(var, lb, ub)

    def evaluate(self, assignment):
        return self.operator.evaluate(assignment)

    def rename(self, translation):
        return LinearTest(self.operator.rename(translation))

    def to_canonical(self, child_true, child_false):
        if isinstance(self.operator, (GreaterThan, LessThan)):
            return LinearTest(self._negated_operator.to_canonical()), child_false, child_true
        else:
            self._operator = self.operator.to_canonical()
            return self, child_true, child_false

    def __repr__(self):
        return str(self.operator)

    def __hash__(self):
        return hash(self.operator)

    def __eq__(self, other):
        return isinstance(other, LinearTest) and self.operator == other.operator


class BinaryTest(Test):
    def __init__(self, var):
        self._var = var

    @property
    def var(self):
        return self._var

    @property
    def variables(self):
        return [self.var]

    def evaluate(self, assignment):
        return bool(assignment(self.var))

    def rename(self, translation):
        if self.var in translation:
            return BinaryTest(translation[self.var])
        return self

    def to_canonical(self, child_true, child_false):
        return self, child_true, child_false

    def __repr__(self):
        return str(self.var)

    def __hash__(self):
        return hash(self.var)

    def __eq__(self, other):
        return isinstance(other, BinaryTest) and self.var == other.var
