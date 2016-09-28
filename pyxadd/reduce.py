import sympy

from pyxadd.diagram import TerminalNode, InternalNode
from pyxadd.test import Operators
from pyxadd.walk import DepthFirstWalker

# TODO Same procedure with SMT solver


# noinspection PyPep8Naming
class LinearReduction(object):
    def __init__(self, pool):
        self.pool = pool
        self.variables = None

    @property
    def columns(self):
        return len(self.variables)

    def reduce(self, node_id, variables):
        self.variables = variables
        return self._reduce(node_id, [[] * len(variables)], [])

    def _reduce(self, node_id, coefficients, constants):
        node = self.pool.get_node(node_id)
        if isinstance(node, TerminalNode):

            return node_id
        elif isinstance(node, InternalNode):
            true_coefficients, true_constants = self._combine(coefficients, constants, node.test, True)
            if not self._is_feasible(true_coefficients, true_constants):
                # Only false branch is true
                return self._reduce(node.child_false, coefficients, constants)

            false_coefficients, false_constants = self._combine(coefficients, constants, node.test, False)
            if not self._is_feasible(false_coefficients, false_constants):
                # Only true branch is true
                return self._reduce(node.child_true, coefficients, constants)

            true_reduced = self._reduce(node.child_true, true_coefficients, true_constants)
            false_reduced = self._reduce(node.child_false, false_coefficients, false_constants)
            return self.pool.internal(node.test, true_reduced, false_reduced)
        else:
            raise RuntimeError("Unexpected node {} of type {}".format(node, type(node)))

    def _combine(self, coefficients, constants, test, test_true):
        new_coefficients, new_constant = self._test_to_linear_leq_constraint(test, test_true)
        combined_coefficients = list(coefficients[i] + [new_coefficients[i]] for i in range(0, len(coefficients)))
        combined_constants = constants + [new_constant]
        return combined_coefficients, combined_constants

    def _test_to_linear_leq_constraint(self, test, test_true):
        # Assumes integer variables / constraints
        expression = test.expression
        operator = test.operator
        if not test_true:
            operator = operator.flip_int()
        expression, operator = operator.weak_int(expression)
        if operator == Operators.get(">="):
            expression *= -1
        elif operator != Operators.get("<="):
            raise RuntimeError("Unexpected operator: {}".format(operator))
        f = sympy.lambdify(tuple(self.variables), -expression)
        constant = float(f(*([0] * len(self.variables))))
        coefficients = list(float(expression.coeff(var, 1)) for var in self.variables)
        print(coefficients)
        return coefficients, constant

    def _is_feasible(self, coefficients, constants):
        # TODO substitute variable for value if it can be only one value
        import cvxopt
        print("Coefficients & constants", coefficients, constants)
        cvxopt.solvers.options["show_progress"] = False
        A = cvxopt.matrix(coefficients)
        b = cvxopt.matrix(constants)
        c = cvxopt.matrix([0.0] * len(self.variables))
        status = cvxopt.solvers.lp(c, A, b, solver="cvxopt_glpk")["status"]
        return status == "optimal"

