from __future__ import print_function

import sympy

from pyxadd.diagram import Diagram, DefaultCache, Pool
from pyxadd.operation import Summation, Multiplication
from pyxadd.test import Test
from pyxadd.variables import VariableFinder
from pyxadd.walk import DownUpWalker


class SummationCache(DefaultCache):

    name = "summation-cache"

    def __init__(self):
        self.lb = sympy.S("lb")
        self.ub = sympy.S("ub")

        def calculator(pool, var_node):
            """
            Symbolically sums the expression of the node.
            :return: a lambda expression that substitutes the given lower- and upper bound in the sum
            """
            variable, node_id = var_node
            expression = pool.get_node(node_id).expression
            variables = {str(v): v for v in expression.free_symbols}
            v = variables[variable] if variable in variables else sympy.S(variable)
            try:
                result = sympy.simplify(sympy.Sum(expression, (v, self.lb, self.ub)).doit())
                return lambda lb, ub: result.subs({self.lb: lb, self.ub: ub})
            except sympy.BasePolynomialError as e:
                print("Problem trying to sum the expression {} for variable {}"
                      .format(expression, v))
                raise e

        super(SummationCache, self).__init__(calculator)

    @classmethod
    def initialize(cls, pool):
        if not pool.has_cache(cls.name):
            pool.add_cache(cls.name, cls())


class SummationWalker(DownUpWalker):
    def __init__(self, diagram, variable):
        DownUpWalker.__init__(self, diagram)
        self.variable = str(variable)
        # The node cache keeps track of the updated bounds per test
        self.node_cache = dict()
        self.sum_cache = dict()
        SummationCache.initialize(diagram.pool)

    def visit_internal_down(self, internal_node, parent_message):
        operator = internal_node.test.operator.to_canonical()
        # expression = internal_node.test.expression

        # Initialize bounds
        if parent_message is not None:
            lb, ub, bounds = parent_message
        else:
            lb, ub, bounds = -float("inf"), float("inf"), []

        if operator.is_singular() and self.variable in operator.variables:
            # Test on exactly the given variable: update bounds for the two children (node will be collapsed)
            lb_t, ub_t = internal_node.test.update_bounds(self.variable, lb, ub, test=True)
            lb_f, ub_f = internal_node.test.update_bounds(self.variable, lb, ub, test=False)
            return (lb_t, ub_t, bounds), (lb_f, ub_f, bounds)

        elif len(operator.variables) > 1 and self.variable in operator.variables:
            # Test that includes the given variable and others: rewrite and pass both options (node will be collapsed)
            def to_exp(op):
                expression = sympy.sympify(op.rhs)
                for k, v in op.lhs.items():
                    if k != self.variable:
                        expression = -sympy.S(k) * v + expression
                return expression

            rewritten_positive = operator.times(1 / operator.coefficient(self.variable)).weak()
            exp_pos = to_exp(rewritten_positive)

            rewritten_negative = (~operator).times(1 / operator.coefficient(self.variable)).weak()
            exp_neg = to_exp(rewritten_negative)

            true_bound = (rewritten_positive.symbol, exp_pos)
            false_bound = (rewritten_negative.symbol, exp_neg)
            return (lb, ub, bounds + [true_bound]), (lb, ub, bounds + [false_bound])

        else:
            # Test that does not include the given variable (node test will be maintained)
            self.node_cache[internal_node.node_id] = internal_node.test
            return (lb, ub, bounds), (lb, ub, bounds)

    def visit_internal_aggregate(self, internal_node, true_result, false_result):
        pool = self._diagram.pool
        if internal_node.node_id in self.node_cache:
            # The node test is maintained and the two results become the new child nodes
            test_node = pool.bool_test(self.node_cache[internal_node.node_id])
            result = pool.apply(Summation,
                                pool.apply(Multiplication, test_node, true_result),
                                pool.apply(Multiplication, pool.invert(test_node), false_result))
            return result

        result = pool.apply(Summation, true_result, false_result)
        return result

    def visit_terminal(self, terminal_node, parent_message):
        if parent_message is None:
            return terminal_node.node_id

        pool = self._diagram.pool
        lb_natural, ub_natural, bounds = parent_message

        if terminal_node.expression == 0 or lb_natural > ub_natural:
            return pool.zero_id

        lower_bounds = [lb_natural]
        upper_bounds = [ub_natural]
        for bound in bounds:
            # [var] [operator] [expression]
            operator, expression = bound

            if operator == ">=":
                lower_bounds.append(expression)
            elif operator == "<=":
                upper_bounds.append(expression)
            else:
                raise RuntimeError("Cannot handle operator {}".format(operator))

        return self._build_terminal(terminal_node, 0, 1, lower_bounds, 0, 1, upper_bounds)

    def _build_terminal(self, terminal_node, lb_i, lb_c, lower_bounds, ub_i, ub_c, upper_bounds):
        pool = self._diagram.pool
        assert isinstance(pool, Pool)

        # Check lower bounds
        if lb_c == len(lower_bounds):
            # Check upper bounds
            if ub_c == len(upper_bounds):
                lb = lower_bounds[lb_i]
                ub = upper_bounds[ub_i]
                import time
                # result = sympy.nsimplify(sympy.Sum(sympy.nsimplify(expression), (self.variable, lb, ub)).doit())
                node_id = terminal_node.node_id
                f = pool.get_cached(SummationCache.name, (self.variable, node_id))
                result = f(lb, ub)
                # TODO: simplify result numerically??
                bound_integrity_check = Test(lb, "<=", ub)
                if bound_integrity_check.operator.is_tautology():
                    return pool.terminal(result) if bound_integrity_check.evaluate({}) else pool.zero_id
                else:
                    node_id = pool.bool_test(bound_integrity_check)
                    return pool.apply(Multiplication, node_id, pool.terminal(result))
            else:
                # Add upper bound check
                test = Test(upper_bounds[ub_i] - upper_bounds[ub_c], "<=")
                child_true = self._build_terminal(terminal_node, lb_i, lb_c, lower_bounds, ub_i, ub_c + 1, upper_bounds)
                child_false = self._build_terminal(terminal_node, lb_i, lb_c, lower_bounds, ub_c, ub_c + 1, upper_bounds)
        else:
            # Add lower bound check
            test = Test(lower_bounds[lb_i] - lower_bounds[lb_c], ">=")
            child_true = self._build_terminal(terminal_node, lb_i, lb_c + 1, lower_bounds, ub_i, ub_c, upper_bounds)
            child_false = self._build_terminal(terminal_node, lb_c, lb_c + 1, lower_bounds, ub_i, ub_c, upper_bounds)

        if test.operator.is_tautology():
            return child_true if test.evaluate({}) else child_false

        test_node = pool.bool_test(test)
        return pool.apply(Summation,
                          pool.apply(Multiplication, test_node, child_true),
                          pool.apply(Multiplication, pool.invert(test_node), child_false))


def matrix_multiply(pool, root1, root2, variables):
    return sum_out(pool, pool.apply(Multiplication, root1, root2), variables)


def sum_out(pool, root, variables):
    variables = list(str(v) for v in variables)
    diagram = pool.diagram(root)
    result = diagram
    for var in variables:
        result = pool.diagram(SummationWalker(result, var).walk())
    _check_output(diagram, result, variables)
    return result.root_node.node_id


def _check_output(start_diagram, end_diagram, variables):
    present = set([str(v) for v in VariableFinder(start_diagram).walk()])
    remaining = set([str(v) for v in VariableFinder(end_diagram).walk()])
    variables = [str(v) for v in variables]
    if not remaining.issubset(present):
        raise RuntimeError("New variables ({}) have been introduced in the diagram (which contained {}) after summation"
                           .format(list(remaining - present), present))
    if not present.issubset(remaining | set(variables)):
        raise RuntimeError("Some variables ({}) have been erroneously removed: they are not present in the diagram "
                           "after summation ({}) nor in the variables to remove ({})"
                           .format(list(present - remaining - set(variables)), list(remaining), variables))
    if len(remaining & set(variables)) != 0:
        raise RuntimeError("Some variables ({}) that should be summed out ({}) still appear in the diagram after "
                           "summation. The diagram started out with variables {} and now contains {}"
                           .format(list(remaining & set(variables)), variables, list(present), list(remaining)),
                           end_diagram)
