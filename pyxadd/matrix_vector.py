import sympy

from pyxadd.diagram import Diagram
from pyxadd.operation import Summation, Multiplication
from pyxadd.test import Operators, Test
from pyxadd.walk import DownUpWalker


class SummationWalker(DownUpWalker):
    def __init__(self, diagram, variable):
        DownUpWalker.__init__(self, diagram)
        self.variable = sympy.sympify(variable)
        # The node cache keeps track of the updated bounds per test
        self.node_cache = dict()

    def visit_internal_down(self, internal_node, parent_message):
        expression = internal_node.test.expression

        # Initialize bounds
        if parent_message is not None:
            lb, ub, bounds = parent_message
        else:
            lb, ub, bounds = -float("inf"), float("inf"), []

        if len(expression.free_symbols) == 1 and self.variable in expression.free_symbols:
            # Test on exactly the given variable: update bounds for the two children (node will be collapsed)
            lb_t, ub_t = internal_node.test.update_bounds(self.variable, lb, ub, test=True)
            lb_f, ub_f = internal_node.test.update_bounds(self.variable, lb, ub, test=False)
            return (lb_t, ub_t, bounds), (lb_f, ub_f, bounds)

        elif len(expression.free_symbols) > 1 and self.variable in expression.free_symbols:
            # Test that includes the given variable and others: rewrite and pass both options (node will be collapsed)
            expression, operator_t = internal_node.test.rewrite(self.variable)
            operator_f = operator_t.flip_int()
            return (lb, ub, bounds + [(operator_t, expression)]), (lb, ub, bounds + [(operator_f, expression)])

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
        if terminal_node.expression == 0:
            return pool.zero_id

        lb_natural, ub_natural, bounds = parent_message
        lower_bounds = [lb_natural]
        upper_bounds = [ub_natural]
        for bound in bounds:
            # [var] [operator] [expression]
            operator, expression = bound
            expression, operator = operator.flip().weak_int(expression)
            operator = operator.flip()

            if operator == Operators.get(">="):
                lower_bounds.append(expression)
            elif operator == Operators.get("<="):
                upper_bounds.append(expression)
            else:
                raise RuntimeError("Cannot handle operator {}".format(operator))

        return self._build_terminal(terminal_node.expression, 0, 1, lower_bounds, 0, 1, upper_bounds)

    def _build_terminal(self, expression, lb_i, lb_c, lower_bounds, ub_i, ub_c, upper_bounds):
        pool = self._diagram.pool

        # Check lower bounds
        if lb_c == len(lower_bounds):
            # Check upper bounds
            if ub_c == len(upper_bounds):
                lb = lower_bounds[lb_i]
                ub = upper_bounds[ub_i]

                result = sympy.nsimplify(sympy.Sum(sympy.nsimplify(expression), (self.variable, lb, ub)).doit())
                print("Bounds", lb, ub, "Expression", expression, "Result", result)
                return pool.terminal(result)
            else:
                # Add upper bound check
                test = Test(upper_bounds[ub_i] - upper_bounds[ub_c], "<=")
                child_true = self._build_terminal(expression, lb_i, lb_c, lower_bounds, ub_i, ub_c + 1, upper_bounds)
                child_false = self._build_terminal(expression, lb_i, lb_c, lower_bounds, ub_c, ub_c + 1, upper_bounds)
        else:
            # Add lower bound check
            test = Test(lower_bounds[lb_i] - lower_bounds[lb_c], ">=")
            child_true = self._build_terminal(expression, lb_i, lb_c + 1, lower_bounds, ub_i, ub_c, upper_bounds)
            child_false = self._build_terminal(expression, lb_c, lb_c + 1, lower_bounds, ub_i, ub_c, upper_bounds)

        if len(test.expression.free_symbols) == 0:
            return child_true if test.operator.test(test.expression, 0) else child_false

        test_node = pool.bool_test(test)
        return pool.apply(Summation,
                          pool.apply(Multiplication, test_node, child_true),
                          pool.apply(Multiplication, pool.invert(test_node), child_false))


def matrix_multiply(pool, root1, root2, variables):
    result = Diagram(pool, pool.apply(Multiplication, root1, root2))
    for var in variables:
        result = SummationWalker(result, var).walk()
    return result
