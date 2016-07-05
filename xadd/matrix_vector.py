import sympy

from xadd.walk import DownUpWalker


class SummationWalker(DownUpWalker):
    def __init__(self, diagram, variable):
        super().__init__(diagram)
        self.variable = sympy.sympify(variable)

    def visit_internal_down(self, internal_node, parent_message):
        expression = internal_node.test.expression
        if parent_message is not None:
            lb, ub, bounds = parent_message
        else:
            lb, ub, bounds = -float("inf"), float("inf"), ()
        print("Visit {}".format(internal_node.test), expression.free_symbols)
        if len(expression.free_symbols) == 1 and self.variable in expression.free_symbols:
            # variable bound
            lb_t, ub_t = internal_node.test.update_bounds(self.variable, lb, ub, test=True)
            lb_f, ub_f = internal_node.test.update_bounds(self.variable, lb, ub, test=False)
            return (lb_t, ub_t, bounds), (lb_f, ub_f, bounds)
        if len(expression.free_symbols) > 1 and self.variable in expression.free_symbols:
            expression, operator_t = internal_node.test.rewrite(self.variable)
            operator_f = operator_t.flip_int()
            return (lb, ub, (bounds, (expression, operator_t))), (lb, ub, (bounds, (expression, operator_f)))
        return (lb, ub, bounds), (lb, ub, bounds)

    def visit_internal_aggregate(self, internal_node, true_result, false_result):
        pass

    def visit_terminal(self, terminal_node, parent_message):
        print(parent_message, terminal_node.expression)
