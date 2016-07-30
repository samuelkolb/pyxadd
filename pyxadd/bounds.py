import sympy

from pyxadd.walk import BottomUpWalker, DownUpWalker


class BoundsWalker(DownUpWalker):
    def __init__(self, variable, diagram):
        DownUpWalker.__init__(self, diagram)
        self.variable = variable

    def visit_internal_aggregate(self, internal_node, true_result, false_result):
        lb_t, ub_t = true_result
        lb_f, ub_f = false_result
        return min(lb_t, lb_f), max(ub_t, ub_f)

    def visit_internal_down(self, internal_node, parent_message):
        if parent_message is not None:
            lb, ub = parent_message
        else:
            lb, ub = -sympy.oo, sympy.oo

        expression = internal_node.test.expression
        if len(expression.free_symbols) == 1 and self.variable in expression.free_symbols:
            lb_t, ub_t = internal_node.test.update_bounds(self.variable, lb, ub, test=True)
            lb_f, ub_f = internal_node.test.update_bounds(self.variable, lb, ub, test=False)
            return (lb_t, ub_t), (lb_f, ub_f)
        return parent_message, parent_message

    def visit_terminal(self, terminal_node, parent_message):
        if terminal_node.expression == 0:
            return sympy.oo, -sympy.oo
        return parent_message


def get_bounds(variable, diagram):
    walker = BoundsWalker(variable, diagram)
    return walker.walk()


def get_domain_size(variable, diagram):
    v_type = diagram.pool.get_var_type(variable)
    if v_type == "int":
        lb, ub = get_bounds(variable, diagram)
        return max(0, ub - lb)
    raise RuntimeError("Domain size currently only supported for int variables, not for {} variables".format(v_type))
