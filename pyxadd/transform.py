import numpy
from pyxadd.bounds import get_bounds
from pyxadd.build import Builder
from pyxadd.matrix.matrix import assignments
from pyxadd.reduce import LinearReduction
from pyxadd.test import Test
from pyxadd.walk import DownUpWalker
from sympy import oo


class ConstantWalker(DownUpWalker):
    def __init__(self, diagram, diagram_vars):
        DownUpWalker.__init__(self, diagram)
        self._diagram_vars = diagram_vars
        self._reduction = LinearReduction(self.diagram.pool)

    def visit_internal_aggregate(self, internal_node, true_result, false_result):
        node_id = self._reduction.reduce((true_result + false_result).root_node.node_id, self._diagram_vars)
        return self.diagram.pool.diagram(node_id)

    def visit_internal_down(self, internal_node, parent_message):
        if parent_message is None:
            parent_message = self._initial_message()

        bounds, path = parent_message
        bounds_t = dict(bounds)
        bounds_f = dict(bounds)

        # Avoiding having to find all integer points in complex areas by only looking at "simple" constraints
        operator = internal_node.test.operator
        if operator.is_singular():
            var = operator.variables[0]
            lb, ub = bounds[var] if var in bounds else (-oo, oo)
            bounds_t[var] = internal_node.test.update_bounds(var, lb, ub, test=True)
            bounds_f[var] = internal_node.test.update_bounds(var, lb, ub, test=False)

        pool = self.diagram.pool
        path_t = path & pool.diagram(pool.internal(Test(operator), pool.one_id, pool.zero_id))
        path_f = path & pool.diagram(pool.internal(Test(~operator), pool.one_id, pool.zero_id))

        return (bounds_t, path_t), (bounds_f, path_f)

    def visit_terminal(self, terminal_node, parent_message):
        print(parent_message, terminal_node.expression)
        if parent_message is None:
            parent_message = self._initial_message()

        bounds, path = parent_message

        pool = self.diagram.pool
        if terminal_node.expression == 0:
            return pool.diagram(pool.zero_id)
        elif len(terminal_node.expression.free_symbols) == 0:
            return path * pool.diagram(terminal_node.node_id)
        else:
            variables_present = terminal_node.expression.free_symbols
            # FIXME todo
            return pool.diagram(pool.zero_id)

    def _initial_message(self):
        return {}, self.diagram.pool.diagram(self.diagram.pool.one_id)


def to_constant(diagram):
    b = Builder(diagram.pool)
    from pyxadd.variables import variables
    all_variables = variables(diagram)
    print("Variables: {}".format(all_variables))
    bounds = [(v, ) + get_bounds(v, diagram) for v in all_variables]

    # Approach 1:
    # Per path, find all included points, merge based on expression (constant all, one var => test only on that, ...)

    # Approach 2:
    # Ground to matrix, learn structure

    return ConstantWalker(diagram, all_variables).walk()


def to_ground_tensor(diagram, bounds):
    tensor = numpy.empty(shape=tuple(ub - lb + 1 for _, lb, ub in bounds))
    for a in assignments(bounds):
        tensor.itemset(tuple(a[t[0]] - t[1] for t in bounds), diagram.evaluate(a))
    return tensor

# TODO Substitute bottom up

def learn(tensor, variables):
    """
    :param tensor: The tensor to learn from
    :param variables: The variables and their bounds
    :return: A diagram corresponding to the tensor
    """
