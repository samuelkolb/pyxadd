import sympy

from pyxadd.diagram import Diagram
from pyxadd.operation import Summation, Multiplication
from pyxadd.variables import VariableFinder
from pyxadd.walk import BottomUpWalker


class SquareWalker(BottomUpWalker):
    def visit_terminal(self, terminal_node):
        expression = terminal_node.expression
        return self._diagram.pool.terminal(expression * expression)

    def visit_internal(self, internal_node, true_message, false_message):
        pool = self._diagram.pool
        test_node = pool.bool_test(internal_node.test)
        return pool.apply(Summation,
                          pool.apply(Multiplication, test_node, true_message),
                          pool.apply(Multiplication, pool.invert(test_node), false_message))


class AbsoluteValueWalker(BottomUpWalker):
    def visit_terminal(self, terminal_node):
        expression = terminal_node.expression
        if len(expression.free_symbols) == 0:
            return terminal_node.node_id if expression >= 0 else self._diagram.pool.terminal(-expression)
        else:
            raise NotImplementedError("Absolute value not currently implemented for expressions")

    def visit_internal(self, internal_node, true_message, false_message):
        pool = self._diagram.pool
        test_node = pool.bool_test(internal_node.test)
        return pool.apply(Summation,
                          pool.apply(Multiplication, test_node, true_message),
                          pool.apply(Multiplication, pool.invert(test_node), false_message))


def norm(variables, diagram, l_norm=None):
    from pyxadd.matrix_vector import sum_out
    if l_norm is None:
        l_norm = 2
    elif l_norm != 2 and l_norm != 1:
        raise RuntimeError("Unsupported norm:{}, should be 1 or 2 [default: None => 2]".format(norm))

    if l_norm == 2:
        node_id = SquareWalker(diagram, diagram.profile).walk()
    else:
        node_id = AbsoluteValueWalker(diagram, diagram.profile).walk()
    normed = sum_out(diagram.pool, node_id, variables)
    diagram = Diagram(diagram.pool, normed)
    try:
        value = diagram.evaluate({})
        if value < 0:
            raise RuntimeError("The squared sum is negative ({})".format(value))
        if l_norm == 2:
            return sympy.sqrt(value)
        else:
            return value
    except RuntimeError as e:
        found = VariableFinder(diagram).walk()
        if len(found) > 0:
            raise RuntimeError("Variables {f} were not eliminated (given variables were {v})"
                               .format(f=list(found), v=variables))
        else:
            raise e
