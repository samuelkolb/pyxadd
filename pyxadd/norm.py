import sympy

from pyxadd.diagram import Diagram
from pyxadd.matrix_vector import SummationWalker
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


def norm(variables, diagram):
    from matrix_vector import sum_out
    squared = SquareWalker(diagram, diagram.profile).walk()
    normed = sum_out(diagram.pool, squared, variables)
    diagram = Diagram(diagram.pool, normed)
    try:
        value = diagram.evaluate({})
        return sympy.sqrt(value)
    except RuntimeError as e:
        found = VariableFinder(diagram).walk()
        if len(found) > 0:
            raise RuntimeError("Variables {f} were not eliminated (given variables were {v})"
                               .format(f=list(found), v=variables))
        else:
            raise e
