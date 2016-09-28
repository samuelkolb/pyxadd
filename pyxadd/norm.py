import sympy

from pyxadd.diagram import Diagram
from pyxadd.matrix_vector import SummationWalker
from pyxadd.operation import Summation, Multiplication
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
    result = Diagram(diagram.pool, SquareWalker(diagram, diagram.profile).walk())
    for var in variables:
        result = SummationWalker(result, var).walk()
    print(Diagram(diagram.pool, result).evaluate({}))
    return sympy.sqrt(Diagram(diagram.pool, result).evaluate({}))
