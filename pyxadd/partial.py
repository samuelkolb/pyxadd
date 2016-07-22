from pyxadd.test import Test
from pyxadd.walk import BottomUpWalker
from pyxadd.diagram import Diagram


class PartialWalker(BottomUpWalker):
    def __init__(self, diagram, assignment):
        BottomUpWalker.__init__(self, diagram, diagram.profile)
        self._assignment = assignment

    def visit_internal(self, internal_node, true_message, false_message):
        expression = self._partial(internal_node.test.expression)
        operator = internal_node.test.operator
        if len(expression.free_symbols) == 0:
            if operator.test(expression, 0):
                return true_message
            else:
                return false_message
        else:
            return self._diagram.pool.internal(Test(expression, operator), true_message, false_message)

    def visit_terminal(self, terminal_node):
        return self._diagram.pool.terminal(self._partial(terminal_node.expression))

    def _partial(self, expression):
        return expression.subs(self._assignment)

    def walk(self):
        return Diagram(self._diagram.pool, BottomUpWalker.walk(self))
