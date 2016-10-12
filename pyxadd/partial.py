from pyxadd.test import Test, Operator
from pyxadd.walk import BottomUpWalker
from pyxadd.diagram import Diagram


class PartialWalker(BottomUpWalker):
    def __init__(self, diagram, assignment):
        BottomUpWalker.__init__(self, diagram, diagram.profile)
        self._assignment = assignment

    def visit_internal(self, internal_node, true_message, false_message):
        partial = internal_node.test.operator.partial(self._assignment)
        assert isinstance(partial, Operator)

        if partial.is_tautology():
            if partial.evaluate({}):
                return true_message
            else:
                return false_message
        else:
            return self._diagram.pool.internal(Test(partial), true_message, false_message)

    def visit_terminal(self, terminal_node):
        return self._diagram.pool.terminal(terminal_node.expression.subs(self._assignment))

    def walk(self):
        return Diagram(self._diagram.pool, BottomUpWalker.walk(self))
