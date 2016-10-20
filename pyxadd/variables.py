from pyxadd.diagram import Diagram
from pyxadd.walk import BottomUpWalker


class VariableFinder(BottomUpWalker):
    def visit_internal(self, internal_node, true_message, false_message):
        return true_message | false_message | set(internal_node.test.operator.variables)

    def visit_terminal(self, terminal_node):
        return set([str(v) for v in terminal_node.expression.free_symbols])


def variables(diagram_or_node, pool=None):
    if not isinstance(diagram_or_node, Diagram):
        diagram_or_node = Diagram(pool, diagram_or_node)
    return VariableFinder(diagram_or_node).walk()
