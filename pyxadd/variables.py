from pyxadd import diagram as core
from pyxadd import walk


class VariableFinder(walk.BottomUpWalker):
    def visit_internal(self, internal_node, true_message, false_message):
        return true_message | false_message | set(internal_node.test.variables)

    def visit_terminal(self, terminal_node):
        return set([str(v) for v in terminal_node.expression.free_symbols])


class TopDownVariableFinder(walk.DepthFirstUniqueWalker):
    def __init__(self, diagram):
        walk.DepthFirstUniqueWalker.__init__(self, diagram)
        self._variables = None

    def visit_internal(self, internal_node, parent_message):
        self._variables |= set(internal_node.test.variables)
        return None, None

    def visit_terminal(self, terminal_node, parent_message):
        self._variables |= set([str(v) for v in terminal_node.expression.free_symbols])
        return None

    def walk(self):
        self._variables = set()
        walk.DepthFirstUniqueWalker.walk(self)
        found = self._variables
        self._variables = None
        return found


def variables(diagram_or_node, pool=None):
    if not isinstance(diagram_or_node, core.Diagram):
        diagram_or_node = pool.diagram(diagram_or_node)
    return VariableFinder(diagram_or_node).walk()
