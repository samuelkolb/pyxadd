from pyxadd import test
from pyxadd.walk import BottomUpWalker


class RenameWalker(BottomUpWalker):
    def __init__(self, diagram, translation, profile=None):
        BottomUpWalker.__init__(self, diagram, profile)
        self.translation = translation

    def visit_internal(self, internal_node, true_message, false_message):
        return self.diagram.pool.internal(internal_node.test.rename(self.translation), true_message, false_message)

    def visit_terminal(self, terminal_node):
        return self.diagram.pool.terminal(terminal_node.expression.subs(self.translation))


class SubstitutionWalker(BottomUpWalker):
    def __init__(self, diagram, translation, profile=None):
        BottomUpWalker.__init__(self, diagram, profile)
        self.translation = translation

    def visit_internal(self, internal_node, true_message, false_message):
        current_test = internal_node.test
        if isinstance(current_test, test.LinearTest):
            renamed_test = test.LinearTest(current_test.operator.substitute_expressions(self.translation))
        elif isinstance(current_test, test.BinaryTest):
            renamed_test = current_test.rename(self.translation)
        else:
            raise RuntimeError("Substitution not yet implemented for {} of type {}"
                               .format(current_test, type(current_test)))
        return self.diagram.pool.internal(renamed_test, true_message, false_message)

    def visit_terminal(self, terminal_node):
        return self.diagram.pool.terminal(terminal_node.expression.subs(self.translation))


def rename(root, translation, pool=None):
    from pyxadd.diagram import Diagram
    if isinstance(root, Diagram):
        pool = root.pool
        root = root.root_node.node_id

    walker = RenameWalker(pool.diagram(root), translation)
    return walker.walk()


def substitute(root, translation, pool=None):
    from pyxadd.diagram import Diagram
    if isinstance(root, Diagram):
        pool = root.pool
        root = root.root_node.node_id

    walker = RenameWalker(pool.diagram(root), translation)
    return walker.walk()
