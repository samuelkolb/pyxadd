from pyxadd.walk import BottomUpWalker


class RenameWalker(BottomUpWalker):
    def __init__(self, diagram, translation, profile=None):
        BottomUpWalker.__init__(self, diagram, profile)
        self.translation = translation

    def visit_internal(self, internal_node, true_message, false_message):
        return self.diagram.pool.internal(internal_node.test.rename(self.translation), true_message, false_message)

    def visit_terminal(self, terminal_node):
        return self.diagram.pool.terminal(terminal_node.expression.subs(self.translation))


def rename(root, translation, pool=None):
    from pyxadd.diagram import Diagram
    if isinstance(root, Diagram):
        pool = root.pool
        root = root.root_node.node_id

    walker = RenameWalker(pool.diagram(root), translation)
    return walker.walk()