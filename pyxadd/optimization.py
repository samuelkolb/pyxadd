"""

per node:
n + n! = sum of parents or 1 if root

NEED:
parents per node

"""
from pyxadd.build import Builder
from pyxadd.diagram import Pool
from pyxadd.walk import BottomUpWalker


class ParentsWalker(BottomUpWalker):
    def __init__(self, diagram, profile=None):
        BottomUpWalker.__init__(self, diagram, profile)
        self.parents = {}

    def visit_internal(self, internal_node, true_message, false_message):
        node_id = internal_node.node_id
        self.parents[node_id] = []
        self.parents[true_message].append((node_id, True))
        self.parents[false_message].append((node_id, False))
        return node_id

    def visit_terminal(self, terminal_node):
        node_id = terminal_node.node_id
        self.parents[node_id] = []
        return node_id

    def walk(self):
        BottomUpWalker.walk(self)
        return self.parents
