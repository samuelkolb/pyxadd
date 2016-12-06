import unittest

from pyxadd.build import Builder
from pyxadd.diagram import Pool, Diagram
from pyxadd.reduce import LinearReduction, SmtReduce
from pyxadd.test import LinearTest
from pyxadd.view import export
from pyxadd.walk import map_leaves, DepthFirstWalker


def build_diagram_1():
    b = Builder(Pool())
    b.ints("r", "c")
    lb = 1
    ub = 10
    bounds = b.limit("r", lb, ub) & b.limit("c", lb, ub)
    diagonal = b.test("r", "<=", "c") & b.test("r", ">=", "c")
    block_1 = b.test("r", ">", lb + (ub - lb) / 2) & b.test("c", "<=", lb + (ub - lb) / 2)
    block_2 = b.test("r", "<=", lb + (ub - lb) / 2) & b.test("c", ">", lb + (ub - lb) / 2)
    return bounds * (diagonal * b.exp(6) + (block_1 | block_2))


def get_terminal_count(diagram):
    counter = lambda *args: 1
    result = map_leaves(counter, diagram)
    return len(result)


class PathCounter(DepthFirstWalker):
    def __init__(self, diagram):
        DepthFirstWalker.__init__(self, diagram)
        self.count = 0

    def visit_terminal(self, terminal_node, parent_message):
        self.count += 1

    def visit_internal(self, internal_node, parent_message):
        return None, None


def get_path_count(diagram):
    counter = PathCounter(diagram)
    counter.walk()
    return counter.count


class TestReduce(unittest.TestCase):
    def test_reduce(self):
        diagram = build_diagram_1()
        result = LinearReduction(diagram.pool).reduce(diagram.root_node.node_id, ["r", "c"])
        result = diagram.pool.diagram(result)
        self.assertEqual(3, get_terminal_count(result))


