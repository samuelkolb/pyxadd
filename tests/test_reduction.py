import unittest

from pyxadd.build import Builder
from pyxadd.diagram import Pool, Diagram
from pyxadd.reduce import LinearReduction, SmtReduce, SimpleBoundReducer
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


def build_diagram_2():
    b = Builder(Pool())
    b.ints("r", "c")
    lb, ub = 1, 10
    bounds = b.limit("r", lb, ub) & b.limit("c", lb, ub)
    t1 = b.ite(b.test("r", "<=", 0), 2, 4)
    t2 = b.ite(b.test("c", ">=", 11), 3, 5)
    return bounds * (t1 + t2)


def get_terminal_count(diagram):
    counter = lambda *args: 1
    result = map_leaves(counter, diagram)
    return len(result)


class PathCounter(DepthFirstWalker):
    def __init__(self, diagram, count_zero=True):
        DepthFirstWalker.__init__(self, diagram)
        self.count = 0
        self.count_zero = count_zero

    def visit_terminal(self, terminal_node, parent_message):
        if self.count_zero or terminal_node.expression != 0:
            self.count += 1

    def visit_internal(self, internal_node, parent_message):
        return None, None


def get_path_count(diagram, count_zero=True):
    counter = PathCounter(diagram, count_zero=count_zero)
    counter.walk()
    return counter.count


def get_solvers(simple=False):
    if simple:
        return [SimpleBoundReducer, LinearReduction, SmtReduce]
    else:
        return [LinearReduction]


class TestReduce(unittest.TestCase):
    def test_reduce(self):
        diagram = build_diagram_1()
        result = LinearReduction(diagram.pool).reduce(diagram.root_id, ["r", "c"])
        result = diagram.pool.diagram(result)
        self.assertEqual(3, get_terminal_count(result))

    def test_simple_bounds_simple_reduce(self):
        diagram = build_diagram_2()
        for solver_type in get_solvers(simple=True):
            solver = solver_type(diagram.pool)
            result = diagram.pool.diagram(solver.reduce(diagram.root_id))
            self.assertEqual(2, get_terminal_count(result))
            self.assertEqual(1, get_path_count(result, count_zero=False))
