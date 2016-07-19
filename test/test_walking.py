import unittest

from xadd.diagram import Diagram, InternalNode, Pool
from xadd.test import Test, Operators
from xadd.walk import DepthFirstWalker, DownUpWalker, ParentsWalker, WalkingProfile, BottomUpWalker


class DummyDepthFirstWalker(DepthFirstWalker):
    def __init__(self, diagram):
        super().__init__(diagram)
        self.numbers = []

    def visit_internal(self, internal_node, parent_message):
        if parent_message is None:
            parent_message = []
        true_message = parent_message + [internal_node.node_id]
        false_message = parent_message + [-internal_node.node_id]
        return true_message, false_message

    def visit_terminal(self, terminal_node, parent_message):
        self.numbers.append((terminal_node.node_id, parent_message))


class DummyDownUpWalker(DownUpWalker):
    def visit_internal_down(self, internal_node, parent_message):
        if parent_message is None:
            parent_message = []
        true_message = parent_message + [internal_node.node_id]
        false_message = parent_message + [-internal_node.node_id]
        return true_message, false_message

    def visit_terminal(self, terminal_node, parent_message):
        return [(terminal_node.node_id, parent_message)]

    def visit_internal_aggregate(self, internal_node, true_result, false_result):
        return true_result + false_result


class DummyBottomUpWalker(BottomUpWalker):
    def visit_terminal(self, terminal_node):
        return terminal_node.node_id

    def visit_internal(self, internal_node, true_message, false_message):
        return true_message - false_message


class TestWalking(unittest.TestCase):
    def setUp(self):
        self.diagram = TestWalking.construct_diagram()

    @staticmethod
    def construct_diagram():
        pool = Pool(empty=True)
        x = pool.terminal("x")
        zero = pool.terminal("0")
        test1 = pool.internal(Test("x - 5", Operators.get("<=")), x, zero)
        test2 = pool.internal(Test("x + 1", Operators.get(">=")), test1, zero)
        test3 = pool.internal(Test("x + 2", Operators.get("=")), x, test2)
        root = pool.internal(Test("x", Operators.get(">=")), test1, test3)
        return Diagram(pool, root)

    def test_depth_first(self):
        dummy = DummyDepthFirstWalker(self.diagram)
        dummy.walk()
        self.assertEqual(dummy.numbers, [
            (1, [6, 3]),
            (2, [6, -3]),
            (1, [-6, 5]),
            (1, [-6, -5, 4, 3]),
            (2, [-6, -5, 4, -3]),
            (2, [-6, -5, -4])])

    def test_down_up(self):
        dummy = DummyDownUpWalker(self.diagram)
        result = dummy.walk()
        self.assertEqual(result, [
            (1, [6, 3]),
            (2, [6, -3]),
            (1, [-6, 5]),
            (1, [-6, -5, 4, 3]),
            (2, [-6, -5, 4, -3]),
            (2, [-6, -5, -4])])

    def test_parents_walker(self):
        walker = ParentsWalker(self.diagram)
        parents = walker.walk()
        self.assertEqual(parents, {1: {3, 5}, 2: {3, 4}, 3: {4, 6}, 4: {5}, 5: {6}, 6: set()})

    def test_walking_profile(self):
        ordering = list(WalkingProfile.extract_cache(ParentsWalker(self.diagram).walk(), self.diagram))
        for node_id in [1, 2, 3, 4, 5, 6]:
            node = self.diagram.node(node_id)
            if isinstance(node, InternalNode):
                self_position = ordering.index(node_id)
                children_max = max(ordering.index(node.child_true), ordering.index(node.child_false))
                self.assertTrue(self_position > children_max)

    def test_bottom_up_walker(self):
        profile = WalkingProfile(self.diagram)
        result = DummyBottomUpWalker(self.diagram, profile).walk()
        self.assertEqual(result, -5)


if __name__ == '__main__':
    unittest.main()
