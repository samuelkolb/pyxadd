from pyxadd.diagram import Pool, Diagram, InternalNode


# TODO cache variables per node?
from pyxadd.operation import Operation, Summation, Multiplication
from pyxadd.walk import BottomUpWalker


class OrderTest(BottomUpWalker):
    def visit_terminal(self, terminal_node):
        return True, None  # (Ordered, test)

    def visit_internal(self, internal_node, true_message, false_message):
        """
        :type internal_node: InternalNode
        """
        if not true_message[0] or not false_message[0]:
            # One of the sub-diagrams is not ordered
            return False, None

        def test_message(message):
            test = message[1]
            if test is None:
                return True
            return not self.diagram.pool.test_smaller_eq(test, internal_node.test)

        return test_message(true_message) and test_message(false_message), internal_node.test


class Order(BottomUpWalker):
    def visit_internal(self, internal_node, true_message, false_message):
        """
        Order internal node (apply trick)
        :type internal_node: InternalNode
        """
        pool = self.diagram.pool
        true_test = pool.internal(internal_node.test, pool.terminal(1), pool.terminal(0))
        false_test = pool.internal(internal_node.test, pool.terminal(0), pool.terminal(1))
        return pool.apply(Summation, pool.apply(Multiplication, true_message, true_test),
                              pool.apply(Multiplication, false_message, false_test))

    def visit_terminal(self, terminal_node):
        """
        Order terminal (trivial)
        :type terminal_node: pyxadd.diagram.TerminalNode
        """
        return terminal_node.node_id


def is_ordered(diagram):
    """
    Test if the given diagram is ordered
    :type diagram: Diagram
    :rtype: bool
    """
    return OrderTest(diagram).walk()[0]


def order(diagram):
    return diagram.pool.diagram(Order(diagram).walk())