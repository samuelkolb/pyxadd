import re

import sympy

from pyxadd.operation import Summation, Multiplication
from pyxadd.test import Operators, Test


def check_node_id(node_id, name="Node id"):
    if not isinstance(node_id, int):
        raise RuntimeError("{} must be integer, was {} of type {}".format(name, node_id, type(node_id)))
    return node_id


class Node:
    def __init__(self, node_id):
        self._node_id = node_id

    @property
    def node_id(self):
        return self._node_id


class TerminalNode(Node):
    def __init__(self, node_id, expression):
        Node.__init__(self, node_id)
        if type(expression) == str:
            expression = sympy.sympify(expression)
        self._expression = expression

    @property
    def expression(self):
        return self._expression

    def __repr__(self):
        return "T(id: {}, expression: {})".format(self.node_id, self.expression)


class InternalNode(Node):
    def __init__(self, node_id, test, child_true, child_false):
        Node.__init__(self, node_id)
        self._test = test
        check_node_id(child_true, "Child (true)")
        self._child_true = child_true
        check_node_id(child_false, "Child (false)")
        self._child_false = child_false

    @property
    def test(self):
        return self._test

    @property
    def child_true(self):
        return self._child_true

    @property
    def child_false(self):
        return self._child_false

    def __repr__(self):
        return "I(id: {}, test: {}, true: {}, false: {})"\
            .format(self.node_id, self.test, self.child_true, self.child_false)


class Pool:
    def __init__(self, empty=False):
        self._counter = 1
        self._nodes = dict()
        self._internal_map = dict()
        self._expressions = dict()
        self._tests = dict()
        self.vars = dict()
        if not empty:
            self.zero_id = self.terminal("0")
            self.one_id = self.terminal("1")
            self.pos_inf_id = self.terminal(sympy.oo)
            self.neg_inf_id = self.terminal(-sympy.oo)

    def _get_test_id(self, test):
        return self._tests.get(test, None)

    def _add_test(self, test):
        if test not in self._tests:
            self._tests[test] = len(self._tests)
        return self._tests[test]

    def get_node(self, node_id):
        check_node_id(node_id)
        if node_id in self._nodes:
            return self._nodes.get(node_id)
        else:
            raise RuntimeError("No node in pool with id {}.".format(node_id))

    def int_var(self, *args):
        for name in args:
            self.vars[sympy.sympify(name)] = "int"

    def add_var(self, name, v_type):
        if v_type == "int":
            self.int_var(name)

    def terminal(self, expression, v_type=None):
        if not isinstance(expression, sympy.Basic):
            expression = sympy.sympify(expression)
        if expression in self._expressions:
            return self._expressions[expression]
        for var in expression.free_symbols:
            if var not in self.vars:
                if v_type is None:
                    raise RuntimeError("Variable {} not declared".format(var))
                else:
                    self.add_var(var, v_type)
        node_id = self._register(lambda n_id: TerminalNode(n_id, expression))
        self._expressions[expression] = node_id
        return node_id

    def internal(self, test, child_true, child_false, v_type=None):
        check_node_id(child_true, "Child (true)")
        check_node_id(child_false, "Child (false)")
        if child_true == child_false:
            return child_true
        test_id = self._add_test(test)
        key = (test_id, child_true, child_false)
        node_id = self._internal_map.get(key, None)
        for var in test.expression.free_symbols:
            if var not in self.vars:
                if v_type is None:
                    raise RuntimeError("Variable {} not declared".format(var))
                else:
                    self.add_var(var, v_type)
        if node_id is None:
            node_id = self._register(lambda n_id: InternalNode(n_id, test, child_true, child_false))
            self._internal_map[key] = node_id
        return node_id

    def bool_test(self, test):
        return self.internal(test, self.one_id, self.zero_id)

    def _register(self, constructor):
        node_id = self._counter
        self._counter += 1
        self._nodes[node_id] = constructor(node_id)
        return node_id

    def apply(self, operation, root1, root2):
        # TODO check apply cache
        node1 = self.get_node(root1)
        node2 = self.get_node(root2)

        result = operation.compute_terminal(self, node1, node2)

        if result is None:
            # Find minimal node (or only internal node)
            if isinstance(node1, InternalNode):
                if isinstance(node2, InternalNode):
                    if self._get_test_id(node1.test) <= self._get_test_id(node2.test):
                        selected_test = node1.test
                    else:
                        selected_test = node2.test
                else:
                    selected_test = node1.test
            else:
                selected_test = node2.test

            if isinstance(node1, InternalNode) and node1.test == selected_test:
                children1 = (node1.child_true, node1.child_false)
            else:
                children1 = (node1.node_id, node1.node_id)

            if isinstance(node2, InternalNode) and node2.test == selected_test:
                children2 = (node2.child_true, node2.child_false)
            else:
                children2 = (node2.node_id, node2.node_id)

            child_true = self.apply(operation, children1[0], children2[0])
            child_false = self.apply(operation, children1[1], children2[1])

            result = self.internal(selected_test, child_true, child_false)
        # TODO update cache
        return result

    def invert(self, node_id):
        minus_one = self.terminal("-1")
        return self.apply(Multiplication, self.apply(Summation, node_id, minus_one), minus_one)


class Diagram:
    def __init__(self, pool, root_node):
        self._pool = pool
        if isinstance(root_node, Node):
            self._root_node = root_node
        else:
            self._root_node = pool.get_node(root_node)
        self._profile = None

    @property
    def root_node(self):
        return self._root_node

    @property
    def pool(self):
        return self._pool

    @property
    def profile(self):
        from pyxadd.walk import WalkingProfile
        if self._profile is None:
            self._profile = WalkingProfile(self)
        return self._profile

    def node(self, node_id):
        return self._pool.get_node(node_id)

    def evaluate(self, assignment):
        return self._evaluate(assignment, self.root_node.node_id)

    def _evaluate(self, assignment, node_id):
        node = self.node(node_id)
        if isinstance(node, TerminalNode):
            return node.expression.subs(assignment)
        elif isinstance(node, InternalNode):
            if node.test.operator.test(node.test.expression.subs(assignment), 0):
                return self._evaluate(assignment, node.child_true)
            else:
                return self._evaluate(assignment, node.child_false)
        else:
            raise RuntimeError("Unexpected node type: {}".format(type(node)))

    def __invert__(self):
        return Diagram(self.pool, self.pool.invert(self.root_node.node_id))

    def __add__(self, other):
        if not isinstance(other, Diagram):
            raise TypeError("Cannot sum Diagram with {}".format(type(other)))
        if self.pool != other.pool:
            raise RuntimeError("Can only add diagrams from the same pool")
        return Diagram(self.pool, self.pool.apply(Summation, self.root_node.node_id, other.root_node.node_id))

    def __mul__(self, other):
        if not isinstance(other, Diagram):
            raise TypeError("Cannot multiply Diagram with {}".format(type(other)))
        if self.pool != other.pool:
            raise RuntimeError("Can only multiply diagrams from the same pool")
        return Diagram(self.pool, self.pool.apply(Multiplication, self.root_node.node_id, other.root_node.node_id))

    # T	1	0	null
    # T	2	1	null
    # E	5	(1 * y) > 0
    # I	17	5	1	2
    # E	4	(1 + (-1 * x) + (-1 * y)) > 0
    # I	19	4	1	17
    # E	3	(1 + (-1 * y)) > 0
    # I	27	3	19	2
    # E	2	((-1 * x) + (1 * y)) > 0
    # I	28	2	19	27
    # E	1	(1 * x) > 0
    # I	29	1	1	28
    # F	7	5	(#nodes and #decisions)

    @staticmethod
    def import_from_string(string, pool=None):
        # TODO implement
        pattern = re.compile(r"(.*) (<|<=|>|>=|=) (.*)")
        tests = dict()
        if pool is None:
            pool = Pool()
        root_id = None
        for line in string.split("\n"):
            parts = line.split("\t")
            if parts[0] == "T":
                root_id = pool.register_node(TerminalNode(int(parts[1]), sympy.sympify(parts[2])))
            elif parts[0] == "E":
                match = pattern.match(parts[2])
                expression = sympy.sympify(match.group(1))
                operator = Operators.get(match.group(2))
                tests[int(parts[1])] = Test(expression, operator)
            elif parts[0] == "I":
                pool.register_node(InternalNode(int(parts[1]), tests[int(parts[2])], int(parts[3]), int(parts[4])))
