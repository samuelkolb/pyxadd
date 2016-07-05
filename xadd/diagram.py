import re

import sympy

from xadd.test import Operators, Test


class Node:
    def __init__(self, node_id):
        self._node_id = node_id

    @property
    def node_id(self):
        return self._node_id


class TerminalNode(Node):
    def __init__(self, node_id, expression):
        super().__init__(node_id)
        if type(expression) == str:
            expression = sympy.sympify(expression)
        self._expression = expression

    @property
    def expression(self):
        return self._expression

    def __repr__(self):
        return "T\t" + str(self.node_id)


class InternalNode(Node):
    def __init__(self, node_id, test, child_true, child_false):
        super().__init__(node_id)
        self._test = test
        self._child_true = child_true
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
        return "I\t" + str(self.node_id)


class Pool:
    def __init__(self, empty=False):
        self._counter = 1
        self._nodes = dict()
        self._expressions = dict()
        self._tests = dict()
        if not empty:
            self.zero_id = self.terminal("0")
            self.one_id = self.terminal("1")
            self.pos_inf_id = self.terminal("+inf")
            self.neg_inf_id = self.terminal("-inf")

    def get_node(self, node_id):
        if node_id in self._nodes:
            return self._nodes.get(node_id)
        else:
            raise RuntimeError("No node in pool with id {}.".format(node_id))

    def terminal(self, expression):
        if expression in self._expressions:
            return self._expressions[expression]
        node_id = self._register(lambda n_id: TerminalNode(n_id, expression))
        self._expressions[expression] = node_id
        return node_id

    def internal(self, test, child_true, child_false):
        t = (test, child_true, child_false)
        if t in self._tests:
            return self._tests[t]
        node_id = self._register(lambda n_id: InternalNode(n_id, test, child_true, child_false))
        self._tests[t] = node_id
        return node_id

    def _register(self, constructor):
        node_id = self._counter
        self._counter += 1
        self._nodes[node_id] = constructor(node_id)
        return node_id


    def apply(self, operation, root1, root2):
        # TODO check cache

        # TODO compute terminal
        # TODO deal with NaN?


        # TODO compute recursive

        # TODO update cache
        pass


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
        from xadd.walk import WalkingProfile
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
