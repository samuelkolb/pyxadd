import re
import warnings

import sympy

from pyxadd.operation import Summation, Multiplication, LogicalOr, LogicalAnd
from pyxadd.test import LinearTest, Test


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

    def is_terminal(self):
        raise NotImplementedError()


class TerminalNode(Node):
    def __init__(self, node_id, expression):
        Node.__init__(self, node_id)
        if type(expression) == str:
            expression = sympy.sympify(expression)
        self._expression = expression
        self._symbols = tuple(expression.free_symbols)
        self._f = None

    @property
    def expression(self):
        return self._expression

    def _get_f(self):
        if self._f is None:
            self._f = sympy.lambdify(self._symbols, self._expression)
        return self._f

    def evaluate(self, assignment):
        try:
            return self._get_f()(*[assignment[str(v)] for v in self._symbols])
        except KeyError as e:
            raise RuntimeError(("The assignment {a} contains no value for variable {v} [node {n} ]"
                  .format(a=assignment, v=e.args[0], n=self)))

    def __repr__(self):
        return "T(id: {}, expression: {})".format(self.node_id, self.expression)

    def is_terminal(self):
        return True


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

    def is_terminal(self):
        return False


class DefaultCache(object):
    def __init__(self, calculator):
        """
        :param callable calculator:
        """
        self._cache = dict()
        self._calculator = calculator
        self.hits = 0
        self.misses = 0

    def get(self, pool, key):
        """
        :param Pool pool: The pool this cache is used for
        :param key: The key to compute a value for
        :return: The value
        """
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        else:
            self.misses += 1
            value = self._calculator(pool, key)
            self._cache[key] = value
            return value

    def contains(self, key):
        """
        :param key: The key to lookup
        :return: True if there is a value cached for the given key, False otherwise
        """
        return key in self._cache

    def clear(self):
        """
        Clears the cache
        """
        self._cache = dict()
        self.hits = 0
        self.misses = 0


class Ordering(object):
    def test_smaller_eq(self, test_id1, test1, test_id2, test2):
        raise NotImplementedError()


class Pool:
    def __init__(self, empty=False, ordering=None):
        self._counter = 1
        self._nodes = dict()
        self._internal_map = dict()
        self._expressions = dict()
        self._tests = dict()
        self.vars = dict()
        self.caches = dict()
        self._apply_cache = dict()
        self._ordering = ordering
        if not empty:
            self.zero_id = self.terminal("0")
            self.one_id = self.terminal("1")
            self.pos_inf_id = self.terminal(sympy.oo)
            self.neg_inf_id = self.terminal(-sympy.oo)

        self.add_cache("diagram", DefaultCache(lambda pool, node_id: Diagram(pool, pool.get_node(node_id))))

    def has_cache(self, name):
        return name in self.caches

    def add_cache(self, name, cache):
        if name in self.caches:
            raise RuntimeError("There is already a cache with name {}".format(name))
        self.caches[name] = cache

    def get_cached(self, name, key):
        return self.caches[name].get(self, key)

    def is_cached(self, name, key):
        return self.caches[name].contains(key)

    def _get_test_id(self, test):
        return self._tests.get(test, None)

    def _add_test(self, test):
        if test not in self._tests:
            self._tests[test] = len(self._tests)
        return self._tests[test]

    def get_node(self, node_id):
        """
        Returns the node object associated with the given node_id
        :param int node_id: The node id
        :return: Node The node object
        """
        check_node_id(node_id)
        if node_id in self._nodes:
            return self._nodes.get(node_id)
        else:
            raise RuntimeError("No node in pool with id {}.".format(node_id))

    def int_var(self, *args):
        for name in args:
            self.vars[str(name)] = "int"

    def bool_var(self, *args):
        for name in args:
            self.vars[str(name)] = "bool"

    def add_var(self, name, v_type):
        mapping = {
            "int": self.int_var,
            "bool": self.bool_var,
        }
        if v_type not in mapping:
            raise RuntimeError("Type {} is not supported".format(v_type))
        else:
            mapping[v_type](name)

    def get_var_type(self, name):
        return self.vars[str(name)]

    def terminal(self, expression, v_type=None):
        """
        :type expression: sympy.Basic|str|int
        :type v_type: None|str
        :rtype: int
        """
        if not isinstance(expression, sympy.Basic):
            expression = sympy.sympify(expression)
        if expression in self._expressions:
            return self._expressions[expression]
        for var in expression.free_symbols:
            if str(var) not in self.vars:
                if v_type is None:
                    raise RuntimeError("Variable {} not declared".format(var))
                else:
                    self.add_var(var, v_type)
        node_id = self._register(lambda n_id: TerminalNode(n_id, expression))
        self._expressions[expression] = node_id
        return node_id

    def internal(self, test, child_true, child_false, v_type=None):
        """
        Creates an internal "test" node and returns its id
        :type test: Test
        :type child_true: int
        :type child_false: int
        :type v_type: None|str
        :rtype: int
        """
        # type: (Test, int, int, None|str) -> int
        check_node_id(child_true, "Child (true)")
        check_node_id(child_false, "Child (false)")
        if child_true == child_false:
            return child_true

        test, child_true, child_false = test.to_canonical(child_true, child_false)
        test_id = self._add_test(test)
        key = (test_id, child_true, child_false)
        node_id = self._internal_map.get(key, None)
        for var in test.variables:
            if str(var) not in self.vars:
                if v_type is None:
                    raise RuntimeError("Variable {} not declared".format(var))
                else:
                    self.add_var(var, v_type)
        if node_id is None:
            node_id = self._register(lambda n_id: InternalNode(n_id, test, child_true, child_false))
            self._internal_map[key] = node_id
        return node_id

    def bool_test(self, test, v_type=None):
        """
        Creates an internal child node with child nodes 1 and 0 for the true and false branch, respectively
        :type test: Test
        :type v_type: None|str
        :rtype: int
        """
        return self.internal(test, self.one_id, self.zero_id, v_type=v_type)

    def _register(self, constructor):
        node_id = self._counter
        self._counter += 1
        self._nodes[node_id] = constructor(node_id)
        return node_id

    def apply(self, operation, root1, root2):
        """
        :type operation: pyxadd.operation.Operation
        :type root1: int
        :type root2: int
        :rtype: int
        """
        key = (operation, root1, root2)
        if key in self._apply_cache:
            return self._apply_cache[key]

        node1 = self.get_node(root1)
        node2 = self.get_node(root2)

        result = operation.compute_terminal(self, node1, node2)

        if result is None:
            # Find minimal node (or only internal node)
            if isinstance(node1, InternalNode):
                if isinstance(node2, InternalNode):
                    if self.test_smaller_eq(node1.test, node2.test):
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

        self._apply_cache[key] = result
        return result

    def test_smaller_eq(self, test1, test2):
        test_id1 = self._get_test_id(test1)
        test_id2 = self._get_test_id(test2)
        if self._ordering is None:
            return test_id1 <= test_id2
        else:
            return self._ordering.test_smaller_eg(test_id1, test1, test_id2, test2)

    def invert(self, node_id):
        """
        Performs a logical inversion on the diagram
        :type node_id: int
        :rtype: int
        """
        warnings.warn("This method is slowish and does not check for non-boolean leaves.", DeprecationWarning)
        minus_one = self.terminal("-1")
        return self.apply(Multiplication, self.apply(Summation, node_id, minus_one), minus_one)

    def diagram(self, node_id):
        """
        :type node_id: int
        :rtype: Diagram
        """
        return self.get_cached("diagram", node_id)

    @staticmethod
    def to_json(pool):
        """
        Serializes this pool object and returns a JSON string representation that contains:
        1) Variables
        2) Tests
        3) Expressions
        4) Nodes
        :type pool: Pool
        :rtype: string
        """
        import json
        representation = {
            "vars": [(name, kind) for name, kind in pool.vars.items()],
            "tests": [(Test.export_test(test), test_id) for test, test_id in pool._tests.items()],
            "expressions": [(str(expression), exp_id) for expression, exp_id in pool._expressions.items()],
            "nodes": [(key, node_id) for key, node_id in pool._internal_map.items()],
        }
        import json
        return json.dumps(representation)

    @staticmethod
    def from_json(json_string):
        import json
        representation = json.loads(json_string)
        pool = Pool()
        for name, kind in representation["vars"]:
            pool.add_var(name, kind)
        tests = [(Test.import_test(test_string), test_id) for test_string, test_id in representation["tests"]]
        tests = [t[0] for t in sorted(tests, key=lambda p: p[1])]

        nodes = representation["nodes"] + representation["expressions"]
        nodes = [t[0] for t in sorted(nodes, key=lambda p: p[1])]

        for node in nodes:
            if isinstance(node, list):
                test_id, high, low = node
                pool.internal(tests[test_id], high, low)
            else:
                pool.terminal(node)
        return pool


class Diagram:
    def __init__(self, pool, root_node):
        self._pool = pool
        if isinstance(root_node, Node):
            self._root_node = root_node
        elif isinstance(root_node, (int, long)):
            self._root_node = pool.get_node(root_node)
        else:
            raise RuntimeError("Unexpected root node {} of type {}".format(root_node, type(root_node)))
        self._profile = None

    @property
    def root_node(self):
        """
        :rtype: Node
        """
        return self._root_node

    @property
    def root_id(self):
        """
        Returns the id of the root node of this diagram
        :rtype: int
        """
        return self.root_node.node_id

    @property
    def pool(self):
        """
        Returns the pool of this diagram
        :rtype: Pool
        """
        return self._pool

    @property
    def profile(self):
        """
        Returns the profile of this diagram, creates a profile if none exists
        :rtype: pyxadd.walk.WalkingProfile
        """
        from pyxadd.walk import WalkingProfile
        if self._profile is None:
            self._profile = WalkingProfile(self)
        return self._profile

    def node(self, node_id):
        """
        Returns the node associated with the given node_id in the pool of this diagram
        :type node_id: int
        :rtype: Node
        """
        return self._pool.get_node(node_id)

    def evaluate(self, assignment):
        assignment = {str(k): v for k, v in assignment.items()}
        node = self.root_node

        while True:
            if isinstance(node, InternalNode):
                if node.test.evaluate(assignment):  # node.test.operator.test(node.test.expression.subs(assignment), 0):
                    node = self.node(node.child_true)
                else:
                    node = self.node(node.child_false)
            elif isinstance(node, TerminalNode):
                return node.evaluate(assignment)
            else:
                raise RuntimeError("Unexpected node type {} of node {}".format(type(node), node))

    def reduce(self, variables=None, method="linear"):
        if method == "linear":
            from pyxadd.reduce import LinearReduction
            reducer = LinearReduction(self.pool)
        elif method == "smt":
            from pyxadd.reduce import SmtReduce
            reducer = SmtReduce(self.pool)
        else:
            raise RuntimeError("Unknown reduction method {} (valid options are 'linear' or 'smt')".format(method))

        return Diagram(self.pool, reducer.reduce(self.root_node.node_id, variables))

    def __invert__(self):
        return Diagram(self.pool, self.pool.invert(self.root_node.node_id))

    def __add__(self, other):
        if not isinstance(other, Diagram):
            raise TypeError("Cannot sum diagram with {}".format(type(other)))
        if self.pool != other.pool:
            raise RuntimeError("Can only add diagrams from the same pool")
        return Diagram(self.pool, self.pool.apply(Summation, self.root_node.node_id, other.root_node.node_id))

    def __sub__(self, other):
        if not isinstance(other, Diagram):
            raise TypeError("Cannot subtract {} from diagram".format(type(other)))
        if self.pool != other.pool:
            raise RuntimeError("Can only substract diagrams from the same pool")
        minus_one = self.pool.terminal("-1")
        return self + Diagram(self.pool, self.pool.apply(Multiplication, minus_one, other.root_node.node_id))

    def __mul__(self, other):
        if not isinstance(other, Diagram):
            raise TypeError("Cannot multiply diagram with {}".format(type(other)))
        if self.pool != other.pool:
            raise RuntimeError("Can only multiply diagrams from the same pool")
        return Diagram(self.pool, self.pool.apply(Multiplication, self.root_node.node_id, other.root_node.node_id))

    def __or__(self, other):
        if not isinstance(other, Diagram):
            raise TypeError("Cannot perform or on diagram with {}".format(type(other)))
        if self.pool != other.pool:
            raise RuntimeError("Can only operate on diagrams from the same pool")
        return Diagram(self.pool, self.pool.apply(LogicalOr, self.root_node.node_id, other.root_node.node_id))

    def __and__(self, other):
        if not isinstance(other, Diagram):
            raise TypeError("Cannot perform and on diagram with {}".format(type(other)))
        if self.pool != other.pool:
            raise RuntimeError("Can only operate on diagrams from the same pool")
        return Diagram(self.pool, self.pool.apply(LogicalAnd, self.root_node.node_id, other.root_node.node_id))

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
                operator = match.group(2)
                tests[int(parts[1])] = LinearTest(expression, operator)
            elif parts[0] == "I":
                pool.register_node(InternalNode(int(parts[1]), tests[int(parts[2])], int(parts[3]), int(parts[4])))
