import picos
from pyxadd.diagram import TerminalNode, InternalNode
from pyxadd.operation import Operation, LogicalOr, LogicalAnd
from pyxadd.sympy_conversion import SympyConverter
from pyxadd.test import LinearTest
from pyxadd.view import export
from pyxadd.walk import BottomUpWalker, walk_leaves, DepthFirstWalker


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


class PicosConverter(SympyConverter):
    def __init__(self, registration):
        self.registration = registration

    def custom(self, value):
        raise RuntimeError("Could not convert {} of type {}".format(value, type(value)))

    def int(self, value):
        return value

    def times_batch(self, values):
        return super(PicosConverter, self).times_batch([1] + values)

    def times(self, val1, val2):
        return val1 * val2

    def float(self, value):
        return value

    def symbol(self, value):
        return self.registration(value)

    def add_batch(self, values):
        return super(PicosConverter, self).add_batch([0] + values)

    def add(self, val1, val2):
        return val1 + val2


def find_optimum(diagram, maximize=True, m=1000000):
    problem = picos.Problem()
    parent_map = ParentsWalker(diagram).walk()
    variables = dict()
    assignment_variables = []

    def registration(value, v_type="integer"):
        if value not in variables:
            variables[value] = problem.add_variable(value, size=1, vtype=v_type)
            assignment_variables.append(value)
        return variables[value]

    converter = PicosConverter(registration)

    for node_id in parent_map:
        node = diagram.pool.get_node(node_id)
        if isinstance(node, TerminalNode):
            variables[node_id] = problem.add_variable("t{}".format(node_id), vtype="binary")
        elif isinstance(node, InternalNode):
            variables[(node_id, True)] = problem.add_variable("i{}t".format(node_id), vtype="binary")
            variables[(node_id, False)] = problem.add_variable("i{}f".format(node_id), vtype="binary")
        else:
            raise RuntimeError("Unknown node {} of type {}".format(node, type(node)))

    s = problem.add_variable("s", vtype="integer")
    problem.set_objective("max" if maximize else "min", s)

    for node_id, parents in parent_map.items():
        if len(parents) > 0:
            parent_sum = 0
            for i in range(0, len(parents)):
                parent_sum += variables[parents[i]]
        else:
            parent_sum = 1

        if node_id in variables:
            problem.add_constraint(variables[node_id] == parent_sum)
        else:
            problem.add_constraint(variables[(node_id, True)] + variables[(node_id, False)] == parent_sum)

    for node_id in parent_map:
        node = diagram.pool.get_node(node_id)
        if isinstance(node, InternalNode):
            true_test = node.test.operator
            lhs = 0
            for var, coefficient in true_test.lhs.items():
                lhs += registration(var) * coefficient
            problem.add_constraint(lhs < (1 - variables[(node_id, True)]) * m + true_test.rhs)

            false_test = (~node.test.operator).to_canonical()
            lhs = 0
            for var, coefficient in false_test.lhs.items():
                lhs += registration(var) * coefficient
            problem.add_constraint(lhs < (1 - variables[(node_id, False)]) * m + false_test.rhs)
        elif isinstance(node, TerminalNode):
            problem.add_constraint(s < (1 - variables[node_id]) * m + converter.convert(node.expression))

    problem.solve(verbose=False)
    return s.value[0, 0], {v: variables[v].value[0, 0] for v in assignment_variables}


class NodeFinder(BottomUpWalker):
    def __init__(self, node_id, diagram, profile=None):
        BottomUpWalker.__init__(self, diagram, profile)
        self.node_id = node_id

    def visit_terminal(self, terminal_node):
        if terminal_node.node_id == self.node_id:
            return self.diagram.pool.terminal(1)
        else:
            return self.diagram.pool.terminal(0)

    def visit_internal(self, internal_node, true_message, false_message):
        pool = self.diagram.pool
        one = pool.terminal(1)
        zero = pool.terminal(0)
        if internal_node.node_id == self.node_id:
            return one
        return pool.apply(LogicalOr,
                          pool.apply(LogicalAnd, pool.internal(internal_node.test, one, zero), true_message),
                          pool.apply(LogicalAnd, pool.internal(internal_node.test, zero, one), false_message))


class IntervalFinder(DepthFirstWalker):
    def __init__(self, variables, diagram):
        DepthFirstWalker.__init__(self, diagram)
        self.variables = variables
        self.bounds = []

    def visit_internal(self, internal_node, parent_message):
        if parent_message is None:
            parent_message = self.variables
        test = internal_node.test
        if isinstance(test, LinearTest):
            true_message = [(var,) + test.update_bounds(var, lb, ub, True) for var, lb, ub in parent_message]
            false_message = [(var,) + test.update_bounds(var, lb, ub, False) for var, lb, ub in parent_message]
        else:
            raise RuntimeError("Currently non-linear tests (e.g. boolean tests) are unsupported")
        return true_message, false_message

    def visit_terminal(self, terminal_node, parent_message):
        """
        :type terminal_node: TerminalNode
        :type parent_message: tuple[]
        """
        if parent_message is None:
            parent_message = self.variables
        if terminal_node.evaluate({}) == 1:
            self.bounds.append(parent_message)

    def walk(self):
        DepthFirstWalker.walk(self)
        return self.bounds


def find_optimal_conditions(diagram, variables, maximize=True):
    def better(value1, value2):
        if maximize:
            return value1 > value2
        else:
            return value1 < value2

    class Optimal:
        value = None
        node_id = None

    def leaf_walker(pool, node):
        value = node.evaluate({})
        if Optimal.value is None or better(value, Optimal.value):
            Optimal.value = value
            Optimal.node_id = node.node_id

    walk_leaves(leaf_walker, diagram)
    path_diagram = diagram.pool.diagram(NodeFinder(Optimal.node_id, diagram).walk())
    return IntervalFinder(variables, path_diagram).walk()

