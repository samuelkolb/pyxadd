import picos
from pyxadd.diagram import TerminalNode, InternalNode
from pyxadd.sympy_conversion import SympyConverter
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

