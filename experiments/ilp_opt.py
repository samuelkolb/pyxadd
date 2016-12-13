import picos
from pyxadd.build import Builder
from pyxadd.diagram import Pool, InternalNode, TerminalNode
from pyxadd.optimization import ParentsWalker
from pyxadd.sympy_conversion import SympyConverter
from pyxadd.view import export

pool = Pool()
b = Builder(pool)
b.ints("x")
d = b.ite(b.test("x", "<=", 3), b.ite(b.test("x", "<=", 2), b.exp(1), b.exp("2*x")), b.exp(1))
export(d, "visual/ilp_opt/d.dot")


def get_ilp_var(number, branch=None):
    if isinstance(pool.get_node(number), InternalNode):
        return "i{}{}".format(number, "t" if branch else "f")
    else:
        return "t{}".format(number)

problem = picos.Problem()
M = 1000000

parent_map = ParentsWalker(d, d.profile).walk()
variables = dict()


class PicosConverter(SympyConverter):
    def custom(self, value):
        raise RuntimeError("Could not convert {} of type {}".format(value, type(value)))

    def int(self, value):
        return value

    def times(self, val1, val2):
        return val1 * val2

    def float(self, value):
        return value

    def symbol(self, value):
        if value not in variables:
            variables[value] = problem.add_variable(value, "integer")
        return variables[value]

    def add(self, val1, val2):
        return val1 + val2

converter = PicosConverter()

for node_id in parent_map:
    node = pool.get_node(node_id)
    if isinstance(node, TerminalNode):
        variables[node_id] = problem.add_variable("t{}".format(node_id), vtype="binary")
    else:
        variables[(node_id, True)] = problem.add_variable("i{}t".format(node_id), vtype="binary")
        variables[(node_id, False)] = problem.add_variable("i{}f".format(node_id), vtype="binary")

s = problem.add_variable("s", vtype="integer")
problem.set_objective("max", s)

for node_id, parents in parent_map.items():
    if len(parents) > 0:
        parent_sum = 0
        for i in range(0, len(parents)):
            parent_sum += variables[parents[i]]
    else:
        parent_sum = 1

    print("Node {} has parents {} = ".format(node_id, parents, parent_sum))

    # parent_sum = " + ".join([get_ilp_var(p, b) for p, b in parents]) if len(parents) > 0 else 1

    if node_id in variables:
        problem.add_constraint(variables[node_id] == parent_sum)
    else:
        problem.add_constraint(variables[(node_id, True)] + variables[(node_id, False)] == parent_sum)

for node_id in parent_map:
    node = pool.get_node(node_id)
    if isinstance(node, InternalNode):
        true_test = node.test.operator
        lhs = 0
        for var, coefficient in true_test.lhs.items():
            if var not in variables:
                variables[var] = problem.add_variable(var, vtype="integer")
            lhs += variables[var] * coefficient
        problem.add_constraint(lhs < (1 - variables[(node_id, True)]) * M + true_test.rhs)

        false_test = (~node.test.operator).to_canonical()
        lhs = 0
        for var, coefficient in false_test.lhs.items():
            if var not in variables:
                variables[var] = problem.add_variable(var, vtype="integer")
            lhs += variables[var] * coefficient
        problem.add_constraint(lhs < (1 - variables[(node_id, False)]) * M + false_test.rhs)
    elif isinstance(node, TerminalNode):
        problem.add_constraint(s < (1 - variables[node_id]) * M + converter.convert(node.expression))

# print("Maximize: s".format(" + ".join(objective)))
# print("Subject to:")
# print("\n".join(["\t{}".format(c) for c in constraints]))

problem.solve(verbose=False)
print(problem)
print(", ".join(["{}: {}".format(v.name, v.value[0, 0]) for k, v in variables.items()]))
print("Optimal value: {}".format(s.value[0, 0]))

"""
max (r if t7 else 0):

max s
st.
    // restrict to r if t7, else "unbounded"
    s <= r + (1 - t7) * M
    s >= r - (1 - t7) * M
    // restrict to 0 if !t7, else "unbounded"
    s <= 0 + t7 * M
    s >= 0 - t7 * M
"""
