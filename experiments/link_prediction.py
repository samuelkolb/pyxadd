import os

import sklearn as sk
from experiments import pagerank
from pyxadd.build import Builder
from pyxadd.diagram import Pool
from pyxadd.matrix.matrix import Matrix
from tests.export import Exporter

pool = Pool()
build = Builder(pool)
size = 8
variables = [("i", 1, size)]
for var in variables:
    name = var[0]
    build.ints("r_{}".format(name), "c_{}".format(name), name)
limits = build.limit("r_i", 1, size) & build.limit("c_i", 1, size)
block1 = (build.limit("r_i", 1, size / 2) & build.limit("c_i", 1, size / 2))
block2 = (build.limit("r_i", size * 3 / 4 + 1, size) & build.limit("c_i", 1, size / 2))
block3 = (build.limit("r_i", size / 4 + 1, size * 3 / 4) & build.limit("c_i", size / 4 + 1, size))
diagram = limits & (block1 | block2 | block3)
row_variables = [("r_" + name, lb, ub) for name, lb, ub in variables]
column_variables = [("c_" + name, lb, ub) for name, lb, ub in variables]
matrix = Matrix(diagram, row_variables, column_variables).reduce()

# exporter = Exporter(os.path.join(os.path.dirname(os.path.realpath(__file__)), "visual"), "link_prediction")
# exporter.export(matrix, "matrix")
# matrix.print_ground()

projected = matrix.project(False)
# projected.print_ground()

inverted = projected.transform_leaves(lambda terminal, d: diagram.pool.terminal(1.0 / terminal.expression))
# inverted.print_ground()

stochastic = matrix.element_product(inverted)
# stochastic.print_ground()

dampened = pagerank.dampen(stochastic, 0.85)
# dampened.print_ground()

result, iterations = pagerank.page_rank(dampened, variables)
print("{} iterations".format(iterations))
# result.print_ground()
