from experiments.pagerank import pagerank
from pyxadd.order import order
from pyxadd.diagram import Pool, Ordering
from pyxadd.matrix.matrix import Matrix

size = 500000
pool_file = "pool_{}.txt".format(size)
root_id = 1663

variables = [('f0', 0, 1658), ('f1', 0, 964)]
row_vars = [('r_f0', 0, 1658), ('r_f1', 0, 964)]
col_vars = [('c_f0', 0, 1658), ('c_f1', 0, 964)]

with open(pool_file, "r") as stream:
    json_input = stream.readline()

exported_pool = Pool.from_json(json_input)
diagram = exported_pool.diagram(root_id)

matrix = Matrix(diagram, row_vars, col_vars)
matrix.export("test.dot")


class CustomOrdering(Ordering):
    def test_smaller_eq(self, test_id1, test1, test_id2, test2):
        pass


exported_pool.change_order(CustomOrdering())
reordered_matrix = Matrix(order(diagram), row_vars, col_vars)

pagerank(matrix, 0.85, variables, iterations=2, delta=0, norm=1)
