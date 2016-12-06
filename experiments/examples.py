from pyxadd.build import Builder
from pyxadd.diagram import Pool
from pyxadd.view import export

pool = Pool()
b = Builder(pool)
b.ints("r", "c")

xadd_1 = b.terminal("5 + r * c")
export(xadd_1, "visual/examples/xadd_1.dot")

xadd_2 = b.limit("r", 1, 2) & b.limit("c", 1, 2) * xadd_1
export(xadd_2, "visual/examples/xadd_2.dot")
